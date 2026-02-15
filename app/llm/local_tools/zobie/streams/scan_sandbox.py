# FILE: app/llm/local_tools/zobie/streams/scan_sandbox.py
"""SCAN SANDBOX stream generator (scope="sandbox") - DB + RAG chunks + embeddings.

Extracted from zobie_tools.py for modularity.
No logic changes - exact same behavior and SSE output format.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from sqlalchemy.orm import Session

from app.llm.audit_logger import RoutingTrace
from app.memory import schemas as memory_schemas
from app.memory import service as memory_service

from ..config import (
    SANDBOX_CONTROLLER_URL,
    SANDBOX_SCAN_ROOTS,
    SANDBOX_CONTENT_BATCH_SIZE,
    SANDBOX_MAX_CONTENT_SIZE,
    SANDBOX_CONTENT_EXTENSIONS,
    SANDBOX_SKIP_FILENAMES,
    SANDBOX_SKIP_PATTERNS,
)
from ..sse import sse_token, sse_error, sse_done
from ..sandbox_client import call_fs_tree, call_fs_contents
from ..db_ops import (
    ARCH_MODELS_AVAILABLE,
    save_scan_incremental_to_db,
    count_files_by_zone,
)
from ..filter_utils import filter_scan_results
from ..rag_helpers import signatures_to_db

logger = logging.getLogger(__name__)


async def generate_sandbox_structure_scan_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """
    Scan sandbox environment (C:\\Users + D:\\ areas) and save to DB.
    
    v4.3: FULL RAG INGEST
    - Scans file tree (incremental: mtime+size)
    - Fetches contents for NEW/CHANGED code files only
    - Extracts signatures → creates ArchCodeChunk entries
    - Queues background embedding job
    - NO .architecture outputs (DB only)
    """
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    
    yield sse_token("🔍 [SCAN_SANDBOX] Scanning sandbox environment...\n")
    yield sse_token(f"📡 Controller: {SANDBOX_CONTROLLER_URL}\n")
    yield sse_token(f"📂 Roots: {', '.join(SANDBOX_SCAN_ROOTS)}\n\n")
    
    # Check if models available
    if not ARCH_MODELS_AVAILABLE:
        yield sse_error(
            "Architecture models not available. "
            "Run: alembic upgrade head to create tables."
        )
        yield sse_done(
            provider="local",
            model="sandbox_scanner",
            success=False,
            error="models_not_available",
        )
        return
    
    # ==========================================================================
    # Phase 1: Scan file tree
    # ==========================================================================
    yield sse_token("📊 Phase 1: Scanning file tree...\n")
    
    status, data, error_msg = await loop.run_in_executor(
        None,
        lambda: call_fs_tree(SANDBOX_SCAN_ROOTS, max_files=200000),
    )
    
    if status != 200 or data is None:
        logger.error(f"[SCAN_SANDBOX] Failed: status={status}, error={error_msg}")
        
        if status == 404:
            yield sse_error(
                f"Sandbox controller /fs/tree not found at {SANDBOX_CONTROLLER_URL}\n"
                f"Please update sandbox_controller to v0.3.0 or later."
            )
        elif status is None:
            yield sse_error(
                f"Could not connect to sandbox controller at {SANDBOX_CONTROLLER_URL}\n"
                f"Error: {error_msg}\n"
                f"Is the sandbox running?"
            )
        else:
            yield sse_error(f"Scan failed (status={status}): {error_msg}")
        
        yield sse_done(
            provider="local",
            model="sandbox_scanner",
            success=False,
            error=f"status={status}",
        )
        return
    
    # Extract file data
    raw_files_data = data.get("files", [])
    roots_scanned = data.get("roots_scanned", SANDBOX_SCAN_ROOTS)
    scan_time_ms = data.get("scan_time_ms", 0)
    truncated = data.get("truncated", False)
    
    yield sse_token(f"   Found {len(raw_files_data)} files in {scan_time_ms}ms\n")
    
    # v4.2: Apply exclusion filtering
    files_data, excluded_count = filter_scan_results(raw_files_data)
    yield sse_token(f"   Excluded {excluded_count} junk files (caches, node_modules, binaries, etc.)\n")
    yield sse_token(f"   Keeping {len(files_data)} relevant files\n")
    
    if truncated:
        yield sse_token("   ⚠️ Results truncated (max files limit reached)\n")
    
    # ==========================================================================
    # Phase 2: Save file metadata to DB (incremental)
    # ==========================================================================
    yield sse_token("\n💾 Phase 2: Saving metadata to DB (incremental)...\n")
    
    scan_id = None
    stats = {}
    
    try:
        scan_id, stats = save_scan_incremental_to_db(
            db=db,
            scope="sandbox",
            files_data=files_data,
            roots_scanned=roots_scanned,
            scan_time_ms=scan_time_ms,
        )
        
        if scan_id:
            zone_counts = count_files_by_zone(db, scan_id) if count_files_by_zone else {}
            
            yield sse_token(f"   ✅ Metadata saved (scan_id={scan_id})\n")
            yield sse_token(f"   New: {stats.get('new_files', 0)} | Updated: {stats.get('updated_files', 0)} | Unchanged: {stats.get('unchanged_files', 0)}\n")
            
            # Log to console for visibility
            logger.info(f"[SCAN_SANDBOX] scanned_files={len(files_data)}")
            logger.info(f"[SCAN_SANDBOX] db_upserts={scan_id}")
        else:
            yield sse_token("   ⚠️ Could not save to DB (models not available)\n")
            yield sse_done(
                provider="local",
                model="sandbox_scanner",
                success=False,
                error="models_not_available",
            )
            return
            
    except Exception as e:
        logger.exception(f"[SCAN_SANDBOX] DB save failed: {e}")
        yield sse_error(f"Failed to save to DB: {e}")
        yield sse_done(
            provider="local",
            model="sandbox_scanner",
            success=False,
            error=str(e),
        )
        return
    
    # ==========================================================================
    # Phase 3: Fetch contents for NEW/CHANGED files only (incremental)
    # ==========================================================================
    yield sse_token("\n📖 Phase 3: Fetching contents (incremental)...\n")
    
    # Build set of paths that need content fetch (only new or changed)
    # Use incremental logic: only NEW files and UPDATED files need content fetch
    new_count = stats.get("new_files", 0)
    updated_count = stats.get("updated_files", 0)
    
    if new_count == 0 and updated_count == 0:
        yield sse_token("   ⏭️ No new/changed files - skipping content fetch\n")
        paths_to_read = []
        eligible_paths = []  # For logging
        # Required logging for acceptance test (no changes = no fetch needed)
        yield sse_token(f"   [SCAN_SANDBOX] eligible_for_content=0\n")
        yield sse_token(f"   [SCAN_SANDBOX] incremental_fetch_count=0\n")
    else:
        # v4.4: TRUE INCREMENTAL content fetch
        # Build set of changed paths from Phase 2 (new + updated files only)
        changed_set = set(stats.get("new_paths", [])) | set(stats.get("updated_paths", []))
        
        yield sse_token(f"   {new_count} new + {updated_count} changed files detected\n")
        
        # Build list of content-eligible files (apply size/extension/secret filters)
        eligible_paths = []
        for f in files_data:
            ext = (f.get("ext") or "").lower()
            size = f.get("size_bytes") or 0
            name = (f.get("name") or "").lower()
            path = f.get("path", "")
            
            # v4.3: Hard size cap (1MB) for safety
            if size > SANDBOX_MAX_CONTENT_SIZE:
                continue
            
            # Skip sensitive files
            if name in SANDBOX_SKIP_FILENAMES:
                continue
            
            # Skip files with sensitive patterns in name
            if any(p in name for p in SANDBOX_SKIP_PATTERNS):
                continue
            
            # Include by extension
            if ext in SANDBOX_CONTENT_EXTENSIONS:
                eligible_paths.append(path)
                continue
            
            # Include special files without matching extension
            if name in (".gitignore", ".gitattributes", "dockerfile", "makefile"):
                eligible_paths.append(path)
        
        # v4.4: INCREMENTAL - Only fetch paths that are BOTH eligible AND changed
        paths_to_read = [p for p in eligible_paths if p in changed_set]
        
        # Required logging for acceptance test
        yield sse_token(f"   [SCAN_SANDBOX] eligible_for_content={len(eligible_paths)}\n")
        yield sse_token(f"   [SCAN_SANDBOX] incremental_fetch_count={len(paths_to_read)}\n")
    
    # Fetch contents in batches (v4.3: batch_size=25 for sandbox scale)
    contents_data: List[Dict[str, Any]] = []
    
    if paths_to_read:
        batch_size = SANDBOX_CONTENT_BATCH_SIZE  # Default 25
        total_batches = (len(paths_to_read) + batch_size - 1) // batch_size
        
        # Required logging for acceptance test
        yield sse_token(f"   [SCAN_SANDBOX] batches={total_batches}\n")
        
        for i in range(0, len(paths_to_read), batch_size):
            batch_paths = paths_to_read[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            status, resp, error_msg = await loop.run_in_executor(
                None,
                lambda bp=batch_paths: call_fs_contents(
                    bp, 
                    max_file_size=SANDBOX_MAX_CONTENT_SIZE,
                    include_line_numbers=False,  # Don't need line numbers for signatures
                ),
            )
            
            if status == 200 and resp:
                batch_files = resp.get("files", [])
                contents_data.extend(batch_files)
                
                if batch_num % 10 == 0 or batch_num == total_batches:
                    yield sse_token(f"   Batch {batch_num}/{total_batches}: {len(contents_data)} files fetched\n")
            else:
                yield sse_token(f"   Batch {batch_num}: Failed - {error_msg}\n")
        
        files_with_content = sum(1 for f in contents_data if f.get("content") and not f.get("error"))
        yield sse_token(f"   ✅ Fetched {files_with_content} files with content\n")
        
        # v4.5: Self-report incremental fetch file list
        yield sse_token(f"   [SCAN_SANDBOX] incremental_fetch_files={len(paths_to_read)}\n")
        # Build lookup of content fetch results
        content_results = {c.get("path", ""): c for c in contents_data}
        
        yield sse_token("   [SCAN_SANDBOX] incremental_fetch_list:\n")
        max_to_print = 50
        for idx, path in enumerate(paths_to_read[:max_to_print]):
            ext = os.path.splitext(path)[1] or "(no ext)"
            result = content_results.get(path, {})
            has_content = "yes" if (result.get("content") and not result.get("error")) else "no"
            yield sse_token(f"    - {path} ({ext}) content={has_content}\n")
        
        if len(paths_to_read) > max_to_print:
            yield sse_token(f"    … ({len(paths_to_read) - max_to_print} more)\n")
    else:
        # No paths to fetch - log batches=0 for acceptance test
        yield sse_token(f"   [SCAN_SANDBOX] batches=0\n")
        yield sse_token(f"   [SCAN_SANDBOX] incremental_fetch_files=0\n")
        yield sse_token("   [SCAN_SANDBOX] incremental_fetch_list: <none>\n")
    
    # ==========================================================================
    # Phase 4: Extract signatures → ArchCodeChunk entries (DEDUP + FAST)
    # ==========================================================================
    yield sse_token("\n🔗 Phase 4: Extracting signatures for RAG...\n")
    
    chunks_created = 0
    rag_scan_id = None
    
    if contents_data:
        try:
            rag_scan_id = signatures_to_db(db, contents_data, SANDBOX_SCAN_ROOTS[0])
            if rag_scan_id:
                from app.rag.models import ArchCodeChunk
                chunks_created = db.query(ArchCodeChunk).filter(
                    ArchCodeChunk.scan_id == rag_scan_id
                ).count()
                yield sse_token(f"   ✅ {chunks_created} code chunks written (rag_scan_id={rag_scan_id})\n")
                logger.info(f"[SCAN_SANDBOX] chunks_written={chunks_created}")
            else:
                yield sse_token("   ⚠️ Signature DB write returned None (check logs)\n")
        except Exception as e:
            logger.exception(f"[SCAN_SANDBOX] Signature extraction failed: {e}")
            yield sse_token(f"   ⚠️ Signature extraction failed (non-fatal): {e}\n")
    else:
        yield sse_token("   ⏭️ No content to process - skipping signature extraction\n")
    
    # ==========================================================================
    # Phase 5: Queue background embedding job
    # ==========================================================================
    yield sse_token("\n🚀 Phase 5: Queueing embedding job...\n")
    
    embedding_queued = False
    
    if chunks_created > 0:
        try:
            from app.rag.jobs.embedding_job import queue_embedding_job, EMBEDDING_AUTO_ENABLED
            from app.db import get_db_session
            
            if not EMBEDDING_AUTO_ENABLED:
                yield sse_token("   ⚠️ Auto-embedding disabled (ORB_EMBEDDING_AUTO=false)\n")
            else:
                # Queue embedding ONLY for chunks from THIS scan (incremental)
                # This prevents re-embedding all pending chunks from all time
                embedding_queued = queue_embedding_job(
                    db_session_factory=get_db_session,
                    scan_id=rag_scan_id,  # ✅ Embed only chunks from THIS scan
                )
                logger.info(f"[SCAN_SANDBOX] Embedding queued for rag_scan_id={rag_scan_id}")
                
                if embedding_queued:
                    yield sse_token(f"   ✅ Embedding job queued for scan_id={rag_scan_id} (background)\n")
                    yield sse_token("   📊 Priority: Tier1 (routers) → Tier2 (pipeline) → Tier3 (services) → ...\n")
                else:
                    yield sse_token("   ⚠️ Embedding job not queued (may already be running)\n")
                    
        except ImportError as ie:
            logger.warning(f"[SCAN_SANDBOX] Embedding module not available: {ie}")
            yield sse_token(f"   ⚠️ Embedding module not available: {ie}\n")
        except Exception as emb_err:
            logger.warning(f"[SCAN_SANDBOX] Failed to queue embedding job (non-fatal): {emb_err}")
            yield sse_token(f"   ⚠️ Embedding queue failed (non-fatal): {emb_err}\n")
    else:
        yield sse_token("   ⏭️ No chunks to embed\n")
    
    logger.info(f"[SCAN_SANDBOX] embeddings_queued={embedding_queued}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    
    # Record in memory service
    try:
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id,
                role="assistant",
                content=f"[SCAN_SANDBOX] Indexed {len(files_data)} files, {chunks_created} chunks (scan_id={scan_id}, rag_scan_id={rag_scan_id})",
                provider="local",
                model="sandbox_scanner",
            ),
        )
    except Exception:
        pass
    
    duration_ms = int(loop.time() * 1000) - started_ms
    
    if trace:
        trace.log_model_call(
            "local_tool", "local", "sandbox_scanner", "scan_sandbox",
            0, 0, duration_ms, success=True, error=None,
        )
    
    # Final summary with required logging format
    summary = (
        f"\n✅ [SCAN_SANDBOX] Complete\n"
        f"   scanned_files={len(files_data)}\n"
        f"   db_upserts={scan_id}\n"
        f"   chunks_written={chunks_created}\n"
        f"   embeddings_queued={embedding_queued}\n"
        f"   duration={duration_ms}ms\n"
    )
    yield sse_token(summary)
    
    yield sse_done(
        provider="local",
        model="sandbox_scanner",
        total_length=len(files_data),
        meta={
            "scan_id": scan_id,
            "rag_scan_id": rag_scan_id,
            "files": len(files_data),
            "chunks_created": chunks_created,
            "embeddings_queued": embedding_queued,
            "roots": roots_scanned,
            "scope": "sandbox",
            "new_files": stats.get("new_files", 0),
            "updated_files": stats.get("updated_files", 0),
            "unchanged_files": stats.get("unchanged_files", 0),
        },
    )
