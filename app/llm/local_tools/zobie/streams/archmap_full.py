# FILE: app/llm/local_tools/zobie/streams/archmap_full.py
"""CREATE ARCHITECTURE MAP - FULL (ALL CAPS) stream generator.

Scan + out folder + map generation.

Extracted from zobie_tools.py for modularity.
No logic changes - exact same behavior and SSE output format.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from sqlalchemy.orm import Session

from app.llm.audit_logger import RoutingTrace
from app.memory import schemas as memory_schemas
from app.memory import service as memory_service

from app.llm.local_tools.archmap_helpers import (
    ARCHMAP_PROVIDER,
    ARCHMAP_MODEL,
    ARCHMAP_SYSTEM_PROMPT,
    ARCHMAP_SENTINEL,
    ARCHMAP_MAX_CONTINUATION_ROUNDS,
    build_continuation_prompt,
    has_sentinel,
)

# Import stage config for proper max_tokens/timeout
try:
    from app.llm.stage_models import get_archmap_config
    _STAGE_MODELS_AVAILABLE = True
except ImportError:
    _STAGE_MODELS_AVAILABLE = False
    get_archmap_config = None

from ..config import (
    SANDBOX_CONTROLLER_URL,
    CODE_SCAN_ROOTS,
    MAX_CONTENT_FILE_SIZE,
    FULL_ARCHMAP_OUTPUT_DIR,
    FULL_ARCHMAP_OUTPUT_FILE,
    FULL_CODEBASE_OUTPUT_FILE,
)
from ..sse import sse_token, sse_error, sse_done
from ..sandbox_client import call_fs_tree, call_fs_contents
from ..db_ops import save_scan_with_contents_to_db
from ..rag_helpers import (
    generate_signatures_json,
    generate_index_for_rag,
    generate_codebase_md,
)

# Reuse the prompt builder from archmap_db
from .archmap_db import _build_db_archmap_prompt

logger = logging.getLogger(__name__)


async def generate_full_architecture_map_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """
    Full architecture map: Scan D:\\Orb + D:\\orb-desktop, fetch contents, save to out folder.
    
    This is the ALL CAPS "CREATE ARCHITECTURE MAP" command.
    
    Outputs to .architecture/:
    - INDEX.json: File tree metadata
    - CODEBASE.md: All source code with line numbers (for Claude context)
    - ARCHITECTURE_MAP.md: Generated overview (optional, via Opus)
    
    Also saves to DB with full file contents for future RAG.
    """
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    
    # v4.2: Debug - verify output path is correct
    logger.info(f"[full_archmap] Output directory: {FULL_ARCHMAP_OUTPUT_DIR!r}")
    print(f"[full_archmap] Output directory: {FULL_ARCHMAP_OUTPUT_DIR!r}")
    
    # Resolve to absolute path to ensure correctness
    output_dir = Path(FULL_ARCHMAP_OUTPUT_DIR).resolve()
    logger.info(f"[full_archmap] Resolved output path: {output_dir}")
    
    yield sse_token("🔍 FULL ARCHITECTURE SCAN: Capturing codebase...\n")
    yield sse_token(f"📡 Controller: {SANDBOX_CONTROLLER_URL}\n")
    yield sse_token(f"📂 Roots: {', '.join(CODE_SCAN_ROOTS)}\n")
    yield sse_token(f"📤 Output: {output_dir}\n\n")
    
    # ===========================================================================
    # Phase 1: Scan file tree
    # ===========================================================================
    yield sse_token("📊 Phase 1: Scanning file tree...\n")
    
    status, data, error_msg = await loop.run_in_executor(
        None,
        lambda: call_fs_tree(CODE_SCAN_ROOTS, max_files=100000),
    )
    
    if status != 200 or data is None:
        logger.error(f"[full_archmap] Scan failed: status={status}, error={error_msg}")
        yield sse_error(f"Scan failed: {error_msg}")
        yield sse_done(
            provider="local",
            model="architecture_scanner",
            success=False,
            error="scan_failed",
        )
        return
    
    files_data = data.get("files", [])
    scan_time_ms = data.get("scan_time_ms", 0)
    
    yield sse_token(f"   Found {len(files_data)} files in {scan_time_ms}ms\n\n")
    
    # ===========================================================================
    # Phase 2: Fetch file contents
    # ===========================================================================
    yield sse_token("📖 Phase 2: Reading file contents...\n")
    
    # Filter to files that should have content captured
    # Based on extension and size
    content_extensions = {
        ".py", ".pyw", ".pyi",
        ".js", ".mjs", ".cjs", ".jsx",
        ".ts", ".tsx", ".mts", ".cts",
        ".json", ".jsonc",
        ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
        ".html", ".htm", ".css", ".scss", ".sass", ".less",
        ".sql", ".sh", ".bash", ".zsh", ".ps1", ".psm1", ".bat", ".cmd",
        ".md", ".markdown", ".rst", ".txt",
        "",  # Files without extension (like Dockerfile, Makefile)
    }
    
    # Files to NEVER capture (secrets, credentials, keys)
    skip_filenames = {
        ".env", ".env.local", ".env.production", ".env.development",
        ".env.example",  # Often contains real values
        "secrets.json", "credentials.json", "config.secret.json",
        ".npmrc", ".pypirc",  # Can contain auth tokens
        "id_rsa", "id_ed25519", "id_ecdsa",  # SSH keys
        ".pem", ".key", ".crt", ".p12", ".pfx",  # Certificates
    }
    
    # Patterns to skip
    skip_patterns = {"secret", "credential", "password", "token", "apikey", "api_key"}
    
    paths_to_read = []
    for f in files_data:
        ext = (f.get("ext") or "").lower()
        size = f.get("size_bytes") or 0
        name = (f.get("name") or "").lower()
        
        # Skip large files
        if size > MAX_CONTENT_FILE_SIZE:
            continue
        
        # Skip sensitive files
        if name in skip_filenames:
            continue
        
        # Skip files with sensitive patterns in name
        if any(p in name for p in skip_patterns):
            continue
        
        # Include by extension
        if ext in content_extensions:
            paths_to_read.append(f.get("path"))
            continue
        
        # Include special files without matching extension (but not .env)
        if name in (".gitignore", ".gitattributes", "dockerfile", "makefile"):
            paths_to_read.append(f.get("path"))
    
    yield sse_token(f"   Reading {len(paths_to_read)} source files...\n")
    
    # Fetch contents in batches to avoid timeout
    contents_data: List[Dict[str, Any]] = []
    batch_size = 100
    
    for i in range(0, len(paths_to_read), batch_size):
        batch_paths = paths_to_read[i:i + batch_size]
        
        status, resp, error_msg = await loop.run_in_executor(
            None,
            lambda bp=batch_paths: call_fs_contents(bp, include_line_numbers=True),
        )
        
        if status == 200 and resp:
            batch_files = resp.get("files", [])
            contents_data.extend(batch_files)
            
            batch_lines = sum(f.get("line_count", 0) for f in batch_files if not f.get("error"))
            yield sse_token(f"   Batch {i // batch_size + 1}: {len(batch_files)} files, {batch_lines:,} lines\n")
        else:
            yield sse_token(f"   Batch {i // batch_size + 1}: Failed - {error_msg}\n")
    
    # Stats
    total_lines = sum(f.get("line_count", 0) for f in contents_data if not f.get("error"))
    total_bytes = sum(f.get("size_bytes", 0) for f in contents_data if not f.get("error"))
    files_with_content = sum(1 for f in contents_data if f.get("content") and not f.get("error"))
    
    yield sse_token(f"\n   Total: {files_with_content} files, {total_lines:,} lines, {total_bytes / 1024 / 1024:.2f}MB\n\n")
    
    # ===========================================================================
    # Phase 3: Save to output folder
    # ===========================================================================
    yield sse_token("💾 Phase 3: Saving to output folder...\n")
    
    try:
        # Use resolved output_dir for all file operations
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[full_archmap] Created/verified output directory: {output_dir}")
        
        # Save INDEX.json (legacy - always overwritten)
        index_path = output_dir / "INDEX.json"
        index_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "roots": CODE_SCAN_ROOTS,
            "scan_time_ms": scan_time_ms,
            "total_files": len(files_data),
            "files_with_content": files_with_content,
            "total_lines": total_lines,
            "total_bytes": total_bytes,
            "files": files_data,
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)
        yield sse_token(f"   Saved: INDEX.json (legacy)\n")
        
        # Generate timestamp for RAG-compatible files
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M")
        
        # Save INDEX_<timestamp>.json (RAG expected format)
        index_rag_path = output_dir / f"INDEX_{timestamp_str}.json"
        index_rag_data = generate_index_for_rag(files_data, contents_data, CODE_SCAN_ROOTS[0])
        with open(index_rag_path, "w", encoding="utf-8") as f:
            json.dump(index_rag_data, f, indent=2)
        yield sse_token(f"   Saved: INDEX_{timestamp_str}.json (RAG)\n")
        
        # Save SIGNATURES_<timestamp>.json (RAG required)
        signatures_path = output_dir / f"SIGNATURES_{timestamp_str}.json"
        signatures_data = generate_signatures_json(contents_data, CODE_SCAN_ROOTS[0])
        with open(signatures_path, "w", encoding="utf-8") as f:
            json.dump(signatures_data, f, indent=2)
        total_sigs = signatures_data.get("total_signatures", 0)
        yield sse_token(f"   Saved: SIGNATURES_{timestamp_str}.json ({total_sigs} signatures)\n")
        
        # Save CODEBASE.md (full source code)
        codebase_path = output_dir / FULL_CODEBASE_OUTPUT_FILE
        codebase_content = generate_codebase_md(files_data, contents_data)
        with open(codebase_path, "w", encoding="utf-8") as f:
            f.write(codebase_content)
        codebase_size_mb = len(codebase_content.encode("utf-8")) / 1024 / 1024
        yield sse_token(f"   Saved: {FULL_CODEBASE_OUTPUT_FILE} ({codebase_size_mb:.2f}MB)\n")
        
    except Exception as e:
        logger.exception(f"[full_archmap] Save failed: {e}")
        yield sse_error(f"Failed to save: {e}")
        yield sse_done(
            provider="local",
            model="architecture_scanner",
            success=False,
            error=str(e),
        )
        return
    
    # ===========================================================================
    # Phase 4: Save to database
    # ===========================================================================
    yield sse_token("\n💾 Phase 4: Saving to database...\n")
    
    scan_id = None
    try:
        scan_id = save_scan_with_contents_to_db(
            db=db,
            scope="code",
            files_data=files_data,
            contents_data=contents_data,
            roots_scanned=CODE_SCAN_ROOTS,
            scan_time_ms=scan_time_ms,
        )
        if scan_id:
            yield sse_token(f"   Saved to DB: scan_id={scan_id}\n")
        else:
            yield sse_token("   DB save skipped (models not available)\n")
    except Exception as e:
        logger.exception(f"[full_archmap] DB save failed: {e}")
        yield sse_token(f"   DB save failed: {e}\n")
    
    # ===========================================================================
    # Phase 4.5: Queue background embedding job (non-blocking)
    # ===========================================================================
    # Embeddings build in background after command returns.
    # This keeps CREATE ARCHITECTURE MAP fast while enabling semantic search.
    # Wrapped in try/except so embedding failures never crash the main workflow.
    
    yield sse_token("\n🔗 Phase 4.5: Queueing background embedding job...\n")
    
    embedding_queued = False
    try:
        from app.rag.jobs.embedding_job import queue_embedding_job, EMBEDDING_AUTO_ENABLED
        from app.db import get_db_session
        
        if not EMBEDDING_AUTO_ENABLED:
            yield sse_token("   ⚠️ Auto-embedding disabled (ORB_EMBEDDING_AUTO=false)\n")
        else:
            # get_db_session() returns a Session directly (not a generator)
            # Pass it as the session factory callable
            # NOTE: Do NOT pass scan_id here!
            # ArchCodeChunk.scan_id references arch_scan_runs.id (RAG pipeline's scan tracking)
            # but _save_scan_with_contents_to_db creates architecture_scan_runs.id
            # These are different tables, so passing scan_id would filter out all chunks.
            # Instead, embed ALL pending ArchCodeChunk rows regardless of origin.
            embedding_queued = queue_embedding_job(
                db_session_factory=get_db_session,
                scan_id=None,  # Embed all pending chunks, not filtered by scan
            )
            
            if embedding_queued:
                yield sse_token("   ✅ Embedding job queued (background)\n")
                yield sse_token("   📊 Priority: Tier1 (routers) → Tier2 (pipeline) → Tier3 (services) → ...\n")
                yield sse_token("   💡 Use `embedding status` to check progress\n")
            else:
                yield sse_token("   ⚠️ Embedding job not queued (may already be running)\n")
                
    except ImportError as ie:
        logger.warning(f"[full_archmap] Embedding module not available: {ie}")
        yield sse_token(f"   ⚠️ Embedding module not available: {ie}\n")
    except Exception as emb_err:
        # Never let embedding errors crash the main workflow
        logger.warning(f"[full_archmap] Failed to queue embedding job (non-fatal): {emb_err}")
        yield sse_token(f"   ⚠️ Embedding queue failed (non-fatal): {emb_err}\n")
    
    # ===========================================================================
    # Phase 5: Generate architecture map with Opus (with sentinel + continuation)
    # ===========================================================================
    yield sse_token("\n🤖 Phase 5: Generating architecture overview with Claude Opus...\n\n")
    
    # Get config from stage_models (proper max_tokens/timeout)
    if _STAGE_MODELS_AVAILABLE and get_archmap_config:
        config = get_archmap_config()
        use_provider = config.provider
        use_model = config.model
        use_max_tokens = config.max_output_tokens
        use_timeout = config.timeout_seconds
        logger.info(f"[ARCHMAP] Using stage_models config: provider={use_provider} model={use_model} max_tokens={use_max_tokens} timeout={use_timeout}s")
    else:
        use_provider = ARCHMAP_PROVIDER
        use_model = ARCHMAP_MODEL
        use_max_tokens = 60000  # Default from ENV spec
        use_timeout = 300
        logger.info(f"[ARCHMAP] Using fallback config: provider={use_provider} model={use_model} max_tokens={use_max_tokens} timeout={use_timeout}s")
    
    yield sse_token(f"   [ARCHMAP] provider={use_provider} model={use_model} max_tokens={use_max_tokens} timeout={use_timeout}s\n\n")
    
    # Build prompt from file tree
    zone_counts: Dict[str, int] = {}
    for f in files_data:
        zone = f.get("zone", "other")
        zone_counts[zone] = zone_counts.get(zone, 0) + 1
    
    file_list = [
        {
            "path": f.get("path", ""),
            "name": f.get("name", ""),
            "ext": f.get("ext", ""),
            "zone": f.get("zone", "other"),
            "size": f.get("size_bytes"),
            "root": f.get("root", ""),
        }
        for f in files_data
    ]
    
    prompt = _build_db_archmap_prompt(file_list, zone_counts)
    map_content = ""
    round_num = 1
    
    map_path = output_dir / FULL_ARCHMAP_OUTPUT_FILE
    
    try:
        from app.llm.streaming import stream_llm
        
        while round_num <= ARCHMAP_MAX_CONTINUATION_ROUNDS:
            # Build messages for this round
            if round_num == 1:
                messages = [{"role": "user", "content": prompt}]
                system_prompt = ARCHMAP_SYSTEM_PROMPT
            else:
                # Continuation round
                continuation_prompt = build_continuation_prompt(map_content)
                messages = [{"role": "user", "content": continuation_prompt}]
                system_prompt = ARCHMAP_SYSTEM_PROMPT
            
            logger.info(f"[ARCHMAP] Round {round_num}: Starting LLM call")
            chunk_content = ""
            
            async for event in stream_llm(
                messages=messages,
                system_prompt=system_prompt,
                provider=use_provider,
                model=use_model,
                max_tokens=use_max_tokens,
                timeout_seconds=use_timeout,
            ):
                event_type = event.get("type")
                if event_type == "token":
                    text = event.get("text", "")
                    chunk_content += text
                    map_content += text
                    yield sse_token(text)
                elif event_type == "error":
                    yield sse_token(f"\n⚠️ Opus error: {event.get('message')}\n")
                    break
                elif event_type == "done":
                    break
            
            logger.info(f"[ARCHMAP] Round {round_num}: Received {len(chunk_content)} chars (total: {len(map_content)} chars)")
            
            # Crash-safe incremental write:
            # Round 1 = overwrite (fresh run), Rounds 2+ = append (continuation)
            if chunk_content:
                write_mode = "w" if round_num == 1 else "a"
                with open(map_path, write_mode, encoding="utf-8") as f:
                    f.write(chunk_content)
                logger.info(f"[ARCHMAP] Round {round_num}: Wrote {len(chunk_content)} chars to disk (mode={write_mode})")
                yield sse_token(f"\n   💾 Round {round_num}: wrote {len(chunk_content)} chars (mode={write_mode})\n")
            
            # Check for sentinel
            if has_sentinel(map_content):
                logger.info(f"[ARCHMAP] Sentinel found in round {round_num} - completed")
                yield sse_token(f"\n   ✅ Sentinel detected - map complete (round {round_num}, total {len(map_content)} chars)\n")
                break
            
            if round_num >= ARCHMAP_MAX_CONTINUATION_ROUNDS:
                logger.warning(f"[ARCHMAP] Max rounds ({ARCHMAP_MAX_CONTINUATION_ROUNDS}) reached without sentinel - keeping partial file")
                yield sse_token(f"\n   ⚠️ Max rounds ({ARCHMAP_MAX_CONTINUATION_ROUNDS}) reached without sentinel - partial file kept ({len(map_content)} chars)\n")
                break
            
            # Need continuation
            logger.info(f"[ARCHMAP] Sentinel missing after round {round_num} ({len(map_content)} chars) -> requesting continuation")
            yield sse_token(f"   📝 Round {round_num} complete - continuing...\n\n")
            round_num += 1
        
        # Final summary (file already written incrementally)
        yield sse_token(f"\n   📄 Final: {FULL_ARCHMAP_OUTPUT_FILE} ({len(map_content)} chars, {round_num} round(s))\n")
        
    except Exception as e:
        logger.exception(f"[full_archmap] Opus call failed: {e}")
        yield sse_token(f"\n⚠️ Opus failed: {e}\n")
        if map_content:
            yield sse_token(f"   💾 Partial content preserved on disk ({len(map_content)} chars)\n")
    
    # ===========================================================================
    # Done
    # ===========================================================================
    
    # Record in memory
    try:
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id,
                role="assistant",
                content=f"[architecture_scan] Full scan: {len(files_data)} files, {files_with_content} with content, {total_lines:,} lines",
                provider="local",
                model="architecture_scanner",
            ),
        )
    except Exception:
        pass
    
    duration_ms = int(loop.time() * 1000) - started_ms
    
    if trace:
        trace.log_model_call(
            "local_tool", "local", "architecture_scanner", "full_architecture_map",
            len(prompt), len(map_content), duration_ms, success=True, error=None,
        )
    
    # Build embedding status for summary
    embedding_status_str = "🔗 Embeddings: queued (background)" if embedding_queued else "🔗 Embeddings: not queued"
    
    summary = (
        f"\n✅ Architecture scan complete.\n"
        f"📂 Output: {output_dir}\n"
        f"📊 Files: {len(files_data)} ({files_with_content} with content)\n"
        f"📝 Lines: {total_lines:,}\n"
        f"📦 Size: {total_bytes / 1024 / 1024:.2f}MB\n"
        f"🗺️ Outputs: INDEX.json, INDEX_{timestamp_str}.json, SIGNATURES_{timestamp_str}.json, {FULL_CODEBASE_OUTPUT_FILE}, {FULL_ARCHMAP_OUTPUT_FILE}\n"
        f"💾 DB scan_id: {scan_id}\n"
        f"{embedding_status_str}\n"
        f"⏱️ Duration: {duration_ms}ms\n"
    )
    yield sse_token(summary)
    
    yield sse_done(
        provider="local",
        model="architecture_scanner",
        total_length=len(codebase_content),
        meta={
            "output_dir": str(output_dir),
            "files": len(files_data),
            "files_with_content": files_with_content,
            "total_lines": total_lines,
            "total_bytes": total_bytes,
            "scan_id": scan_id,
            "zones": zone_counts,
            "rag_index": f"INDEX_{timestamp_str}.json",
            "rag_signatures": f"SIGNATURES_{timestamp_str}.json",
            "embedding_queued": embedding_queued,
        },
    )
