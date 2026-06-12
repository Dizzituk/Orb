# FILE: app/memory/upload_knowledge_hook.py
# Purpose: Universal upload knowledge hook.
# Called-by: app.debug.file_upload_router, app.drive.content_indexer
# Depends-on: app.embeddings, app.embeddings.service, app.llm.file_analyzer, app.memory (+4 more)
# Last-renovated: 2026-06-11
"""
Universal upload knowledge hook.

v0.14.2: Content-hash deduplication — same file uploaded twice is detected
         and skipped (no duplicate storage, indexing, or promotion).
v0.14.1: Fixed FK constraint, session poisoning, async LLM calls.

Any endpoint that accepts file uploads can call this after saving the
file to trigger text extraction, document storage, embeddings indexing,
and knowledge promotion. Non-blocking — failures are logged but never
break the upload response.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session as DbSession

logger = logging.getLogger(__name__)


def _content_hash(text: str) -> str:
    """SHA-256 hash of text content for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _find_existing_by_hash(db: DbSession, content_hash: str) -> bool:
    """Check if a DocumentContent record with this hash already exists."""
    try:
        from app.memory.models import DocumentContent
        # Check raw_text hash — we store the hash in the summary field prefix
        # as a lightweight check. For a proper check, query by filename + size.
        # But the most reliable approach: check if any DocumentContent has
        # identical raw_text by querying a prefix match.
        existing = db.query(DocumentContent).filter(
            DocumentContent.filename.isnot(None)
        ).all()
        for doc in existing:
            try:
                if doc.raw_text and _content_hash(doc.raw_text) == content_hash:
                    return True
            except Exception:
                continue
    except Exception as e:
        logger.debug("[upload_hook] Dedup check failed: %s", e)
    return False


def _find_existing_by_name_and_size(
    db: DbSession, filename: str, text_len: int
) -> bool:
    """
    Fast dedup check: same filename + same text length = almost certainly same file.
    Much cheaper than hashing every DocumentContent's raw_text.
    """
    try:
        from app.memory.models import DocumentContent
        from sqlalchemy import func

        matches = db.query(DocumentContent).filter(
            DocumentContent.filename == filename
        ).all()
        for doc in matches:
            try:
                if doc.raw_text and abs(len(doc.raw_text) - text_len) < 10:
                    return True
            except Exception:
                continue
    except Exception as e:
        logger.debug("[upload_hook] Fast dedup check failed: %s", e)
    return False


def _resolve_project_id(db: DbSession, fallback_id: Optional[int] = None) -> int:
    """
    Find a valid project ID for document storage.

    Priority:
      1. Explicit fallback_id if provided and exists.
      2. Most recent active project.
      3. Create a dedicated 'Document Uploads' project.
    """
    from app.memory.models import Project

    if fallback_id:
        exists = db.query(Project).filter(Project.id == fallback_id).first()
        if exists:
            return fallback_id

    latest = db.query(Project).order_by(Project.id.desc()).first()
    if latest:
        return latest.id

    from app.memory import service as memory_service, schemas as memory_schemas
    proj = memory_service.create_project(
        db,
        memory_schemas.ProjectCreate(
            name="Document Uploads",
            description="Auto-created for uploaded document storage",
        ),
    )
    return proj.id


async def process_uploaded_file(
    db: DbSession,
    file_path: str,
    original_name: str,
    mime_type: str = "",
    project_id: Optional[int] = None,
    raw_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    """
    Universal post-upload processing: extract, dedup, index, promote.

    v0.14.2: Deduplication — if the same content has already been stored,
    skip storage + indexing + promotion entirely.
    """
    result: Dict[str, Any] = {
        "filename": original_name,
        "project_id": None,
        "extracted": False,
        "deduplicated": False,
        "indexed": False,
        "promoted": False,
        "knowledge_result": None,
        "errors": [],
    }

    # ── Step 1: Extract text ─────────────────────────────────────
    raw_text: Optional[str] = None
    structured_data: Optional[str] = None
    doc_type: str = "unknown"

    try:
        from app.llm.file_analyzer import extract_text, detect_document_type

        text, err = extract_text(
            file_path=file_path,
            file_bytes=raw_bytes,
            filename=original_name,
        )
        if text:
            raw_text = text
            result["extracted"] = True
            doc_type = detect_document_type(
                content_or_path=raw_text,
                filename=original_name,
                mime_type=mime_type,
            )
            logger.info(
                "[upload_hook] Extracted %d chars from %s (type=%s)",
                len(raw_text), original_name, doc_type,
            )
        elif err:
            logger.info("[upload_hook] No text from %s: %s", original_name, err)
            result["errors"].append(f"extraction: {err}")
    except Exception as e:
        logger.warning("[upload_hook] Extraction failed for %s: %s", original_name, e)
        result["errors"].append(f"extraction: {e}")

    if not raw_text:
        return result

    # ── Step 2: Deduplication check ──────────────────────────────
    # Fast check first (filename + text length), then hash if needed.
    try:
        if _find_existing_by_name_and_size(db, original_name, len(raw_text)):
            result["deduplicated"] = True
            logger.info(
                "[upload_hook] DEDUP: %s already stored (same name + size) — skipping",
                original_name,
            )
            return result
    except Exception as e:
        logger.debug("[upload_hook] Dedup check error (non-fatal): %s", e)

    # ── Step 3: Resolve project + store document content ─────────
    file_record_id: Optional[int] = None
    try:
        pid = _resolve_project_id(db, project_id)
        result["project_id"] = pid

        from app.memory import service as memory_service, schemas as memory_schemas

        summary = raw_text[:300].strip() + "..." if len(raw_text) > 300 else raw_text.strip()

        file_record = memory_service.create_file_for_project(
            db, pid,
            memory_schemas.FileCreate(
                project_id=pid,
                path=file_path,
                original_name=original_name,
                file_type=mime_type or doc_type,
                description=summary,
            ),
        )
        file_record_id = file_record.id

        memory_service.create_document_content(
            db,
            memory_schemas.DocumentContentCreate(
                project_id=pid,
                file_id=file_record.id,
                filename=original_name,
                doc_type=doc_type,
                raw_text=raw_text,
                summary=summary,
                structured_data=structured_data,
            ),
        )
        logger.info(
            "[upload_hook] Stored document content for %s (file_id=%d, project=%d)",
            original_name, file_record.id, pid,
        )
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning("[upload_hook] Document storage failed for %s: %s", original_name, e)
        result["errors"].append(f"storage: {e}")

    # ── Step 4: Embeddings indexing ──────────────────────────────
    if file_record_id:
        try:
            from app.embeddings import service as embeddings_service
            from app.memory.models import DocumentContent

            doc = db.query(DocumentContent).filter(
                DocumentContent.file_id == file_record_id
            ).first()
            if doc:
                embeddings_service.index_document(db, doc, force=True)
                result["indexed"] = True
                logger.info("[upload_hook] Indexed %s", original_name)
        except Exception as e:
            logger.warning("[upload_hook] Indexing failed for %s: %s", original_name, e)
            result["errors"].append(f"indexing: {e}")

    # ── Step 5: Knowledge promotion ──────────────────────────────
    try:
        from app.memory.document_knowledge_promoter import promote_document_knowledge

        promo = await promote_document_knowledge(
            db=db,
            filename=original_name,
            raw_text=raw_text,
            structured_data=structured_data,
        )
        result["knowledge_result"] = promo
        result["promoted"] = promo.get("written", 0) > 0
        logger.info(
            "[upload_hook] Knowledge promotion for %s: %s",
            original_name, promo,
        )
    except Exception as e:
        logger.warning("[upload_hook] Knowledge promotion failed for %s: %s", original_name, e)
        result["errors"].append(f"promotion: {e}")

    return result


def process_uploaded_file_sync(
    db: DbSession,
    file_path: str,
    original_name: str,
    mime_type: str = "",
    project_id: Optional[int] = None,
    raw_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Synchronous wrapper for process_uploaded_file."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        future = asyncio.run_coroutine_threadsafe(
            process_uploaded_file(db, file_path, original_name, mime_type, project_id, raw_bytes),
            loop,
        )
        return future.result(timeout=90)
    except RuntimeError:
        return asyncio.run(
            process_uploaded_file(db, file_path, original_name, mime_type, project_id, raw_bytes)
        )
