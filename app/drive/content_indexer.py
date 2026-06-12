# FILE: app/drive/content_indexer.py
# Purpose: Content indexer — Tier 2 of the boot scan.
# Called-by: app.drive.boot_scan
# Depends-on: app.drive.manifest_models, app.embeddings, app.embeddings.service, app.llm.file_analyzer (+6 more)
# Last-renovated: 2026-06-11
"""
Content indexer — Tier 2 of the boot scan.

Processes new/modified DOCUMENT files (txt, md, docx, pdf, xlsx, pptx,
csv, json, etc.) by extracting their text, generating embeddings via
Gemini Embedding 2, and storing in the RAG system.

Does NOT process media files (images, audio, video) — those are
handled by Tier 3 (media cataloguer, not yet implemented).

Runs as a background task after the manifest scan completes.
Designed to be interruptible — progress is saved per-file.

Called from: app/drive/boot_scan.py → run_content_indexing()
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy.orm import Session as DbSession

from app.drive.manifest_models import DriveFileManifest

logger = logging.getLogger(__name__)

# File classes that get content-indexed (text extraction + embeddings)
INDEXABLE_CLASSES = {"document"}

# Max file size for content indexing (5MB — skip huge files)
MAX_INDEX_SIZE = 5 * 1024 * 1024

# No cap — index everything on first boot, deltas only after that.
# The manifest tracks content_indexed per file, so subsequent boots
# only process genuinely new or modified documents.


@dataclass
class IndexReport:
    """Result of content indexing pass."""
    files_processed: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_errored: int = 0
    files_promoted: int = 0
    duration_ms: int = 0
    errors: List[str] = field(default_factory=list)


def run_content_indexing(db: DbSession) -> IndexReport:
    """
    Index new/modified document files from the manifest.

    Finds files in drive_file_manifest where:
      - file_class = 'document'
      - content_indexed = False
      - size_bytes < MAX_INDEX_SIZE

    For each: extract text → store as DocumentContent → embed → promote knowledge.

    Returns IndexReport with counts.
    """
    start = time.monotonic()
    report = IndexReport()

    # Find unindexed document files
    candidates = (
        db.query(DriveFileManifest)
        .filter(
            DriveFileManifest.file_class.in_(INDEXABLE_CLASSES),
            DriveFileManifest.content_indexed == False,
            DriveFileManifest.size_bytes < MAX_INDEX_SIZE,
        )
        .order_by(DriveFileManifest.size_bytes.asc())  # Small files first
        .all()
    )

    if not candidates:
        logger.info("[content_indexer] No new documents to index")
        return report

    logger.info(
        "[content_indexer] Found %d unindexed document(s) to process",
        len(candidates),
    )

    for manifest_rec in candidates:
        report.files_processed += 1
        filepath = manifest_rec.path
        filename = manifest_rec.filename

        try:
            # Step 1: Extract text
            from app.llm.file_analyzer import extract_text, detect_document_type

            text, err = extract_text(file_path=filepath, filename=filename)
            if not text:
                logger.debug(
                    "[content_indexer] No text from %s: %s", filename, err
                )
                manifest_rec.content_indexed = True  # Mark as processed (nothing to index)
                manifest_rec.indexed_at = datetime.utcnow()
                report.files_skipped += 1
                continue

            doc_type = detect_document_type(
                content_or_path=text, filename=filename,
            )

            # Step 2: Store document content (dedup check)
            from app.memory.models import DocumentContent
            existing = db.query(DocumentContent).filter(
                DocumentContent.filename == filename,
            ).all()

            already_stored = False
            for doc in existing:
                try:
                    if doc.raw_text and abs(len(doc.raw_text) - len(text)) < 10:
                        already_stored = True
                        break
                except Exception:
                    continue

            if not already_stored:
                try:
                    from app.memory.upload_knowledge_hook import _resolve_project_id
                    pid = _resolve_project_id(db)

                    from app.memory import service as memory_service
                    from app.memory import schemas as memory_schemas

                    summary = text[:300].strip() + "..." if len(text) > 300 else text.strip()

                    file_rec = memory_service.create_file_for_project(
                        db, pid,
                        memory_schemas.FileCreate(
                            project_id=pid,
                            path=filepath,
                            original_name=filename,
                            file_type=doc_type,
                            description=summary,
                        ),
                    )

                    memory_service.create_document_content(
                        db,
                        memory_schemas.DocumentContentCreate(
                            project_id=pid,
                            file_id=file_rec.id,
                            filename=filename,
                            doc_type=doc_type,
                            raw_text=text,
                            summary=summary,
                        ),
                    )

                    # Step 3: Generate embeddings
                    try:
                        from app.embeddings import service as embeddings_service
                        doc_content = db.query(DocumentContent).filter(
                            DocumentContent.file_id == file_rec.id
                        ).first()
                        if doc_content:
                            embeddings_service.index_document(db, doc_content, force=True)
                    except Exception as embed_err:
                        logger.warning(
                            "[content_indexer] Embedding failed for %s: %s",
                            filename, embed_err,
                        )

                except Exception as store_err:
                    try:
                        db.rollback()
                    except Exception:
                        pass
                    logger.warning(
                        "[content_indexer] Storage failed for %s: %s",
                        filename, store_err,
                    )
                    report.files_errored += 1
                    report.errors.append(f"{filename}: {store_err}")
                    continue

            # Step 4: Run knowledge promotion (personal docs only)
            try:
                from app.memory.document_knowledge_promoter import (
                    promote_document_knowledge_sync,
                )
                promo = promote_document_knowledge_sync(
                    db=db, filename=filename, raw_text=text,
                )
                if promo.get("written", 0) > 0:
                    report.files_promoted += 1
            except Exception as promo_err:
                logger.debug(
                    "[content_indexer] Promotion skipped for %s: %s",
                    filename, promo_err,
                )

            # Mark as indexed
            manifest_rec.content_indexed = True
            manifest_rec.indexed_at = datetime.utcnow()
            report.files_indexed += 1

            logger.info(
                "[content_indexer] Indexed: %s (%d chars, type=%s)",
                filename, len(text), doc_type,
            )

        except Exception as e:
            logger.warning(
                "[content_indexer] Failed to process %s: %s", filename, e
            )
            report.files_errored += 1
            report.errors.append(f"{filename}: {e}")

    db.commit()

    elapsed = int((time.monotonic() - start) * 1000)
    report.duration_ms = elapsed

    logger.info(
        "[content_indexer] Complete: %d processed, %d indexed, "
        "%d skipped, %d errored, %d promoted (%dms)",
        report.files_processed, report.files_indexed,
        report.files_skipped, report.files_errored,
        report.files_promoted, elapsed,
    )

    return report
