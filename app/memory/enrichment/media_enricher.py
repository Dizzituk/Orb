# FILE: app/memory/enrichment/media_enricher.py
# Purpose: Generate + store short descriptions for images, documents and music.
# Called-by: app.memory.enrichment.backfill_job, app.drive.content_indexer, app.memory.upload_knowledge_hook
# Depends-on: app.llm.gemini_vision, app.memory.enrichment.id3_reader, app.memory.service, app.astra_memory
# Last-renovated: 2026-06-12
"""
Media & document enrichment (memory architecture Job 3, 2026-06-12).

Embeddings give search; descriptions give the LLM something to read. Every
image and document record gets a short stored description; music gets
metadata-derived tags (never an audio-listening model).

Storage strategy: enrichment REUSES the existing document pipeline — a File
record (carries the path) + a DocumentContent record (summary = description)
+ a hot-index row (record_type="document") so conversational retrieval can
surface it and say "we have a file about this — here it is". No new tables.

All LLM-dependent steps degrade gracefully when no API key is configured
(descriptions fall back to deterministic text; embedding is skipped) so the
backfill can run offline and upgrade later.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 1-2 sentence caption prompt for the vision model (cheapest adequate tier)
_IMAGE_CAPTION_PROMPT = (
    "Describe this image in 1-2 plain sentences for a search index: what it "
    "shows, any visible text worth noting, and what kind of image it is "
    "(photo, screenshot, diagram...). No preamble — just the description."
)

_ABSTRACT_SYSTEM_PROMPT = (
    "You write 2-3 sentence abstracts of documents for a personal search "
    "index. First line: a short title. Then the abstract: what the document "
    "is, what it covers, anything identifying (dates, parties, amounts). "
    "No preamble, no markdown headers."
)


def llm_keys_available() -> bool:
    """True when a Gemini key is present (keys sync from DB at live boot)."""
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


# =========================================================================
# Enrichers (pure: produce description + tags, no storage)
# =========================================================================

def enrich_image(path: str, mime_type: Optional[str] = None) -> Dict:
    """Vision-caption an image. Returns {description, tags, enriched}.

    enriched=False means the vision call failed/unavailable and the
    description is a deterministic placeholder (retry later).
    """
    filename = Path(path).name
    try:
        from app.llm.gemini_vision import analyze_image
        result = analyze_image(path, mime_type=mime_type,
                               user_prompt=_IMAGE_CAPTION_PROMPT)
        if result and result.get("summary") and not result.get("error"):
            return {
                "description": result["summary"].strip()[:600],
                "tags": ["image"] + list(result.get("tags") or [])[:5],
                "enriched": True,
            }
    except Exception as exc:
        logger.debug("[enrich] vision failed for %s: %s", filename, exc)
    return {
        "description": f"Image file: {filename} (no description yet)",
        "tags": ["image"],
        "enriched": False,
    }


def enrich_music(path: str) -> Dict:
    """Metadata-derived music description. Never calls an LLM.

    'unknown' values are acceptable per the directive.
    """
    from app.memory.enrichment.id3_reader import read_audio_tags
    filename = Path(path).name
    meta = read_audio_tags(path)
    title = meta["title"] if meta["title"] != "unknown" else filename
    bits = [f"Music file: {title}"]
    if meta["artist"] != "unknown":
        bits.append(f"by {meta['artist']}")
    detail = []
    if meta["album"] != "unknown":
        detail.append(f"album: {meta['album']}")
    if meta["genre"] != "unknown":
        detail.append(f"genre: {meta['genre']}")
    description = " ".join(bits) + (f" ({'; '.join(detail)})" if detail else "")
    tags = ["music", "audio"]
    if meta["genre"] != "unknown":
        tags.append(meta["genre"].lower())
    return {"description": description, "tags": tags,
            "meta": meta, "enriched": True}


async def enrich_document_abstract(filename: str, raw_text: str) -> Optional[str]:
    """2-3 sentence LLM abstract (+title) of a document. None on failure."""
    if not raw_text or not raw_text.strip() or not llm_keys_available():
        return None
    try:
        from app.llm._streaming_utils_3 import call_llm_text
        provider = "google"
        model = os.getenv("SUMMARY_MODEL", "gemini-2.5-flash-lite")
        raw = await call_llm_text(
            provider=provider,
            model=model,
            system_prompt=_ABSTRACT_SYSTEM_PROMPT,
            user_prompt=f"Filename: {filename}\n\n{raw_text[:6000]}",
            max_tokens=160,
            timeout_seconds=20,
        )
        abstract = (raw or "").strip()
        return abstract[:700] if abstract else None
    except Exception as exc:
        logger.debug("[enrich] abstract failed for %s: %s", filename, exc)
        return None


# =========================================================================
# Storage (File + DocumentContent + hot index + best-effort embedding)
# =========================================================================

def hot_index_document(db, doc, description: Optional[str] = None,
                       path: Optional[str] = None,
                       extra_tags: Optional[List[str]] = None) -> bool:
    """Upsert a hot-index row for a DocumentContent record.

    record_id is the FILE id (str) — the same key the embeddings table and
    the semantic channel use for source_type='file', so vector matches and
    keyword matches resolve to the same record.
    """
    try:
        from app.astra_memory.retrieval import upsert_hot_index
        from app.astra_memory.preference_models import RetrievalCost
        from app.astra_memory.topic_tagger import extract_tags, extract_entities

        one_liner = (description or doc.summary or "").strip()[:240]
        basis = f"{doc.filename or ''}\n{one_liner}\n{(doc.raw_text or '')[:1500]}"
        tags = sorted(set(extract_tags(basis)) | set(extra_tags or []) | {"document"})[:8]
        entities = extract_entities(basis)

        upsert_hot_index(
            db=db,
            record_type="document",
            record_id=str(doc.file_id),
            title=doc.filename or f"document {doc.id}",
            one_liner=one_liner or (doc.filename or ""),
            tags=tags,
            entities=entities,
            cold_storage_path=path,
            retrieval_priority=0.55,
            retrieval_cost=RetrievalCost.TINY,
        )
        return True
    except Exception as exc:
        logger.warning("[enrich] hot-index failed for doc %s: %s",
                       getattr(doc, "id", "?"), exc)
        return False


def store_media_document(
    db,
    path: str,
    filename: str,
    kind: str,
    description: str,
    raw_text: Optional[str] = None,
    tags: Optional[List[str]] = None,
    project_id: Optional[int] = None,
) -> Optional[int]:
    """Persist an enriched media file through the document pipeline.

    Returns the File id, or None on failure. Dedupes on an existing
    DocumentContent with the same filename + doc_type.
    """
    try:
        from app.memory.models import DocumentContent
        from app.memory import service as memory_service, schemas as memory_schemas
        from app.memory.upload_knowledge_hook import _resolve_project_id

        existing = db.query(DocumentContent).filter(
            DocumentContent.filename == filename,
            DocumentContent.doc_type == kind,
        ).first()
        if existing is not None:
            # Already stored — refresh placeholder/empty summaries + hot row
            placeholder = (existing.summary or "").startswith(
                ("Image file:", "Music file:")
            )
            if description and (not existing.summary or placeholder):
                existing.summary = description
                db.commit()
            hot_index_document(db, existing, description or existing.summary,
                               path, tags)
            return existing.file_id

        pid = _resolve_project_id(db, project_id)
        file_rec = memory_service.create_file_for_project(
            db, pid,
            memory_schemas.FileCreate(
                project_id=pid,
                path=path,
                original_name=filename,
                file_type=kind,
                description=description[:300],
            ),
        )
        memory_service.create_document_content(
            db,
            memory_schemas.DocumentContentCreate(
                project_id=pid,
                file_id=file_rec.id,
                filename=filename,
                doc_type=kind,
                raw_text=raw_text or f"{description}\nPath: {path}",
                summary=description,
            ),
        )
        doc = db.query(DocumentContent).filter(
            DocumentContent.file_id == file_rec.id
        ).first()

        if doc is not None:
            hot_index_document(db, doc, description, path, tags)
            # Best-effort embedding — silently skipped when keys are absent
            if llm_keys_available():
                try:
                    from app.embeddings import service as embeddings_service
                    embeddings_service.index_document(db, doc, force=True)
                except Exception as embed_exc:
                    logger.debug("[enrich] embedding skipped for %s: %s",
                                 filename, embed_exc)
        return file_rec.id
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning("[enrich] store failed for %s: %s", filename, exc)
        return None


__all__ = [
    "enrich_image",
    "enrich_music",
    "enrich_document_abstract",
    "store_media_document",
    "hot_index_document",
    "llm_keys_available",
]
