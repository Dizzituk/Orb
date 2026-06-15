# FILE: app/memory/enrichment/backfill_job.py
# Purpose: Resumable, throttled backfill of media/document enrichment.
# Called-by: CLI (python -m app.memory.enrichment.backfill_job), app.memory.enrichment
# Depends-on: app.memory.enrichment.media_enricher, app.drive.manifest_models, app.db
# Last-renovated: 2026-06-12
"""
Enrichment backfill — resumable, throttled, never a one-shot loop.

Phases (each independently resumable via a progress cursor):
  1. documents_hot_index — hot-index every DocumentContent record so existing
     documents become retrievable in conversation. No API key needed.
  2. audio — ID3-metadata descriptions for drive audio files. No API key.
  3. images — vision captions for drive images. NEEDS a Gemini key; the
     phase is skipped (state-noted) when keys are absent so the job can run
     offline and be re-run after live boot syncs keys.
  4. document_abstracts — upgrade truncation summaries to 2-3 sentence LLM
     abstracts. Needs a key; skipped when absent.
  5. embed_missing — embed DocumentContent records that have no vectors.
     Needs a key; skipped when absent.

State: data/enrichment/backfill_state.json (cursor + counts per phase).
Errors: data/enrichment/backfill_errors.log (one line per failure).

Usage:
  .venv\\Scripts\\python.exe -m app.memory.enrichment.backfill_job
  .venv\\Scripts\\python.exe -m app.memory.enrichment.backfill_job --phase images --limit 200
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

STATE_DIR = Path("data/enrichment")
STATE_FILE = STATE_DIR / "backfill_state.json"
ERROR_LOG = STATE_DIR / "backfill_errors.log"

BATCH_SIZE = 10
BATCH_SLEEP_SECONDS = 1.0  # throttle between batches (rate-limit headroom)

PHASES = ["documents_hot_index", "audio", "images",
          "document_abstracts", "embed_missing"]


# =========================================================================
# State management
# =========================================================================

def _load_state() -> Dict:
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("[backfill] state load failed (fresh start): %s", exc)
    return {}


def _save_state(state: Dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _log_error(phase: str, item: str, exc) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with ERROR_LOG.open("a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now(timezone.utc).isoformat()} [{phase}] "
                     f"{item}: {exc}\n")
    except Exception:
        pass


def _phase_state(state: Dict, phase: str) -> Dict:
    return state.setdefault(phase, {
        "cursor": 0, "done": 0, "errors": 0, "note": "",
    })


# =========================================================================
# Phase implementations — each returns (processed, done, errors)
# =========================================================================

def _phase_documents_hot_index(db, ps: Dict, limit: int) -> int:
    """Hot-index DocumentContent rows after the cursor. No API needed."""
    from app.memory.models import DocumentContent, File
    from app.memory.enrichment.media_enricher import hot_index_document

    rows = (
        db.query(DocumentContent)
        .filter(DocumentContent.id > ps["cursor"])
        .order_by(DocumentContent.id.asc())
        .limit(limit)
        .all()
    )
    for doc in rows:
        try:
            path = None
            try:
                file_rec = db.query(File).filter(File.id == doc.file_id).first()
                path = file_rec.path if file_rec else None
            except Exception:
                pass
            if hot_index_document(db, doc, doc.summary, path):
                ps["done"] += 1
            else:
                ps["errors"] += 1
        except Exception as exc:
            ps["errors"] += 1
            _log_error("documents_hot_index", f"doc {doc.id}", exc)
        ps["cursor"] = doc.id
    return len(rows)


def _manifest_rows(db, file_class: str, cursor: int, limit: int):
    from app.drive.manifest_models import DriveFileManifest
    return (
        db.query(DriveFileManifest)
        .filter(
            DriveFileManifest.file_class == file_class,
            DriveFileManifest.content_indexed == False,  # noqa: E712
            DriveFileManifest.id > cursor,
        )
        .order_by(DriveFileManifest.id.asc())
        .limit(limit)
        .all()
    )


def _phase_audio(db, ps: Dict, limit: int) -> int:
    """Metadata-derived music descriptions. No API needed."""
    from app.memory.enrichment.media_enricher import enrich_music, store_media_document

    rows = _manifest_rows(db, "audio", ps["cursor"], limit)
    for rec in rows:
        try:
            info = enrich_music(rec.path)
            file_id = store_media_document(
                db, path=rec.path, filename=rec.filename, kind="audio",
                description=info["description"],
                raw_text=(
                    f"{info['description']}\n"
                    f"Title: {info['meta']['title']}\n"
                    f"Artist: {info['meta']['artist']}\n"
                    f"Album: {info['meta']['album']}\n"
                    f"Genre: {info['meta']['genre']}\n"
                    f"Path: {rec.path}"
                ),
                tags=info["tags"],
            )
            if file_id:
                rec.content_indexed = True
                rec.indexed_at = datetime.utcnow()
                db.commit()
                ps["done"] += 1
            else:
                ps["errors"] += 1
        except Exception as exc:
            ps["errors"] += 1
            _log_error("audio", rec.path, exc)
        ps["cursor"] = rec.id
    return len(rows)


def _phase_images(db, ps: Dict, limit: int) -> int:
    """Vision captions for drive images. Skipped without API keys."""
    from app.memory.enrichment.media_enricher import (
        enrich_image, store_media_document, llm_keys_available,
    )

    if not llm_keys_available():
        ps["note"] = "skipped: no Gemini API key in environment (re-run after live boot)"
        return 0

    rows = _manifest_rows(db, "image", ps["cursor"], limit)
    for rec in rows:
        try:
            info = enrich_image(rec.path)
            if not info["enriched"]:
                # Vision failed — log, leave unindexed so a re-run retries it
                ps["errors"] += 1
                _log_error("images", rec.path, "vision call failed")
                ps["cursor"] = rec.id
                continue
            file_id = store_media_document(
                db, path=rec.path, filename=rec.filename, kind="image",
                description=info["description"],
                raw_text=f"{info['description']}\nPath: {rec.path}",
                tags=info["tags"],
            )
            if file_id:
                rec.content_indexed = True
                rec.indexed_at = datetime.utcnow()
                db.commit()
                ps["done"] += 1
            else:
                ps["errors"] += 1
        except Exception as exc:
            ps["errors"] += 1
            _log_error("images", rec.path, exc)
        ps["cursor"] = rec.id
    return len(rows)


def _phase_document_abstracts(db, ps: Dict, limit: int) -> int:
    """Upgrade truncation summaries to LLM abstracts. Skipped without keys."""
    from app.memory.models import DocumentContent
    from app.memory.enrichment.media_enricher import (
        enrich_document_abstract, hot_index_document, llm_keys_available,
    )

    if not llm_keys_available():
        ps["note"] = "skipped: no Gemini API key in environment (re-run after live boot)"
        return 0

    rows = (
        db.query(DocumentContent)
        .filter(
            DocumentContent.id > ps["cursor"],
            DocumentContent.doc_type.notin_(["image", "audio"]),
        )
        .order_by(DocumentContent.id.asc())
        .limit(limit)
        .all()
    )
    for doc in rows:
        try:
            # Only upgrade summaries that are raw-text truncations
            is_truncation = bool(doc.summary) and (
                doc.summary.endswith("...")
                or (doc.raw_text or "").startswith(doc.summary[:80])
            )
            if doc.raw_text and (is_truncation or not doc.summary):
                abstract = asyncio.run(
                    enrich_document_abstract(doc.filename or "", doc.raw_text)
                )
                if abstract:
                    doc.summary = abstract
                    db.commit()
                    hot_index_document(db, doc, abstract)
                    ps["done"] += 1
        except Exception as exc:
            ps["errors"] += 1
            _log_error("document_abstracts", f"doc {doc.id}", exc)
        ps["cursor"] = doc.id
    return len(rows)


def _phase_embed_missing(db, ps: Dict, limit: int) -> int:
    """Embed DocumentContent records with no vectors. Skipped without keys."""
    from sqlalchemy import text as _sql
    from app.memory.models import DocumentContent
    from app.memory.enrichment.media_enricher import llm_keys_available

    if not llm_keys_available():
        ps["note"] = "skipped: no Gemini API key in environment (re-run after live boot)"
        return 0

    from app.embeddings import service as embeddings_service

    rows = (
        db.query(DocumentContent)
        .filter(DocumentContent.id > ps["cursor"])
        .order_by(DocumentContent.id.asc())
        .limit(limit)
        .all()
    )
    for doc in rows:
        try:
            existing = db.execute(_sql(
                "SELECT 1 FROM embeddings WHERE source_type='file' "
                "AND source_id=:fid LIMIT 1"
            ), {"fid": doc.file_id}).fetchone()
            if existing is None:
                created = embeddings_service.index_document(db, doc, force=False)
                if created:
                    ps["done"] += 1
        except Exception as exc:
            ps["errors"] += 1
            _log_error("embed_missing", f"doc {doc.id}", exc)
        ps["cursor"] = doc.id
    return len(rows)


_PHASE_FUNCS = {
    "documents_hot_index": _phase_documents_hot_index,
    "audio": _phase_audio,
    "images": _phase_images,
    "document_abstracts": _phase_document_abstracts,
    "embed_missing": _phase_embed_missing,
}


# =========================================================================
# Runner
# =========================================================================

def run_backfill(
    phases: Optional[List[str]] = None,
    limit: Optional[int] = None,
    db=None,
) -> Dict:
    """Run the backfill, resuming from the saved state.

    Args:
        phases: subset of PHASES to run (default: all, in order)
        limit: max items per phase this invocation (default: unlimited)
        db: optional session (tests); a fresh one is opened otherwise
    """
    close_db = False
    if db is None:
        from app.db import SessionLocal
        db = SessionLocal()
        close_db = True

    state = _load_state()
    selected = phases or PHASES

    try:
        for phase in selected:
            func = _PHASE_FUNCS.get(phase)
            if func is None:
                logger.warning("[backfill] unknown phase: %s", phase)
                continue
            ps = _phase_state(state, phase)
            ps["note"] = ""
            remaining = limit if limit is not None else 10_000_000
            logger.info("[backfill] phase %s from cursor %s", phase, ps["cursor"])

            while remaining > 0:
                batch = min(BATCH_SIZE, remaining)
                processed = func(db, ps, batch)
                _save_state(state)
                remaining -= processed
                if processed < batch:
                    break  # phase exhausted
                time.sleep(BATCH_SLEEP_SECONDS)

            logger.info(
                "[backfill] phase %s: done=%s errors=%s cursor=%s %s",
                phase, ps["done"], ps["errors"], ps["cursor"],
                ps.get("note", ""),
            )
        return state
    finally:
        if close_db:
            db.close()


def reset_state(phase: Optional[str] = None) -> None:
    """Clear saved cursors (all phases, or one)."""
    state = _load_state()
    if phase:
        state.pop(phase, None)
    else:
        state = {}
    _save_state(state)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Enrichment backfill")
    parser.add_argument("--phase", choices=PHASES, default=None,
                        help="run a single phase (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="max items per phase this run")
    parser.add_argument("--reset", action="store_true",
                        help="clear saved cursors first")
    args = parser.parse_args()

    if args.reset:
        reset_state(args.phase)

    final_state = run_backfill(
        phases=[args.phase] if args.phase else None,
        limit=args.limit,
    )
    print(json.dumps(final_state, indent=2))
