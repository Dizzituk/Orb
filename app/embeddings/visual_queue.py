# FILE: app/embeddings/visual_queue.py
# Purpose: Visual-embed ingest queue — items wait here while the local multimodal model is unloaded.
# Called-by: app.content.video_pipeline.asset_library, app.idle.tasks_visual, app.gpu.orchestrator (queue depth)
# Depends-on: app.db, app.embeddings.provider_router, app.memory.rag_entries_model
# Last-renovated: 2026-07-02 (LANE E)
"""
Visual-embed ingest queue (LANE E, Task 2).

The visual collection (video assets, drive images, screenshots) embeds via
the multimodal role. With EMBEDDINGS_MULTIMODAL_PROVIDER=local the model is
LOAD-ON-DEMAND ONLY (never resident outside BACKGROUND_INGEST), so producers
enqueue items here at ingest; the visual_embed_drain idle task embeds them
while the model is resident. With provider=gemini the drain can run in any
idle window (API path).

One row per pending item. kind decides the writeback:
  - video_asset: vector + stamp written onto rag_entries row `ref_id`
  (future: drive_image, screenshot — add a writeback handler, nothing else)
"""
from __future__ import annotations

import json
import logging
import struct
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import Session

from app.db import Base

logger = logging.getLogger(__name__)

STATUS_PENDING = "pending"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


class VisualEmbedItem(Base):
    """A visual item waiting for (or finished with) multimodal embedding."""
    __tablename__ = "visual_embed_queue"

    id = Column(Integer, primary_key=True, index=True)
    kind = Column(String(30), nullable=False)          # "video_asset" | future kinds
    ref_id = Column(Integer, nullable=True)            # target row id (per kind)
    text = Column(Text, nullable=True)                 # text side of the embed
    image_path = Column(String(1000), nullable=True)   # optional image side
    payload_json = Column(Text, nullable=True)         # kind-specific extras
    status = Column(String(20), nullable=False, default=STATUS_PENDING, index=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    embedded_at = Column(DateTime, nullable=True)


def enqueue_visual_item(
    db: Session,
    kind: str,
    *,
    ref_id: Optional[int] = None,
    text: Optional[str] = None,
    image_path: Optional[str] = None,
    payload: Optional[dict] = None,
) -> VisualEmbedItem:
    """Queue a visual item for embedding at the next BACKGROUND_INGEST window.
    Dedupe: an already-pending row for the same (kind, ref_id) absorbs it."""
    if ref_id is not None:
        existing = (
            db.query(VisualEmbedItem)
            .filter(
                VisualEmbedItem.kind == kind,
                VisualEmbedItem.ref_id == ref_id,
                VisualEmbedItem.status == STATUS_PENDING,
            )
            .first()
        )
        if existing is not None:
            return existing
    item = VisualEmbedItem(
        kind=kind,
        ref_id=ref_id,
        text=text,
        image_path=image_path,
        payload_json=json.dumps(payload or {}, ensure_ascii=False),
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    logger.info("[visual_queue] queued %s #%s (item %d)", kind, ref_id, item.id)
    return item


def pending_count(db: Session) -> int:
    return (
        db.query(VisualEmbedItem)
        .filter(VisualEmbedItem.status == STATUS_PENDING)
        .count()
    )


def _writeback_video_asset(db: Session, item: VisualEmbedItem, vector, spec) -> bool:
    """Write the vector + model stamp onto the rag_entries row."""
    from app.memory.rag_entries_model import RAGEntry

    entry = db.query(RAGEntry).filter(RAGEntry.id == item.ref_id).first()
    if entry is None:
        item.error = f"rag_entries row {item.ref_id} gone"
        return False
    entry.embedding = struct.pack(f"{len(vector)}f", *vector)
    entry.embedding_model = spec.label
    entry.embedding_dims = len(vector)
    return True


_WRITEBACKS = {
    "video_asset": _writeback_video_asset,
}


def drain_pending(db: Session, limit: int = 25) -> dict:
    """Embed up to `limit` pending items via the multimodal role. Caller
    (idle task) is responsible for model residency (BACKGROUND_INGEST) and
    checkpointing between calls. Returns counts."""
    from app.embeddings import provider_router

    stats = {"embedded": 0, "failed": 0, "skipped_unavailable": 0}
    if not provider_router.multimodal_ready():
        stats["skipped_unavailable"] = 1
        return stats

    spec = provider_router.multimodal_write_spec()
    items = (
        db.query(VisualEmbedItem)
        .filter(VisualEmbedItem.status == STATUS_PENDING)
        .order_by(VisualEmbedItem.id.asc())
        .limit(limit)
        .all()
    )
    for item in items:
        try:
            vector = provider_router.embed_multimodal(
                text=item.text or None,
                image_path=item.image_path or None,
            )
            writeback = _WRITEBACKS.get(item.kind)
            if vector and writeback and writeback(db, item, vector, spec):
                item.status = STATUS_DONE
                item.embedded_at = datetime.utcnow()
                stats["embedded"] += 1
            else:
                item.status = STATUS_FAILED
                item.error = item.error or (
                    "no vector" if not vector else f"no writeback for kind={item.kind!r}"
                )
                stats["failed"] += 1
        except Exception as exc:
            item.status = STATUS_FAILED
            item.error = str(exc)[:500]
            stats["failed"] += 1
        db.commit()
    return stats
