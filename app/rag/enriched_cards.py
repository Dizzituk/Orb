# FILE: app/rag/enriched_cards.py
# Purpose: Ingest enriched architecture cards (.architecture/enriched/*.md) as retrieval chunks.
# Called-by: app.rag.rescan
# Depends-on: app.rag.models, app.rag.descriptors.descriptor_gen
# Last-renovated: 2026-07-02
"""
Enriched architecture cards -> ArchCodeChunk rows (2026-07-02).

The enriched cards (D:\\Orb\\.architecture\\enriched\\*.md) are ASTRA's
system-level self-knowledge: SYSTEM_OVERVIEW, capability ledger, subsystem /
endpoint-map / data-flow / drift cards. Unlike code chunks they are generated
on and read from the HOST — host is source of truth for the map; the sandbox
clone may lag (see the drift_note card). One card = ONE chunk, at card
granularity: front matter is parsed into fields (never shredded from the
body), the full body is stored in `docstring`, and `descriptor` (the text
that gets embedded) is pre-set here so reindex/embedding jobs treat a card
as a single retrieval unit.

Tier mapping lives in app/rag/jobs/_embedding_priority.py:
overview + capability_ledger -> Tier1_Critical, all other kinds -> Tier2_High.

Safety: cards are .md so rescan's .py-scoped diff never quarantines them;
removal is handled HERE (file gone -> chunk quarantined -> its vector is
pruned by cleanup_orphaned_embeddings). Never raises into the caller.
"""

import hashlib
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from sqlalchemy import and_
from sqlalchemy.orm import Session

from app.rag.models import ArchCodeChunk, ChunkType

logger = logging.getLogger(__name__)

ENRICHED_CARDS_DIR = Path(
    os.getenv("ASTRA_ENRICHED_CARDS_DIR", r"D:\Orb\.architecture\enriched")
)

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_DESCRIPTOR_MAX_CHARS = 1800  # embedded text: card head, safe for the embed model


def _parse_front_matter(text: str) -> Tuple[Dict[str, str], str]:
    """Split YAML-ish front matter from the body. Flat key: value lines only —
    no yaml dependency; unknown lines are ignored."""
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    meta: Dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line and not line.lstrip().startswith("-"):
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, text[match.end():]


def _latest_scan_id(db: Session) -> int:
    """Mirror rescan_codebase's scan_id resolution (latest complete run)."""
    from app.rag.models import ArchScanRun
    latest = (
        db.query(ArchScanRun)
        .filter(ArchScanRun.status == "complete")
        .order_by(ArchScanRun.id.desc())
        .first()
    )
    if latest:
        return latest.id
    from app.memory.architecture_models import ArchitectureScanRun
    latest_arch = (
        db.query(ArchitectureScanRun)
        .filter(ArchitectureScanRun.status == "completed")
        .order_by(ArchitectureScanRun.id.desc())
        .first()
    )
    return latest_arch.id if latest_arch else 1


def _card_descriptor(card_id: str, kind: str, body: str) -> str:
    """Embeddable text for a card: identity line + flattened head of the body."""
    head = " ".join(body.split())[:_DESCRIPTOR_MAX_CHARS]
    return f"ARCHITECTURE CARD [{kind}] {card_id} | {head}"


def ingest_enriched_cards(db: Session, scan_id: Optional[int] = None) -> Dict[str, int]:
    """Upsert one ArchCodeChunk per enriched card file; quarantine chunks whose
    card file is gone. Changed cards are marked embedded=False so the next
    reindex/embedding pass re-embeds them. Returns counts; never raises."""
    stats = {"seen": 0, "added": 0, "updated": 0, "unchanged": 0, "removed": 0, "errors": 0}
    try:
        if not ENRICHED_CARDS_DIR.is_dir():
            return stats
        if scan_id is None:
            scan_id = _latest_scan_id(db)

        existing = {
            c.file_path: c
            for c in db.query(ArchCodeChunk).filter(
                and_(
                    ArchCodeChunk.chunk_type == ChunkType.ARCHITECTURE_CARD,
                    ArchCodeChunk.status == "active",
                )
            ).all()
        }
        on_disk = set()

        for path in sorted(ENRICHED_CARDS_DIR.glob("*.md")):
            stats["seen"] += 1
            file_path = str(path)
            on_disk.add(file_path)
            try:
                meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
                card_id = meta.get("card_id") or path.stem
                kind = meta.get("kind", "subsystem")
                body = body.strip()
                signature = (
                    f"ARCHITECTURE CARD: {card_id} (kind={kind}, repo={meta.get('repo', '?')}, "
                    f"source=host, generated={meta.get('generated_at', '?')})"
                )
                # Same shape as jobs._embedding_job_utils_7.compute_content_hash
                content_hash = hashlib.sha256(
                    f"{card_id}|{signature}|{body}".encode()
                ).hexdigest()[:32]

                chunk = existing.get(file_path)
                if chunk is None:
                    from app.rag.descriptors.descriptor_gen import estimate_tokens
                    descriptor = _card_descriptor(card_id, kind, body)
                    db.add(ArchCodeChunk(
                        scan_id=scan_id,
                        file_path=file_path,
                        file_abs_path=file_path,
                        chunk_type=ChunkType.ARCHITECTURE_CARD,
                        chunk_name=card_id,
                        qualified_name=card_id,
                        start_line=1,
                        end_line=body.count("\n") + 1,
                        signature=signature,
                        docstring=body,
                        descriptor=descriptor,
                        descriptor_tokens=estimate_tokens(descriptor),
                        content_hash=content_hash,
                        embedded=False,
                        status="active",
                        created_at=datetime.utcnow(),
                    ))
                    stats["added"] += 1
                elif chunk.content_hash != content_hash:
                    from app.rag.descriptors.descriptor_gen import estimate_tokens
                    descriptor = _card_descriptor(card_id, kind, body)
                    chunk.chunk_name = card_id
                    chunk.qualified_name = card_id
                    chunk.signature = signature
                    chunk.docstring = body
                    chunk.end_line = body.count("\n") + 1
                    chunk.descriptor = descriptor
                    chunk.descriptor_tokens = estimate_tokens(descriptor)
                    chunk.content_hash = content_hash
                    chunk.embedded = False
                    stats["updated"] += 1
                else:
                    stats["unchanged"] += 1
            except Exception as exc:
                stats["errors"] += 1
                logger.warning("[enriched_cards] card %s failed: %s", path.name, exc)

        for file_path, chunk in existing.items():
            if file_path not in on_disk:
                chunk.status = "quarantined"
                stats["removed"] += 1

        db.commit()
        logger.info(
            "[enriched_cards] ingest: %d seen, %d added, %d updated, %d unchanged, "
            "%d removed, %d errors",
            stats["seen"], stats["added"], stats["updated"],
            stats["unchanged"], stats["removed"], stats["errors"],
        )
    except Exception as exc:
        logger.warning("[enriched_cards] ingest failed: %s", exc)
        try:
            db.rollback()
        except Exception:
            pass
    return stats


__all__ = ["ingest_enriched_cards", "ENRICHED_CARDS_DIR"]
