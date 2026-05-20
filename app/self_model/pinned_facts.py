# FILE: app/self_model/pinned_facts.py
"""
User-pinned facts — explicitly-stored memory.

Triggered by user commands like:
  - "remember that ..."
  - "save this to memory"
  - "don't forget that ..."
  - "make a note that ..."
  - "keep in mind that ..."

These bypass passive capture entirely. When a user gives an explicit
memory command, the content is stored as a pinned fact, always injected
into every prompt, and can be reviewed/deleted via the API.

Pinned facts are different from identity (fixed schema) and fragments
(passive + decay). Pinned facts are:
  - Free-form user-authored statements
  - Never decay
  - Always inject
  - Removable only by explicit user command ("forget about X")

Storage: data/self_model/pinned_facts.json
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "self_model"
_PINNED_FILE = _DATA_DIR / "pinned_facts.json"
_lock = Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PinnedFact:
    fact_id: str = field(default_factory=lambda: str(uuid4())[:10])
    statement: str = ""
    tag: str = ""          # optional category hint (e.g. "location", "relationship")
    created_at: str = field(default_factory=_utc_now)
    source: str = ""       # which session/message it came from
    superseded_by: Optional[str] = None   # if replaced by another pinned fact

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PinnedFact":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class PinnedFactStore:
    def __init__(self) -> None:
        self._facts: Dict[str, PinnedFact] = {}
        self._ensure_storage()
        self._load()

    def _ensure_storage(self) -> None:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not _PINNED_FILE.exists():
            _PINNED_FILE.write_text(
                json.dumps({"facts": []}, indent=2), encoding="utf-8",
            )

    def _load(self) -> None:
        try:
            raw = json.loads(_PINNED_FILE.read_text(encoding="utf-8"))
            for item in raw.get("facts", []):
                f = PinnedFact.from_dict(item)
                self._facts[f.fact_id] = f
        except Exception as exc:
            logger.error("[pinned_facts] load failed: %s", exc)

    def _persist(self) -> None:
        payload = {"facts": [f.to_dict() for f in self._facts.values()]}
        _PINNED_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def pin(self, statement: str, tag: str = "", source: str = "") -> PinnedFact:
        with _lock:
            # Dedup by normalised statement text (case-insensitive trim)
            key = statement.strip().lower()
            for f in self._facts.values():
                if f.statement.strip().lower() == key and not f.superseded_by:
                    logger.info("[pinned_facts] duplicate ignored: %s", statement[:60])
                    return f
            f = PinnedFact(statement=statement.strip(), tag=tag, source=source)
            self._facts[f.fact_id] = f
            self._persist()
            logger.info(
                "[pinned_facts] pinned %s: %r (tag=%s)",
                f.fact_id, statement[:80], tag or "-",
            )
            return f

    def unpin(self, fact_id: str) -> bool:
        with _lock:
            if fact_id not in self._facts:
                return False
            del self._facts[fact_id]
            self._persist()
            logger.info("[pinned_facts] unpinned %s", fact_id)
            return True

    def supersede(self, old_id: str, new_statement: str, source: str = "") -> Optional[PinnedFact]:
        """Mark an old fact as superseded by a new one; both remain queryable."""
        with _lock:
            old = self._facts.get(old_id)
            if not old:
                return None
            new = PinnedFact(statement=new_statement.strip(), tag=old.tag, source=source)
            self._facts[new.fact_id] = new
            old.superseded_by = new.fact_id
            self._persist()
            logger.info(
                "[pinned_facts] %s superseded by %s",
                old_id, new.fact_id,
            )
            return new

    def all(self) -> List[PinnedFact]:
        return list(self._facts.values())

    def active(self) -> List[PinnedFact]:
        return [f for f in self._facts.values() if not f.superseded_by]

    def get(self, fact_id: str) -> Optional[PinnedFact]:
        return self._facts.get(fact_id)

    def summary(self) -> Dict[str, Any]:
        return {
            "total":    len(self._facts),
            "active":   len(self.active()),
            "storage":  str(_PINNED_FILE),
        }


_store: Optional[PinnedFactStore] = None


def get_pinned_store() -> PinnedFactStore:
    global _store
    if _store is None:
        _store = PinnedFactStore()
    return _store