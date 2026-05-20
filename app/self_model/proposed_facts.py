# FILE: app/self_model/proposed_facts.py
"""
Staging store for proposed identity-store writes.

Every Tier 1 write proposed by an auto-writer (identity_capture,
knowledge_extractor, save_to_memory, etc.) lands here first. In Phase 3
shadow mode, that's the ONLY place it lands — the proposal is logged
and the human (Taz) reviews. In Phase 4 enforce mode, accepted
proposals get committed to identity.json by the arbiter.

This file is intentionally narrow — pure storage + lifecycle. The
arbiter (write_arbiter.py) is what decides whether to call our methods.
The reviewer (proposal_review_router.py) is what surfaces them to the user.

Storage: data/self_model/proposed_facts.json.

Authoritative since Phase 3 (2026-05-02).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "self_model"
_PROPOSED_FILE = _DATA_DIR / "proposed_facts.json"
_lock = Lock()


# =============================================================================
# Statuses
# =============================================================================

# Lifecycle: shadow_logged → (queued OR auto_accepted OR rejected) → ...
#
# shadow_logged    — Phase 3 default. Arbiter saw it but did nothing.
# queued           — Phase 4 enforce: waiting on Taz to accept/reject.
# auto_accepted    — Phase 4 enforce: arbiter committed without confirmation
#                    (low-risk preference write).
# user_accepted    — Taz approved via review API; written to identity.json.
# user_rejected    — Taz rejected via review API; not written.
# expired          — Aged out (default 30 days).
# rejected_validation     — Failed schema validator; never offered to user.
# rejected_contradiction  — Conflicts with high-confidence existing value.

VALID_STATUSES = frozenset({
    "shadow_logged",
    "queued",
    "auto_accepted",
    "user_accepted",
    "user_rejected",
    "expired",
    "rejected_validation",
    "rejected_contradiction",
})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ProposedFact:
    """One proposed write to the identity store."""

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    field_name: str = ""
    proposed_value: Any = None
    current_value: Any = None  # what identity.json had at proposal time
    source: str = ""           # e.g. "identity_capture", "knowledge_extractor:session:53"
    proposed_at: str = field(default_factory=_utc_now)
    status: str = "shadow_logged"

    # What the arbiter would have done if not in shadow mode.
    would_have_status: Optional[str] = None
    would_have_reason: str = ""

    # Filled when status moves out of shadow_logged / queued.
    decided_at: Optional[str] = None
    decided_by: Optional[str] = None  # "user" or "auto" or "expiry"

    # Optional context for the reviewer.
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProposedFact":
        return cls(
            id=d.get("id") or uuid4().hex[:12],
            field_name=d.get("field_name", ""),
            proposed_value=d.get("proposed_value"),
            current_value=d.get("current_value"),
            source=d.get("source", ""),
            proposed_at=d.get("proposed_at", _utc_now()),
            status=d.get("status", "shadow_logged"),
            would_have_status=d.get("would_have_status"),
            would_have_reason=d.get("would_have_reason", ""),
            decided_at=d.get("decided_at"),
            decided_by=d.get("decided_by"),
            evidence=d.get("evidence", {}) or {},
        )


# =============================================================================
# Store
# =============================================================================

class ProposedFactsStore:
    """
    Thread-safe append-mostly store for ProposedFact records.

    Backed by a single JSON file; reads load the whole list, writes
    rewrite the whole list. Fine for the expected volume (~tens to
    low hundreds of records — old ones expire).
    """

    def __init__(self) -> None:
        self._items: List[ProposedFact] = []
        self._ensure_storage()
        self._load()

    def _ensure_storage(self) -> None:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not _PROPOSED_FILE.exists():
            _PROPOSED_FILE.write_text("[]", encoding="utf-8")

    def _load(self) -> None:
        try:
            raw = json.loads(_PROPOSED_FILE.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                logger.warning("[proposed_facts] file not a list; ignoring")
                raw = []
        except Exception as exc:
            logger.error("[proposed_facts] load failed: %s", exc)
            raw = []
        self._items = [ProposedFact.from_dict(d) for d in raw]

    def _persist(self) -> None:
        payload = [p.to_dict() for p in self._items]
        _PROPOSED_FILE.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    # ── Public API ─────────────────────────────────────────────

    def add(self, proposal: ProposedFact) -> ProposedFact:
        with _lock:
            self._items.append(proposal)
            self._persist()
        logger.info(
            "[proposed_facts] +%s field=%s status=%s would=%s",
            proposal.id, proposal.field_name, proposal.status,
            proposal.would_have_status,
        )
        return proposal

    def list(
        self,
        status: Optional[str] = None,
        field_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[ProposedFact]:
        out = list(self._items)
        if status:
            out = [p for p in out if p.status == status]
        if field_name:
            out = [p for p in out if p.field_name == field_name]
        # Newest first.
        out.sort(key=lambda p: p.proposed_at, reverse=True)
        return out[:limit]

    def get(self, proposal_id: str) -> Optional[ProposedFact]:
        for p in self._items:
            if p.id == proposal_id:
                return p
        return None

    def update_status(
        self,
        proposal_id: str,
        new_status: str,
        decided_by: str = "user",
    ) -> Optional[ProposedFact]:
        if new_status not in VALID_STATUSES:
            raise ValueError(f"unknown status {new_status!r}")
        with _lock:
            target = next((p for p in self._items if p.id == proposal_id), None)
            if not target:
                return None
            target.status = new_status
            target.decided_at = _utc_now()
            target.decided_by = decided_by
            self._persist()
        logger.info(
            "[proposed_facts] %s -> %s by=%s", proposal_id, new_status, decided_by,
        )
        return target

    def expire_old(self, max_age_days: int = 30) -> int:
        """
        Move stale shadow_logged / queued proposals to expired.
        Returns count expired.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        cutoff_iso = cutoff.isoformat()
        count = 0
        with _lock:
            for p in self._items:
                if p.status in ("shadow_logged", "queued") and p.proposed_at < cutoff_iso:
                    p.status = "expired"
                    p.decided_at = _utc_now()
                    p.decided_by = "expiry"
                    count += 1
            if count:
                self._persist()
        if count:
            logger.info("[proposed_facts] expired %d stale proposals", count)
        return count

    def stats(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for p in self._items:
            out[p.status] = out.get(p.status, 0) + 1
        out["_total"] = len(self._items)
        return out


# =============================================================================
# Singleton accessor
# =============================================================================

_store: Optional[ProposedFactsStore] = None


def get_proposed_facts_store() -> ProposedFactsStore:
    global _store
    if _store is None:
        _store = ProposedFactsStore()
    return _store
