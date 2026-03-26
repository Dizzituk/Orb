# FILE: app/self_model/journal.py
"""
Evolution Journal

A plain-language running record of what ASTRA has observed, learned,
and suggested over time. Not logs. Not code. A readable history of
how ASTRA is evolving as a system and as a partner.

Each entry records what happened, what was suggested, whether it was
approved or rejected, and what changed as a result.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.self_model.models import JournalEntry

logger = logging.getLogger(__name__)

_JOURNAL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "self_model")
_JOURNAL_FILE = os.path.join(_JOURNAL_DIR, "evolution_journal.ndjson")


class EvolutionJournal:
    """Append-only evolution journal for ASTRA."""

    def __init__(self, journal_path: Optional[str] = None) -> None:
        self._path = journal_path or _JOURNAL_FILE
        self._entries: List[JournalEntry] = []
        self._ensure_dir()
        self._load()

    def _ensure_dir(self) -> None:
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        """Load existing journal entries from disk."""
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self._entries.append(JournalEntry(**data))
            logger.info("[self_model] Loaded %d journal entries", len(self._entries))
        except Exception as e:
            logger.error("[self_model] Failed to load journal: %s", e)

    def _persist(self, entry: JournalEntry) -> None:
        """Append a single entry to the journal file."""
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as e:
            logger.error("[self_model] Failed to persist journal entry: %s", e)

    def record(
        self,
        event_type: str,
        summary: str,
        details: str = "",
        suggestion_id: Optional[str] = None,
        outcome: str = "",
    ) -> JournalEntry:
        """Record a new journal entry."""
        entry = JournalEntry(
            event_type=event_type,
            summary=summary,
            details=details,
            suggestion_id=suggestion_id,
            outcome=outcome,
        )
        self._entries.append(entry)
        self._persist(entry)
        logger.info("[self_model] Journal entry: [%s] %s", event_type, summary)
        return entry

    def record_observation(self, what_was_noticed: str, details: str = "") -> JournalEntry:
        return self.record("observation", what_was_noticed, details)

    def record_suggestion(self, suggestion_id: str, title: str, description: str) -> JournalEntry:
        return self.record("suggestion", f"Suggested: {title}", description, suggestion_id=suggestion_id)

    def record_approval(self, suggestion_id: str, title: str, outcome: str = "") -> JournalEntry:
        return self.record("approval", f"Approved: {title}", suggestion_id=suggestion_id, outcome=outcome)

    def record_rejection(self, suggestion_id: str, title: str) -> JournalEntry:
        return self.record("rejection", f"Rejected: {title}", suggestion_id=suggestion_id)

    def record_deferral(self, suggestion_id: str, title: str) -> JournalEntry:
        return self.record("deferral", f"Deferred: {title}", suggestion_id=suggestion_id)

    def record_learning(self, what_was_learned: str, source: str = "") -> JournalEntry:
        return self.record("learning", what_was_learned, source)

    def record_capability_change(self, what_changed: str, details: str = "") -> JournalEntry:
        return self.record("capability_change", what_changed, details)

    def get_recent(self, n: int = 10) -> List[JournalEntry]:
        return self._entries[-n:]

    def get_by_type(self, event_type: str) -> List[JournalEntry]:
        return [e for e in self._entries if e.event_type == event_type]

    def get_all(self) -> List[JournalEntry]:
        return list(self._entries)

    def summary(self) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        for e in self._entries:
            by_type[e.event_type] = by_type.get(e.event_type, 0) + 1
        return {
            "total_entries": len(self._entries),
            "by_type": by_type,
            "earliest": self._entries[0].date if self._entries else None,
            "latest": self._entries[-1].date if self._entries else None,
        }

    def to_plain_language(self, n: int = 10) -> str:
        recent = self.get_recent(n)
        if not recent:
            return "The evolution journal is empty — we are just getting started."
        lines = [f"Last {len(recent)} entries from the evolution journal:\n"]
        for e in recent:
            prefix = {
                "observation": "Noticed",
                "suggestion": "Suggested",
                "approval": "Approved",
                "rejection": "Rejected",
                "deferral": "Deferred",
                "learning": "Learned",
                "capability_change": "Changed",
            }.get(e.event_type, e.event_type.title())
            lines.append(f"- [{e.date}] **{prefix}**: {e.summary}")
            if e.outcome:
                lines.append(f"  Result: {e.outcome}")
        return "\n".join(lines)


# ── Singleton ─────────────────────────────────────────────

_journal: Optional[EvolutionJournal] = None


def get_journal() -> EvolutionJournal:
    global _journal
    if _journal is None:
        _journal = EvolutionJournal()
    return _journal
