# Purpose: observer
# Called-by: app.self_model, app.self_model.chat_integration, app.self_model.hooks, app.self_model.router
# Depends-on: app.self_model.journal, app.self_model.models, app.self_model.suggestions
# Last-renovated: 2026-06-11
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.self_model.models import ObservedPattern

logger = logging.getLogger(__name__)
PATTERN_THRESHOLD = 3
_DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "self_model" / "patterns.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PatternObserver:
    def __init__(self) -> None:
        self._patterns: Dict[str, ObservedPattern] = {}
        self._ensure_storage()
        self._load()

    def _ensure_storage(self) -> None:
        _DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not _DATA_FILE.exists():
            _DATA_FILE.write_text("[]", encoding="utf-8")

    def _load(self) -> None:
        try:
            raw = json.loads(_DATA_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("[self_model] Failed to load patterns: %s", exc)
            raw = []
        self._patterns = {}
        for item in raw:
            pattern = ObservedPattern(**item)
            self._patterns[self._make_key(pattern.domain, pattern.description)] = pattern

    def _persist(self) -> None:
        payload = [pattern.to_dict() for pattern in self.get_all_patterns()]
        _DATA_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _make_key(domain: str, description: str) -> str:
        return f"{domain}:{description}"

    def record_event(self, domain: str, event_type: str, detail: str) -> ObservedPattern:
        description = f"Recurring {event_type} in {domain}"
        pattern_key = self._make_key(domain, description)
        now = _utc_now()
        existing = self._patterns.get(pattern_key)
        became_actionable = False
        if existing:
            existing.frequency += 1
            existing.last_observed = now
            existing.evidence.append(detail)
            existing.evidence = existing.evidence[-20:]
            if existing.frequency >= PATTERN_THRESHOLD and not existing.actionable:
                existing.actionable = True
                became_actionable = True
            pattern = existing
        else:
            pattern = ObservedPattern(
                domain=domain,
                description=description,
                frequency=1,
                first_observed=now,
                last_observed=now,
                evidence=[detail],
                actionable=False,
                suggested=False,
            )
            self._patterns[pattern_key] = pattern
        self._persist()
        if became_actionable:
            self._on_pattern_became_actionable(pattern)
        return pattern

    def _on_pattern_became_actionable(self, pattern: ObservedPattern) -> None:
        logger.info("[self_model] Pattern became actionable: %s", pattern.description)
        try:
            from app.self_model.journal import get_journal
            from app.self_model.suggestions import get_suggestion_engine
        except Exception:
            return
        get_journal().record_observation(pattern.description, pattern.evidence[-1] if pattern.evidence else "")
        suggestion = get_suggestion_engine().from_pattern(pattern)
        if suggestion:
            pattern.suggested = True
            self._persist()
            get_journal().record_suggestion(suggestion.suggestion_id, suggestion.title, suggestion.description)

    def record_routing_miss(self, intended_domain: str, actual_domain: str, user_input: str) -> ObservedPattern:
        detail = f"Input '{user_input[:80]}' was routed to {actual_domain} but should have been {intended_domain}"
        return self.record_event("routing", f"misroute_{intended_domain}", detail)

    def record_memory_miss(self, query: str, expected_topic: str) -> ObservedPattern:
        detail = f"Query about '{expected_topic}' did not retrieve relevant memory for: {query[:80]}"
        return self.record_event("memory", f"retrieval_miss_{expected_topic}", detail)

    def record_repeated_question(self, topic: str) -> ObservedPattern:
        return self.record_event("understanding", f"repeat_{topic}", f"User had to repeat or rephrase about: {topic}")

    def record_subsystem_failure(self, subsystem: str, error_summary: str) -> ObservedPattern:
        return self.record_event(subsystem, "failure", f"{subsystem} failed: {error_summary[:100]}")

    def record_user_correction(self, domain: str, what_was_wrong: str) -> ObservedPattern:
        return self.record_event(domain, "user_correction", f"User correction in {domain}: {what_was_wrong[:100]}")

    def get_actionable_patterns(self) -> List[ObservedPattern]:
        return [p for p in self.get_all_patterns() if p.actionable]

    def get_all_patterns(self) -> List[ObservedPattern]:
        return sorted(self._patterns.values(), key=lambda pattern: pattern.last_observed, reverse=True)

    def get_patterns_by_domain(self, domain: str) -> List[ObservedPattern]:
        return [p for p in self.get_all_patterns() if p.domain == domain]

    def mark_suggested(self, pattern_id: str) -> None:
        for pattern in self._patterns.values():
            if pattern.pattern_id == pattern_id:
                pattern.suggested = True
                self._persist()
                return

    def summary(self) -> Dict[str, Any]:
        patterns = self.get_all_patterns()
        actionable = [p for p in patterns if p.actionable]
        return {
            "total_patterns": len(patterns),
            "actionable": len(actionable),
            "domains_affected": sorted({p.domain for p in patterns}),
            "top_patterns": [
                {"domain": p.domain, "description": p.description, "frequency": p.frequency}
                for p in sorted(patterns, key=lambda item: item.frequency, reverse=True)[:5]
            ],
            "storage_path": str(_DATA_FILE),
        }

    def to_plain_language(self) -> str:
        actionable = self.get_actionable_patterns()
        if not actionable:
            return "I have not noticed any significant recurring patterns yet."
        lines = [f"I have noticed {len(actionable)} recurring pattern(s):\n"]
        for pattern in actionable:
            lines.append(f"- **{pattern.domain}**: {pattern.description} (seen {pattern.frequency} times)")
            if pattern.evidence:
                lines.append(f"  Latest: {pattern.evidence[-1]}")
        return "\n".join(lines)


_observer: Optional[PatternObserver] = None


def get_observer() -> PatternObserver:
    global _observer
    if _observer is None:
        _observer = PatternObserver()
    return _observer
