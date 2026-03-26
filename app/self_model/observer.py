# FILE: app/self_model/observer.py
"""
Pillar 3: Pattern Observer

Detects recurring friction, failures, missed opportunities, and
behavioural patterns across ASTRA's subsystems. Feeds observations
into the suggestion engine.

The observer watches but never acts. It collects evidence and
flags patterns that cross a threshold of significance.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.self_model.models import ObservedPattern

logger = logging.getLogger(__name__)

# Minimum occurrences before a pattern is considered significant
PATTERN_THRESHOLD = 3


class PatternObserver:
    """Observes and records patterns across ASTRA's behaviour."""

    def __init__(self) -> None:
        self._patterns: Dict[str, ObservedPattern] = {}

    def record_event(self, domain: str, event_type: str, detail: str) -> Optional[ObservedPattern]:
        """Record an event and check if it forms or reinforces a pattern."""
        pattern_key = f"{domain}:{event_type}"
        now = datetime.now(timezone.utc).isoformat()

        existing = self._patterns.get(pattern_key)
        if existing:
            existing.frequency += 1
            existing.last_observed = now
            existing.evidence.append(detail)
            if len(existing.evidence) > 20:
                existing.evidence = existing.evidence[-20:]
            if existing.frequency >= PATTERN_THRESHOLD and not existing.actionable:
                existing.actionable = True
                logger.info(
                    "[self_model] Pattern became actionable: %s (freq=%d)",
                    pattern_key, existing.frequency,
                )
            return existing

        pattern = ObservedPattern(
            domain=domain,
            description=f"Recurring {event_type} in {domain}",
            frequency=1,
            first_observed=now,
            last_observed=now,
            evidence=[detail],
        )
        self._patterns[pattern_key] = pattern
        return pattern

    def record_routing_miss(self, intended_domain: str, actual_domain: str, user_input: str) -> ObservedPattern:
        """Record when routing sends a message to the wrong domain."""
        detail = f"Input '{user_input[:80]}...' was routed to {actual_domain} but should have been {intended_domain}"
        return self.record_event("routing", f"misroute_{intended_domain}", detail)

    def record_memory_miss(self, query: str, expected_topic: str) -> ObservedPattern:
        """Record when memory retrieval misses relevant context."""
        detail = f"Query about '{expected_topic}' did not retrieve relevant memory for: {query[:80]}"
        return self.record_event("memory", f"retrieval_miss_{expected_topic}", detail)

    def record_repeated_question(self, topic: str) -> ObservedPattern:
        """Record when the user has to repeat or rephrase something."""
        detail = f"User had to repeat/rephrase about: {topic}"
        return self.record_event("understanding", f"repeat_{topic}", detail)

    def record_subsystem_failure(self, subsystem: str, error_summary: str) -> ObservedPattern:
        """Record a subsystem failure for trend detection."""
        detail = f"{subsystem} failed: {error_summary[:100]}"
        return self.record_event(subsystem, "failure", detail)

    def record_user_correction(self, domain: str, what_was_wrong: str) -> ObservedPattern:
        """Record when the user corrects ASTRA's behaviour."""
        detail = f"User correction in {domain}: {what_was_wrong[:100]}"
        return self.record_event(domain, "user_correction", detail)

    def get_actionable_patterns(self) -> List[ObservedPattern]:
        """Get patterns that have crossed the significance threshold."""
        return [p for p in self._patterns.values() if p.actionable and not p.suggested]

    def get_all_patterns(self) -> List[ObservedPattern]:
        return list(self._patterns.values())

    def get_patterns_by_domain(self, domain: str) -> List[ObservedPattern]:
        return [p for p in self._patterns.values() if p.domain == domain]

    def mark_suggested(self, pattern_id: str) -> None:
        """Mark a pattern as having been turned into a suggestion."""
        for p in self._patterns.values():
            if p.pattern_id == pattern_id:
                p.suggested = True
                break

    def summary(self) -> Dict[str, Any]:
        patterns = self.get_all_patterns()
        actionable = self.get_actionable_patterns()
        domains = set(p.domain for p in patterns)
        return {
            "total_patterns": len(patterns),
            "actionable": len(actionable),
            "domains_affected": sorted(domains),
            "top_patterns": [
                {"domain": p.domain, "description": p.description, "frequency": p.frequency}
                for p in sorted(patterns, key=lambda x: x.frequency, reverse=True)[:5]
            ],
        }

    def to_plain_language(self) -> str:
        actionable = self.get_actionable_patterns()
        if not actionable:
            return "I have not noticed any significant recurring patterns yet."
        lines = [f"I have noticed {len(actionable)} recurring pattern(s):\n"]
        for p in actionable:
            lines.append(f"- **{p.domain}**: {p.description} (seen {p.frequency} times)")
            if p.evidence:
                lines.append(f"  Latest: {p.evidence[-1]}")
        return "\n".join(lines)


# ── Singleton ─────────────────────────────────────────────

_observer: Optional[PatternObserver] = None


def get_observer() -> PatternObserver:
    global _observer
    if _observer is None:
        _observer = PatternObserver()
    return _observer
