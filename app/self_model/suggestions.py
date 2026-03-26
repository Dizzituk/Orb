# FILE: app/self_model/suggestions.py
"""
Suggestion Engine

Converts observed patterns into plain-language suggestions for the user.
Every suggestion is a question — it proposes, never executes.

Suggestions are categorised (capability, quality, personalisation,
maintenance, opportunity) and tracked through their lifecycle
(pending, approved, rejected, deferred).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.self_model.models import (
    ObservedPattern,
    Suggestion,
    SuggestionCategory,
    SuggestionStatus,
)

logger = logging.getLogger(__name__)


class SuggestionEngine:
    """Generates and manages suggestions from observed patterns."""

    def __init__(self) -> None:
        self._suggestions: Dict[str, Suggestion] = {}

    def create_suggestion(
        self,
        category: SuggestionCategory,
        title: str,
        description: str,
        reasoning: str,
        what_we_gain: str,
        source_patterns: Optional[List[str]] = None,
    ) -> Suggestion:
        """Create a new suggestion. It starts as pending."""
        suggestion = Suggestion(
            category=category,
            title=title,
            description=description,
            reasoning=reasoning,
            what_we_gain=what_we_gain,
            source_patterns=source_patterns or [],
        )
        self._suggestions[suggestion.suggestion_id] = suggestion
        logger.info("[self_model] New suggestion: [%s] %s", category.value, title)
        return suggestion

    def from_pattern(self, pattern: ObservedPattern) -> Optional[Suggestion]:
        """Generate a suggestion from an observed pattern."""
        if pattern.suggested:
            return None

        category = self._classify_pattern(pattern)
        title, description, reasoning, gain = self._describe_pattern(pattern)

        suggestion = self.create_suggestion(
            category=category,
            title=title,
            description=description,
            reasoning=reasoning,
            what_we_gain=gain,
            source_patterns=[pattern.pattern_id],
        )
        return suggestion

    def _classify_pattern(self, pattern: ObservedPattern) -> SuggestionCategory:
        domain = pattern.domain.lower()
        desc = pattern.description.lower()
        if "misroute" in desc or "failure" in desc or "retrieval_miss" in desc:
            return SuggestionCategory.QUALITY
        if "correction" in desc or "repeat" in desc:
            return SuggestionCategory.PERSONALISATION
        if domain in ("routing", "memory", "bridge"):
            return SuggestionCategory.QUALITY
        return SuggestionCategory.MAINTENANCE

    def _describe_pattern(self, pattern: ObservedPattern) -> tuple[str, str, str, str]:
        domain = pattern.domain
        freq = pattern.frequency
        latest = pattern.evidence[-1] if pattern.evidence else "no details"

        if "misroute" in pattern.description:
            target = pattern.description.split("misroute_")[-1] if "misroute_" in pattern.description else domain
            return (
                f"{target.title()} routing could be more accurate",
                f"I have noticed {freq} times that messages about {target} are being sent to the wrong handler. Want me to look into tightening the routing rules?",
                f"Latest example: {latest}",
                f"Your {target} requests would be understood correctly more often.",
            )

        if "retrieval_miss" in pattern.description:
            return (
                "Memory retrieval could be improved",
                f"I have noticed {freq} times that relevant context was not retrieved from memory when it should have been. Want me to investigate the retrieval scoring?",
                f"Latest example: {latest}",
                "Past conversations and decisions would appear more reliably when they are relevant.",
            )

        if "repeat" in pattern.description:
            return (
                "I am making you repeat yourself too often",
                f"I have noticed you have had to rephrase or repeat something {freq} times in this area. Want me to look at how I can understand this better?",
                f"Latest example: {latest}",
                "Less friction — I would understand what you mean the first time.",
            )

        if "user_correction" in pattern.description:
            return (
                f"Recurring corrections in {domain}",
                f"You have corrected me {freq} times about similar things in {domain}. Want me to learn from these corrections more aggressively?",
                f"Latest example: {latest}",
                "I would stop making the same mistakes.",
            )

        if "failure" in pattern.description:
            return (
                f"{domain.title()} has recurring failures",
                f"I have noticed {freq} failures in the {domain} subsystem. Want me to investigate what is going wrong?",
                f"Latest example: {latest}",
                f"The {domain} feature would be more reliable.",
            )

        return (
            f"Pattern detected in {domain}",
            f"I have noticed a recurring pattern in {domain} ({freq} occurrences). Want me to look into it?",
            f"Latest example: {latest}",
            f"The {domain} experience could be smoother.",
        )

    def approve(self, suggestion_id: str) -> Optional[Suggestion]:
        s = self._suggestions.get(suggestion_id)
        if s:
            s.status = SuggestionStatus.APPROVED
            s.resolved_at = datetime.now(timezone.utc).isoformat()
            logger.info("[self_model] Suggestion approved: %s", s.title)
        return s

    def reject(self, suggestion_id: str) -> Optional[Suggestion]:
        s = self._suggestions.get(suggestion_id)
        if s:
            s.status = SuggestionStatus.REJECTED
            s.resolved_at = datetime.now(timezone.utc).isoformat()
            logger.info("[self_model] Suggestion rejected: %s", s.title)
        return s

    def defer(self, suggestion_id: str) -> Optional[Suggestion]:
        s = self._suggestions.get(suggestion_id)
        if s:
            s.status = SuggestionStatus.DEFERRED
            logger.info("[self_model] Suggestion deferred: %s", s.title)
        return s

    def get_pending(self) -> List[Suggestion]:
        return [s for s in self._suggestions.values() if s.status == SuggestionStatus.PENDING]

    def get_all(self) -> List[Suggestion]:
        return list(self._suggestions.values())

    def summary(self) -> Dict[str, Any]:
        all_s = self.get_all()
        return {
            "total": len(all_s),
            "pending": len([s for s in all_s if s.status == SuggestionStatus.PENDING]),
            "approved": len([s for s in all_s if s.status == SuggestionStatus.APPROVED]),
            "rejected": len([s for s in all_s if s.status == SuggestionStatus.REJECTED]),
            "deferred": len([s for s in all_s if s.status == SuggestionStatus.DEFERRED]),
        }

    def to_plain_language(self) -> str:
        pending = self.get_pending()
        if not pending:
            return "No pending suggestions right now."
        lines = [f"I have {len(pending)} suggestion(s) for you:\n"]
        for s in pending:
            lines.append(f"**{s.title}**")
            lines.append(f"{s.description}")
            lines.append(f"_What we would gain: {s.what_we_gain}_\n")
        return "\n".join(lines)


# ── Singleton ─────────────────────────────────────────────

_suggestion_engine: Optional[SuggestionEngine] = None


def get_suggestion_engine() -> SuggestionEngine:
    global _suggestion_engine
    if _suggestion_engine is None:
        _suggestion_engine = SuggestionEngine()
    return _suggestion_engine
