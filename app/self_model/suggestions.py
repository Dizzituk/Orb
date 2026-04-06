from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.self_model.models import ObservedPattern, Suggestion, SuggestionCategory, SuggestionStatus

logger = logging.getLogger(__name__)
_DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "self_model" / "suggestions.json"


class SuggestionEngine:
    def __init__(self) -> None:
        self._suggestions: Dict[str, Suggestion] = {}
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
            logger.error("[self_model] Failed to load suggestions: %s", exc)
            raw = []
        self._suggestions = {}
        for item in raw:
            item = dict(item)
            item["category"] = SuggestionCategory(item.get("category", "quality"))
            item["status"] = SuggestionStatus(item.get("status", "pending"))
            suggestion = Suggestion(**item)
            self._suggestions[suggestion.suggestion_id] = suggestion

    def _persist(self) -> None:
        payload = [suggestion.to_dict() for suggestion in self.get_all()]
        _DATA_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _find_existing_for_pattern(self, pattern_id: str) -> Optional[Suggestion]:
        for suggestion in self._suggestions.values():
            if pattern_id in suggestion.source_patterns:
                return suggestion
        return None

    def create_suggestion(self, category: SuggestionCategory, title: str, description: str, reasoning: str, what_we_gain: str, source_patterns: Optional[List[str]] = None) -> Suggestion:
        suggestion = Suggestion(category=category, title=title, description=description, reasoning=reasoning, what_we_gain=what_we_gain, source_patterns=source_patterns or [])
        self._suggestions[suggestion.suggestion_id] = suggestion
        self._persist()
        logger.info("[self_model] New suggestion: [%s] %s", category.value, title)
        return suggestion

    def from_pattern(self, pattern: ObservedPattern) -> Optional[Suggestion]:
        if pattern.suggested:
            return None
        existing = self._find_existing_for_pattern(pattern.pattern_id)
        if existing:
            return existing
        category = self._classify_pattern(pattern)
        title, description, reasoning, gain = self._describe_pattern(pattern)
        return self.create_suggestion(category, title, description, reasoning, gain, [pattern.pattern_id])

    def _classify_pattern(self, pattern: ObservedPattern) -> SuggestionCategory:
        desc = pattern.description.lower()
        if any(token in desc for token in ["misroute", "failure", "retrieval_miss"]):
            return SuggestionCategory.QUALITY
        if any(token in desc for token in ["correction", "repeat"]):
            return SuggestionCategory.PERSONALISATION
        return SuggestionCategory.MAINTENANCE

    def _describe_pattern(self, pattern: ObservedPattern) -> tuple[str, str, str, str]:
        latest = pattern.evidence[-1] if pattern.evidence else "No evidence recorded."
        freq = pattern.frequency
        desc = pattern.description.lower()
        if "misroute" in desc:
            return (
                "Routing is drifting",
                f"I have noticed {freq} routing mistakes in the same area. Want me to inspect the routing rules?",
                f"Latest example: {latest}",
                "You would spend less time correcting me and more time moving forward.",
            )
        if "retrieval_miss" in desc:
            return (
                "Memory retrieval needs attention",
                f"I have missed relevant memory {freq} times in the same area. Want me to inspect retrieval scoring and indexing?",
                f"Latest example: {latest}",
                "Past context would show up more reliably when it matters.",
            )
        if "user_correction" in desc or "repeat" in desc:
            return (
                "I am creating repeated friction",
                f"You have had to correct or repeat yourself {freq} times around this pattern. Want me to tighten how I learn from it?",
                f"Latest example: {latest}",
                "I would adapt faster to how you work and think.",
            )
        if "failure" in desc:
            return (
                f"{pattern.domain.title()} has recurring failures",
                f"I have seen {freq} failures in {pattern.domain}. Want me to investigate the failure chain?",
                f"Latest example: {latest}",
                f"The {pattern.domain} subsystem would become more reliable.",
            )
        return (
            f"Pattern detected in {pattern.domain}",
            f"I have seen a recurring pattern in {pattern.domain} {freq} times. Want me to inspect it?",
            f"Latest example: {latest}",
            f"The {pattern.domain} experience could become smoother.",
        )

    def approve(self, suggestion_id: str) -> Optional[Suggestion]:
        suggestion = self._suggestions.get(suggestion_id)
        if suggestion:
            suggestion.status = SuggestionStatus.APPROVED
            suggestion.resolved_at = datetime.now(timezone.utc).isoformat()
            self._persist()
        return suggestion

    def reject(self, suggestion_id: str) -> Optional[Suggestion]:
        suggestion = self._suggestions.get(suggestion_id)
        if suggestion:
            suggestion.status = SuggestionStatus.REJECTED
            suggestion.resolved_at = datetime.now(timezone.utc).isoformat()
            self._persist()
        return suggestion

    def defer(self, suggestion_id: str) -> Optional[Suggestion]:
        suggestion = self._suggestions.get(suggestion_id)
        if suggestion:
            suggestion.status = SuggestionStatus.DEFERRED
            self._persist()
        return suggestion

    def get_pending(self) -> List[Suggestion]:
        return [item for item in self.get_all() if item.status == SuggestionStatus.PENDING]

    def get_all(self) -> List[Suggestion]:
        return sorted(self._suggestions.values(), key=lambda item: item.created_at, reverse=True)

    def summary(self) -> Dict[str, Any]:
        all_items = self.get_all()
        return {
            "total": len(all_items),
            "pending": len([s for s in all_items if s.status == SuggestionStatus.PENDING]),
            "approved": len([s for s in all_items if s.status == SuggestionStatus.APPROVED]),
            "rejected": len([s for s in all_items if s.status == SuggestionStatus.REJECTED]),
            "deferred": len([s for s in all_items if s.status == SuggestionStatus.DEFERRED]),
            "storage_path": str(_DATA_FILE),
        }

    def to_plain_language(self) -> str:
        pending = self.get_pending()
        if not pending:
            return "No pending suggestions right now."
        lines = [f"I have {len(pending)} suggestion(s) for you:\n"]
        for suggestion in pending:
            lines.append(f"**{suggestion.title}**")
            lines.append(suggestion.description)
            lines.append(f"_What we would gain: {suggestion.what_we_gain}_\n")
        return "\n".join(lines)


_suggestion_engine: Optional[SuggestionEngine] = None


def get_suggestion_engine() -> SuggestionEngine:
    global _suggestion_engine
    if _suggestion_engine is None:
        _suggestion_engine = SuggestionEngine()
    return _suggestion_engine
