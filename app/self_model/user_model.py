# FILE: app/self_model/user_model.py
"""
Pillar 2: User Model

Transparent model of what ASTRA knows about the user. Built from
memory extraction, conversation patterns, and explicit corrections.

The user model makes memory visible and conversational. Instead of
knowledge being a silent background process, ASTRA can surface what
it knows, explain where it came from, and accept corrections.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.self_model.models import ConfidenceLevel, UserFact

logger = logging.getLogger(__name__)


class UserModel:
    """Transparent model of user knowledge."""

    def __init__(self) -> None:
        self._facts: Dict[str, UserFact] = {}
        self._seed_known_facts()

    def _seed_known_facts(self) -> None:
        """Seed with facts ASTRA already knows from memory."""
        now = datetime.now(timezone.utc).isoformat()
        known = [
            UserFact(
                category="biographical", key="name", value="Taz",
                confidence=ConfidenceLevel.HIGH, reinforcement_count=50,
                source="multiple sessions", first_seen="2025-11-01", last_seen=now,
            ),
            UserFact(
                category="biographical", key="location", value="Cornwall, UK",
                confidence=ConfidenceLevel.HIGH, reinforcement_count=20,
                source="multiple sessions", first_seen="2025-11-01", last_seen=now,
            ),
            UserFact(
                category="biographical", key="occupation", value="Self-employed delivery driver (Yodel) building ASTRA",
                confidence=ConfidenceLevel.HIGH, reinforcement_count=30,
                source="multiple sessions", first_seen="2025-11-01", last_seen=now,
            ),
            UserFact(
                category="biographical", key="background", value="17 years catering management, 10 years personal training",
                confidence=ConfidenceLevel.HIGH, reinforcement_count=10,
                source="early sessions", first_seen="2025-11-01", last_seen=now,
            ),
            UserFact(
                category="preference", key="file_size", value="Target 20KB, max 30KB per file. Split automatically if exceeded.",
                confidence=ConfidenceLevel.HIGH, reinforcement_count=50,
                source="enforced rule", first_seen="2025-12-01", last_seen=now,
            ),
            UserFact(
                category="preference", key="communication_style", value="Plain language, no jargon. Direct and honest. Explain like a partner, not a teacher.",
                confidence=ConfidenceLevel.HIGH, reinforcement_count=20,
                source="multiple sessions", first_seen="2025-11-01", last_seen=now,
            ),
            UserFact(
                category="preference", key="pushback", value="Always challenge views, present counter-arguments, steelman the other side. Never agree by default.",
                confidence=ConfidenceLevel.HIGH, reinforcement_count=10,
                source="explicit instruction", first_seen="2026-01-01", last_seen=now,
            ),
            UserFact(
                category="philosophy", key="partnership", value="ASTRA and user are partners. ASTRA never acts autonomously. Everything requires approval.",
                confidence=ConfidenceLevel.HIGH, reinforcement_count=5,
                source="self-model spec discussion", first_seen="2026-03-26", last_seen=now,
            ),
            UserFact(
                category="philosophy", key="git_rule", value="ASTRA never runs git commands. All version control is done by the user.",
                confidence=ConfidenceLevel.HIGH, reinforcement_count=30,
                source="hard rule", first_seen="2025-12-01", last_seen=now,
            ),
            UserFact(
                category="project", key="driver_copilot", value="First commercial app. OBD2 van health, OCR receipts, route optimisation, accounting. Android target.",
                confidence=ConfidenceLevel.HIGH, reinforcement_count=15,
                source="multiple sessions", first_seen="2026-01-01", last_seen=now,
            ),
            UserFact(
                category="project", key="portugal_move", value="Long-term goal: relocate to Portugal via D7 visa funded by ASTRA passive income.",
                confidence=ConfidenceLevel.MEDIUM, reinforcement_count=5,
                source="multiple sessions", first_seen="2026-01-01", last_seen=now,
            ),
            UserFact(
                category="pattern", key="work_style", value="Works from van during delivery routes using voice-to-text. Desktop sessions at home.",
                confidence=ConfidenceLevel.HIGH, reinforcement_count=20,
                source="multiple sessions", first_seen="2025-11-01", last_seen=now,
            ),
            UserFact(
                category="pattern", key="thinking_style", value="Cross-domain pattern recognition. Neurodivergent (dyslexia, ADHD). Systems thinker.",
                confidence=ConfidenceLevel.HIGH, reinforcement_count=10,
                source="multiple sessions", first_seen="2025-11-01", last_seen=now,
            ),
        ]
        for fact in known:
            self._facts[f"{fact.category}:{fact.key}"] = fact

    def get_all(self) -> List[UserFact]:
        return list(self._facts.values())

    def get_by_category(self, category: str) -> List[UserFact]:
        return [f for f in self._facts.values() if f.category == category]

    def get_fact(self, category: str, key: str) -> Optional[UserFact]:
        return self._facts.get(f"{category}:{key}")

    def add_fact(self, fact: UserFact) -> None:
        composite_key = f"{fact.category}:{fact.key}"
        existing = self._facts.get(composite_key)
        if existing:
            existing.value = fact.value
            existing.reinforcement_count += 1
            existing.last_seen = datetime.now(timezone.utc).isoformat()
            if existing.reinforcement_count >= 3:
                existing.confidence = ConfidenceLevel.HIGH
            elif existing.reinforcement_count >= 2:
                existing.confidence = ConfidenceLevel.MEDIUM
            logger.info("[self_model] Reinforced user fact: %s (count=%d)", composite_key, existing.reinforcement_count)
        else:
            fact.first_seen = fact.first_seen or datetime.now(timezone.utc).isoformat()
            fact.last_seen = datetime.now(timezone.utc).isoformat()
            self._facts[composite_key] = fact
            logger.info("[self_model] New user fact: %s", composite_key)

    def correct_fact(self, category: str, key: str, new_value: str) -> Optional[UserFact]:
        composite_key = f"{category}:{key}"
        fact = self._facts.get(composite_key)
        if fact:
            old_value = fact.value
            fact.value = new_value
            fact.last_seen = datetime.now(timezone.utc).isoformat()
            logger.info("[self_model] Corrected fact %s: '%s' -> '%s'", composite_key, old_value, new_value)
        return fact

    def remove_fact(self, category: str, key: str) -> bool:
        composite_key = f"{category}:{key}"
        if composite_key in self._facts:
            del self._facts[composite_key]
            logger.info("[self_model] Removed fact: %s", composite_key)
            return True
        return False

    def summary(self) -> Dict[str, Any]:
        facts = self.get_all()
        by_category: Dict[str, int] = {}
        for f in facts:
            by_category[f.category] = by_category.get(f.category, 0) + 1
        return {
            "total_facts": len(facts),
            "by_category": by_category,
            "high_confidence": len([f for f in facts if f.confidence == ConfidenceLevel.HIGH]),
            "medium_confidence": len([f for f in facts if f.confidence == ConfidenceLevel.MEDIUM]),
            "low_confidence": len([f for f in facts if f.confidence == ConfidenceLevel.LOW]),
        }

    def to_plain_language(self) -> str:
        lines = ["Here is what I currently know about you:\n"]
        for category in ["biographical", "preference", "philosophy", "project", "pattern"]:
            facts = self.get_by_category(category)
            if facts:
                lines.append(f"**{category.title()}:**")
                for f in facts:
                    conf = f.confidence.value
                    lines.append(f"- {f.key}: {f.value} (confidence: {conf}, reinforced {f.reinforcement_count} times)")
                lines.append("")
        return "\n".join(lines)


# ── Singleton ─────────────────────────────────────────────

_user_model: Optional[UserModel] = None


def get_user_model() -> UserModel:
    global _user_model
    if _user_model is None:
        _user_model = UserModel()
    return _user_model
