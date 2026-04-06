from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.self_model.models import ConfidenceLevel, UserFact

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "self_model"
_FACTS_FILE = _DATA_DIR / "user_facts.json"
_BOOTSTRAP_FILE = _DATA_DIR / "bootstrap_user_facts.json"

_BOOTSTRAP_FACTS: List[Dict[str, Any]] = [
    {"category": "biographical", "key": "name", "value": "Taz", "confidence": "high", "source": "bootstrap", "reinforcement_count": 1},
    {"category": "preference", "key": "communication_style", "value": "Plain language, direct, partner-like, no jargon.", "confidence": "high", "source": "bootstrap", "reinforcement_count": 1},
    {"category": "preference", "key": "file_size", "value": "Target 20KB per file, hard max 30KB, split when needed.", "confidence": "high", "source": "bootstrap", "reinforcement_count": 1},
    {"category": "philosophy", "key": "partnership", "value": "ASTRA and Taz are partners. ASTRA proposes and waits for approval.", "confidence": "high", "source": "bootstrap", "reinforcement_count": 1},
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _confidence_from_value(value: str) -> ConfidenceLevel:
    try:
        return ConfidenceLevel(value)
    except Exception:
        return ConfidenceLevel.LOW


class UserModel:
    def __init__(self) -> None:
        self._facts: Dict[str, UserFact] = {}
        self._ensure_storage()
        self._load()

    def _ensure_storage(self) -> None:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not _BOOTSTRAP_FILE.exists():
            _BOOTSTRAP_FILE.write_text(json.dumps(_BOOTSTRAP_FACTS, indent=2), encoding="utf-8")
        if not _FACTS_FILE.exists():
            _FACTS_FILE.write_text(json.dumps(_BOOTSTRAP_FACTS, indent=2), encoding="utf-8")

    def _load(self) -> None:
        try:
            raw = json.loads(_FACTS_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("[self_model] Failed to load user facts: %s", exc)
            raw = list(_BOOTSTRAP_FACTS)
        self._facts = {}
        now = _utc_now()
        for item in raw:
            fact = UserFact(
                category=item.get("category", "unknown"),
                key=item.get("key", "unknown"),
                value=item.get("value", ""),
                confidence=_confidence_from_value(item.get("confidence", "low")),
                source=item.get("source", ""),
                reinforcement_count=int(item.get("reinforcement_count", 0)),
                first_seen=item.get("first_seen") or now,
                last_seen=item.get("last_seen") or now,
            )
            self._facts[self._make_key(fact.category, fact.key)] = fact

    def _persist(self) -> None:
        payload = [fact.to_dict() for fact in self.get_all()]
        _FACTS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _make_key(category: str, key: str) -> str:
        return f"{category}:{key}"

    def get_all(self) -> List[UserFact]:
        return sorted(self._facts.values(), key=lambda fact: (fact.category, fact.key))

    def get_by_category(self, category: str) -> List[UserFact]:
        return [f for f in self.get_all() if f.category == category]

    def get_fact(self, category: str, key: str) -> Optional[UserFact]:
        return self._facts.get(self._make_key(category, key))

    def add_fact(self, fact: UserFact) -> UserFact:
        composite_key = self._make_key(fact.category, fact.key)
        existing = self._facts.get(composite_key)
        now = _utc_now()
        if existing:
            existing.value = fact.value
            existing.source = fact.source or existing.source
            existing.reinforcement_count += 1
            existing.last_seen = now
            if existing.reinforcement_count >= 5:
                existing.confidence = ConfidenceLevel.HIGH
            elif existing.reinforcement_count >= 2:
                existing.confidence = ConfidenceLevel.MEDIUM
            self._persist()
            logger.info("[self_model] Reinforced user fact: %s", composite_key)
            return existing
        fact.first_seen = fact.first_seen or now
        fact.last_seen = now
        fact.reinforcement_count = max(1, fact.reinforcement_count)
        self._facts[composite_key] = fact
        self._persist()
        logger.info("[self_model] Added user fact: %s", composite_key)
        return fact

    def correct_fact(self, category: str, key: str, new_value: str) -> Optional[UserFact]:
        fact = self.get_fact(category, key)
        if not fact:
            return None
        fact.value = new_value
        fact.last_seen = _utc_now()
        fact.source = "user_correction"
        fact.confidence = ConfidenceLevel.HIGH
        fact.reinforcement_count = max(fact.reinforcement_count, 1)
        self._persist()
        logger.info("[self_model] Corrected fact: %s", self._make_key(category, key))
        return fact

    def remove_fact(self, category: str, key: str) -> bool:
        composite_key = self._make_key(category, key)
        if composite_key not in self._facts:
            return False
        del self._facts[composite_key]
        self._persist()
        logger.info("[self_model] Removed fact: %s", composite_key)
        return True

    def summary(self) -> Dict[str, Any]:
        facts = self.get_all()
        by_category: Dict[str, int] = {}
        for fact in facts:
            by_category[fact.category] = by_category.get(fact.category, 0) + 1
        return {
            "total_facts": len(facts),
            "by_category": by_category,
            "high_confidence": len([f for f in facts if f.confidence == ConfidenceLevel.HIGH]),
            "medium_confidence": len([f for f in facts if f.confidence == ConfidenceLevel.MEDIUM]),
            "low_confidence": len([f for f in facts if f.confidence == ConfidenceLevel.LOW]),
            "storage_path": str(_FACTS_FILE),
        }

    def to_plain_language(self) -> str:
        lines = ["Here is what I currently know about you:\n"]
        for category in ["biographical", "preference", "philosophy", "project", "pattern", "learning"]:
            facts = self.get_by_category(category)
            if not facts:
                continue
            lines.append(f"**{category.title()}:**")
            for fact in facts:
                lines.append(
                    f"- {fact.key}: {fact.value} (confidence: {fact.confidence.value}, source: {fact.source or 'unknown'}, updated: {fact.last_seen})"
                )
            lines.append("")
        return "\n".join(lines)


_user_model: Optional[UserModel] = None


def get_user_model() -> UserModel:
    global _user_model
    if _user_model is None:
        _user_model = UserModel()
    return _user_model
