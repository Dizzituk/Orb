# FILE: app/optimize/pattern_learner.py
# Purpose: Pattern Learner — codifies successful optimisations into reusable rules.
# Called-by: app.optimize.orchestrator, app.optimize.router
# Depends-on: app.intelligent_memory.graph
# Last-renovated: 2026-06-11
"""
Pattern Learner — codifies successful optimisations into reusable rules.

When an optimisation passes BVL and gets promoted, the pattern learner:
  1. Extracts the anti-pattern that was detected
  2. Extracts the fix that was applied
  3. Creates a pattern rule for the Scaffold Engine
  4. Registers the pattern in the Knowledge Graph

This creates the compound improvement loop: every optimisation makes
every future build better because the Scaffold Engine avoids the
anti-pattern from the start.

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-OPT-001.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PATTERNS_PATH = os.getenv(
    "ASTRA_PATTERNS_PATH",
    os.path.join("D:", os.sep, "Orb", "data", "optimization_patterns.json"),
)


@dataclass
class PatternRule:
    """A learned optimisation pattern."""
    pattern_id: str
    anti_pattern: str       # What was wrong
    fix_description: str    # What was done to fix it
    category: str           # Proposal category that generated this
    detection_hint: str     # How to detect this in future code
    confidence: float = 0.7
    times_applied: int = 0
    source_proposal_id: str = ""
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "anti_pattern": self.anti_pattern,
            "fix_description": self.fix_description,
            "category": self.category,
            "detection_hint": self.detection_hint,
            "confidence": self.confidence,
            "times_applied": self.times_applied,
            "source_proposal_id": self.source_proposal_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatternRule":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class PatternLearner:
    """Learns and stores optimisation patterns."""

    def __init__(self, patterns_path: str = PATTERNS_PATH):
        self._patterns_path = patterns_path
        self._patterns: Dict[str, PatternRule] = {}
        self._load()

    def learn_from_execution(
        self,
        proposal: Any,
        execution_result: Any,
    ) -> Optional[PatternRule]:
        """Learn a pattern from a successful optimisation execution.

        Only learns from executions that passed BVL.
        """
        if not execution_result.success:
            return None

        import hashlib
        pid = hashlib.sha256(
            f"{proposal.category.value}:{proposal.title}".encode()
        ).hexdigest()[:10]
        pattern_id = f"pat-{pid}"

        # Don't duplicate
        if pattern_id in self._patterns:
            self._patterns[pattern_id].times_applied += 1
            self._persist()
            return self._patterns[pattern_id]

        rule = PatternRule(
            pattern_id=pattern_id,
            anti_pattern=proposal.description,
            fix_description=proposal.predicted_improvement,
            category=proposal.category.value,
            detection_hint=_build_detection_hint(proposal),
            confidence=proposal.confidence,
            source_proposal_id=proposal.proposal_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self._patterns[pattern_id] = rule
        self._persist()
        self._register_in_graph(rule)

        logger.info("[pattern_learner] Learned: %s — %s",
                     pattern_id, rule.anti_pattern[:60])
        return rule

    def get_patterns(
        self,
        category: str = "",
        min_confidence: float = 0.0,
    ) -> List[PatternRule]:
        """Get learned patterns, optionally filtered."""
        results = []
        for rule in self._patterns.values():
            if category and rule.category != category:
                continue
            if rule.confidence < min_confidence:
                continue
            results.append(rule)
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def get_scaffold_rules(self) -> List[Dict[str, str]]:
        """Format patterns for Scaffold Engine injection.

        Returns a list of {anti_pattern, fix, detection_hint} dicts
        that the Scaffold Engine can check against generated code.
        """
        return [
            {
                "anti_pattern": r.anti_pattern[:200],
                "fix": r.fix_description[:200],
                "detection": r.detection_hint[:100],
            }
            for r in self._patterns.values()
            if r.confidence >= 0.6
        ]

    def get_stats(self) -> Dict[str, Any]:
        cats: Dict[str, int] = {}
        for r in self._patterns.values():
            cats[r.category] = cats.get(r.category, 0) + 1
        return {
            "total_patterns": len(self._patterns),
            "categories": cats,
            "path": self._patterns_path,
        }

    def _register_in_graph(self, rule: PatternRule) -> None:
        """Register the pattern in the Knowledge Graph."""
        try:
            from app.intelligent_memory.graph import (
                get_knowledge_graph, Entity, EntityType,
            )
            graph = get_knowledge_graph()
            graph.add_entity(Entity(
                entity_id=f"pattern:{rule.pattern_id}",
                entity_type=EntityType.PATTERN,
                name=rule.anti_pattern[:80],
                attributes={
                    "fix": rule.fix_description,
                    "category": rule.category,
                    "confidence": rule.confidence,
                },
                tags=["optimisation", rule.category],
            ))
        except Exception as e:
            logger.debug("[pattern_learner] Graph registration failed: %s", e)

    def _load(self) -> None:
        path = Path(self._patterns_path)
        if not path.exists():
            return
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            for item in data.get("patterns", []):
                rule = PatternRule.from_dict(item)
                self._patterns[rule.pattern_id] = rule
            logger.info("[pattern_learner] Loaded %d patterns", len(self._patterns))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("[pattern_learner] Load failed: %s", e)

    def _persist(self) -> None:
        try:
            path = Path(self._patterns_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            output = {
                "patterns": [r.to_dict() for r in self._patterns.values()],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        except OSError as e:
            logger.error("[pattern_learner] Persist failed: %s", e)


def _build_detection_hint(proposal: Any) -> str:
    """Build a detection hint from the proposal for future scanning."""
    cat = proposal.category.value
    if cat == "dead_code":
        return "File with zero dependents and no entry point role"
    if cat == "file_split_merge":
        return f"File exceeding {20}KB with multiple responsibilities"
    if cat == "dependency_cleanup":
        return "Module with >15 total coupling edges"
    if cat == "logic_simplification":
        return "Function with cyclomatic complexity >40"
    if cat == "model_call_reduction":
        return "LLM call on a common/predictable path"
    return f"Pattern from category: {cat}"


_instance: Optional[PatternLearner] = None

def get_pattern_learner() -> PatternLearner:
    global _instance
    if _instance is None:
        _instance = PatternLearner()
    return _instance
