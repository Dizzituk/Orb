from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class EvidenceScore:
    metric_deltas: Dict[str, float] = field(default_factory=dict)
    regression_count: int = 0
    confidence: float = 0.0
    improvement_score: float = 0.0
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_evidence(
    before_metrics: Dict[str, float] | None,
    after_metrics: Dict[str, float] | None,
    test_results: Dict[str, Any] | None,
) -> EvidenceScore:
    before_metrics = before_metrics or {}
    after_metrics = after_metrics or {}
    test_results = test_results or {}

    metric_deltas: Dict[str, float] = {}
    weighted_gain = 0.0
    compared = 0

    for key in sorted(set(before_metrics) | set(after_metrics)):
        before = float(before_metrics.get(key, 0.0) or 0.0)
        after = float(after_metrics.get(key, 0.0) or 0.0)
        delta = after - before
        metric_deltas[key] = round(delta, 4)
        if before > 0:
            relative = (before - after) / before
        else:
            relative = 0.0 if after == 0 else -1.0
        weighted_gain += relative
        compared += 1

    regression_count = int(test_results.get("regression_count", 0) or 0)
    passed = int(test_results.get("passed", 0) or 0)
    failed = int(test_results.get("failed", 0) or 0)
    total_tests = max(1, passed + failed)
    pass_ratio = passed / total_tests

    average_gain = weighted_gain / max(1, compared)
    improvement_score = max(0.0, min(1.0, (average_gain * 0.7) + (pass_ratio * 0.3) - (regression_count * 0.15)))
    confidence = max(0.0, min(1.0, (pass_ratio * 0.6) + (min(compared, 5) / 5 * 0.2) + (max(0.0, average_gain) * 0.2)))

    if regression_count > 0:
        summary = f"Evidence is mixed: {regression_count} regression(s) were detected."
    elif improvement_score >= 0.65:
        summary = "Evidence is strong enough to justify another optimisation pass."
    elif improvement_score >= 0.35:
        summary = "Evidence shows some improvement, but gains are starting to level off."
    else:
        summary = "Evidence suggests the system is close to its current optimum."

    return EvidenceScore(
        metric_deltas=metric_deltas,
        regression_count=regression_count,
        confidence=round(confidence, 4),
        improvement_score=round(improvement_score, 4),
        summary=summary,
    )


def is_at_optimum(score: EvidenceScore) -> bool:
    return score.regression_count == 0 and score.improvement_score < 0.35
