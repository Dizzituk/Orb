# FILE: app/content/video_pipeline/_resolver_budget.py
# Purpose: Per-job spend ceiling tracker for paid asset tiers (fal.ai / HeyGen).
# Called-by: app.content.video_pipeline.asset_resolver, app.content.video_pipeline._resolver_fetchers
# Depends-on: (stdlib only)
# Last-renovated: 2026-06-20
"""
BudgetTracker - cumulative spend tracking against a per-job budget ceiling.

Extracted verbatim from asset_resolver.py on 2026-06-20 (split campaign,
batch 2). Logic byte-identical to the pre-split module.
"""
import logging

logger = logging.getLogger(__name__)


class BudgetTracker:
    """Tracks cumulative spend against a per-job budget ceiling."""

    def __init__(self, max_budget_usd: float = 10.0):
        self.max_budget = max_budget_usd
        self.spent = 0.0
        self.line_items = []

    @property
    def remaining(self) -> float:
        return max(0, self.max_budget - self.spent)

    def can_spend(self, amount: float) -> bool:
        return self.spent + amount <= self.max_budget

    def record(self, source: str, segment_id: str, amount: float):
        self.spent += amount
        self.line_items.append({
            "source": source,
            "segment_id": segment_id,
            "amount": amount,
        })
        logger.info(
            f"[budget] +${amount:.3f} ({source}/{segment_id}) "
            f"— total: ${self.spent:.2f} / ${self.max_budget:.2f}"
        )
