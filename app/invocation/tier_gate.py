# FILE: app/invocation/tier_gate.py
"""
Tier Promotion Gate.

Decides whether an invocation should be promoted from its default tier
(T1 mini or T2 standard) up to T4 Opus 4.7 based on cognitive load.

Deterministic and microsecond-fast. No LLM involved in the gate itself.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from app.invocation.models import (
    Tier,
    TierDecision,
    PromotionReason,
    InvocationRequest,
    ClassifiedIntent,
)
from app.invocation.classifier import intent_to_default_tier


@dataclass
class TierGateConfig:
    context_kb_opus_threshold: int = 30
    source_count_opus_threshold: int = 3
    tool_rounds_opus_threshold: int = 10
    demotion_context_kb_max: int = 15

    architectural_keywords: List[re.Pattern] = field(
        default_factory=lambda: [
            re.compile(p, re.IGNORECASE) for p in [
                r"\broot\s+cause\b",
                r"\bwhy\s+is\s+this\s+failing\b",
                r"\btrace\s+through\b",
                r"\bdebug\s+this\s+across\b",
                r"\bwhy\s+isn't\s+the\s+pipeline\b",
                r"\barchitectural\b",
                r"\blook\s+at\s+the\s+logs\s+and\b",
                r"\bfigure\s+out\s+what's\s+wrong\b",
                r"\bcross[-\s]?reference\b.*\bcode\b",
                r"\banalyse?\s+the\s+whole\b",
            ]
        ]
    )

    explicit_promotion_phrases: List[re.Pattern] = field(
        default_factory=lambda: [
            re.compile(p, re.IGNORECASE) for p in [
                r"\buse\s+opus\b",
                r"\buse\s+(the\s+)?big\s+model\b",
                r"\buse\s+claude\s+4(\.\d+)?\b",
                r"\bgive\s+this\s+the\s+full\s+treatment\b",
            ]
        ]
    )

    cost_per_1k_input: dict = field(default_factory=lambda: {
        Tier.T1_MINI:  0.00015,
        Tier.T2_STD:   0.0025,
        Tier.T3_MULTI: 0.00125,
        Tier.T4_OPUS:  0.005,
    })
    cost_per_1k_output: dict = field(default_factory=lambda: {
        Tier.T1_MINI:  0.0006,
        Tier.T2_STD:   0.015,
        Tier.T3_MULTI: 0.005,
        Tier.T4_OPUS:  0.025,
    })


_DEFAULT_CONFIG = TierGateConfig()


def _estimate_cost_range(tier: Tier, input_bytes: int, cfg: TierGateConfig):
    """Rough cost estimate for the pre-send preview."""
    tokenizer_factor = 1.2 if tier == Tier.T4_OPUS else 1.0
    input_tokens = max(1, int(input_bytes / 4 * tokenizer_factor))
    output_lo, output_hi = 200, 800

    in_per_1k = cfg.cost_per_1k_input[tier]
    out_per_1k = cfg.cost_per_1k_output[tier]

    lo = (input_tokens / 1000) * in_per_1k + (output_lo / 1000) * out_per_1k
    hi = (input_tokens / 1000) * in_per_1k + (output_hi / 1000) * out_per_1k
    return (round(lo, 4), round(hi, 4))


def decide(
    req: InvocationRequest,
    intent: ClassifiedIntent,
    cfg: TierGateConfig = _DEFAULT_CONFIG,
) -> TierDecision:
    """Decide the final tier for an invocation."""
    starting = intent_to_default_tier(intent.intent)
    reasons: List[PromotionReason] = []

    if req.explicit_tier is not None:
        final = req.explicit_tier
        reasons.append(PromotionReason.EXPLICIT_PROMOTION)
    else:
        final = starting
        transcript = (req.transcript or "").strip()

        if any(p.search(transcript) for p in cfg.explicit_promotion_phrases):
            final = Tier.T4_OPUS
            reasons.append(PromotionReason.EXPLICIT_PROMOTION)

        if req.context_bytes >= cfg.context_kb_opus_threshold * 1024:
            final = Tier.T4_OPUS
            reasons.append(PromotionReason.CONTEXT_VOLUME)

        if req.source_count >= cfg.source_count_opus_threshold:
            final = Tier.T4_OPUS
            reasons.append(PromotionReason.CROSS_SOURCE)

        if any(p.search(transcript) for p in cfg.architectural_keywords):
            final = Tier.T4_OPUS
            reasons.append(PromotionReason.ARCHITECTURAL_KEYWORD)

        # Demotion check
        if (
            final == Tier.T4_OPUS
            and PromotionReason.EXPLICIT_PROMOTION not in reasons
            and req.context_bytes < cfg.demotion_context_kb_max * 1024
            and req.source_count < 2
        ):
            final = Tier.T2_STD
            reasons.append(PromotionReason.DEMOTED_SMALL_CONTEXT)

        # Hard floor: T1_MINI cannot jump to T4 without explicit override
        if starting == Tier.T1_MINI and final == Tier.T4_OPUS:
            if PromotionReason.EXPLICIT_PROMOTION not in reasons:
                final = Tier.T2_STD

        if not reasons:
            reasons.append(PromotionReason.DEFAULT)

    lo, hi = _estimate_cost_range(final, req.context_bytes, cfg)

    return TierDecision(
        tier=final,
        starting_tier=starting,
        reasons=reasons,
        cost_estimate_usd_low=lo,
        cost_estimate_usd_high=hi,
        notes=None,
    )