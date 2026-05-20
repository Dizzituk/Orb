# FILE: app/invocation/__init__.py
"""Ambient invocation pipeline.

Thin layer that turns a user trigger (wake word, hotkey, tray click)
into a classified intent, a tiered model decision, and a dispatched
LLM call.
"""

from app.invocation.models import (
    Intent,
    Tier,
    PromotionReason,
    TierDecision,
    ClassifiedIntent,
    InvocationRequest,
    InvocationResponse,
)
from app.invocation.classifier import classify, intent_to_default_tier
from app.invocation.tier_gate import decide, TierGateConfig
from app.invocation.router import route

__all__ = [
    "Intent",
    "Tier",
    "PromotionReason",
    "TierDecision",
    "ClassifiedIntent",
    "InvocationRequest",
    "InvocationResponse",
    "classify",
    "intent_to_default_tier",
    "decide",
    "TierGateConfig",
    "route",
]