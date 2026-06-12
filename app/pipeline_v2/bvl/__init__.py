# FILE: app/pipeline_v2/bvl/__init__.py
# Purpose: Behavioral Verification Layer (BVL).
# Called-by: app.pipeline_v2.verifier_agent.perception_tools
# Depends-on: app.pipeline_v2.bvl.bvl_models
# Last-renovated: 2026-06-11
"""
Behavioral Verification Layer (BVL).

Three-tier verification cascade for ASTRA pipeline builds:
  Tier 1 — Sanity Gate: build, install, launch, screenshot
  Tier 2 — Behavioral Gate: interactive flow testing via ADB/UI Automator
  Tier 3 — Adversarial Gate: fuzzing, lifecycle stress, navigation chaos

Each tier is progressively harder. Code must clear all three to be signed off.
Failed components enter a surgical retry loop (max 3 attempts per tier).

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-BVL-001.
"""
from app.pipeline_v2.bvl.bvl_models import (
    TierName,
    TierStatus,
    TierResult,
    TestAction,
    TestAssertion,
    TestFlow,
    FlowCoverage,
    RetryContext,
    ComponentResult,
    BVLReport,
)

__all__ = [
    "TierName",
    "TierStatus",
    "TierResult",
    "TestAction",
    "TestAssertion",
    "TestFlow",
    "FlowCoverage",
    "RetryContext",
    "ComponentResult",
    "BVLReport",
]
