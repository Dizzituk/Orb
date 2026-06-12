# FILE: app/pipeline_v2/__init__.py
# Purpose: ASTRA v2 Pipeline.
# Called-by: app.pipeline_v2.llm_tools, app.pipeline_v2.spec_review.context, app.pipeline_v2.stages.builder, tests.test_clone_freshness_smoke (+2 more)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
ASTRA v2 Pipeline.

Five stages, clean separation:
  1. Weaver     — captures user intent (existing, unchanged)
  2. SpecGate   — grounded specification (existing, unchanged)
  3. Architect   — layered build plan (GPT-5.4)
  4. Builder    — code per tier (Claude Opus 4.6)
  5. Verifier   — visual QA (Gemini 3.1 Pro)

The orchestrator routes data between stages. It is not smart.
It is reliable.
"""
