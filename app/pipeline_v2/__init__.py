# FILE: app/pipeline_v2/__init__.py
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
