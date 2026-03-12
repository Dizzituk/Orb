# FILE: app/pipeline_v2/config.py
"""
ASTRA v2.2 Pipeline Configuration.

v2.1: Simplified. Scaffold Engine (no LLM) + Agentic Builder (GPT-5.4)
      + Verification Model (cheap vision) + Opus fallback.
v2.2: Multi-project targeting. Build target profiles replace hardcoded paths.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

V2_ENABLED = os.getenv("ASTRA_V2_PIPELINE", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Build target — which project the pipeline builds into
# ---------------------------------------------------------------------------

# Override with env var or pass explicitly per job.
# Values: "astra-backend", "astra-frontend", "driver-copilot"
DEFAULT_BUILD_TARGET = os.getenv("ASTRA_V2_BUILD_TARGET", "astra-backend")

# ---------------------------------------------------------------------------
# Model assignments
# ---------------------------------------------------------------------------

# Stage 1: Weaver (existing — configured elsewhere)
# Stage 2: SpecGate (existing — configured elsewhere)
# Stage 3: Scaffold Engine — no LLM (deterministic)

# Stage 4: Agentic Builder — one model, one loop, full tool access
BUILDER_PROVIDER = os.getenv("ASTRA_V2_BUILDER_PROVIDER", "openai")
BUILDER_MODEL = os.getenv("ASTRA_V2_BUILDER_MODEL", "gpt-5.4")
BUILDER_MAX_OUTPUT = int(os.getenv("ASTRA_V2_BUILDER_MAX_OUTPUT", "128000"))

# Fallback builder for hard problems
FALLBACK_BUILDER_PROVIDER = os.getenv("ASTRA_V2_FALLBACK_PROVIDER", "anthropic")
FALLBACK_BUILDER_MODEL = os.getenv("ASTRA_V2_FALLBACK_MODEL", "claude-opus-4-6")

# Verification Model — cheap, fast, vision-capable
VERIFIER_PROVIDER = os.getenv("ASTRA_V2_VERIFIER_PROVIDER", "google")
VERIFIER_MODEL = os.getenv("ASTRA_V2_VERIFIER_MODEL", "gemini-2.5-flash")
VERIFIER_MAX_OUTPUT = int(os.getenv("ASTRA_V2_VERIFIER_MAX_OUTPUT", "4000"))

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

# Max verify loops before accepting with issues
MAX_VERIFY_LOOPS = int(os.getenv("ASTRA_V2_MAX_VERIFY_LOOPS", "3"))

# Enhanced checkout — multi-step verification (boot, render, navigate, smoke test)
# Set to "true" to use enhanced checkout instead of simple screenshot verify
ENHANCED_CHECKOUT = os.getenv("ASTRA_V2_ENHANCED_CHECKOUT", "true").lower() == "true"

# Max agentic builder tool calls per session before forcing handover
# GPT-5.4 context: 272K default (1.05M opt-in), 128K max output
# A 19-file Education job used 88 tool calls and hit the 50 cap.
# Bigger jobs (30+ files) need room for read→analyse→write→verify per file.
MAX_TOOL_CALLS = int(os.getenv("ASTRA_V2_MAX_TOOL_CALLS", "150"))

# Token budget — trigger handover at this % of context window
HANDOVER_THRESHOLD_PCT = float(os.getenv("ASTRA_V2_HANDOVER_PCT", "0.80"))

# Max fallback escalations before giving up
MAX_FALLBACK_ATTEMPTS = int(os.getenv("ASTRA_V2_MAX_FALLBACK", "1"))

# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------

SANDBOX_URL = os.getenv("ASTRA_SANDBOX_URL", "http://192.168.250.2:8765")

# ---------------------------------------------------------------------------
# Behavioral Verification Layer (BVL)
# ---------------------------------------------------------------------------

# Enable BVL for Android/emulator builds (replaces compilation-only check)
BVL_ENABLED = os.getenv("ASTRA_BVL_ENABLED", "true").lower() == "true"

# Max retries per component per tier before marking BLOCKED
BVL_MAX_RETRIES = int(os.getenv("ASTRA_BVL_MAX_RETRIES", "3"))

# Tier 3 (Adversarial) Monkey runner event count
BVL_MONKEY_EVENTS = int(os.getenv("ASTRA_BVL_MONKEY_EVENTS", "500"))

# Emulator boot timeout (seconds)
BVL_EMULATOR_BOOT_TIMEOUT = int(os.getenv("ASTRA_BVL_BOOT_TIMEOUT", "120"))
