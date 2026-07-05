# FILE: app/pipeline_v2/config.py
# Purpose: ASTRA v2.2 Pipeline Configuration.
# Called-by: app.orchestrator.segment_loop, app.pipeline_v2.agentic_builder, app.pipeline_v2.bvl.tier2_test_generator, app.pipeline_v2.checkout (+7 more)
# Depends-on: app.llm.frontier_models
# Last-renovated: 2026-07-04 (Derek phase 2: builder/eyes resolve via stage_roles, stale defaults removed)
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

# Stage 4: Agentic Builder — one model, one loop, full tool access.
# DEREK phase 2 (2026-07-04): resolves through the stage-role registry
# (ASTRA_STAGE_BUILDER_MAIN_*, with ASTRA_V2_BUILDER_* as deprecated
# aliases). The old code defaults ("gpt-5.4" / "claude-opus-4-6") LIED —
# they silently selected models .env never named. Now: resolve or fail
# loudly at import with the exact env key to set.

def _resolve_role_or_die(role: str) -> tuple:
    from app.llm.stage_roles import resolve_stage_role
    r = resolve_stage_role(role)  # raises StageRoleResolutionError naming the key
    return r.provider, r.model


BUILDER_PROVIDER, BUILDER_MODEL = _resolve_role_or_die("BUILDER_MAIN")
BUILDER_MAX_OUTPUT = int(os.getenv("ASTRA_V2_BUILDER_MAX_OUTPUT", "128000"))

# Fallback builder for hard problems (escalation target). Env-only: the
# stale "claude-opus-4-6" default is gone — blank means "no fallback",
# which the orchestrator already tolerates (single-attempt build).
FALLBACK_BUILDER_PROVIDER = os.getenv("ASTRA_V2_FALLBACK_PROVIDER", "anthropic")
FALLBACK_BUILDER_MODEL = (
    os.getenv("ASTRA_V2_FALLBACK_MODEL", "").strip()
    or os.getenv("FRONTIER_ANTHROPIC_OPUS_MODEL", "").strip()
)

# Verification Model — cheap, fast, vision-capable (the CHECKOUT_EYES role;
# ASTRA_V2_VERIFIER_* remain as deprecated aliases via the registry).
VERIFIER_PROVIDER, VERIFIER_MODEL = _resolve_role_or_die("CHECKOUT_EYES")
VERIFIER_MAX_OUTPUT = int(os.getenv("ASTRA_V2_VERIFIER_MAX_OUTPUT", "4000"))

# ---------------------------------------------------------------------------
# Agentic Verifier (Jobs 11-12, 2026-06-10)
# ---------------------------------------------------------------------------
# The Claude final-checkout stage that drives the real app. OFF by default
# because it spends real tokens; enable with ASTRA_AGENTIC_VERIFIER=1.
# The model resolves through the verifier-family alias in
# app/llm/frontier_models.py (claude-fable-5); one .env line moves the
# whole verifier family: ASTRA_VERIFIER_FAMILY_MODEL=...

VERIFIER_AGENT_ENABLED = os.getenv("ASTRA_AGENTIC_VERIFIER", "0").strip().lower() in ("1", "true", "yes")


def _resolve_verifier_model() -> str:
    explicit = os.getenv("ASTRA_VERIFIER_MODEL", "").strip()
    if explicit:
        return explicit
    try:
        from app.llm.frontier_models import resolve_model_alias
        return resolve_model_alias("anthropic:frontier-verifier")
    except Exception:
        return "claude-fable-5"


VERIFIER_AGENT_MODEL = _resolve_verifier_model()
VERIFIER_FALLBACK_MODEL = os.getenv("ASTRA_VERIFIER_FALLBACK", "claude-opus-4-8")

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

# ---------------------------------------------------------------------------
# Decision Ledger (Piece 1 of v2.2 stability upgrade)
# ---------------------------------------------------------------------------

# Enable the decision ledger — binds pipeline stages to a shared record of
# architectural decisions, with conflict detection on write.
LEDGER_ENABLED = os.getenv("ASTRA_V2_LEDGER_ENABLED", "true").lower() == "true"

# Max chars of ledger content injected into each stage prompt.
LEDGER_PROMPT_MAX_CHARS = int(os.getenv("ASTRA_V2_LEDGER_PROMPT_MAX", "8000"))

