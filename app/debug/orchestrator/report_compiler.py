# FILE: app/debug/orchestrator/report_compiler.py
# Purpose: Compile N sub-agent fan-out reports into ONE short brief before they reach
#          the Debug-tab orchestrator brain, so the brain reflects on a crisp digest
#          (keeps its context clean + its narration reflection line short and real).
#          Cheap, .env-sourced model; ALWAYS falls back to the raw serialised reports.
# Called-by: app.debug.orchestrator.spawn_tool (terminal result of a fan-out)
# Depends-on: app.providers.registry (llm_call), app.debug.debug_model_config
# Last-renovated: 2026-06-17
"""report_compiler -- digest sub-agent reports into a short orchestrator brief.

Phase-narration spec (debug-visibility), mechanism step 3: "Before results reach the
brain, a compiler agent digests N agent/worker outputs into one short brief. The brain
reflects on the digest, not the raw dumps." This is that compiler.

Model: get_compiler_model() -- DEBUG_COMPILER_MODEL, falling back to the cheap executor
tier (gpt-5.4-mini); no hardcoded id. The call is non-streaming and bounded. If it
fails for ANY reason -- model down, timeout, empty, exception -- we return the original
serialised reports unchanged, so the fan-out result still reaches the brain (never a
broken loop, never a silent drop).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_COMPILER_SYSTEM = (
    "You compress sub-agent investigation reports into ONE short brief for an "
    "orchestrator that will decide the next move. Be faithful and concrete:\n"
    "- Lead with the combined finding(s) / root cause, highest confidence first.\n"
    "- KEEP concrete file:line evidence and any files_modified verbatim -- the "
    "orchestrator needs them to act.\n"
    "- Note any conflicts between agents, failures, or dead-ends in one clause.\n"
    "- No preamble, no restating the task, no markdown headers. <=150 words.\n"
    "This brief REPLACES the raw reports; do not drop information the orchestrator "
    "would need to apply a fix."
)

_MAX_INPUT = 24000   # serialised reports are already capped near here upstream
_MAX_TOKENS = 512
_MIN_COMPILE_CHARS = 600   # below this the fan-out is trivial -- skip the LLM round


async def compile_reports(serialised_reports: str) -> str:
    """Return a short prose brief digesting the serialised fan-out reports.

    Never raises. On any failure (or a trivially small input) returns
    ``serialised_reports`` unchanged -- the mechanical digest -- so the orchestrator
    always receives the fan-out outcome.
    """
    src = (serialised_reports or "").strip()
    if not src or len(src) < _MIN_COMPILE_CHARS:
        return serialised_reports

    try:
        from app.debug.debug_model_config import get_compiler_model
        from app.providers.registry import llm_call

        model = get_compiler_model()
        result = await llm_call(
            provider_id="openai",
            model_id=model,
            messages=[{"role": "user", "content": src[:_MAX_INPUT]}],
            system_prompt=_COMPILER_SYSTEM,
            temperature=0.2,
            max_tokens=_MAX_TOKENS,
            timeout_seconds=60,
            enable_tools=False,
        )
        status = getattr(getattr(result, "status", None), "value", "")
        if status == "success":
            brief = (getattr(result, "content", "") or "").strip()
            if brief:
                logger.info(
                    "[report_compiler] digested %d -> %d chars (model=%s)",
                    len(src), len(brief), model,
                )
                return brief
        logger.warning(
            "[report_compiler] compile non-success (%s); using raw reports", status or "unknown",
        )
    except Exception as e:
        logger.warning("[report_compiler] compile failed (%s); using raw reports", e)
    return serialised_reports
