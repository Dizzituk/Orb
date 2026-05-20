# FILE: app/pipeline_v2/spec_review/reviewer.py
"""
Always-on Spec Reviewer — the main entry point.

Runs after the builder (regardless of BVL outcome). Reads the spec +
every file the builder wrote + build output + BVL results, and emits a
structured ReviewReport with specific findings.

v1.1 (2026-04-18): Routed through providers.registry.llm_call() instead
    of pipeline_v2.llm_caller.call_llm(). The registry path threads
    Opus 4.7's adaptive thinking (thinking={"type":"adaptive"} +
    output_config={"effort":"high"}) correctly, handles the temperature
    strip, sets adequate max_tokens headroom, and runs through the
    pipeline's cost/budget tracking (stage="SPEC_REVIEW").

v1.0 (2026-04-18): Initial implementation for Stage 2 verifier work.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from app.pipeline_v2.spec_review.context import assemble_review_context
from app.pipeline_v2.spec_review.models import (
    Category,
    Finding,
    ReviewReport,
    Severity,
    Verdict,
)
from app.pipeline_v2.spec_review.parser import parse_review_response
from app.pipeline_v2.spec_review.prompt import (
    REVIEWER_SYSTEM_PROMPT,
    REVIEWER_USER_TEMPLATE,
)

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile
    from app.pipeline_v2.models import BuildResult

logger = logging.getLogger(__name__)


async def run_spec_review(
    spec: Dict[str, Any],
    build_result: "BuildResult",
    profile: "BuildTargetProfile",
    intent_text: str = "",
    bvl_report: Optional[Any] = None,
    build_output: str = "",
    emit: Optional[Callable[[str], None]] = None,
    job_id: Optional[str] = None,
) -> ReviewReport:
    """Run the always-on spec reviewer over the builder's output.

    Never raises. On any failure (config issue, LLM error, parser failure)
    it returns a ReviewReport with the failure recorded as an info
    finding so the pipeline can continue.
    """
    emit = emit or (lambda msg: None)
    started = time.time()

    emit("\n" + "=" * 60)
    emit("🔍 STAGE: SPEC REVIEW (always-on Opus verifier w/ adaptive thinking)")
    emit("=" * 60)

    # ── Get stage config ────────────────────────────────────────────
    try:
        from app.llm.stage_models import get_stage_config
        config = get_stage_config("SPEC_REVIEW")
    except Exception as exc:
        logger.error("[spec_review] Failed to load SPEC_REVIEW config: %s", exc)
        return _early_failure_report(f"Stage config unavailable: {exc}")

    provider = config.provider
    model = config.model
    max_tokens = config.max_output_tokens
    timeout = config.timeout_seconds
    reasoning = config.reasoning  # {"effort": "high"} from STAGE_REASONING

    emit(f"   📐 Reviewer: {provider}/{model}")
    emit(f"   🧠 Reasoning: {reasoning or 'off'}")
    emit(f"   💼 Budget: max_tokens={max_tokens}, timeout={timeout}s")

    # ── Assemble context ────────────────────────────────────────────
    try:
        ctx = await assemble_review_context(
            spec=spec,
            build_result=build_result,
            profile=profile,
            intent_text=intent_text,
            bvl_report=bvl_report,
            build_output=build_output,
        )
    except Exception as exc:
        logger.exception("[spec_review] Context assembly failed: %s", exc)
        return _early_failure_report(f"Could not assemble review context: {exc}")

    emit(
        f"   📦 Context: {len(ctx.get('source_concatenation', ''))} chars of source, "
        f"{len(ctx.get('spec_text', ''))} chars of spec"
    )

    user_prompt = REVIEWER_USER_TEMPLATE.format(**ctx)

    # ── Call Opus 4.7 with adaptive thinking via providers.registry ─
    # v1.1: providers.registry.llm_call() is the only call path that
    # correctly wires Anthropic extended thinking. pipeline_v2/llm_caller
    # does NOT thread the reasoning kwarg through to stream_anthropic,
    # which means this reviewer only gets Opus's regular output — no
    # scratchpad. For a verifier job where we're asking the model to
    # walk the data flow and ground every finding, the thinking
    # scratchpad materially changes output quality.
    emit("   🤔 Calling reviewer (adaptive thinking on)...")
    try:
        from app.providers.registry import llm_call, LlmCallStatus

        result = await llm_call(
            provider_id=provider,
            model_id=model,
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            # temperature is stripped automatically when reasoning is set.
            # max_tokens is bumped to at least 32000 for effort=high inside
            # providers.registry._call_anthropic (see v2.3 logic), so we
            # don't have to do that here ourselves.
            max_tokens=max_tokens,
            timeout_seconds=timeout,
            reasoning=reasoning,
            stage="SPEC_REVIEW",
            job_envelope={"job_id": job_id} if job_id else None,
        )
    except Exception as exc:
        logger.exception("[spec_review] Registry call failed: %s", exc)
        report = _early_failure_report(f"Reviewer LLM call failed: {exc}")
        report.model_used = f"{provider}/{model}"
        report.duration_seconds = time.time() - started
        return report

    if result.status != LlmCallStatus.SUCCESS:
        logger.error(
            "[spec_review] LLM call non-success: status=%s error=%s",
            result.status, result.error_message,
        )
        report = _early_failure_report(
            f"Reviewer LLM call returned {result.status.value}: "
            f"{result.error_message or '(no message)'}",
        )
        report.model_used = f"{provider}/{model}"
        report.duration_seconds = time.time() - started
        return report

    response = result.content or ""

    # ── Parse response ─────────────────────────────────────────────
    report = parse_review_response(response)
    report.duration_seconds = time.time() - started
    report.model_used = f"{provider}/{model}"

    # v1.1: Pull real token counts from the registry's usage dataclass
    # rather than estimating from char counts. The registry records the
    # authoritative cost via record_llm_cost; this is just for the
    # per-call line in the progress log + the ReviewReport.
    usage = getattr(result, "usage", None)
    if usage is not None:
        report.total_input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        report.total_output_tokens = getattr(usage, "completion_tokens", 0) or 0
    else:
        report.total_input_tokens = max(1, len(user_prompt) // 4)
        report.total_output_tokens = max(1, len(response) // 4)

    report.estimated_cost_usd = _estimate_cost(
        provider=provider,
        input_tokens=report.total_input_tokens,
        output_tokens=report.total_output_tokens,
    )

    # ── Emit summary ───────────────────────────────────────────────
    emit(f"   ✅ Review complete in {report.duration_seconds:.1f}s")
    emit(f"   📋 {report.summary_line()}")
    emit(
        f"   🧮 Tokens: {report.total_input_tokens:,} in / "
        f"{report.total_output_tokens:,} out (~${report.estimated_cost_usd:.2f})"
    )

    if report.findings:
        emit("   ── Findings ──")
        for finding in report.findings[:10]:
            emit(f"      • {finding.one_line()}")
        if len(report.findings) > 10:
            emit(f"      … and {len(report.findings) - 10} more")

    if report.requirements_unmet:
        emit("   ── Unmet requirements ──")
        for req in report.requirements_unmet[:6]:
            emit(f"      ✗ {req}")

    return report


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _early_failure_report(reason: str) -> ReviewReport:
    """Build a report for a failure that happened before the LLM call.

    Marks verdict as PASS_WITH_WARNINGS (not critical) so the build
    doesn't look broken just because the reviewer itself couldn't run.
    """
    return ReviewReport(
        verdict=Verdict.PASS_WITH_WARNINGS,
        summary=(
            "Spec reviewer could not complete. Build outputs are NOT "
            "verified. Treat this as 'review inconclusive' and run a "
            "manual audit before shipping."
        ),
        findings=[
            Finding(
                severity=Severity.INFO,
                category=Category.OTHER,
                title="Spec review did not run",
                description=reason,
            )
        ],
    )


def _estimate_cost(provider: str, input_tokens: int, output_tokens: int) -> float:
    """Rough per-call cost estimate.

    Authoritative cost lives in app.cost.cost_recorder. This is just for
    the per-call progress log line.
    """
    rates = {
        "anthropic": (15.0, 75.0),   # Opus 4.7 input / output per 1M tokens
        "openai":    (10.0, 40.0),   # GPT-5.4 estimate
        "google":    (1.25, 5.0),    # Gemini flash
    }
    in_rate, out_rate = rates.get(provider.lower(), (5.0, 25.0))
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate
