# FILE: app/llm/pipeline/high_stakes.py
"""High-stakes critique pipeline - Main orchestrator.

Implements Blocks 4, 5, 6 of the PoT (Proof of Thought) system:
  Block 4: Architecture generation as versioned artifact with spec traceability
  Block 5: Structured JSON critique with blocking/non-blocking issues (critique.py)
  Block 6: Revision loop until critique passes (revision.py)

Helpers in _high_stakes_helpers.py. Utilities in _high_stakes_utils.py.
Prompt building in _high_stakes_prompt_build.py.
Pipelines in _high_stakes_pipelines.py.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from app.llm.schemas import LLMResult, LLMTask
from app.jobs.schemas import JobEnvelope
from app.providers.registry import llm_call as registry_llm_call
from app.llm.job_classifier import compute_modality_flags
from app.llm.gemini_vision import transcribe_video_for_context

# Sibling re-exports
from app.llm.pipeline.critique import (
    call_json_critic,
    store_critique_artifact,
    call_gemini_critic,
    build_critique_prompt,
)
from app.llm.pipeline.revision import (
    call_revision,
    run_revision_loop,
    call_opus_revision,
    _map_to_phase4_job_type,
)
from app.llm.pipeline._high_stakes_utils import (
    _format_force_resolve,
    _format_fulfilled_evidence,
    _maybe_complete_trace,
    _trace_error,
    _trace_step,
)
from app.llm.pipeline._high_stakes_helpers import (
    MIN_CRITIQUE_CHARS,
    _get_architecture_draft_config,
    OPUS_DRAFT_MAX_TOKENS,
    OPUS_TIMEOUT_SECONDS,
    HIGH_STAKES_JOB_TYPES,
    _maybe_start_trace,
    get_environment_context,
    normalize_job_type_for_high_stakes,
    is_high_stakes_job,
    is_opus_model,
    is_long_enough_for_critique,
    store_architecture_artifact,
)
from app.llm.pipeline._high_stakes_prompt_build import build_draft_messages
from app.llm.pipeline._high_stakes_pipelines import run_artifact_pipeline, run_legacy_pipeline

# v3.2: Evidence fulfillment loop
try:
    from app.llm.pipeline.evidence_loop import (
        run_stage_with_evidence,
        parse_evidence_requests,
        StageResult,
        JobContext,
    )
    _EVIDENCE_LOOP_AVAILABLE = True
except ImportError:
    _EVIDENCE_LOOP_AVAILABLE = False

# v2.0: Evidence contract availability check
try:
    from app.llm.pipeline.evidence_contract_prompt import EVIDENCE_CONTRACT_PROMPT
    _EVIDENCE_CONTRACT_AVAILABLE = True
except ImportError:
    _EVIDENCE_CONTRACT_AVAILABLE = False
    EVIDENCE_CONTRACT_PROMPT = ""

logger = logging.getLogger(__name__)


# =============================================================================
# Main Pipeline Entry Point
# =============================================================================

async def run_high_stakes_with_critique(
    task: LLMTask,
    provider_id: str,
    model_id: str,
    envelope: JobEnvelope,
    job_type_str: str,
    file_map: Optional[str] = None,
    *,
    db=None,
    spec_id: Optional[str] = None,
    spec_hash: Optional[str] = None,
    spec_json: Optional[str] = None,
    spec_markdown: Optional[str] = None,
    use_json_critique: bool = True,
    segment_contract_markdown: Optional[str] = None,
    segment_file_scope: Optional[List[str]] = None,
    enrichment_markdown: Optional[str] = None,
    # v3.0: Deterministic verdict parameters
    segment_id: Optional[str] = None,
    segment_spec: Optional[Dict[str, Any]] = None,
    skeleton_contract: Optional[Dict[str, Any]] = None,
    skeleton_file_scope_det: Optional[List[str]] = None,
    enrichment_data: Optional[Dict[str, Any]] = None,
    manifest_dict: Optional[Dict[str, Any]] = None,
    needle_estimate: Optional[int] = None,
) -> LLMResult:
    """Run high-stakes critique pipeline.

    Delegates to Block 4-6 artifact pipeline or legacy prose critique
    depending on whether spec_id/spec_hash are provided.
    """
    logger.info("[critic] High-stakes pipeline: job_type=%s model=%s", job_type_str, model_id)

    audit_logger, trace = _maybe_start_trace(
        task, envelope, job_type_str=job_type_str, provider_id=provider_id, model_id=model_id,
    )

    # Pre-step: Video transcription
    transcripts_text = await _transcribe_videos(task)

    # Build draft messages with all injections
    draft_messages = build_draft_messages(
        envelope_messages=list(envelope.messages),
        spec_markdown=spec_markdown,
        spec_json=spec_json,
        spec_id=spec_id,
        spec_hash=spec_hash,
        transcripts_text=transcripts_text,
        file_map=file_map,
    )

    if trace:
        _trace_step(trace, 'draft')

    # Generate architecture draft (evidence loop or single-pass)
    _, _, arch_max_tokens, arch_timeout = _get_architecture_draft_config()
    draft = await _generate_draft(
        draft_messages=draft_messages,
        provider_id=provider_id,
        model_id=model_id,
        envelope=envelope,
        arch_max_tokens=arch_max_tokens,
        arch_timeout=arch_timeout,
        trace=trace,
        audit_logger=audit_logger,
    )

    if draft.error_message:
        return draft

    if trace:
        _trace_step(trace, 'draft_done')

    # Check if critique needed
    if not is_long_enough_for_critique(draft.content):
        logger.warning("[critic] Draft too short for critique")
        _maybe_complete_trace(audit_logger, trace, success=True)
        draft.routing_decision = {"job_type": job_type_str, "provider": provider_id, "model": model_id, "reason": "draft too short"}
        return draft

    # Route to appropriate pipeline
    if spec_id and spec_hash and use_json_critique:
        return await run_artifact_pipeline(
            draft=draft, task=task, provider_id=provider_id, model_id=model_id,
            envelope=envelope, job_type_str=job_type_str, db=db,
            spec_id=spec_id, spec_hash=spec_hash, spec_json=spec_json,
            spec_markdown=spec_markdown,
            segment_contract_markdown=segment_contract_markdown,
            segment_file_scope=segment_file_scope,
            enrichment_markdown=enrichment_markdown,
            trace=trace, audit_logger=audit_logger,
            # v3.0: Deterministic verdict parameters
            segment_id=segment_id,
            segment_spec=segment_spec,
            skeleton_contract=skeleton_contract,
            skeleton_file_scope=skeleton_file_scope_det,
            enrichment_data=enrichment_data,
            manifest_dict=manifest_dict,
            needle_estimate=needle_estimate,
        )
    else:
        return await run_legacy_pipeline(
            draft=draft, task=task, provider_id=provider_id, model_id=model_id,
            envelope=envelope, job_type_str=job_type_str, spec_json=spec_json,
            trace=trace, audit_logger=audit_logger,
        )


# =============================================================================
# Internal helpers
# =============================================================================

async def _transcribe_videos(task: LLMTask) -> str:
    """Transcribe any video attachments."""
    attachments = task.attachments or []
    flags = compute_modality_flags(attachments)
    video_attachments = flags.get("video_attachments", [])
    parts = []
    for video_att in video_attachments:
        try:
            video_path = getattr(video_att, "path", None)
            if video_path:
                transcript = await transcribe_video_for_context(video_path)
                parts.append(f"\n\n=== Video: {video_att.filename} ===\n{transcript}")
        except Exception:
            pass
    return "".join(parts)


async def _generate_draft(
    draft_messages: List[Dict],
    provider_id: str,
    model_id: str,
    envelope: JobEnvelope,
    arch_max_tokens: int,
    arch_timeout: int,
    trace: Any,
    audit_logger: Any,
) -> LLMResult:
    """Generate architecture draft via evidence loop or single-pass."""
    _use_evidence_loop = (
        _EVIDENCE_LOOP_AVAILABLE
        and _EVIDENCE_CONTRACT_AVAILABLE
        and os.getenv("ASTRA_EVIDENCE_LOOP_ENABLED", "1") == "1"
    )

    if _use_evidence_loop:
        return await _draft_with_evidence_loop(
            draft_messages, provider_id, model_id, envelope,
            arch_max_tokens, arch_timeout, trace, audit_logger,
        )
    else:
        return await _draft_single_pass(
            draft_messages, provider_id, model_id, envelope,
            arch_max_tokens, arch_timeout, trace, audit_logger,
        )


async def _draft_with_evidence_loop(
    draft_messages: List[Dict],
    provider_id: str,
    model_id: str,
    envelope: JobEnvelope,
    arch_max_tokens: int,
    arch_timeout: int,
    trace: Any,
    audit_logger: Any,
) -> LLMResult:
    """v3.2: Draft with evidence fulfillment loop."""
    logger.info("[high_stakes] v3.2 Evidence loop ENABLED")

    _total_prompt = 0
    _total_completion = 0
    _total_cost = 0.0
    _raw = None

    async def _stage_fn(ctx: JobContext) -> StageResult:
        nonlocal _total_prompt, _total_completion, _total_cost, _raw
        messages = list(draft_messages)

        if ctx.fulfilled_evidence:
            messages.append({"role": "system", "content": _format_fulfilled_evidence(ctx)})
        if ctx.force_resolve_only and ctx.force_resolve:
            messages.append({"role": "system", "content": _format_force_resolve(ctx)})

        try:
            result = await registry_llm_call(
                provider_id=provider_id, model_id=model_id, messages=messages,
                job_envelope=envelope, max_tokens=arch_max_tokens,
                timeout_seconds=arch_timeout, stage="architecture",
            )
            _total_prompt += result.usage.prompt_tokens
            _total_completion += result.usage.completion_tokens
            _total_cost += result.usage.cost_estimate
            _raw = result.raw_response
            return StageResult(output=result.content, success=True)
        except Exception as exc:
            return StageResult(output="", success=False, error=str(exc))

    # Load evidence bundle
    # v3.2: Codebase report DISABLED — stale bulk dump that adds noise.
    # Architecture gets arch_map + segment-scoped evidence + evidence ledger.
    # Overwatcher retains full project visibility for cross-segment oversight.
    evidence_bundle = None
    try:
        from app.pot_spec.evidence_collector import load_evidence
        evidence_bundle = load_evidence(include_codebase_report=False)
    except Exception:
        pass

    max_loops = int(os.getenv("ASTRA_CP_EVIDENCE_MAX_LOOPS", os.getenv("ASTRA_EVIDENCE_MAX_LOOPS", "6")))  # v3.3: Separate CP ceiling, inherits from global
    stage_result = await run_stage_with_evidence(
        stage_name="critical", stage_fn=_stage_fn,
        context=JobContext(evidence_bundle=evidence_bundle), max_loops=max_loops,
    )

    if not stage_result.success and stage_result.error:
        err_msg = f"High-stakes draft failed: {stage_result.error}"
        if trace:
            _trace_error(trace, 'draft', err_msg)
        _maybe_complete_trace(audit_logger, trace, success=False, error_message=err_msg)
        return LLMResult(
            content=err_msg, provider=provider_id, model=model_id,
            finish_reason="error", error_message=err_msg,
            prompt_tokens=0, completion_tokens=0, total_tokens=0, cost_usd=0.0, raw_response=None,
        )

    if stage_result.unresolved_human_required:
        logger.warning("[high_stakes] %d HUMAN_REQUIRED items", len(stage_result.unresolved_human_required))

    return LLMResult(
        content=stage_result.output, provider=provider_id, model=model_id,
        finish_reason="stop", error_message=None,
        prompt_tokens=_total_prompt, completion_tokens=_total_completion,
        total_tokens=_total_prompt + _total_completion, cost_usd=_total_cost,
        raw_response=_raw,
    )


async def _draft_single_pass(
    draft_messages: List[Dict],
    provider_id: str,
    model_id: str,
    envelope: JobEnvelope,
    arch_max_tokens: int,
    arch_timeout: int,
    trace: Any,
    audit_logger: Any,
) -> LLMResult:
    """Legacy single-pass draft (no evidence loop)."""
    try:
        result = await registry_llm_call(
            provider_id=provider_id, model_id=model_id, messages=draft_messages,
            job_envelope=envelope, max_tokens=arch_max_tokens,
            timeout_seconds=arch_timeout, stage="architecture",
        )
    except Exception as exc:
        err_msg = f"High-stakes draft failed: {exc}"
        if trace:
            _trace_error(trace, 'draft', err_msg)
        _maybe_complete_trace(audit_logger, trace, success=False, error_message=err_msg)
        return LLMResult(
            content=err_msg, provider=provider_id, model=model_id,
            finish_reason="error", error_message=err_msg,
            prompt_tokens=0, completion_tokens=0, total_tokens=0, cost_usd=0.0, raw_response=None,
        )

    return LLMResult(
        content=result.content, provider=provider_id, model=model_id,
        finish_reason="stop", error_message=None,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        total_tokens=result.usage.total_tokens,
        cost_usd=result.usage.cost_estimate,
        raw_response=result.raw_response,
    )


__all__ = [
    "HIGH_STAKES_JOB_TYPES",
    "MIN_CRITIQUE_CHARS",
    "normalize_job_type_for_high_stakes",
    "is_high_stakes_job",
    "is_opus_model",
    "is_long_enough_for_critique",
    "get_environment_context",
    "_map_to_phase4_job_type",
    "store_architecture_artifact",
    "call_json_critic",
    "store_critique_artifact",
    "call_gemini_critic",
    "build_critique_prompt",
    "call_revision",
    "run_revision_loop",
    "call_opus_revision",
    "run_high_stakes_with_critique",
]
