# FILE: app/llm/pipeline/_high_stakes_pipelines.py
"""
High-stakes pipeline: Block 4-6 artifact pipeline and legacy critique pipeline.

Extracted from high_stakes.py to keep file sizes modular.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.llm.schemas import LLMResult, LLMTask
from app.jobs.schemas import JobEnvelope

logger = logging.getLogger(__name__)


async def run_artifact_pipeline(
    draft: LLMResult,
    task: LLMTask,
    provider_id: str,
    model_id: str,
    envelope: JobEnvelope,
    job_type_str: str,
    db: Any,
    spec_id: str,
    spec_hash: str,
    spec_json: Optional[str],
    spec_markdown: Optional[str],
    segment_contract_markdown: Optional[str],
    segment_file_scope: Optional[List[str]],
    enrichment_markdown: Optional[str],
    trace: Any,
    audit_logger: Any,
    # v3.0: Deterministic verdict parameters
    segment_id: Optional[str] = None,
    segment_spec: Optional[Dict[str, Any]] = None,
    skeleton_contract: Optional[Dict[str, Any]] = None,
    skeleton_file_scope: Optional[List[str]] = None,
    enrichment_data: Optional[Dict[str, Any]] = None,
    manifest_dict: Optional[Dict[str, Any]] = None,
    needle_estimate: Optional[int] = None,
) -> LLMResult:
    """Run Block 4-6 artifact pipeline with JSON critique and revision loop."""
    from app.llm.pipeline._high_stakes_helpers import (
        HIGH_STAKES_JOB_TYPES,
        get_environment_context,
        store_architecture_artifact,
    )
    from app.llm.pipeline._high_stakes_utils import (
        _maybe_complete_trace,
        _trace_step,
    )
    from app.llm.pipeline.revision import run_revision_loop

    logger.info("[critic] Using Block 4-6 artifact pipeline")

    job_id = str(envelope.job_id)
    project_id = int(getattr(envelope, "project_id", 0))

    # v5.18: Sanitise draft before storing
    _sanitise_draft(draft, segment_file_scope, envelope)

    # Store initial architecture (Block 4)
    arch_id, arch_hash, _ = store_architecture_artifact(
        db=db,
        job_id=job_id,
        project_id=project_id,
        arch_content=draft.content,
        spec_id=spec_id,
        spec_hash=spec_hash,
        arch_version=1,
        model=model_id,
    )

    if trace:
        _trace_step(trace, 'arch_stored', arch_id=arch_id)

    env_context = get_environment_context(spec_json=spec_json) if job_type_str in HIGH_STAKES_JOB_TYPES else None

    # Run revision loop (Block 5 + 6)
    final_content, final_version, passed, final_critique = await run_revision_loop(
        db=db,
        job_id=job_id,
        project_id=project_id,
        arch_content=draft.content,
        arch_id=arch_id,
        spec_id=spec_id,
        spec_hash=spec_hash,
        spec_json=spec_json,
        spec_markdown=spec_markdown,
        original_request=_extract_original_request(task),
        opus_model_id=model_id,
        envelope=envelope,
        env_context=env_context,
        store_architecture_fn=store_architecture_artifact,
        segment_contract_markdown=segment_contract_markdown,
        enrichment_markdown=enrichment_markdown,
        # v3.0: Deterministic verdict parameters
        segment_id=segment_id,
        segment_spec=segment_spec,
        skeleton_contract=skeleton_contract,
        skeleton_file_scope=skeleton_file_scope,
        enrichment_data=enrichment_data,
        manifest_dict=manifest_dict,
        needle_estimate=needle_estimate,
    )

    if trace:
        _trace_step(trace, 'revision_loop_done', version=final_version, passed=passed)
    _maybe_complete_trace(audit_logger, trace, success=True)

    # v2.2: Cache successful architectures
    if passed:
        _cache_architecture(spec_json, segment_file_scope, final_content, spec_hash, model_id)

    return LLMResult(
        content=final_content,
        provider=provider_id,
        model=model_id,
        finish_reason="stop",
        error_message=None,
        prompt_tokens=draft.prompt_tokens,
        completion_tokens=draft.completion_tokens,
        total_tokens=draft.total_tokens,
        cost_usd=draft.cost_usd,
        raw_response=None,
        routing_decision={
            "job_type": job_type_str,
            "provider": provider_id,
            "model": model_id,
            "reason": f"Block 4-6 pipeline: v{final_version}, passed={passed}",
            "arch_id": arch_id,
            "final_version": final_version,
            "critique_passed": passed,
            "blocking_issues": len(final_critique.blocking_issues),
        },
    )


async def run_legacy_pipeline(
    draft: LLMResult,
    task: LLMTask,
    provider_id: str,
    model_id: str,
    envelope: JobEnvelope,
    job_type_str: str,
    spec_json: Optional[str],
    trace: Any,
    audit_logger: Any,
) -> LLMResult:
    """Run legacy prose critique + single revision pipeline."""
    from app.llm.pipeline._high_stakes_helpers import (
        HIGH_STAKES_JOB_TYPES,
        get_environment_context,
    )
    from app.llm.pipeline._high_stakes_utils import (
        _maybe_complete_trace,
        _trace_step,
    )
    from app.llm.pipeline.critique import call_gemini_critic
    from app.llm.pipeline.revision import call_opus_revision

    logger.info("[critic] Using legacy prose critique pipeline")

    env_context = get_environment_context(spec_json=spec_json) if job_type_str in HIGH_STAKES_JOB_TYPES else None
    critique = await call_gemini_critic(
        original_task=task,
        draft_result=draft,
        job_type_str=job_type_str,
        envelope=envelope,
        env_context=env_context,
    )

    if not critique:
        logger.warning("[critic] Critique failed; returning draft")
        _maybe_complete_trace(audit_logger, trace, success=True)
        draft.routing_decision = {"job_type": job_type_str, "provider": provider_id, "model": model_id, "reason": "critique failed"}
        return draft

    if trace:
        _trace_step(trace, 'critique_done')

    revision = await call_opus_revision(
        original_task=task, draft_result=draft, critique_result=critique,
        opus_model_id=model_id, envelope=envelope,
    )

    if not revision:
        logger.warning("[critic] Revision failed; returning draft")
        _maybe_complete_trace(audit_logger, trace, success=True)
        draft.routing_decision = {"job_type": job_type_str, "provider": provider_id, "model": model_id, "reason": "revision failed"}
        return draft

    if trace:
        _trace_step(trace, 'revision_done')
    _maybe_complete_trace(audit_logger, trace, success=True)

    revision.routing_decision = {
        "job_type": job_type_str,
        "provider": provider_id,
        "model": revision.model,
        "reason": "Legacy: Opus draft → Gemini critique → Opus revision",
        "critique_pipeline": {
            "draft_tokens": draft.total_tokens,
            "critique_tokens": critique.total_tokens,
            "revision_tokens": revision.total_tokens,
            "total_cost": draft.cost_usd + critique.cost_usd + revision.cost_usd,
        },
    }
    return revision


# =============================================================================
# HELPERS
# =============================================================================

def _extract_original_request(task: LLMTask) -> str:
    user_messages = [m for m in task.messages if m.get("role") == "user"]
    return user_messages[-1].get("content", "") if user_messages else ""


def _sanitise_draft(draft: LLMResult, file_scope: Optional[List[str]], envelope: Any) -> None:
    """v5.18: Sanitise draft before storing."""
    try:
        from app.orchestrator.architecture_sanitiser import sanitise_architecture
        sanitised, result = sanitise_architecture(
            arch_text=draft.content,
            file_scope=file_scope,
            segment_id=str(getattr(envelope, 'job_id', 'unknown')),
        )
        if result.had_fixes:
            draft.content = sanitised
            logger.info("[high_stakes] v5.18 Sanitiser applied %d fix(es)", result.fix_count)
    except ImportError:
        pass
    except Exception as err:
        logger.warning("[high_stakes] v5.18 Sanitiser error: %s", err)


def _cache_architecture(spec_json, file_scope, content, spec_hash, model_id) -> None:
    """v2.2: Cache successful architecture."""
    try:
        from app.orchestrator.architecture_cache import store_architecture as _store
        _goal = ""
        if spec_json:
            _data = json.loads(spec_json) if isinstance(spec_json, str) else spec_json
            _goal = _data.get("goal", "")
        _store(
            goal=_goal,
            file_targets=file_scope or [],
            arch_content=content,
            spec_hash=spec_hash or "",
            model_used=model_id,
            critique_passed=True,
        )
    except Exception:
        pass
