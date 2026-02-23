# FILE: app/llm/critical_pipeline/stream_handler.py
"""
Main SSE stream handler for Critical Pipeline execution.

Orchestrates the full pipeline flow:
1. Load validated spec from DB
2. Classify job type (MICRO / SCAN_ONLY / ARCHITECTURE)
3a. MICRO:        plan -> quickcheck -> ready for Overwatcher
3b. SCAN_ONLY:    plan -> quickcheck -> ready for execution
3c. ARCHITECTURE: evidence -> prompt -> Block 4-6 pipeline -> stream result
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy.orm import Session

from app.llm.critical_pipeline.config import (
    PIPELINE_AVAILABLE,
    SCHEMAS_AVAILABLE,
    SPECS_SERVICE_AVAILABLE,
    get_spec,
    get_latest_validated_spec,
    get_pipeline_model_config,
    memory_service,
    memory_schemas,
    run_high_stakes_with_critique,
    LLMTask,
    JobType,
    JobEnvelope,
    Phase4JobType,
    Importance,
    DataSensitivity,
    Modality,
    JobBudget,
    OutputContract,
)
from app.llm.critical_pipeline.evidence import (
    gather_critical_pipeline_evidence,
)
from app.llm.critical_pipeline.job_classification import (
    JobKind,
    classify_job_kind,
)
from app.llm.critical_pipeline.quickcheck_micro import (
    micro_quickcheck,
)
from app.llm.critical_pipeline.quickcheck_scan import (
    scan_quickcheck,
)
from app.llm.critical_pipeline.plan_micro import (
    generate_micro_execution_plan,
)
from app.llm.critical_pipeline.plan_scan import (
    generate_scan_execution_plan,
)
from app.llm.critical_pipeline.artifact_binding import (
    extract_artifact_bindings,
)
from app.llm.critical_pipeline.prompt_builder import (
    is_refactor_job,
    extract_original_request,
    extract_spec_constraints,
    build_architecture_system_prompt,
)
from app.llm.critical_pipeline._stream_handler_utils_1 import _build_segment_critique_spec, _format_enrichment_for_critique, _handle_micro, _handle_scan
from app.llm.critical_pipeline._stream_handler_utils_2 import _done, _save_to_memory, _token, generate_critical_pipeline_stream
from app.llm.critical_pipeline._stream_handler_utils import _sse
from app.llm.critical_pipeline._segment_prompt_builder import build_segment_injection

logger = logging.getLogger(__name__)


# =============================================================================
# SSE helpers
# =============================================================================


# =============================================================================
# Memory persistence helper
# =============================================================================


# =============================================================================
# Segment-Scoped Spec Builder (v5.26)
# =============================================================================


# =============================================================================
# Main Stream Generator
# =============================================================================


# =============================================================================
# MICRO handler
# =============================================================================


# =============================================================================
# SCAN handler
# =============================================================================


# =============================================================================
# ARCHITECTURE handler
# =============================================================================

async def _handle_architecture(
    spec_data, message, spec_id, spec_hash, spec_json, spec_markdown,
    job_id, job_kind, project_id, db, trace, conversation_id,
    pipeline_provider, pipeline_model, response_parts,
    segment_context: Optional[dict] = None,
):
    def _emit(text):
        response_parts.append(text)
        return _token(text)

    yield _emit("\n\ud83c\udfd7\ufe0f **Architecture Mode:** Full design pipeline required.\n\n")

    if not job_id:
        job_id = f"cp-{uuid4().hex[:8]}"

    binding_ctx = {
        "job_id": job_id,
        "job_root": os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"),
        "repo_root": os.getenv("REPO_ROOT", "."),
    }
    artifact_bindings = extract_artifact_bindings(spec_data, binding_ctx)

    yield _emit(f"\ud83d\udcc1 **Job ID:** `{job_id}`\n")

    if artifact_bindings:
        binding_msg = f"\ud83d\udce6 **Artifact Bindings:** {len(artifact_bindings)} output(s)\n"
        for b in artifact_bindings[:3]:
            binding_msg += f"  - `{b['path']}`\n"
        if len(artifact_bindings) > 3:
            binding_msg += f"  - ... and {len(artifact_bindings) - 3} more\n"
        yield _emit(binding_msg)

    # --- Evidence ---
    yield _emit("\ud83d\udcda **Gathering evidence...**\n")

    refactor = is_refactor_job(spec_data, message)
    if refactor:
        logger.info("[critical_pipeline] Codebase report: INJECTED (refactor job)")
    else:
        logger.info("[critical_pipeline] Codebase report: SKIPPED (non-refactor job)")

    cp_evidence = gather_critical_pipeline_evidence(
        spec_data=spec_data, message=message,
        include_arch_map=True, include_codebase_report=refactor,
        include_file_evidence=True,
        arch_map_max_lines=800, codebase_max_lines=500,
    )

    evidence_status = []
    if cp_evidence.arch_map_loaded:
        evidence_status.append(f"Architecture map ({len(cp_evidence.arch_map_content or '')} chars)")
    if cp_evidence.codebase_report_loaded:
        evidence_status.append(f"Codebase report ({len(cp_evidence.codebase_report_content or '')} chars)")
    if cp_evidence.file_evidence_loaded:
        evidence_status.append(f"File evidence ({len(cp_evidence.multi_target_files)} files)")

    if evidence_status:
        yield _emit("\u2705 **Evidence loaded:** " + ", ".join(evidence_status) + "\n")
    else:
        yield _emit("\u26a0\ufe0f **Limited evidence available**\n")

    for err in cp_evidence.errors[:3]:
        yield _emit(f"  \u26a0\ufe0f {err}\n")

    evidence_context = cp_evidence.to_context_string(
        max_arch_chars=12000, max_codebase_chars=8000,
    )

    # --- Prompt ---
    yield _emit("\ud83d\udd27 **Building architecture prompt...**\n\n")

    original_request = extract_original_request(spec_data, message)
    spec_constraints = extract_spec_constraints(spec_data)

    system_prompt = build_architecture_system_prompt(
        spec_id=spec_id,
        spec_hash=spec_hash,
        spec_data=spec_data,
        artifact_bindings=artifact_bindings,
        evidence_context=evidence_context,
        spec_constraints=spec_constraints,
    )

    # =====================================================================
    # v3.0: Inject experience memory into architecture prompt
    # =====================================================================
    _memory_injection = ""
    try:
        from app.experience.retrieval import retrieve_for_stage, format_injection
        _mem_patterns = retrieve_for_stage(
            db, stage="critical_pipeline",
            context=f"Generating architecture for: {original_request[:200]}",
            job_type="refactor" if refactor else None,
            max_results=8,
        )
        if _mem_patterns:
            _memory_injection = format_injection(
                _mem_patterns, stage="critical_pipeline"
            )
            if _memory_injection:
                system_prompt += f"\n\n{_memory_injection}"
                yield _emit(
                    f"\U0001f9e0 **Experience memory:** {len(_mem_patterns)} pattern(s) injected\n"
                )
    except Exception as _mem_err:
        logger.debug("[critical_pipeline] Memory injection skipped: %s", _mem_err)

    # v3.0: Inject user memory context
    try:
        from app.experience.user_memory import get_user_context_for_pipeline
        _user_ctx = get_user_context_for_pipeline(db, project_id=project_id)
        if _user_ctx:
            system_prompt += f"\n\n{_user_ctx}"
    except Exception:
        pass

    # =====================================================================
    # v5.4 PHASE 2B: Inject segment scope + interface contract into prompt
    # =====================================================================
    _segment_injection = ""
    if segment_context:
        _segment_injection = build_segment_injection(segment_context, job_id)

    # v5.4 Phase 2B: Extract contract for critique injection
    _segment_contract_for_critique = ""
    if segment_context:
        _segment_contract_for_critique = segment_context.get("interface_contract", "")

    _user_content = f"Generate architecture for:\n\n{original_request}\n\n"
    if _segment_injection:
        _user_content += f"---\n\n{_segment_injection}\n---\n\n"
    _user_content += f"Spec:\n{json.dumps(spec_data, indent=2)}"

    task_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _user_content},
    ]

    task = LLMTask(
        messages=task_messages,
        job_type=(
            JobType.ARCHITECTURE_DESIGN
            if hasattr(JobType, 'ARCHITECTURE_DESIGN')
            else list(JobType)[0]
        ),
        attachments=[],
    )

    envelope = JobEnvelope(
        job_id=job_id,
        session_id=conversation_id or f"session-{uuid4().hex[:8]}",
        project_id=project_id,
        job_type=getattr(Phase4JobType, "APP_ARCHITECTURE", list(Phase4JobType)[0]),
        importance=Importance.CRITICAL,
        data_sensitivity=DataSensitivity.INTERNAL,
        modalities_in=[Modality.TEXT],
        budget=JobBudget(
            max_tokens=16384,
            max_cost_estimate=1.00,
            max_wall_time_seconds=600,
        ),
        output_contract=OutputContract.TEXT_RESPONSE,
        messages=task_messages,
        metadata={
            "spec_id": spec_id,
            "spec_hash": spec_hash,
            "pipeline": "critical",
            "artifact_bindings": artifact_bindings,
            "content_verbatim": (
                spec_data.get("content_verbatim")
                or spec_data.get("context", {}).get("content_verbatim")
                or spec_data.get("metadata", {}).get("content_verbatim")
            ),
            "location": (
                spec_data.get("location")
                or spec_data.get("context", {}).get("location")
                or spec_data.get("metadata", {}).get("location")
            ),
            "scope_constraints": (
                spec_data.get("scope_constraints")
                or spec_data.get("context", {}).get("scope_constraints")
                or spec_data.get("metadata", {}).get("scope_constraints")
                or []
            ),
        },
        allow_multi_model_review=True,
        needs_tools=[],
    )

    # --- v5.26: Build segment-scoped spec for critique ---
    # When processing a segment, the critique must evaluate against the
    # SEGMENT's contract, not the parent spec. The parent spec describes
    # the whole job ("refactor this file, no new files") which contradicts
    # what individual segments need to do. Each segment has its own
    # requirements, file_scope, and acceptance_criteria — THOSE are the
    # authoritative contract for that segment's architecture.
    _critique_spec_markdown = spec_markdown  # Default: use parent spec
    if segment_context:
        _critique_spec_markdown = _build_segment_critique_spec(
            segment_context=segment_context,
            parent_spec_markdown=spec_markdown,
        )
        logger.info(
            "[critical_pipeline] v5.26 Segment-scoped spec built for critique (%d chars, was %d)",
            len(_critique_spec_markdown), len(spec_markdown or ""),
        )

    # --- Run pipeline ---
    yield _emit(f"\ud83c\udfd7\ufe0f **Starting Block 4-6 Pipeline with {pipeline_model}...**\n\n")
    yield _emit("This may take 2-5 minutes. Stages:\n")
    yield _emit("  1. \ud83d\udcdd Architecture generation\n")
    yield _emit("  2. \ud83d\udd0d Critique (real blockers only)\n")
    yield _emit("  3. \u270f\ufe0f Revision loop (stops early if clean)\n\n")

    yield _sse("pipeline_started",
        stage="critical_pipeline", job_id=job_id, spec_id=spec_id,
        critique_mode="deep", artifact_bindings=len(artifact_bindings),
    )

    try:
        result = await run_high_stakes_with_critique(
            task=task,
            provider_id=pipeline_provider,
            model_id=pipeline_model,
            envelope=envelope,
            job_type_str="architecture_design",
            file_map=None,
            db=db,
            spec_id=spec_id,
            spec_hash=spec_hash,
            spec_json=spec_json,
            spec_markdown=_critique_spec_markdown,
            use_json_critique=True,
            segment_contract_markdown=_segment_contract_for_critique or None,
            segment_file_scope=segment_context.get("file_scope") if segment_context else None,
            enrichment_markdown=_format_enrichment_for_critique(segment_context.get("enrichment")) if segment_context and segment_context.get("enrichment") else None,
        )
    except Exception as e:
        logger.exception("[critical_pipeline] Pipeline failed: %s", e)
        yield _emit(f"\u274c **Pipeline failed:** {e}\n")
        yield _done(
            provider=pipeline_provider, model=pipeline_model,
            total_length=sum(len(p) for p in response_parts), error=str(e),
        )
        return

    if not result or not result.content:
        yield _emit("\u274c **Pipeline returned empty result.**\n")
        yield _done(
            provider=pipeline_provider, model=pipeline_model,
            total_length=sum(len(p) for p in response_parts),
        )
        return

    # --- Stream result ---
    routing = getattr(result, 'routing_decision', {}) or {}
    arch_id = routing.get('arch_id', 'unknown')
    final_version = routing.get('final_version', 1)
    critique_passed = routing.get('critique_passed', False)
    blocking_issues = routing.get('blocking_issues', 0)

    # Journal: architecture + critique result
    try:
        from app.experience.context import journal_emit
        journal_emit(
            stage="critical_pipeline",
            event_type="architecture_decision",
            description=f"Architecture v{final_version} generated. "
                        f"Critique {'passed' if critique_passed else f'failed ({blocking_issues} blockers)'}",
            details={
                "arch_id": arch_id,
                "final_version": final_version,
                "critique_passed": critique_passed,
                "blocking_issues": blocking_issues,
                "provider": getattr(result, 'provider', ''),
                "model": getattr(result, 'model', ''),
                "total_tokens": getattr(result, 'total_tokens', 0),
                "cost_usd": getattr(result, 'cost_usd', 0),
            },
        )
    except Exception:
        pass

    yield _emit("\u2705 **Pipeline Complete**\n\n")
    yield _emit(
        f"**Architecture ID:** `{arch_id}`\n"
        f"**Final Version:** v{final_version}\n"
        f"**Critique Mode:** deep (blocker filtering enabled)\n"
        f'**Critique Status:** {"\u2705 PASSED" if critique_passed else f"\u26a0\ufe0f {blocking_issues} blocking issues"}\n'
        f"**Provider:** {result.provider}\n"
        f"**Model:** {result.model}\n"
        f"**Tokens:** {result.total_tokens:,}\n"
        f"**Cost:** ${result.cost_usd:.4f}\n"
        f"**Artifact Bindings:** {len(artifact_bindings)}\n\n---\n\n"
    )

    yield _emit("### Architecture Document\n\n")

    content = result.content
    chunk_size = 200
    for i in range(0, len(content), chunk_size):
        chunk = content[i:i + chunk_size]
        yield _token(chunk)
        response_parts.append(chunk)
        await asyncio.sleep(0.01)

    yield _sse("work_artifacts",
        spec_id=spec_id, job_id=job_id, arch_id=arch_id,
        final_version=final_version, critique_mode="deep",
        critique_passed=critique_passed,
        artifact_bindings=artifact_bindings,
        artifacts=[f"arch_v{final_version}.md", f"critique_v{final_version}.json"],
    )

    if critique_passed:
        next_step = (
            f"\n\n---\n\u2705 **Ready for Implementation**\n\n"
            f"Architecture approved with {len(artifact_bindings)} artifact binding(s).\n"
            f"Critique mode: deep (blocker filtering enabled, stops early when clean)\n\n"
            f"\ud83d\udd27 **Next Step:** Say **'Astra, command: send to overwatcher'** to implement.\n"
        )
    else:
        next_step = (
            f"\n\n---\n\u26a0\ufe0f **Critique Not Fully Passed**\n\n"
            f"{blocking_issues} blocking issues remain.\n\n"
            f"You may:\n- Re-run with updated spec\n- Proceed to Overwatcher with caution\n"
        )
    yield _emit(next_step)

    full = "".join(response_parts)
    _save_to_memory(db, project_id, full, pipeline_provider, pipeline_model)
    if trace:
        trace.finalize(success=True)

    yield _done(
        provider=pipeline_provider, model=pipeline_model,
        total_length=len(full), spec_id=spec_id, job_id=job_id,
        arch_id=arch_id, final_version=final_version,
        critique_mode="deep", critique_passed=critique_passed,
        artifact_bindings=len(artifact_bindings),
        tokens=result.total_tokens, cost_usd=result.cost_usd,
    )
