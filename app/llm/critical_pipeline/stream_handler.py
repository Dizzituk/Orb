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
from typing import Optional, Any
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

logger = logging.getLogger(__name__)


# =============================================================================
# SSE helpers
# =============================================================================

def _sse(event_type: str, content: str = "", **extra) -> str:
    payload = {"type": event_type}
    if content:
        payload["content"] = content
    payload.update(extra)
    return "data: " + json.dumps(payload) + "\n\n"


def _token(text: str) -> str:
    return _sse("token", text)


def _done(**fields) -> str:
    return _sse("done", **fields)


# =============================================================================
# Memory persistence helper
# =============================================================================

def _save_to_memory(
    db: Session,
    project_id: int,
    content: str,
    provider: str,
    model: str,
) -> None:
    if memory_service and memory_schemas:
        try:
            memory_service.create_message(
                db,
                memory_schemas.MessageCreate(
                    project_id=project_id,
                    role="assistant",
                    content=content,
                    provider=provider,
                    model=model,
                ),
            )
        except Exception as e:
            logger.warning("[critical_pipeline] Failed to save to memory: %s", e)


# =============================================================================
# Main Stream Generator
# =============================================================================

async def generate_critical_pipeline_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[Any] = None,
    conversation_id: Optional[str] = None,
    spec_id: Optional[str] = None,
    spec_hash: Optional[str] = None,
    job_id: Optional[str] = None,
    segment_context: Optional[dict] = None,
):
    """Generate SSE stream for Critical Pipeline execution."""
    response_parts = []
    model_cfg = get_pipeline_model_config()
    pipeline_provider = model_cfg["provider"]
    pipeline_model = model_cfg["model"]

    def _emit(text):
        response_parts.append(text)
        return _token(text)

    try:
        yield _emit("\u2699\ufe0f **Critical Pipeline**\n\n")

        # =================================================================
        # Validation
        # =================================================================
        if not PIPELINE_AVAILABLE:
            msg = (
                "\u274c **Pipeline modules not available.**\n\n"
                "The high-stakes pipeline modules (app.llm.pipeline.*) failed to import.\n"
            )
            yield _emit(msg)
            if trace:
                trace.finalize(success=False, error_message="Pipeline modules not available")
            yield _done(
                provider=pipeline_provider, model=pipeline_model,
                total_length=sum(len(p) for p in response_parts),
            )
            return

        if not SCHEMAS_AVAILABLE:
            yield _emit("\u274c **Schema imports failed.** Check backend logs.\n")
            yield _done(
                provider=pipeline_provider, model=pipeline_model,
                total_length=sum(len(p) for p in response_parts),
            )
            return

        # =================================================================
        # Step 1: Load validated spec
        # =================================================================
        yield _emit("\ud83d\udccb **Loading validated spec...**\n")

        db_spec = None
        if spec_id and SPECS_SERVICE_AVAILABLE and get_spec:
            try:
                db_spec = get_spec(db, spec_id)
            except Exception as e:
                logger.warning("[critical_pipeline] Failed to get spec by ID: %s", e)

        if not db_spec and SPECS_SERVICE_AVAILABLE and get_latest_validated_spec:
            try:
                db_spec = get_latest_validated_spec(db, project_id)
            except Exception as e:
                logger.warning("[critical_pipeline] Failed to get latest validated spec: %s", e)

        if not db_spec:
            yield _emit(
                "\u274c **No validated spec found.**\n\n"
                "Please complete Spec Gate validation first:\n"
                "1. Describe what you want to build\n"
                "2. Say `Astra, command: how does that look all together`\n"
                "3. Say `Astra, command: critical architecture` to validate\n"
                "4. Once validated, retry `run critical pipeline`\n"
            )
            yield _done(
                provider=pipeline_provider, model=pipeline_model,
                total_length=sum(len(p) for p in response_parts),
            )
            return

        spec_id = db_spec.spec_id
        spec_hash = db_spec.spec_hash
        spec_json = db_spec.content_json
        spec_markdown = db_spec.content_markdown

        try:
            spec_data = json.loads(spec_json) if isinstance(spec_json, str) else (spec_json or {})
        except Exception:
            spec_data = {}

        yield _emit(f"\u2705 Spec loaded: `{spec_id[:16]}...`\n")

        # =================================================================
        # v5.1: Segment context info (UI visibility)
        # =================================================================
        if segment_context:
            _seg_id = segment_context.get("segment_id", "unknown")
            _seg_deps = segment_context.get("dependencies", [])
            _seg_files = segment_context.get("file_scope", [])
            _seg_reqs = segment_context.get("requirements", [])
            _seg_exposes = segment_context.get("exposes")
            _seg_consumes = segment_context.get("consumes")

            seg_info = f"\ud83e\udde9 **Segment:** `{_seg_id}`\n"
            if _seg_deps:
                seg_info += f"   \u2514\u2500 Dependencies: {', '.join(f'`{d}`' for d in _seg_deps)}\n"
            if _seg_files:
                seg_info += f"   \u2514\u2500 Files in scope: {len(_seg_files)}\n"
            if _seg_reqs:
                seg_info += f"   \u2514\u2500 Requirements: {len(_seg_reqs)}\n"
            if _seg_exposes:
                seg_info += f"   \u2514\u2500 Exposes: interface contracts for downstream\n"
            if _seg_consumes:
                seg_info += f"   \u2514\u2500 Consumes: interface contracts from upstream\n"
            yield _emit(seg_info)

        # =================================================================
        # Mechanical guard: pending_evidence / blocked / error
        # =================================================================
        validation_status = spec_data.get("validation_status", "validated")

        if validation_status == "pending_evidence":
            # v5.0: Softened from hard block to warning. SpecGate v4.0+ fulfils
            # its own ERs, so pending_evidence should no longer occur. If a legacy
            # spec arrives with this status, warn but proceed — the Critical
            # Pipeline's own evidence_loop in high_stakes.py can attempt to fulfil
            # remaining ERs during architecture generation. The old hard block
            # caused a deadlock where SpecGate and Critical Pipeline each told
            # the user to go to the other.
            warn_msg = (
                "\n\u26a0\ufe0f **Warning: Spec has unfulfilled evidence requirements**\n\n"
                "SpecGate marked this spec as `pending_evidence`. This usually means "
                "evidence fulfilment was partially unsuccessful. Proceeding anyway \u2014 "
                "the architecture stage will attempt to gather remaining evidence.\n\n"
            )
            logger.warning(
                "[critical_pipeline] v5.0 SOFT GUARD: pending_evidence (proceeding), spec_id=%s",
                spec_id,
            )
            yield _emit(warn_msg)

        if validation_status in ("blocked", "error", "needs_clarification"):
            yield _emit(
                f"\n\ud83d\udeab **BLOCKED: Spec status is `{validation_status}`**\n\n"
                "Please resolve the spec issues and re-validate before retrying.\n"
            )
            yield _done(
                provider=pipeline_provider, model=pipeline_model,
                total_length=sum(len(p) for p in response_parts),
                blocked=True, blocked_reason=validation_status, spec_id=spec_id,
            )
            if trace:
                trace.finalize(success=False, error_message=f"Spec status is {validation_status}")
            return

        logger.info("[critical_pipeline] validation_status=%s \u2014 proceeding", validation_status)

        # =================================================================
        # Step 1a: Check for segmented spec (v5.1)
        # =================================================================
        # If SpecGate decomposed this job into segments, the critical pipeline
        # should NOT process the parent spec as a blob. Redirect to segment loop.
        _spec_context = spec_data.get("context", {})
        _is_segmented = (
            _spec_context.get("segmented", False)
            or spec_data.get("total_segments", 0) > 0
        )
        _total_segs = (
            spec_data.get("total_segments", 0)
            or _spec_context.get("total_segments", 0)
        )

        if _is_segmented and not segment_context:
            # This spec has segments but was called directly (not via segment loop).
            # The segment loop passes segment_context when calling per-segment.
            seg_msg = (
                f"\n\u26a0\ufe0f **This spec has been segmented into {_total_segs} segments.**\n\n"
                f"The critical pipeline should process each segment individually, "
                f"not the parent spec as a single blob.\n\n"
                f"Please use: **'Astra, command: run segments'** to execute "
                f"all segments through the pipeline in dependency order.\n"
            )
            logger.warning(
                "[critical_pipeline] v5.1 SEGMENT GUARD: spec %s has %d segments "
                "but was called directly — redirecting user to segment loop",
                spec_id, _total_segs,
            )
            print(f"[DEBUG] [critical_pipeline] v5.1 SEGMENT GUARD: {_total_segs} segments detected, blocking direct execution")
            yield _emit(seg_msg)
            yield _done(
                provider=pipeline_provider, model=pipeline_model,
                total_length=sum(len(p) for p in response_parts),
            )
            if trace:
                trace.finalize(success=False, error_message=f"Segmented spec ({_total_segs} segments) — use 'run segments' command")
            return

        # =================================================================
        # Step 1b: Classify job type
        # =================================================================
        job_kind = classify_job_kind(spec_data, message)
        yield _emit(f"\ud83c\udff7\ufe0f **Job Type:** `{job_kind}`\n")

        # =================================================================
        # MICRO_EXECUTION PATH
        # =================================================================
        if job_kind == JobKind.MICRO_EXECUTION:
            async for chunk in _handle_micro(
                spec_data, message, spec_id, job_id, project_id, db,
                trace, pipeline_provider, pipeline_model, response_parts,
            ):
                yield chunk
            return

        # =================================================================
        # SCAN_ONLY PATH
        # =================================================================
        if job_kind == JobKind.SCAN_ONLY:
            async for chunk in _handle_scan(
                spec_data, message, spec_id, job_id, project_id, db,
                trace, pipeline_provider, pipeline_model, response_parts,
            ):
                yield chunk
            return

        # =================================================================
        # ARCHITECTURE PATH
        # =================================================================
        async for chunk in _handle_architecture(
            spec_data, message, spec_id, spec_hash, spec_json, spec_markdown,
            job_id, job_kind, project_id, db, trace, conversation_id,
            pipeline_provider, pipeline_model, response_parts,
        ):
            yield chunk

    except Exception as e:
        logger.exception("[critical_pipeline] Stream failed: %s", e)
        if trace:
            trace.finalize(success=False, error_message=str(e))
        yield _sse("error", error=str(e))


# =============================================================================
# MICRO handler
# =============================================================================

async def _handle_micro(
    spec_data, message, spec_id, job_id, project_id, db,
    trace, provider, model, response_parts,
):
    def _emit(text):
        response_parts.append(text)
        return _token(text)

    yield _emit("\n\u26a1 **Fast Path:** This is a micro-execution job.\n")
    yield _emit("No architecture design required - generating execution plan...\n\n")

    if not job_id:
        job_id = f"micro-{uuid4().hex[:8]}"

    # Gather evidence (light)
    micro_evidence = gather_critical_pipeline_evidence(
        spec_data=spec_data, message=message,
        include_arch_map=False, include_codebase_report=False,
        include_file_evidence=True,
    )
    if micro_evidence.file_evidence_loaded:
        yield _emit(
            f"\ud83d\udcda **File evidence loaded:** {len(micro_evidence.multi_target_files)} file(s)\n"
        )

    micro_plan = generate_micro_execution_plan(spec_data, job_id)

    # Quickcheck
    yield _emit("\ud83e\uddea **Running Quickcheck...**\n")
    qc = micro_quickcheck(spec_data, micro_plan)

    if qc.passed:
        yield _emit(f"{qc.summary}\n\n")
        yield _emit(micro_plan)

        binding_ctx = {
            "job_id": job_id,
            "job_root": os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"),
            "repo_root": os.getenv("REPO_ROOT", "."),
        }
        bindings = extract_artifact_bindings(spec_data, binding_ctx)

        yield _sse("work_artifacts",
            spec_id=spec_id, job_id=job_id, job_kind=JobKind.MICRO_EXECUTION,
            critique_mode="quickcheck", critique_passed=True,
            artifact_bindings=bindings,
        )

        full = "".join(response_parts)
        _save_to_memory(db, project_id, full, "local", "micro-execution")
        if trace:
            trace.finalize(success=True)

        yield _done(
            provider="local", model="micro-execution",
            total_length=len(full), spec_id=spec_id, job_id=job_id,
            job_kind=JobKind.MICRO_EXECUTION, critique_mode="quickcheck",
            critique_passed=True, artifact_bindings=len(bindings),
        )
    else:
        yield _emit(f"{qc.summary}\n\n")
        for issue in qc.issues:
            yield _emit(f"\u274c **{issue['id']}:** {issue['description']}\n")

        yield _emit("\n### Generated Plan (for review):\n")
        yield _emit(micro_plan)
        yield _emit(
            "\n---\n\u26a0\ufe0f **Quickcheck Failed** \u2014 Job NOT ready for Overwatcher.\n\n"
            "Please check:\n"
            "1. Did SpecGate resolve the input/output paths correctly?\n"
            "2. Is the spec complete with sandbox_input_path and sandbox_output_path?\n"
            "3. If the plan needs to write output, does the spec have a sandbox_generated_reply?\n\n"
            "You may need to re-run Spec Gate with more details about the file locations.\n"
        )

        full = "".join(response_parts)
        _save_to_memory(db, project_id, full, "local", "micro-execution")
        if trace:
            trace.finalize(success=False, error_message="Quickcheck failed")

        yield _done(
            provider="local", model="micro-execution",
            total_length=len(full), spec_id=spec_id, job_id=job_id,
            job_kind=JobKind.MICRO_EXECUTION, critique_mode="quickcheck",
            critique_passed=False, quickcheck_issues=len(qc.issues),
        )


# =============================================================================
# SCAN handler
# =============================================================================

async def _handle_scan(
    spec_data, message, spec_id, job_id, project_id, db,
    trace, provider, model, response_parts,
):
    def _emit(text):
        response_parts.append(text)
        return _token(text)

    yield _emit("\n\ud83d\udd0d **Scan Mode:** Read-only filesystem scan.\n")
    yield _emit("No architecture design required - generating scan execution plan...\n\n")

    if not job_id:
        job_id = f"scan-{uuid4().hex[:8]}"

    scan_evidence = gather_critical_pipeline_evidence(
        spec_data=spec_data, message=message,
        include_arch_map=True, include_codebase_report=False,
        include_file_evidence=False, arch_map_max_lines=300,
    )
    if scan_evidence.arch_map_loaded:
        yield _emit(
            f"\ud83d\udcda **Architecture context loaded:** "
            f"{len(scan_evidence.arch_map_content or '')} chars\n"
        )

    scan_plan = generate_scan_execution_plan(spec_data, job_id)

    yield _emit("\ud83e\uddea **Running Scan Quickcheck...**\n")
    sqc = scan_quickcheck(spec_data, scan_plan)

    if sqc.passed:
        yield _emit(f"{sqc.summary}\n\n")
        for issue in sqc.issues:
            if issue.get("severity") == "warning":
                yield _emit(f"\u26a0\ufe0f **{issue['id']}:** {issue['description']}\n")
        yield _emit(scan_plan)

        yield _sse("work_artifacts",
            spec_id=spec_id, job_id=job_id, job_kind=JobKind.SCAN_ONLY,
            critique_mode="quickcheck", critique_passed=True,
            scan_roots=spec_data.get("scan_roots", []),
            scan_terms=spec_data.get("scan_terms", []),
            artifact_bindings=[],
        )

        full = "".join(response_parts)
        _save_to_memory(db, project_id, full, "local", "scan-only")
        if trace:
            trace.finalize(success=True)

        yield _done(
            provider="local", model="scan-only",
            total_length=len(full), spec_id=spec_id, job_id=job_id,
            job_kind=JobKind.SCAN_ONLY, critique_mode="quickcheck",
            critique_passed=True,
        )
    else:
        yield _emit(f"{sqc.summary}\n\n")
        for issue in sqc.issues:
            icon = "\u274c" if issue.get("severity") == "blocking" else "\u26a0\ufe0f"
            yield _emit(f"{icon} **{issue['id']}:** {issue['description']}\n")
        yield _emit("\n### Generated Plan (for review):\n")
        yield _emit(scan_plan)
        yield _emit(
            "\n---\n\u26a0\ufe0f **Scan Quickcheck Failed** \u2014 Job NOT ready for execution.\n\n"
            "Please check:\n"
            "1. Did SpecGate resolve the scan_roots correctly?\n"
            "2. Did SpecGate extract the scan_terms from your request?\n"
            "3. Is the output_mode set to CHAT_ONLY?\n\n"
            "You may need to re-run Spec Gate with more details about what to scan.\n"
        )

        full = "".join(response_parts)
        _save_to_memory(db, project_id, full, "local", "scan-only")
        if trace:
            trace.finalize(success=False, error_message="Scan quickcheck failed")

        yield _done(
            provider="local", model="scan-only",
            total_length=len(full), spec_id=spec_id, job_id=job_id,
            job_kind=JobKind.SCAN_ONLY, critique_mode="quickcheck",
            critique_passed=False, quickcheck_issues=len(sqc.issues),
        )


# =============================================================================
# ARCHITECTURE handler
# =============================================================================

async def _handle_architecture(
    spec_data, message, spec_id, spec_hash, spec_json, spec_markdown,
    job_id, job_kind, project_id, db, trace, conversation_id,
    pipeline_provider, pipeline_model, response_parts,
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

    task_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Generate architecture for:\n\n{original_request}\n\n"
                f"Spec:\n{json.dumps(spec_data, indent=2)}"
            ),
        },
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
            spec_markdown=spec_markdown,
            use_json_critique=True,
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
