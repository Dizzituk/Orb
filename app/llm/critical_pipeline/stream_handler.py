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
# Segment-Scoped Spec Builder (v5.26)
# =============================================================================

def _build_segment_critique_spec(
    segment_context: Dict[str, Any],
    parent_spec_markdown: Optional[str] = None,
) -> str:
    """Build a segment-scoped spec for critique evaluation.

    When processing a segment within a segmented job, the critique must
    evaluate the architecture against THIS segment's contract, not the
    parent job spec. The parent spec describes the whole job (e.g.
    'refactor this file, no new files') which may contradict what
    individual segments need to do.

    The segment-scoped spec:
    1. Defines the segment's own requirements and acceptance criteria
    2. Lists the segment's file_scope (what files it owns)
    3. Describes what other segments handle (so critique doesn't flag
       'missing' functionality that lives in sibling segments)
    4. Preserves parent spec context as reference, clearly scoped

    v5.26: Fixes critique false-positives where segment architectures
    were flagged for violating parent-level constraints that don't
    apply at the segment level.
    """
    seg_id = segment_context.get("segment_id", "unknown")
    seg_spec = segment_context.get("segment_spec", {})
    file_scope = segment_context.get("file_scope", [])
    requirements = segment_context.get("requirements", [])
    acceptance_criteria = segment_context.get("acceptance_criteria", [])
    dependencies = segment_context.get("dependencies", [])
    exposes = segment_context.get("exposes")
    consumes = segment_context.get("consumes")

    parts: List[str] = []

    # Header — make it clear this is a segment, not a standalone job
    parts.append(f"# Segment Spec: {seg_id}")
    parts.append(f"**Title:** {seg_spec.get('title', seg_id)}")
    parts.append("")
    parts.append(
        "This is ONE SEGMENT within a segmented job. The architecture "
        "should be evaluated against THIS segment's requirements below, "
        "NOT against the parent job spec. Other segments handle other "
        "parts of the work."
    )
    parts.append("")

    # Goal — the segment's actual job
    parts.append("## Goal")
    parts.append(seg_spec.get("title", "Implement this segment's file scope."))
    parts.append("")

    # File scope — authoritative list of files this segment owns
    if file_scope:
        parts.append("## File Scope (ONLY these files)")
        parts.append(
            "This segment is responsible for ONLY these files. "
            "Creating, modifying, or referencing other files is acceptable "
            "if they are dependencies or standard library imports."
        )
        for f in file_scope:
            parts.append(f"- `{f}`")
        parts.append("")

    # Requirements — the segment's own requirements
    if requirements:
        parts.append("## Requirements")
        for r in requirements:
            parts.append(f"- {r}")
        parts.append("")

    # Acceptance criteria — what the segment must achieve
    if acceptance_criteria:
        parts.append("## Acceptance Criteria")
        for ac in acceptance_criteria:
            parts.append(f"- {ac}")
        parts.append("")

    # Dependencies — what other segments this one depends on
    if dependencies:
        parts.append("## Dependencies")
        parts.append(
            "This segment depends on the following sibling segments. "
            "Their output files already exist or will exist when this "
            "segment executes. Importing from them is EXPECTED and CORRECT."
        )
        for dep in dependencies:
            parts.append(f"- `{dep}`")
        parts.append("")

    # v5.27: Available symbols from dependencies (from enrichment)
    # Prevents critique false-positives like "imports X from seg-02
    # but X is not in available exports" by giving the full symbol list.
    _enrichment = segment_context.get("enrichment")
    if _enrichment:
        _consumes = _enrichment.get("consumes", {})
        if _consumes:
            parts.append("## Available Symbols from Dependencies")
            parts.append(
                "These symbols are confirmed available from sibling segments "
                "(extracted from the source monolith via AST parsing). "
                "Importing any of these is CORRECT and should NOT be flagged."
            )
            for _dep_seg, _dep_syms in _consumes.items():
                if isinstance(_dep_syms, list) and _dep_syms:
                    parts.append(f"- From **{_dep_seg}**: {', '.join(f'`{s}`' for s in _dep_syms)}")
            parts.append("")

    if exposes:
        parts.append("## Exposes (what this segment provides to others)")
        if isinstance(exposes, dict):
            for k, v in exposes.items():
                parts.append(f"- **{k}**: {v}")
        else:
            parts.append(str(exposes))
        parts.append("")

    if consumes:
        parts.append("## Consumes (what this segment needs from dependencies)")
        if isinstance(consumes, dict):
            for k, v in consumes.items():
                parts.append(f"- **{k}**: {v}")
        else:
            parts.append(str(consumes))
        parts.append("")

    # Parent spec as REFERENCE (not authoritative for this segment)
    if parent_spec_markdown:
        parts.append("## Parent Job Spec (REFERENCE ONLY)")
        parts.append(
            "The following is the parent job spec for context. "
            "Constraints in the parent spec (e.g. 'no new files', "
            "'refactor single file') apply to the OVERALL job, not "
            "to this individual segment. This segment may legitimately "
            "create new files, import from sibling-created modules, or "
            "perform operations that appear to violate parent-level "
            "constraints but are correct at the segment level."
        )
        parts.append("")
        # Include truncated parent spec for context
        _parent_truncated = parent_spec_markdown[:4000]
        if len(parent_spec_markdown) > 4000:
            _parent_truncated += f"\n... (truncated from {len(parent_spec_markdown):,} chars)"
        parts.append(_parent_truncated)
        parts.append("")

    return "\n".join(parts)


# =============================================================================
# Main Stream Generator
# =============================================================================

def _format_enrichment_for_critique(enrichment: dict) -> str:
    """Format enrichment data into concise markdown for critique/revision.

    v5.18: Gives Gemini critique and Opus revision the same symbol
    awareness that GPT draft gets, so they don't flag missing functions
    that simply weren't in the AST extraction list.
    """
    if not enrichment:
        return ""
    parts = []

    constants = enrichment.get("constants", [])
    if constants:
        parts.append("**Constants:**")
        for c in constants:
            parts.append(f"- `{c.get('name', '?')}`")

    functions = enrichment.get("functions", [])
    if functions:
        parts.append("**Functions:**")
        for f in functions:
            sig = f.get("signature", f.get("name", "?"))
            parts.append(f"- `{sig}`")

    classes = enrichment.get("classes", [])
    if classes:
        parts.append("**Classes:**")
        for cl in classes:
            methods = ", ".join(cl.get("methods", [])[:10])
            parts.append(f"- `class {cl.get('name', '?')}` (methods: {methods})")

    consumed_by = enrichment.get("consumed_by", {})
    if consumed_by:
        parts.append("**Other segments import from this segment:**")
        for seg_id, syms in consumed_by.items():
            if isinstance(syms, list):
                parts.append(f"- {seg_id}: {', '.join(f'`{s}`' for s in syms)}")

    consumes = enrichment.get("consumes", {})
    if consumes:
        parts.append("**This segment imports from:**")
        for seg_id, syms in consumes.items():
            if isinstance(syms, list):
                parts.append(f"- {seg_id}: {', '.join(f'`{s}`' for s in syms)}")

    guidance = enrichment.get("design_guidance", "")
    if guidance:
        parts.append(f"**Design guidance:** {guidance}")

    if not parts:
        return ""
    return "\n".join(parts)


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

    # v5.5 PHASE 3C: Needle-based model selection — override for segments
    _needle_model_override = None
    if segment_context:
        try:
            from app.llm.critical_pipeline.needle_model_selector import select_model_for_segment
            _needle_model_override = select_model_for_segment(
                segment_context=segment_context,
                grounding_data=segment_context.get("_grounding_data"),
                default_config={"provider": pipeline_provider, "model": pipeline_model},
            )
            pipeline_provider = _needle_model_override["provider"]
            pipeline_model = _needle_model_override["model"]
        except (ImportError, Exception) as _nm_err:
            logger.debug("[critical_pipeline] Needle model selector unavailable: %s", _nm_err)

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
        # v5.5 PHASE 3C: Show model selection reason
        if _needle_model_override:
            _nm_tier = _needle_model_override.get("tier", "?")
            _nm_reason = _needle_model_override.get("reason", "")
            yield _emit(f"🧠 **Model tier:** `{_nm_tier}` → `{pipeline_model}`\n")
            if _nm_reason:
                yield _emit(f"   └─ {_nm_reason}\n")

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
        # v5.4: Check for MULTI-segment specs only. Single-segment manifests
        # (Phase 1 always-manifest) should flow through normally.
        _total_segs = (
            spec_data.get("total_segments", 0)
            or _spec_context.get("total_segments", 0)
        )
        _is_multi_segmented = (
            _spec_context.get("segmented", False)
            or _total_segs > 1
        )

        if _is_multi_segmented and not segment_context:
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
            segment_context=segment_context,
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
        _si_parts = []

        # Segment scope: constrain architecture to this segment's files
        _si_files = segment_context.get("file_scope", [])
        _si_reqs = segment_context.get("requirements", [])
        _si_seg_id = segment_context.get("segment_id", "unknown")
        _si_ac = segment_context.get("acceptance_criteria", [])

        _si_parts.append(f"## Segment Scope: {_si_seg_id}\n")
        _si_parts.append(
            "**IMPORTANT**: You are generating architecture for ONE SEGMENT "
            "of a multi-segment job, not the full specification. Only design "
            "and produce code for the files listed below. "
            "Files marked CREATE do not exist on disk yet — do NOT emit "
            "EVIDENCE_REQUEST to read them. All source code you need is provided "
            "in the Source File Evidence and Segment Enrichment sections below.\n"
        )
        if _si_files:
            _si_parts.append("### Files in Scope (ONLY these files)")
            # v5.27: Mark files as CREATE or MODIFY so the model doesn't
            # try to read files that don't exist yet via EVIDENCE_REQUEST.
            # v5.28: Detect the refactor source monolith and mark it READ-ONLY.
            # In a file->package refactor, the monolith is in source_file_evidence
            # as reference material. Segments should extract FROM it, not MODIFY it.
            # The facade segment handles the monolith replacement.
            _si_source_evidence = segment_context.get("source_file_evidence", {})
            _si_source_evidence_norm = {
                k.replace("\\", "/"): v for k, v in _si_source_evidence.items()
            }
            for _f in _si_files:
                _f_norm = _f.replace("\\", "/")
                _exists = _f_norm in _si_source_evidence_norm or any(
                    k == _f_norm for k in _si_source_evidence_norm
                )
                if _exists:
                    # v5.28: Is this the refactor source (monolith)?
                    # If the file is in source_file_evidence AND other files in
                    # file_scope are inside a subpackage of the same stem, this
                    # is the monolith being decomposed — mark READ-ONLY.
                    _f_stem = _f_norm.rsplit(".py", 1)[0]  # e.g. "app/orchestrator/segment_loop"
                    _is_monolith_source = any(
                        other_f.replace("\\", "/").startswith(_f_stem + "/")
                        for other_f in _si_files if other_f != _f
                    )
                    if _is_monolith_source:
                        _si_parts.append(
                            f"- `{_f}` — **READ-ONLY** (source monolith being refactored — "
                            f"provided as evidence only. Do NOT include in File Inventory. "
                            f"Do NOT generate MODIFY operations for this file.)"
                        )
                    else:
                        _op = "MODIFY"
                        _si_parts.append(f"- `{_f}` — **{_op}** (exists on disk)")
                else:
                    _op = "CREATE"
                    _si_parts.append(f"- `{_f}` — **{_op}** (new file — do NOT try to read)")
            _si_parts.append("")
        if _si_reqs:
            _si_parts.append("### Segment Requirements")
            for _r in _si_reqs[:15]:
                _si_parts.append(f"- {_r}")
            if len(_si_reqs) > 15:
                _si_parts.append(f"- ... (+{len(_si_reqs)-15} more)")
            _si_parts.append("")
        if _si_ac:
            _si_parts.append("### Acceptance Criteria")
            for _a in _si_ac[:10]:
                _si_parts.append(f"- {_a}")
            if len(_si_ac) > 10:
                _si_parts.append(f"- ... (+{len(_si_ac)-10} more)")
            _si_parts.append("")

        # Interface contract (Phase 2A output)
        _si_contract = segment_context.get("interface_contract", "")
        if _si_contract:
            _si_parts.append(_si_contract)

        # v2.2: Source file evidence (pre-loaded existing files for refactor jobs)
        _si_source_files = segment_context.get("source_file_evidence", {})
        if _si_source_files:
            _si_parts.append("### Source File Evidence (EXISTING code — copy verbatim)\n")
            _si_parts.append(
                "**CRITICAL**: The following file(s) exist on disk and are being "
                "refactored. You MUST copy all function signatures, constant values, "
                "parameter names, and return types EXACTLY as they appear below. "
                "Do NOT invent, guess, or approximate any values.\n"
            )
            _si_parts.append(
                "**ENUM/TYPE PRESERVATION**: If the source code uses an enum like "
                "`SegmentStatus.COMPLETE.value` or `SegmentStatus.FAILED`, you MUST "
                "use that same enum in your output. Do NOT replace enums with invented "
                "string constants (e.g. never create `SEGMENT_COMPLETE_STATUS = \"complete\"` "
                "when the source uses `SegmentStatus.COMPLETE`). Import the enum from its "
                "original module and use it exactly as the source does.\n"
            )
            # v5.34: Source evidence ownership warning — annotate which functions
            # in the monolith belong to OTHER segments.  The LLM sees the source
            # evidence as "copy verbatim", which overrides the DO NOT DEFINE
            # prohibition. This annotation bridges that gap by telling the LLM
            # which functions to import rather than copy.
            _si_other_seg_symbols = set()
            _enrich_for_ownership = segment_context.get("enrichment", {})
            _consumes_for_ownership = _enrich_for_ownership.get("consumes", {}) if _enrich_for_ownership else {}
            for _own_seg, _own_syms in _consumes_for_ownership.items():
                if isinstance(_own_syms, list):
                    _si_other_seg_symbols.update(_own_syms)
            if _si_other_seg_symbols:
                _si_parts.append("#### ⚠️ Source Evidence Ownership Warning (v5.34)\n")
                _si_parts.append(
                    "The source file below contains functions that belong to "
                    "**OTHER segments**, not yours. When you see these functions "
                    "in the source code, do NOT copy their bodies into your file. "
                    "Instead, IMPORT them from the upstream segment module.\n"
                )
                _si_parts.append("**Functions in source that you must IMPORT, not copy:**")
                for _own_sym in sorted(_si_other_seg_symbols):
                    _si_parts.append(f"  - `{_own_sym}` — IMPORT this, do NOT copy its body")
                _si_parts.append("")

            for _sf_path, _sf_content in _si_source_files.items():
                _si_parts.append(f"**`{_sf_path}`** ({len(_sf_content):,} chars)")
                # Cap per-file injection at 120K chars to leave room for other context
                _sf_inject = _sf_content[:120_000]
                if len(_sf_content) > 120_000:
                    _sf_inject += f"\n... (truncated from {len(_sf_content):,} chars)"
                _si_parts.append(f"```python\n{_sf_inject}\n```\n")

        # v5.17: Stage 4B Segment Enrichment — grounded evidence from AST parsing
        _enrichment = segment_context.get("enrichment")
        if _enrichment:
            _si_parts.append("### Segment Enrichment (Stage 4B \u2014 Grounded Evidence)\n")
            _si_parts.append(
                "**IMPORTANT**: The following symbols were extracted from the original source "
                "file using AST parsing. You MUST include ALL of them in your architecture. "
                "However, this list may NOT be exhaustive \u2014 the source file may contain "
                "additional functions, classes, or helpers that were not captured by AST "
                "extraction. If your segment logically needs a symbol that is not listed "
                "below but exists in the source file, INCLUDE it. Do NOT exclude functions "
                "just because they are absent from this list.\n"
            )

            # Constants \u2014 exact definitions
            _enrich_constants = _enrichment.get("constants", [])
            if _enrich_constants:
                _si_parts.append("#### Constants (MUST preserve exact names and values)")
                for _ec in _enrich_constants:
                    _ec_val = _ec.get("value", "")
                    if _ec_val:
                        _si_parts.append(f"```python\n{_ec_val}\n```")
                    else:
                        _si_parts.append(f"- `{_ec.get('name', '?')}`")
                _si_parts.append("")

            # Function signatures + structure
            # v5.27: Include line counts and internal helpers so the architecture
            # model knows actual sizes and doesn't create phantom helper files.
            _enrich_functions = _enrichment.get("functions", [])
            if _enrich_functions:
                _si_parts.append("#### Functions (MUST preserve exact signatures)")
                _si_parts.append(
                    "Each function's line count is from AST analysis of the source. "
                    "Use these to judge whether a function fits in the target file. "
                    "Do NOT create extra helper files to split a function — keep it "
                    "whole in the target module unless the spec explicitly says otherwise."
                )
                for _ef in _enrich_functions:
                    _sig = _ef.get("signature", _ef.get("name", "?"))
                    _line_range = _ef.get("line_range")
                    _body = _ef.get("body", "")
                    _line_count = 0
                    if _line_range and len(_line_range) == 2:
                        _line_count = _line_range[1] - _line_range[0] + 1
                    elif _body:
                        _line_count = _body.count("\n") + 1
                    _is_async = _ef.get("is_async", False)
                    _async_tag = " (async)" if _is_async else ""
                    if _line_count:
                        _si_parts.append(f"- `{_sig}`{_async_tag} — **{_line_count} lines**")
                    else:
                        _si_parts.append(f"- `{_sig}`{_async_tag}")
                _si_parts.append("")

            # Classes
            _enrich_classes = _enrichment.get("classes", [])
            if _enrich_classes:
                _si_parts.append("#### Classes (MUST preserve)")
                for _ecl in _enrich_classes:
                    _si_parts.append(f"- `class {_ecl.get('name', '?')}` "
                                     f"(methods: {', '.join(_ecl.get('methods', [])[:10])})")
                _si_parts.append("")

            # Cross-segment contract \u2014 who imports what from this segment
            _enrich_consumed_by = _enrichment.get("consumed_by", {})
            if _enrich_consumed_by:
                _si_parts.append("#### Cross-Segment Contract (other segments import these from YOU)")
                for _cb_seg, _cb_syms in _enrich_consumed_by.items():
                    if isinstance(_cb_syms, list):
                        _si_parts.append(f"- **{_cb_seg}** imports: {', '.join(f'`{s}`' for s in _cb_syms)}")
                _si_parts.append("")

            # What this segment needs from others
            _enrich_consumes = _enrichment.get("consumes", {})
            if _enrich_consumes:
                _si_parts.append("#### Dependencies (symbols YOU need from other segments)")
                for _cn_seg, _cn_syms in _enrich_consumes.items():
                    if isinstance(_cn_syms, list):
                        _si_parts.append(f"- From **{_cn_seg}**: {', '.join(f'`{s}`' for s in _cn_syms)}")
                _si_parts.append("")
                # v5.30: Prohibitive instruction — prevent function duplication
                # across segments. This is the #1 source of cohesion failures.
                _all_dep_symbols = []
                for _cn_syms_list in _enrich_consumes.values():
                    if isinstance(_cn_syms_list, list):
                        _all_dep_symbols.extend(_cn_syms_list)
                if _all_dep_symbols:
                    _si_parts.append("#### ⛔ DUPLICATE FUNCTION PROHIBITION (v5.30)")
                    _si_parts.append(
                        "The following symbols are ALREADY DEFINED in your dependency "
                        "segments. You MUST import them — NEVER redefine, copy, or "
                        "re-implement them in your files. Defining a function that "
                        "already exists in a dependency segment creates duplicate "
                        "definitions and breaks the package."
                    )
                    _si_parts.append("")
                    _si_parts.append("**DO NOT define any of these in your code:**")
                    for _ds in _all_dep_symbols:
                        _si_parts.append(f"  - ❌ `{_ds}` — import it, do NOT redefine it")
                    _si_parts.append("")
                    _si_parts.append(
                        "If your function body needs to call `can_execute_segment()`, "
                        "write `from ._dependencies import can_execute_segment` (or "
                        "the appropriate sibling module). NEVER copy the function body "
                        "from the source monolith into your file."
                    )
                    _si_parts.append("")

            # Design guidance from LLM intelligence
            _enrich_guidance = _enrichment.get("design_guidance", "")
            if _enrich_guidance:
                _si_parts.append(f"#### Design Guidance\n{_enrich_guidance}\n")

            # Source extract structure summary
            # v5.27: If source_extract exists, show total lines being transplanted
            # so the architecture model can plan file sizes accurately.
            _source_extract = _enrichment.get("source_extract")
            if _source_extract and isinstance(_source_extract, dict):
                _total_source_lines = sum(
                    v.count("\n") + 1 for v in _source_extract.values() if isinstance(v, str)
                )
                if _total_source_lines > 0:
                    _si_parts.append(f"#### Source Size Budget")
                    _si_parts.append(
                        f"Total source code being transplanted into this segment: "
                        f"**{_total_source_lines} lines** across {len(_source_extract)} function(s). "
                        f"The target file(s) will also need imports, module docstring, and "
                        f"type annotations, so budget ~{int(_total_source_lines * 1.15)} lines total. "
                        f"This FITS in a single file — do NOT split into helper submodules."
                    )
                    _si_parts.append("")

            # Risk flags
            _enrich_risk = _enrichment.get("risk_level", "low")
            if _enrich_risk in ("medium", "high"):
                _enrich_risk_notes = _enrichment.get("risk_notes", "")
                _si_parts.append(f"\u26a0\ufe0f **Risk: {_enrich_risk.upper()}** \u2014 {_enrich_risk_notes}\n")

            # Unresolved symbols warning
            _enrich_unresolved = _enrichment.get("unresolved", [])
            if _enrich_unresolved:
                _si_parts.append("\u274c **UNRESOLVED SYMBOLS (will cause boot failure):**")
                for _eu in _enrich_unresolved:
                    _si_parts.append(f"- {_eu}")
                _si_parts.append("")

        # Upstream evidence (completed segments' output)
        _si_evidence = segment_context.get("evidence", [])
        if _si_evidence:
            _si_parts.append("### Upstream Evidence (from completed segments)\n")
            # Handle both list and dict forms of evidence
            if isinstance(_si_evidence, dict):
                _si_evidence = list(_si_evidence.values()) if _si_evidence else []
            elif not isinstance(_si_evidence, list):
                _si_evidence = []
            for _ev in _si_evidence[:10]:
                if isinstance(_ev, dict):
                    _ev_path = _ev.get("file_path", _ev.get("path", ""))
                    _ev_sigs = _ev.get("signatures", [])
                    _si_parts.append(f"**{_ev_path}**")
                    for _sig in _ev_sigs[:5]:
                        _si_parts.append(f"  - `{_sig}`")
                elif isinstance(_ev, str):
                    _si_parts.append(f"- {_ev}")
            _si_parts.append("")

        # v5.16: Cohesion regen feedback injection
        # When the cohesion check found issues and is re-generating this segment's
        # architecture, inject the specific issues so the LLM knows what to fix.
        _si_cohesion = segment_context.get("cohesion_issues", "")
        if _si_cohesion:
            _si_parts.append("### ⚠️ COHESION ISSUES (from previous architecture attempt)\n")
            _si_parts.append("The previous architecture for this segment had cross-segment compatibility issues.")
            _si_parts.append("You MUST fix these issues in this regeneration:\n")
            _si_parts.append(f"{_si_cohesion}\n")
            _si_parts.append("Ensure all import names, module names, and function signatures match what other segments expect.\n")

        # v5.38 (Fix 15): Import validation failure feedback injection
        # When the deterministic import validator found phantom/wrong-module
        # imports, inject the specific violations so the LLM fixes them.
        _si_import_feedback = segment_context.get("import_validation_feedback", "")
        if _si_import_feedback:
            _si_parts.append("### ❌ IMPORT VALIDATION FAILED (deterministic check)\n")
            _si_parts.append("The previous architecture for this segment contained cross-segment imports")
            _si_parts.append(" that reference symbols which DO NOT EXIST in the target modules.")
            _si_parts.append(" You MUST fix every violation listed below. Only import symbols that")
            _si_parts.append(" appear in the sibling export map provided above.\n")
            _si_parts.append(f"{_si_import_feedback}\n")

        # v5.25: Implementation failure feedback injection
        # When a previous implementation attempt failed (strike-out), inject
        # the specific failure reasons so the architecture LLM can avoid
        # producing designs that cause the same implementation failures.
        _si_impl_feedback = segment_context.get("implementation_feedback", "")
        if _si_impl_feedback:
            _si_parts.append("### PREVIOUS IMPLEMENTATION FAILED\n")
            _si_parts.append("The previous architecture for this segment was implemented but the Implementer")
            _si_parts.append("failed to produce working code. The specific failure reasons were:\n")
            _si_parts.append(f"{_si_impl_feedback}\n")
            _si_parts.append("DO NOT repeat the design patterns that caused these failures.")
            _si_parts.append("If the failure was 'No edits could be applied', the architecture likely asked")
            _si_parts.append("to MODIFY a large file with edit pairs that did not match. Consider using a")
            _si_parts.append("different file operation strategy (e.g. CREATE a new file instead of MODIFY).")
            _si_parts.append("If the failure was a wrong function name or signature, ensure this architecture")
            _si_parts.append("uses the EXACT names from the skeleton contract.\n")

        # =============================================================
        # v5.36: COMPLETE SIBLING EXPORT MAP
        # When a segment's own `consumes` map is empty (enrichment didn't
        # assign cross-segment dependencies), the architecture LLM has no
        # grounded evidence of what functions exist in sibling segments.
        # It then invents function names like `can_execute_segment` that
        # don't exist, causing phantom-symbol cohesion failures.
        #
        # Fix: Load ALL sibling segments' enrichment from disk and inject
        # every exported symbol. The LLM can then pick the correct real
        # names instead of guessing.
        # =============================================================
        try:
            _job_artifact_root = os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs")
            # v5.36 fix: job_id here is the sub-job ID (e.g. "sg-xxx__seg-04-...")
            # but the segments directory lives under the PARENT job ("sg-xxx").
            # Strip the "__seg-*" suffix to get the parent job dir.
            _parent_job_id = job_id.split("__")[0] if "__" in job_id else job_id
            _job_dir_for_siblings = os.path.join(_job_artifact_root, "jobs", _parent_job_id)
            _segments_dir = os.path.join(_job_dir_for_siblings, "segments")
            _sibling_exports: Dict[str, List[str]] = {}  # seg_id -> [symbol_names]
            _current_seg_id = segment_context.get("segment_id", "")

            if os.path.isdir(_segments_dir):
                for _sib_dir_name in sorted(os.listdir(_segments_dir)):
                    if _sib_dir_name == _current_seg_id:
                        continue  # Skip self
                    _sib_enrich_path = os.path.join(
                        _segments_dir, _sib_dir_name, "enrichment.json"
                    )
                    if not os.path.isfile(_sib_enrich_path):
                        continue
                    try:
                        with open(_sib_enrich_path, "r", encoding="utf-8") as _sef:
                            _sib_enrich = json.load(_sef)
                        # Collect exports (explicit list)
                        _sib_symbols = list(_sib_enrich.get("exports", []))
                        # Also collect function names from enrichment
                        for _sf in _sib_enrich.get("functions", []):
                            _fn_name = _sf.get("name", "")
                            if _fn_name and _fn_name not in _sib_symbols:
                                _sib_symbols.append(_fn_name)
                        # Also collect constant names
                        for _sc in _sib_enrich.get("constants", []):
                            _cn_name = _sc.get("name", "")
                            if _cn_name and _cn_name not in _sib_symbols:
                                _sib_symbols.append(_cn_name)
                        if _sib_symbols:
                            _sibling_exports[_sib_dir_name] = _sib_symbols
                    except Exception:
                        pass  # Skip unreadable enrichment files

            if _sibling_exports:
                _si_parts.append("#### \U0001f4cb Complete Sibling Export Map (v5.36)")
                _si_parts.append("")
                _si_parts.append(
                    "**REFERENCE**: These are ALL real function/constant names "
                    "exported by sibling segments, extracted from the source "
                    "monolith via AST parsing. When your code needs to call a "
                    "function that lives in another segment, pick from THIS "
                    "list \u2014 do NOT invent function names that are not here."
                )
                _si_parts.append("")
                _total_sib_symbols = 0
                for _sib_id, _sib_syms in _sibling_exports.items():
                    _si_parts.append(
                        f"- **{_sib_id}** exports: "
                        f"{', '.join(f'`{s}`' for s in _sib_syms)}"
                    )
                    _total_sib_symbols += len(_sib_syms)
                _si_parts.append("")
                _si_parts.append(
                    "If the function you need is NOT in this list, it either "
                    "doesn\u2019t exist (define it yourself) or belongs to a stdlib/"
                    "third-party module (import normally). NEVER invent a sibling "
                    "import to a symbol not listed above."
                )
                _si_parts.append("")
                logger.info(
                    "[stream_handler] v5.36 Sibling export map injected for %s: "
                    "%d segment(s), %d total symbol(s)",
                    _current_seg_id, len(_sibling_exports), _total_sib_symbols,
                )
        except Exception as _sib_err:
            logger.warning(
                "[stream_handler] v5.36 Sibling export map failed (non-fatal): %s",
                _sib_err,
            )

        _segment_injection = "\n".join(_si_parts)

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
