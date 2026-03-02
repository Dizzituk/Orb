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
from app.llm.critical_pipeline._stream_handler_utils import _build_segment_critique_spec, _format_enrichment_for_critique, _handle_micro, _handle_scan
from app.llm.critical_pipeline._stream_handler_utils_1 import _done, _save_to_memory, _token, generate_critical_pipeline_stream
from app.llm.critical_pipeline._stream_handler_utils_2 import _sse
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

    # v3.2: Codebase report DISABLED for architecture stage.
    # The arch map + segment-scoped evidence + evidence ledger files
    # provide all the context needed. The codebase report is a bulk
    # dump that adds ~5.5K tokens of noise per segment and is often
    # stale. Overwatcher has full project visibility if cross-segment
    # issues need catching.
    logger.info("[critical_pipeline] Codebase report: DISABLED (v3.2 - arch map + segment evidence sufficient)")

    cp_evidence = gather_critical_pipeline_evidence(
        spec_data=spec_data, message=message,
        include_arch_map=True, include_codebase_report=False,
        include_file_evidence=True,
        arch_map_max_lines=800,
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
        _is_refactor = is_refactor_job(spec_data, message)
        _mem_patterns = retrieve_for_stage(
            db, stage="critical_pipeline",
            context=f"Generating architecture for: {original_request[:200]}",
            job_type="refactor" if _is_refactor else None,
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

    # =====================================================================
    # v1.0 (Job 4): Architecture Template Engine — pre-fill deterministic sections
    # =====================================================================
    _arch_template = ""
    if segment_context:
        try:
            from app.orchestrator.arch_template.engine import (
                generate_architecture_template,
                build_llm_instruction_prefix,
            )
            # Load skeleton contracts
            _tmpl_skeleton = None
            _tmpl_all_skeletons = None
            _tmpl_seg_id = segment_context.get("segment_id", "")
            _tmpl_parent_job = job_id.split("__")[0] if "__" in job_id else job_id
            try:
                _skel_path = os.path.join(
                    os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"),
                    "jobs", _tmpl_parent_job, "segments", "skeleton_contract.json",
                )
                if os.path.exists(_skel_path):
                    with open(_skel_path, "r", encoding="utf-8") as _sf:
                        _skel_full = json.load(_sf)
                    _tmpl_all_skeletons = _skel_full.get("skeletons", [])
                    for _sk in _tmpl_all_skeletons:
                        if _sk.get("segment_id") == _tmpl_seg_id:
                            _tmpl_skeleton = _sk
                            break
            except Exception as _sk_err:
                logger.debug("[arch_template] Skeleton load: %s", _sk_err)

            if _tmpl_skeleton:
                # Build evidence context from file evidence
                _tmpl_evidence = {}
                if cp_evidence and cp_evidence.multi_target_files:
                    for _tf in cp_evidence.multi_target_files:
                        if hasattr(_tf, 'path') and hasattr(_tf, 'content'):
                            _tmpl_evidence[_tf.path] = _tf.content
                        elif isinstance(_tf, dict):
                            _tmpl_evidence[_tf.get('path', '')] = _tf.get('content', '')

                # Build design tokens — prefer cached registry, fallback to evidence CSS
                _tmpl_tokens = None
                try:
                    from app.orchestrator.arch_template.token_cache import get_or_build_registry
                    _tmpl_job_dir = os.path.join(
                        os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"),
                        "jobs", _tmpl_parent_job,
                    )
                    _tmpl_tokens = get_or_build_registry(
                        job_dir=_tmpl_job_dir,
                        frontend_base=os.getenv("ORB_FRONTEND_ROOT", r"D:\orb-desktop"),
                    )
                except Exception:
                    # Fallback: extract from evidence CSS
                    try:
                        from app.orchestrator.arch_template.token_registry import (
                            build_token_registry_from_content,
                        )
                        _css_evidence = {
                            k: v for k, v in _tmpl_evidence.items()
                            if k.endswith(".css")
                        }
                        if _css_evidence:
                            _tmpl_tokens = build_token_registry_from_content(_css_evidence)
                    except Exception:
                        pass

                # Build segment spec dict
                # Try multiple sources for grounding_data
                _tmpl_grounding = segment_context.get("_grounding_data", {})
                if not _tmpl_grounding:
                    _tmpl_ss = segment_context.get("segment_spec", {})
                    _tmpl_grounding = _tmpl_ss.get("grounding_data", {}) if isinstance(_tmpl_ss, dict) else {}
                # Use skeleton dependencies (segment_context.dependencies is often empty)
                _tmpl_deps = _tmpl_skeleton.get("dependencies", [])
                if not _tmpl_deps:
                    _tmpl_deps = segment_context.get("dependencies", [])
                _tmpl_seg_spec = {
                    "segment_id": _tmpl_seg_id,
                    "file_scope": segment_context.get("file_scope", []),
                    "requirements": segment_context.get("requirements", []),
                    "dependencies": _tmpl_deps,
                    "grounding_data": _tmpl_grounding,
                }

                template = generate_architecture_template(
                    segment_spec=_tmpl_seg_spec,
                    skeleton=_tmpl_skeleton,
                    all_skeletons=_tmpl_all_skeletons,
                    spec_id=spec_id,
                    spec_hash=spec_hash,
                    design_tokens=_tmpl_tokens,
                    evidence_context=_tmpl_evidence,
                    requirements=segment_context.get("requirements"),
                )
                _arch_template = build_llm_instruction_prefix(template)
                yield _emit(
                    f"\U0001f9e9 **Architecture template:** "
                    f"{len(template)} chars pre-filled from skeleton contracts\n"
                )
                logger.info(
                    "[arch_template] Template generated: %d chars for %s",
                    len(template), _tmpl_seg_id,
                )
        except Exception as _tmpl_err:
            logger.warning("[arch_template] Template generation failed (non-fatal): %s", _tmpl_err)

    _user_content = f"Generate architecture for:\n\n{original_request}\n\n"
    if _arch_template:
        _user_content += f"{_arch_template}\n\n---\n\n"
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

    # v3.0: Extract deterministic verdict data from segment_context
    _det_segment_id = segment_context.get("segment_id") if segment_context else None
    _det_segment_spec = segment_context.get("segment_spec") if segment_context else None
    _det_skeleton_file_scope = segment_context.get("file_scope") if segment_context else None
    _det_enrichment_data = None
    _det_skeleton_contract = None
    _det_manifest_dict = None
    _det_needle_estimate = None
    if segment_context:
        # Extract needle from grounding data
        _gd = segment_context.get("_grounding_data")
        if _gd and isinstance(_gd, dict):
            _ne = _gd.get("needle_estimate")
            if isinstance(_ne, dict):
                _det_needle_estimate = _ne.get("needle_estimate")
            elif isinstance(_ne, int):
                _det_needle_estimate = _ne
        # Try to load skeleton contract and manifest from job dir
        try:
            _seg_spec = segment_context.get("segment_spec", {})
            _parent_spec_id = _seg_spec.get("parent_spec_id", "") if isinstance(_seg_spec, dict) else ""
            if _parent_spec_id and job_id:
                import os as _os
                _jobs_base = _os.path.join("jobs", "jobs")
                # Try parent job dir (for skeleton + manifest)
                for _candidate in [_parent_spec_id, job_id]:
                    _seg_dir = _os.path.join(_jobs_base, _candidate, "segments")
                    _skel_path = _os.path.join(_seg_dir, "skeleton_contract.json")
                    _man_path = _os.path.join(_seg_dir, "manifest.json")
                    if _os.path.isfile(_skel_path) and not _det_skeleton_contract:
                        import json as _json
                        with open(_skel_path, "r", encoding="utf-8") as _f:
                            _det_skeleton_contract = _json.load(_f)
                    if _os.path.isfile(_man_path) and not _det_manifest_dict:
                        import json as _json
                        with open(_man_path, "r", encoding="utf-8") as _f:
                            _det_manifest_dict = _json.load(_f)
        except Exception as _skel_err:
            logger.debug("[critical_pipeline] v3.0 Could not load skeleton/manifest: %s", _skel_err)
        # Enrichment data — try to load all segment enrichments
        if _det_manifest_dict:
            try:
                import os as _os, json as _json
                _det_enrichment_data = {}
                _jobs_base = _os.path.join("jobs", "jobs")
                for _seg_entry in _det_manifest_dict.get("segments", []):
                    _sid = _seg_entry.get("segment_id", "")
                    for _candidate in [_parent_spec_id, job_id]:
                        _enr_path = _os.path.join(_jobs_base, _candidate, "segments", _sid, "enrichment.json")
                        if _os.path.isfile(_enr_path):
                            with open(_enr_path, "r", encoding="utf-8") as _f:
                                _det_enrichment_data[_sid] = _json.load(_f)
                            break
            except Exception as _enr_err:
                logger.debug("[critical_pipeline] v3.0 Could not load enrichment: %s", _enr_err)

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
            # v3.0: Deterministic verdict parameters
            segment_id=_det_segment_id,
            segment_spec=_det_segment_spec,
            skeleton_contract=_det_skeleton_contract,
            skeleton_file_scope_det=_det_skeleton_file_scope,
            enrichment_data=_det_enrichment_data,
            manifest_dict=_det_manifest_dict,
            needle_estimate=_det_needle_estimate,
        )
    except Exception as e:
        logger.exception("[critical_pipeline] Pipeline failed: %s", e)
        yield _emit(f"\u274c **Pipeline failed:** {e}\n")
        result = None  # v3.4-fix: Fall through to empty-result handler for fallback

    if not result or not result.content:
        # v3.4-fix: Retry with fallback provider on empty result.
        # Empty results typically mean an API timeout or malformed response.
        # Try up to 2 fallback providers before giving up.
        _FALLBACK_MODELS = [
            ("anthropic", "claude-sonnet-4-6"),
            ("openai", "gpt-5.2"),
            ("google", "gemini-2.5-flash"),
        ]
        _fallback_attempted = False
        for _fb_provider, _fb_model in _FALLBACK_MODELS:
            # Skip the provider that just failed
            if _fb_provider == pipeline_provider:
                continue
            logger.warning(
                "[critical_pipeline] v3.4 Empty result from %s/%s — "
                "retrying with fallback %s/%s",
                pipeline_provider, pipeline_model, _fb_provider, _fb_model,
            )
            yield _emit(
                f"\u26a0\ufe0f **Empty result from {pipeline_provider}/{pipeline_model}** "
                f"— retrying with {_fb_provider}/{_fb_model}...\n"
            )
            try:
                result = await run_high_stakes_with_critique(
                    task=task,
                    provider_id=_fb_provider,
                    model_id=_fb_model,
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
                    segment_id=_det_segment_id,
                    segment_spec=_det_segment_spec,
                    skeleton_contract=_det_skeleton_contract,
                    skeleton_file_scope_det=_det_skeleton_file_scope,
                    enrichment_data=_det_enrichment_data,
                    manifest_dict=_det_manifest_dict,
                    needle_estimate=_det_needle_estimate,
                )
            except Exception as _fb_exc:
                logger.warning(
                    "[critical_pipeline] v3.4 Fallback %s/%s also failed: %s",
                    _fb_provider, _fb_model, _fb_exc,
                )
                yield _emit(
                    f"\u274c **Fallback {_fb_provider}/{_fb_model} failed:** {_fb_exc}\n"
                )
                continue

            if result and result.content:
                _fallback_attempted = True
                pipeline_provider = _fb_provider
                pipeline_model = _fb_model
                logger.info(
                    "[critical_pipeline] v3.4 Fallback succeeded: %s/%s "
                    "(%d chars)",
                    _fb_provider, _fb_model, len(result.content),
                )
                yield _emit(
                    f"\u2705 **Fallback succeeded:** {_fb_provider}/{_fb_model}\n"
                )
                break
            else:
                logger.warning(
                    "[critical_pipeline] v3.4 Fallback %s/%s also returned empty",
                    _fb_provider, _fb_model,
                )
                yield _emit(
                    f"\u274c **Fallback {_fb_provider}/{_fb_model} also empty.**\n"
                )

        if not _fallback_attempted and (not result or not result.content):
            yield _emit("\u274c **Pipeline returned empty result after all fallbacks.**\n")
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
    _crit_status = "\u2705 PASSED" if critique_passed else f"\u26a0\ufe0f {blocking_issues} blocking issues"
    yield _emit(
        f"**Architecture ID:** `{arch_id}`\n"
        f"**Final Version:** v{final_version}\n"
        f"**Critique Mode:** deep (blocker filtering enabled)\n"
        f'**Critique Status:** {_crit_status}\n'
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
