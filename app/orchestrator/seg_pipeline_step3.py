# FILE: app/orchestrator/seg_pipeline_step3.py
"""Step 3: Overwatcher Pre-Flight + Architecture Executor."""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from app.orchestrator.segment_state import get_job_dir, load_or_init_state
from app.orchestrator._segment_loop_utils_7 import (
    _is_facade_segment, _save_execution_trace,
)

logger = logging.getLogger(__name__)


async def run_preflight(
    seg_id: str,
    arch_text: str,
    segment_context: Dict[str, Any],
    contract_set: Any,
    manifest: Any,
    parent_spec: Any,
    job_id: str,
    job_dir_path: str,
    seg_arch_path: str,
    emit: Any,
) -> Optional[Dict[str, Any]]:
    """
    Step 3a: Overwatcher Coherence Pre-Flight (deterministic).
    Returns a failure result dict if preflight fails, or None to proceed.
    """
    try:
        from app.overwatcher.preflight import run_segment_preflight, save_rejection
    except ImportError:
        logger.debug("[execute_segment] Preflight module not available — skipping")
        return None

    try:
        seg_contract = segment_context.get("interface_contract", "")
        skeleton_json = contract_set.to_json() if contract_set else None
        manifest_dict = manifest.to_dict() if manifest and hasattr(manifest, 'to_dict') else None

        spec_md = ""
        if isinstance(parent_spec, str):
            spec_md = parent_spec
        elif parent_spec:
            try:
                spec_md = json.dumps(parent_spec)
            except Exception:
                pass

        rejection = run_segment_preflight(
            segment_id=seg_id,
            architecture_content=arch_text,
            skeleton_json=skeleton_json,
            manifest_dict=manifest_dict,
            job_id=job_id,
            architecture_path=seg_arch_path,
            skeleton_contract_markdown=seg_contract,
            spec_markdown=spec_md,
            attempt_number=segment_context.get("_attempt_number", 1),
        )

        if rejection:
            emit(f"  ❌ PRE-FLIGHT FAILED for {seg_id}: {rejection.summary}")
            for iss in rejection.issues:
                emit(f"    🚫 [{iss.get('category', '?')}] {iss.get('description', '?')}")
            emit(f"  🔄 Route: back to Critical Pipeline (segment only)")
            try:
                save_rejection(rejection, job_dir_path)
                emit(f"  💾 Rejection saved: {rejection.rejection_id}")
            except Exception as sav_err:
                logger.warning("[execute_segment] Failed to save rejection: %s", sav_err)

            return {
                "success": False,
                "output_files": [],
                "error": None,
                "critique_warnings": [],
                "preflight_failed": True,
                "rejection": rejection.to_dict(),
            }

        emit(f"  ✅ Pre-flight PASSED for {seg_id}")
        return None

    except Exception as pf_err:
        logger.warning("[execute_segment] Pre-flight check error (non-fatal): %s", pf_err)
        emit(f"  ⚠️ Pre-flight check error (non-fatal): {pf_err}")
        return None


async def execute_architecture(
    seg_id: str,
    seg_job_id: str,
    arch_text: str,
    seg_arch_path: str,
    segment_context: Dict[str, Any],
    segment: Any,
    manifest: Any,
    job_id: str,
    job_dir_path: str,
    project_id: int,
    db: Any,
    contract_set: Any,
    quarantine_result: Any,
    emit: Any,
) -> Dict[str, Any]:
    """
    Step 3b: Overwatcher Architecture Execution.
    Returns result dict with success, output_files, error, critique_warnings.
    """
    result = {
        "success": False,
        "output_files": [],
        "error": None,
        "critique_warnings": [],
    }

    try:
        from app.overwatcher.architecture_executor import run_architecture_execution
        from app.overwatcher.spec_resolution import resolve_latest_spec
        from app.llm.overwatcher_stream import create_overwatcher_llm_fn
    except ImportError as ae:
        emit(f"  ⚠️ Architecture executor not available — architecture generated only")
        result["success"] = True
        return result

    try:
        spec = resolve_latest_spec(project_id, db)
        if spec is None:
            emit(f"  ⚠️ No spec found for project {project_id} — skipping Overwatcher")
            result["success"] = True
            return result

        llm_call_fn = create_overwatcher_llm_fn()
        seg_contract = segment_context.get("interface_contract", "")

        # v5.7: Promote quarantined MODIFY->CREATE
        recon_arch_text = _apply_quarantine_promotions(
            arch_text, quarantine_result, emit,
        )

        # v6.0: Implementation Compiler + Brief Validator
        recon_arch_text = _compile_briefs(
            recon_arch_text, arch_text, seg_id, segment, segment_context,
            manifest, job_id, job_dir_path, emit,
        )

        # v5.32: Pass manifest files for import validation
        manifest_all_files = set()
        if manifest:
            for ms in manifest.segments:
                for mf in ms.file_scope:
                    manifest_all_files.add(mf.replace("\\", "/"))

        arch_result = await run_architecture_execution(
            spec=spec,
            architecture_content=recon_arch_text,
            architecture_path=seg_arch_path,
            job_id=seg_job_id,
            llm_call_fn=llm_call_fn,
            artifact_root=os.getenv("ORB_JOB_ARTIFACT_ROOT", "D:/Orb/jobs"),
            interface_contract=seg_contract,
            skip_boot_check=True,
            manifest_all_files=manifest_all_files if manifest_all_files else None,
        )

        if arch_result.get("success", False):
            result["success"] = True
            result["output_files"] = arch_result.get("artifacts_written", [])
            result["critique_warnings"] = [
                e.get("status", "")
                for e in arch_result.get("trace", [])
                if e.get("stage", "").startswith("WARN")
            ]
            emit(
                f"  ✅ Overwatcher + Implementer completed for {seg_id} "
                f"({len(result['output_files'])} artifact(s) written)"
            )
            for of in result['output_files']:
                emit(f"    ✅ {of}")
        else:
            error_msg = arch_result.get("error", "Unknown error")
            result["error"] = f"Architecture execution failed for {seg_id}: {error_msg}"
            emit(f"  ❌ Architecture execution failed for {seg_id}: {error_msg}")
            _save_execution_trace(seg_id, get_job_dir(job_id), arch_result)
            n_trace = len(arch_result.get("trace", []))
            if n_trace:
                emit(
                    f"  💾 Execution trace saved ({n_trace} events) — "
                    f"check segments/{seg_id}/execution_trace/trace.json"
                )

    except Exception as e:
        result["error"] = f"Overwatcher failed for {seg_id}: {e}"
        logger.exception("[SEGMENT_LOOP] Overwatcher error for %s", seg_id)

    return result


def _apply_quarantine_promotions(
    arch_text: str, quarantine_result: Any, emit: Any,
) -> str:
    """v5.7: Promote quarantined MODIFY->CREATE in architecture text."""
    if not quarantine_result or not quarantine_result.has_quarantined:
        return arch_text
    try:
        from app.orchestrator.package_quarantine import promote_quarantined_in_architecture
        orig_len = len(arch_text)
        arch_text = promote_quarantined_in_architecture(
            arch_text, quarantine_result.quarantined_rel_paths,
        )
        if len(arch_text) != orig_len:
            emit(f"  [quarantine] Promoted quarantined file(s) from MODIFY->CREATE")
    except Exception as promo_err:
        logger.warning(
            "[SEGMENT_LOOP] v5.7 Quarantine promotion failed (non-fatal): %s",
            promo_err,
        )
    return arch_text


def _compile_briefs(
    recon_arch_text: str,
    arch_text: str,
    seg_id: str,
    segment: Any,
    segment_context: Dict[str, Any],
    manifest: Any,
    job_id: str,
    job_dir_path: str,
    emit: Any,
) -> str:
    """v6.0: Implementation Compiler + Brief Validator."""
    try:
        from app.orchestrator.implementation_compiler import (
            compile_implementation_briefs, save_compilation_result,
        )
        from app.orchestrator.brief_validator import (
            validate_and_fix_briefs, save_fix_log,
        )
    except ImportError as comp_imp_err:
        logger.info(
            "[SEGMENT_LOOP] v6.0 Implementation compiler not available: %s — legacy fallback",
            comp_imp_err,
        )
        emit(f"  ⚠️ Compiler not available — using legacy injection path")
        return _legacy_injection_path(
            recon_arch_text, arch_text, seg_id, segment, segment_context,
            manifest, job_id, emit,
        )

    try:
        parent_job_id = job_id.split('__')[0]
        from app.orchestrator.segment_state import get_job_dir
        compiler_job_dir = get_job_dir(parent_job_id)
        compiler_enrichment = segment_context.get("enrichment")

        if not compiler_enrichment:
            try:
                from app.orchestrator.extraction_binding import load_segment_enrichment
                compiler_enrichment = load_segment_enrichment(compiler_job_dir, seg_id)
            except Exception:
                pass

        # Gather reconciliation data
        compiler_recon = ""
        try:
            from app.orchestrator.interface_reconciliation import (
                read_dependency_interfaces_from_sandbox,
                inject_reconciliation_into_architecture,
            )
            if segment.dependencies:
                recon_state = load_or_init_state(
                    job_id.split('__')[0], manifest
                ) if manifest else None
                if recon_state:
                    recon_block = read_dependency_interfaces_from_sandbox(
                        segment=segment,
                        completed_segments=recon_state.segments,
                        manifest=manifest,
                    )
                    if recon_block:
                        compiler_recon = recon_block
        except Exception as cr_err:
            logger.warning(
                "[SEGMENT_LOOP] v6.0 Reconciliation gather failed (non-fatal): %s",
                cr_err,
            )

        # Gather sibling enrichments
        sibling_enrichments = {}
        try:
            from app.orchestrator.extraction_binding import (
                load_segment_enrichment as load_sib_enrich,
            )
            if manifest:
                for sib_seg in manifest.segments:
                    if sib_seg.segment_id != seg_id:
                        sib_e = load_sib_enrich(compiler_job_dir, sib_seg.segment_id)
                        if sib_e:
                            sibling_enrichments[sib_seg.segment_id] = sib_e
        except Exception:
            pass

        compilation = compile_implementation_briefs(
            architecture_text=arch_text,
            enrichment=compiler_enrichment,
            segment_id=seg_id,
            source_file_evidence=segment_context.get("source_file_evidence"),
            interface_contract=segment_context.get("interface_contract", ""),
            sibling_interfaces=segment_context.get("sibling_interfaces", ""),
            cohesion_feedback=segment_context.get("cohesion_feedback", ""),
            implementation_feedback=segment_context.get("implementation_feedback", ""),
            import_validation_feedback=segment_context.get("import_validation_feedback", ""),
            sibling_enrichments=sibling_enrichments,
        )

        emit(
            f"  📦 Implementation compiler: {len(compilation.briefs)} brief(s), "
            f"{compilation.total_functions} function(s), "
            f"~{compilation.total_estimated_lines} lines "
            f"(profile={compilation.profile.value})"
        )

        fixed_briefs, fix_log = validate_and_fix_briefs(
            briefs=compilation.briefs,
            enrichment=compiler_enrichment,
            sibling_enrichments=sibling_enrichments,
        )
        compilation.briefs = fixed_briefs

        if fix_log.had_fixes:
            emit(f"  🔧 Brief validator: {fix_log.issues_found} issue(s) found and auto-fixed")
            for fix in fix_log.fixes:
                emit(f"    [{fix.check}] {fix.description}")
        else:
            emit(f"  ✅ Brief validator: all checks passed")

        save_compilation_result(compilation, compiler_job_dir, seg_id)
        save_fix_log(fix_log, compiler_job_dir, seg_id)

        if compilation.briefs:
            brief_sections = ["\n\n---\n"]
            brief_sections.append("## IMPLEMENTATION BRIEFS (v6.0 — Compiler Output)\n")
            brief_sections.append(
                "The following per-file briefs were compiled from enrichment data, "
                "source extractions, and interface contracts. **These briefs are the "
                "primary instruction for implementation.** Follow the directive in each "
                "brief. If a brief says TRANSPLANT VERBATIM, copy the provided source "
                "code exactly — do not rewrite, simplify, or reimagine.\n"
            )
            for brief in compilation.briefs:
                brief_sections.append(brief.to_markdown())
                brief_sections.append("\n---\n")

            briefs_text = "\n".join(brief_sections)
            recon_arch_text = briefs_text + "\n\n" + arch_text
            emit(
                f"  📄 Injected {len(compilation.briefs)} compiled brief(s) "
                f"({len(briefs_text)} chars)"
            )

        # Supplementary reconciliation
        if compiler_recon and compiler_recon not in recon_arch_text:
            try:
                from app.orchestrator.interface_reconciliation import (
                    inject_reconciliation_into_architecture,
                )
                recon_arch_text = inject_reconciliation_into_architecture(
                    recon_arch_text, compiler_recon,
                )
                emit(
                    f"  🧩 Interface reconciliation: supplementary injection "
                    f"({len(compiler_recon)} chars)"
                )
            except Exception:
                pass

        return recon_arch_text

    except Exception as comp_err:
        logger.warning(
            "[SEGMENT_LOOP] v6.0 Implementation compiler error (non-fatal): %s",
            comp_err,
        )
        emit(f"  ⚠️ Compiler error (non-fatal): {comp_err} — using raw architecture")
        return arch_text


def _legacy_injection_path(
    recon_arch_text: str,
    arch_text: str,
    seg_id: str,
    segment: Any,
    segment_context: Dict[str, Any],
    manifest: Any,
    job_id: str,
    emit: Any,
) -> str:
    """Legacy v5.12 + v5.26 injection path (fallback when compiler unavailable)."""
    try:
        from app.orchestrator.interface_reconciliation import (
            read_dependency_interfaces_from_sandbox,
            inject_reconciliation_into_architecture,
        )
        if segment.dependencies:
            recon_state = load_or_init_state(
                job_id.split('__')[0], manifest
            ) if manifest else None
            if recon_state:
                recon_block = read_dependency_interfaces_from_sandbox(
                    segment=segment,
                    completed_segments=recon_state.segments,
                    manifest=manifest,
                )
                if recon_block:
                    recon_arch_text = inject_reconciliation_into_architecture(
                        arch_text, recon_block,
                    )
    except Exception as recon_err:
        logger.warning(
            "[SEGMENT_LOOP] v5.12 Legacy reconciliation failed: %s", recon_err
        )

    try:
        from app.orchestrator.extraction_binding import (
            load_segment_enrichment, build_extraction_block,
            build_facade_export_map, inject_extraction_into_architecture,
        )
        from app.orchestrator.segment_state import get_job_dir
        parent_job_id = job_id.split('__')[0]
        eb_job_dir = get_job_dir(parent_job_id)
        eb_enrichment = load_segment_enrichment(eb_job_dir, seg_id)
        if eb_enrichment:
            is_facade = _is_facade_segment(segment, manifest) if manifest else False
            if is_facade and manifest:
                eb_block = build_facade_export_map(
                    eb_job_dir, manifest.segments, seg_id,
                )
            else:
                eb_block = build_extraction_block(eb_enrichment, seg_id)
            if eb_block:
                recon_arch_text = inject_extraction_into_architecture(
                    recon_arch_text, eb_block,
                )
    except Exception as eb_err:
        logger.warning(
            "[SEGMENT_LOOP] v5.26 Legacy extraction binding failed: %s", eb_err
        )

    return recon_arch_text
