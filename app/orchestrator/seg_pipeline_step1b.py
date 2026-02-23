# FILE: app/orchestrator/seg_pipeline_step1b.py
"""Step 1b: Deterministic Import Validator — HARD GATE (Fix 15, v5.38)."""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def validate_imports_and_regen(
    arch_text: str,
    seg_id: str,
    seg_job_id: str,
    job_id: str,
    segment_context: Dict[str, Any],
    project_id: int,
    db: Any,
    is_deterministic: bool,
    seg_arch_path: str,
    emit: Any,
) -> str:
    """
    v5.38: Zero-LLM-cost import validation. Every cross-segment import must
    reference a symbol that actually exists in a sibling segment's enrichment.
    If violations found, inject feedback and regenerate (max 1 retry).

    Returns (possibly regenerated) arch_text.
    """
    MAX_IMPORT_REGEN = 1

    if is_deterministic:
        emit(f"  ⚡ Skipping import validation (deterministic imports)")
        logger.info(
            "[SEGMENT_LOOP] v6.1 Skipping import validator for deterministic segment %s",
            seg_id,
        )
        return arch_text

    try:
        from app.orchestrator.import_validator import validate_architecture_imports
    except ImportError:
        logger.debug("[SEGMENT_LOOP] import_validator not available — skipping")
        return arch_text

    try:
        parent_jid = job_id.split("__")[0] if "__" in job_id else job_id
        artifact_root = os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs")
        parent_job_dir = os.path.join(artifact_root, "jobs", parent_jid)

        import_result = validate_architecture_imports(
            arch_text=arch_text,
            segment_id=seg_id,
            parent_job_dir=parent_job_dir,
        )

        if import_result.passed:
            emit(
                f"  ✅ Import validation: {import_result.symbols_checked} "
                f"cross-segment import(s) verified"
            )
            return arch_text

        # Violations found
        emit(f"  ❌ Import validation: {len(import_result.violations)} violation(s) found")
        for v in import_result.violations:
            emit(f"    ⚠️ {v.symbol_name}: {v.message}")

        # Regenerate with feedback
        emit(f"  🔄 Import validation regen 1/{MAX_IMPORT_REGEN} — regenerating architecture...")
        logger.info(
            "[SEGMENT_LOOP] v5.38 Import validation regen 1/%d for %s: %d violation(s)",
            MAX_IMPORT_REGEN, seg_id, len(import_result.violations),
        )

        segment_context["import_validation_feedback"] = import_result.format_feedback()

        regen_text = await _regen_architecture(
            seg_job_id, segment_context, project_id, db, emit,
        )

        if regen_text:
            arch_text = regen_text
            emit(f"  ✅ Architecture regenerated ({len(arch_text)} chars)")

            # Re-validate
            import_result_2 = validate_architecture_imports(
                arch_text=arch_text,
                segment_id=seg_id,
                parent_job_dir=parent_job_dir,
            )
            if import_result_2.passed:
                emit(
                    f"  ✅ Import validation (regen): "
                    f"{import_result_2.symbols_checked} import(s) verified"
                )
            else:
                emit(
                    f"  ⚠️ Import validation (regen): "
                    f"{len(import_result_2.violations)} violation(s) remain"
                )
                for v2 in import_result_2.violations:
                    emit(f"    ⚠️ {v2.symbol_name}: {v2.message}")

            # Re-save
            try:
                with open(seg_arch_path, "w", encoding="utf-8") as f:
                    f.write(arch_text)
                emit(f"  💾 Regenerated architecture saved")
            except Exception as save_err:
                logger.warning(
                    "[SEGMENT_LOOP] v5.38 Failed to save regen arch: %s", save_err
                )
        else:
            emit(f"  ⚠️ Regen produced no output — keeping original")

        segment_context.pop("import_validation_feedback", None)

    except Exception as iv_err:
        logger.warning(
            "[SEGMENT_LOOP] v5.38 Import validator error (non-fatal): %s", iv_err
        )

    return arch_text


async def _regen_architecture(
    seg_job_id: str,
    segment_context: Dict[str, Any],
    project_id: int,
    db: Any,
    emit: Any,
) -> Optional[str]:
    """Re-run Critical Pipeline with feedback. Returns arch text or None."""
    try:
        from app.llm.critical_pipeline_stream import generate_critical_pipeline_stream
    except ImportError:
        return None

    parts: List[str] = []
    try:
        async for event in generate_critical_pipeline_stream(
            project_id=project_id,
            message=json.dumps(segment_context.get("segment_spec", {})),
            db=db,
            job_id=seg_job_id,
            segment_context=segment_context,
        ):
            if not isinstance(event, str):
                continue
            for line in event.split("\n"):
                if not line.startswith("data: "):
                    continue
                try:
                    payload = json.loads(line[6:])
                except (json.JSONDecodeError, ValueError):
                    continue
                if payload.get("type") == "token":
                    parts.append(payload.get("content", ""))

        return "".join(parts) if parts else None
    except Exception as e:
        emit(f"  ⚠️ Regen failed: {e} — keeping original")
        logger.warning("[SEGMENT_LOOP] v5.38 Import regen failed: %s", e)
        return None
