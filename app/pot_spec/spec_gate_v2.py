# FILE: app/pot_spec/spec_gate_v2.py
"""
Spec Gate v2 - Main Entrypoint

v2.1 (2026-01-04): Blocking Validation Fix
- Missing steps/outputs/verification now BLOCK validation
- Detects placeholder values
- Returns status='needs_clarification' with specific questions

Split into modules:
- spec_gate_types.py: Types, constants, validation
- spec_gate_parsers.py: Parsing and extraction
- spec_gate_persistence.py: DB and filesystem
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from .spec_gate_types import (
    SpecGateResult,
    validate_blocking_fields,
    derive_required_questions,
)
from .spec_gate_parsers import (
    strip_astra_command,
    extract_weaver_spec,
    best_effort_title_and_objective,
    coerce_output_items,
    coerce_step_items,
    coerce_acceptance_items,
    parse_user_clarification,
    extract_outputs_from_acceptance,
    extract_filename_from_text,
)

# v2.2: Auto-generate steps instead of asking user
try:
    from .step_generator import generate_steps_from_task, should_ask_for_steps
    STEP_GENERATOR_AVAILABLE = True
except ImportError:
    STEP_GENERATOR_AVAILABLE = False
    generate_steps_from_task = None
    should_ask_for_steps = None


def _auto_generate_steps(objective: str, outputs: list, content_verbatim: str, location: str) -> list:
    """
    Built-in step generator - always works, no external dependency.
    The system figures out HOW to do the task, user just says WHAT they want.
    """
    steps = []
    obj_lower = (objective or "").lower()
    
    # Detect task type
    is_create = any(kw in obj_lower for kw in ["create", "write", "make", "new", "add"])
    is_modify = any(kw in obj_lower for kw in ["modify", "edit", "update", "change"])
    is_find = any(kw in obj_lower for kw in ["find", "locate", "search", "where"])
    
    # Get target info
    filename = None
    folder = location or ""
    if outputs and isinstance(outputs, list) and len(outputs) > 0:
        first = outputs[0]
        if isinstance(first, dict):
            filename = first.get("name")
            folder = first.get("path") or location or ""
    
    # Generate steps based on task type
    if is_create or (content_verbatim and filename):
        if folder:
            steps.append(f"Locate target folder: {folder}")
        if filename:
            content_preview = content_verbatim[:30] + "..." if content_verbatim and len(content_verbatim) > 30 else content_verbatim
            if content_verbatim:
                steps.append(f"Create file '{filename}' with content: \"{content_preview}\"")
            else:
                steps.append(f"Create file '{filename}'")
        else:
            steps.append("Create specified output")
        steps.append("Verify file exists with correct content")
    
    elif is_modify:
        if filename:
            steps.append(f"Locate {filename}")
            steps.append(f"Read current contents")
            steps.append("Apply modifications")
            steps.append("Save changes")
            steps.append("Verify modifications applied")
        else:
            steps.append("Locate target file")
            steps.append("Apply modifications")
            steps.append("Verify changes")
    
    elif is_find:
        steps.append("Search filesystem for target")
        steps.append("Report location")
        steps.append("Verify accessibility")
    
    else:
        # Generic fallback
        if folder:
            steps.append(f"Navigate to {folder}")
        steps.append("Execute task operation")
        steps.append("Verify completion")
    
    return steps
from .spec_gate_persistence import (
    write_spec_artifacts,
    compute_spec_hash,
    build_spot_markdown,
    safe_summary_from_objective,
    build_spec_schema,
    persist_spec,
    update_spec_status,
)

logger = logging.getLogger(__name__)


async def run_spec_gate_v2(
    *,
    db: Session,
    job_id: str,
    user_intent: str,
    provider_id: str,
    model_id: str,
    project_id: int,
    constraints_hint: Optional[dict] = None,
    spec_version: int = 1,
) -> SpecGateResult:
    """
    Deterministic Spec Gate with BLOCKING VALIDATION (v2.1):
      - Uses Weaver PoT + user clarification
      - BLOCKS if steps/outputs/verification are missing (rounds 1-2)
      - Round 3: finalizes even if incomplete
      - Persists to DB + writes job artifacts
    """
    try:
        # Normalize round
        round_n = max(1, min(3, int(spec_version or 1)))

        # Extract Weaver spec
        weaver_spec, weaver_prov = extract_weaver_spec(constraints_hint)

        # Parse inputs
        user_text = strip_astra_command(user_intent or "")
        title, objective = best_effort_title_and_objective(weaver_spec, user_text)

        # Parse user's direct answers
        outputs_user, steps_user, verify_user = parse_user_clarification(user_text)

        # Get Weaver metadata for content preservation
        weaver_metadata = {}
        if isinstance(weaver_spec, dict):
            weaver_metadata = weaver_spec.get("metadata", {}) or {}
            if not weaver_metadata:
                weaver_metadata = {
                    "content_verbatim": weaver_spec.get("content_verbatim"),
                    "location": weaver_spec.get("location"),
                    "scope_constraints": weaver_spec.get("scope_constraints", []),
                }

        cv = weaver_metadata.get("content_verbatim") or (weaver_spec or {}).get("content_verbatim")
        loc = weaver_metadata.get("location") or (weaver_spec or {}).get("location")

        # Get raw outputs from Weaver
        raw_outputs = None
        if isinstance(weaver_spec, dict):
            raw_outputs = (
                weaver_spec.get("outputs") or
                weaver_spec.get("metadata", {}).get("outputs") or
                weaver_spec.get("requirements", {}).get("functional", [])
            )
        
        # Coerce steps - check multiple Weaver locations
        steps = steps_user
        if not steps and isinstance(weaver_spec, dict):
            # Try multiple locations where Weaver might store steps
            raw_steps = (
                weaver_spec.get("steps") or
                weaver_spec.get("metadata", {}).get("steps") or
                weaver_spec.get("execution_steps") or
                weaver_spec.get("procedure") or
                []
            )
            steps = coerce_step_items(raw_steps)
            if steps:
                logger.info("[spec_gate_v2] Got %d steps from Weaver spec", len(steps))
        
        acceptance = verify_user or coerce_acceptance_items(
            ((weaver_spec or {}).get("acceptance_criteria") or (weaver_spec or {}).get("acceptance"))
            if isinstance(weaver_spec, dict) else None
        )
        
        # Coerce outputs
        outputs = outputs_user or coerce_output_items(raw_outputs, content_verbatim=cv, location=loc)
        
        # Fallback: extract from acceptance
        if not outputs and acceptance:
            outputs = extract_outputs_from_acceptance(acceptance, cv, loc)
        
        # Fallback: synthesize from content_verbatim + location
        if not outputs and cv and loc:
            filename = extract_filename_from_text(loc) or "output.txt"
            outputs = [{
                "type": "file",
                "name": filename,
                "path": loc,
                "content": cv,
                "action": "add",
                "must_exist": False,
                "description": f"File containing: {cv[:50]}{'...' if len(cv) > 50 else ''}",
            }]
            logger.info("[spec_gate_v2] Synthesized output: %s", filename)

        # v2.2: AUTO-GENERATE STEPS if missing - NEVER ask user
        # The system figures out HOW, user specifies WHAT
        if not steps:
            # Try external step_generator first (more sophisticated)
            if STEP_GENERATOR_AVAILABLE and generate_steps_from_task:
                steps = generate_steps_from_task(
                    objective=objective,
                    outputs=outputs,
                    content_verbatim=cv,
                    location=loc,
                )
            
            # Fall back to built-in generator
            if not steps:
                steps = _auto_generate_steps(objective, outputs, cv, loc)
            
            if steps:
                logger.info("[spec_gate_v2] Auto-generated %d steps (user not asked)", len(steps))

        # Blocking validation
        is_valid, blocking_issues, blocking_questions = validate_blocking_fields(outputs, steps, acceptance)
        
        # Legacy questions
        open_questions = derive_required_questions(outputs, steps, acceptance)
        if not is_valid:
            open_questions = blocking_questions

        # Generate IDs
        spec_id = str(uuid.uuid4())

        # Build context
        context: Dict[str, Any] = {
            "job_id": job_id,
            "provider_id": provider_id,
            "model_id": model_id,
            "weaver_provenance": weaver_prov,
            "latest_user_clarification": user_text,
            "content_verbatim": weaver_metadata.get("content_verbatim"),
            "location": weaver_metadata.get("location"),
            "scope_constraints": weaver_metadata.get("scope_constraints", []),
        }
        if weaver_spec:
            context["weaver_pot"] = weaver_spec

        # Determine finalization
        if round_n >= 3:
            finalized = True
            validation_status = "validated" if is_valid else "validated_with_issues"
        elif is_valid:
            finalized = True
            validation_status = "validated"
        else:
            finalized = False
            validation_status = "needs_clarification"

        # Build open issues for finalized specs
        open_issues: List[str] = []
        if finalized:
            if not outputs:
                open_issues.append("Outputs not fully specified.")
            if not steps:
                open_issues.append("Steps not fully specified.")
            if not acceptance:
                open_issues.append("Acceptance criteria not fully specified.")

        # Build spec payload
        summary = safe_summary_from_objective(objective)
        spec_payload: Dict[str, Any] = {
            "spec_id": spec_id,
            "spec_version": round_n,
            "title": title,
            "summary": summary,
            "objective": objective,
            "outputs": outputs,
            "steps": steps,
            "acceptance_criteria": acceptance,
            "open_issues": open_issues if finalized else [],
            "blocking_issues": blocking_issues,
            "validation_status": validation_status,
            "context": context,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "generator_model": f"{provider_id}/{model_id}",
            "content_verbatim": weaver_metadata.get("content_verbatim"),
            "location": weaver_metadata.get("location"),
            "scope_constraints": weaver_metadata.get("scope_constraints", []),
        }

        spec_hash = compute_spec_hash(spec_payload)
        spec_payload["spec_hash"] = spec_hash

        # Build markdown
        spot_md = build_spot_markdown(
            title=title,
            objective=objective,
            outputs=outputs,
            steps=steps,
            acceptance=acceptance,
            open_issues=open_issues if finalized else [],
            spec_id=spec_id,
            spec_hash=spec_hash,
            spec_version=round_n,
            blocking_issues=blocking_issues if not finalized else [],
        )

        # Write artifacts
        wrote_ok, write_err = write_spec_artifacts(
            job_id=job_id,
            spec_version=round_n,
            spec_payload=spec_payload,
            spot_markdown=spot_md,
            spec_hash=spec_hash,
        )

        # Build schema
        spec_schema = build_spec_schema(
            spec_id=spec_id,
            title=title,
            summary=summary,
            objective=objective,
            outputs=outputs,
            steps=steps,
            acceptance=acceptance,
            context=context,
            job_id=job_id,
            provider_id=provider_id,
            model_id=model_id,
        )

        # DB persistence
        db_ok, db_spec_id, db_spec_hash, persist_err = persist_spec(
            db=db,
            project_id=project_id,
            spec_schema=spec_schema,
            provider_id=provider_id,
            model_id=model_id,
        )

        if db_spec_id:
            spec_id = db_spec_id
        if db_spec_hash:
            spec_hash = db_spec_hash

        # Update status
        status_err = None
        if finalized and db_ok and spec_id:
            _, status_err = update_spec_status(
                db=db,
                spec_id=spec_id,
                provider_id=provider_id,
                model_id=model_id,
            )

        # Collect notes
        notes: List[str] = []
        if write_err:
            notes.append(f"artifact_write_error: {write_err}")
        if persist_err:
            notes.append(f"db_persist_error: {persist_err}")
        if status_err:
            notes.append(f"status_update_error: {status_err}")
        if blocking_issues:
            notes.append(f"blocking_validation: {len(blocking_issues)} issues")

        ready_for_pipeline = bool(finalized and (wrote_ok or db_ok) and spec_id and spec_hash)

        logger.info(
            "[spec_gate_v2] Result: ready=%s, valid=%s, blocking=%d, round=%d",
            ready_for_pipeline, is_valid, len(blocking_issues), round_n
        )

        return SpecGateResult(
            ready_for_pipeline=ready_for_pipeline,
            open_questions=open_questions if (open_questions and not finalized) else [],
            spot_markdown=spot_md if finalized else None,
            db_persisted=bool(db_ok and spec_id),
            spec_id=spec_id,
            spec_hash=spec_hash,
            spec_version=round_n,
            notes=("; ".join(notes) if notes else None),
            blocking_issues=blocking_issues,
            validation_status=validation_status,
        )

    except Exception as e:
        logger.exception("[spec_gate_v2] HARD STOP: %s", e)
        return SpecGateResult(
            ready_for_pipeline=False,
            hard_stopped=True,
            hard_stop_reason=str(e),
            spec_version=int(spec_version) if isinstance(spec_version, int) else None,
            validation_status="error",
        )


__all__ = ["run_spec_gate_v2", "SpecGateResult"]