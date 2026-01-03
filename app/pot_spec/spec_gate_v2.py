# FILE: app/pot_spec/spec_gate_v2.py
"""
Spec Gate v2 - Wraps spec_gate with clarification state management.

CRITICAL FIX (for Overwatcher):
- When persisting a validated spec to DB, persist the *raw spec JSON* (spec_payload)
  into the DB record (spec_json/content). Otherwise Overwatcher sees an empty shell
  and cannot find deliverables / target file.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# Optional Imports (keep module resilient)
# =============================================================================

try:
    from app.pot_spec.spec_gate import run_spec_gate, _artifact_root, _append_event, _utc_ts
    _SPEC_GATE_AVAILABLE = True
except ImportError:
    _SPEC_GATE_AVAILABLE = False
    run_spec_gate = None

    def _artifact_root():
        return os.path.abspath(os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"))

    def _append_event(*args, **kwargs):
        pass

    def _utc_ts():
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

try:
    from app.pot_spec.clarification_state import (
        ClarificationState,
        ClarificationDecision,
        load_clarification_state,
        save_clarification_state,
        format_spec_summary_markdown,
        compute_question_signature_simple,
    )
    _CLARIFICATION_AVAILABLE = True
except ImportError:
    _CLARIFICATION_AVAILABLE = False
    logger.warning("[spec_gate_v2] clarification_state module not available")

try:
    from app.specs.service import create_spec, update_spec_status
    from app.specs.schema import Spec as SpecSchema, SpecStatus
    _SPECS_SERVICE_AVAILABLE = True
    logger.info("[spec_gate_v2] specs.service module loaded successfully")
except ImportError as e:
    _SPECS_SERVICE_AVAILABLE = False
    logger.warning(f"[spec_gate_v2] specs.service not available - validated specs won't persist to DB: {e}")


@dataclass
class SpecGateResult:
    spec_id: str
    spec_hash: str
    open_questions: List[str]
    spec_version: int
    ready_for_pipeline: bool
    hard_stopped: bool = False
    hard_stop_reason: Optional[str] = None
    clarification_round: int = 0
    spec_summary_markdown: Optional[str] = None
    spec_file_path: Optional[str] = None
    db_persisted: bool = False

    def __iter__(self):
        return iter([self.spec_id, self.spec_hash, self.open_questions])

    def __getitem__(self, idx):
        return [self.spec_id, self.spec_hash, self.open_questions][idx]


def _get_artifact_root() -> str:
    if _SPEC_GATE_AVAILABLE:
        return _artifact_root()
    return os.path.abspath(os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"))


def _get_spec_file_path(job_root: str, job_id: str, spec_version: int) -> str:
    # spec_gate.py currently hardcodes v1
    return os.path.join(job_root, "jobs", job_id, "spec", "spec_v1.json")


def _filter_questions_with_clarification(
    questions: List[str],
    job_artifact_root: str,
    job_id: str,
    spec_version: int,
) -> Tuple[List[str], "ClarificationDecision", Optional["ClarificationState"]]:
    if not _CLARIFICATION_AVAILABLE:
        class _StubDecision:
            CONTINUE = "continue"
            READY_FOR_CONFIRM = "ready"
        if questions:
            return questions, _StubDecision.CONTINUE, None
        return [], _StubDecision.READY_FOR_CONFIRM, None

    state = load_clarification_state(job_artifact_root, job_id)
    decision = state.record_questions(questions, spec_version)
    save_clarification_state(job_artifact_root, job_id, state)

    if decision == ClarificationDecision.HARD_STOP:
        return [], decision, state

    if decision == ClarificationDecision.ALREADY_ASKED:
        return [], ClarificationDecision.READY_FOR_CONFIRM, state

    if decision == ClarificationDecision.READY_FOR_CONFIRM:
        return [], decision, state

    if state.rounds:
        return state.rounds[-1].questions_asked, decision, state

    return questions, decision, state


def _persist_validated_spec_to_db(
    db: Any,
    project_id: int,
    spec_id: str,
    spec_hash: str,
    spec_version: int,
    spec_file_path: str,
    job_id: str,
    job_root: str,
) -> bool:
    """
    Persist validated spec to DB AND store raw JSON in DB record
    so Overwatcher can parse deliverables/targets.
    """
    if not _SPECS_SERVICE_AVAILABLE:
        logger.warning("[spec_gate_v2] Cannot persist spec to DB - specs.service not available")
        return False

    if not project_id:
        logger.error("[spec_gate_v2] Cannot persist spec to DB - project_id not provided")
        return False

    try:
        if not os.path.exists(spec_file_path):
            logger.error(f"[spec_gate_v2] Spec file not found: {spec_file_path}")
            return False

        with open(spec_file_path, "r", encoding="utf-8") as f:
            spec_payload = json.load(f)

        # Robust title/goal extraction (handles drift)
        goal = (spec_payload.get("goal") or "").strip()
        title = (goal[:200] if goal else (spec_payload.get("title") or "").strip()[:200]) or f"Spec {spec_id[:8]}"
        created_by_model = spec_payload.get("created_by_model") or "spec_gate_v2"

        # Optional markdown summary
        content_markdown = None
        reqs = spec_payload.get("requirements") or {}
        if isinstance(reqs, dict) and any(reqs.get(k) for k in ("must", "should", "can")):
            md_parts = [f"# {title}", "", "## Goal", goal or "(not provided)", ""]
            if reqs.get("must"):
                md_parts.append("## Must Have")
                for r in reqs["must"]:
                    md_parts.append(f"- {r}")
                md_parts.append("")
            if reqs.get("should"):
                md_parts.append("## Should Have")
                for r in reqs["should"]:
                    md_parts.append(f"- {r}")
                md_parts.append("")
            if reqs.get("can"):
                md_parts.append("## Nice to Have")
                for r in reqs["can"]:
                    md_parts.append(f"- {r}")
                md_parts.append("")
            # Include deliverables if present (Overwatcher-critical)
            dels = spec_payload.get("deliverables")
            if isinstance(dels, list) and dels:
                md_parts.append("## Deliverables")
                for d in dels:
                    if isinstance(d, dict):
                        md_parts.append(f"- {d.get('action')} {d.get('target')} {d.get('filename')}")
                md_parts.append("")
            content_markdown = "\n".join(md_parts)

        raw_json_text = json.dumps(spec_payload, ensure_ascii=False, indent=2)

        # Create SpecSchema (minimal required fields)
        spec_schema = SpecSchema(
            spec_id=spec_id,
            spec_version=str(spec_version),
            title=title,
            summary=(goal[:500] if goal else (spec_payload.get("summary") or "")[:500]),
            objective=(goal or spec_payload.get("objective") or ""),
        )

        # Best-effort attach raw content so DB record carries the real spec
        for attr, val in (
            ("spec_json", spec_payload),
            ("content", raw_json_text),
            ("content_markdown", content_markdown),
        ):
            try:
                setattr(spec_schema, attr, val)
            except Exception:
                pass

        # Provenance (best-effort, don’t explode if schema differs)
        try:
            from app.specs.schema import SpecProvenance
            spec_schema.provenance = SpecProvenance(
                job_id=job_id,
                generator_model=created_by_model,
                created_at=spec_payload.get("created_at", datetime.now(timezone.utc).isoformat()),
            )
        except Exception:
            pass

        db_spec = create_spec(
            db=db,
            project_id=project_id,
            spec_schema=spec_schema,
            generator_model=created_by_model,
        )

        # Ensure DB row also has raw JSON (covers cases where create_spec ignores spec_schema extras)
        try:
            if hasattr(db_spec, "spec_json"):
                db_spec.spec_json = spec_payload
            if hasattr(db_spec, "content"):
                db_spec.content = raw_json_text
            if content_markdown and hasattr(db_spec, "content_markdown"):
                db_spec.content_markdown = content_markdown
            db.add(db_spec)
            db.commit()
        except Exception:
            # Don’t hard-fail persistence if this extra write fails; status still updated below.
            logger.exception("[spec_gate_v2] Failed to attach raw spec payload to DB row (non-fatal)")

        update_spec_status(
            db=db,
            spec_id=spec_id,
            new_status=SpecStatus.VALIDATED.value,
            validation_result={
                "valid": True,
                "validated_at": _utc_ts(),
                "spec_hash": spec_hash,
                "spec_file": spec_file_path,
                "has_deliverables": bool(spec_payload.get("deliverables")),
            },
            triggered_by="spec_gate_v2",
        )

        logger.info(f"[spec_gate_v2] Spec {spec_id} persisted to DB with status=validated (raw JSON stored)")
        return True

    except Exception as e:
        logger.exception(f"[spec_gate_v2] Failed to persist spec to DB: {e}")
        return False


async def run_spec_gate_v2(
    db: Any,
    job_id: str,
    user_intent: str,
    provider_id: str,
    model_id: str,
    project_id: Optional[int] = None,
    repo_snapshot: Optional[dict] = None,
    constraints_hint: Optional[dict] = None,
    spec_version: int = 1,
    **kwargs,
) -> SpecGateResult:
    if not _SPEC_GATE_AVAILABLE:
        raise RuntimeError("spec_gate module not available")

    if project_id is None:
        logger.warning("[spec_gate_v2] project_id not provided - spec won't persist to DB for restart survival")

    job_root = _get_artifact_root()

    effective_intent = user_intent
    if spec_version >= 3:
        round3_directive = """
[FINAL ROUND - MANDATORY SPEC OUTPUT]
This is clarification round 3 of 3. You MUST NOT ask any more questions.
You MUST output the complete, final spec based on all information gathered so far.
If any details are missing, make reasonable assumptions and note them in the spec.
DO NOT return questions. Return the complete spec JSON only.

User context and answers:
"""
        effective_intent = round3_directive + user_intent
        logger.info("[spec_gate_v2] Round 3: Forcing spec output (no questions)")

    spec_id, spec_hash, raw_questions = await run_spec_gate(
        db,
        job_id,
        effective_intent,
        provider_id,
        model_id,
        repo_snapshot=repo_snapshot,
        constraints_hint=constraints_hint,
        **kwargs,
    )

    filtered_questions, decision, clarification_state = _filter_questions_with_clarification(
        questions=raw_questions,
        job_artifact_root=job_root,
        job_id=job_id,
        spec_version=spec_version,
    )

    if spec_version >= 3 and filtered_questions:
        logger.warning("[spec_gate_v2] Round 3 returned questions - forcing ready_for_pipeline")
        decision = ClarificationDecision.READY_FOR_CONFIRM if _CLARIFICATION_AVAILABLE else "ready"
        filtered_questions = []

    ready_for_pipeline = (decision == ClarificationDecision.READY_FOR_CONFIRM) if _CLARIFICATION_AVAILABLE else (decision == "ready")
    hard_stopped = (decision == ClarificationDecision.HARD_STOP) if _CLARIFICATION_AVAILABLE else False
    hard_stop_reason = clarification_state.hard_stop_reason if (hard_stopped and clarification_state) else None

    spec_file_path = _get_spec_file_path(job_root, job_id, spec_version)

    db_persisted = False
    if ready_for_pipeline and project_id:
        db_persisted = _persist_validated_spec_to_db(
            db=db,
            project_id=project_id,
            spec_id=spec_id,
            spec_hash=spec_hash,
            spec_version=spec_version,
            spec_file_path=spec_file_path,
            job_id=job_id,
            job_root=job_root,
        )

    spec_summary_markdown = None
    if ready_for_pipeline and _CLARIFICATION_AVAILABLE:
        try:
            with open(spec_file_path, "r", encoding="utf-8") as f:
                spec_payload = json.load(f)
            spec_payload["spec_hash"] = spec_hash
            spec_summary_markdown = format_spec_summary_markdown(
                spec_payload=spec_payload,
                spec_file_path=spec_file_path,
                clarification_state=clarification_state,
            )
        except Exception as e:
            logger.warning(f"[spec_gate_v2] Failed to build summary: {e}")

    if filtered_questions:
        _append_event(job_root, job_id, {
            "event": "SPEC_QUESTIONS_FILTERED",
            "job_id": job_id,
            "spec_id": spec_id,
            "raw_count": len(raw_questions),
            "filtered_count": len(filtered_questions),
            "clarification_round": clarification_state.current_round if clarification_state else 0,
            "status": "ok",
            "ts": _utc_ts(),
        })

    if ready_for_pipeline:
        _append_event(job_root, job_id, {
            "event": "SPEC_READY_FOR_PIPELINE",
            "job_id": job_id,
            "spec_id": spec_id,
            "spec_version": spec_version,
            "awaiting_confirmation": True,
            "db_persisted": db_persisted,
            "status": "ok",
            "ts": _utc_ts(),
        })

    if hard_stopped:
        _append_event(job_root, job_id, {
            "event": "SPEC_GATE_HARD_STOP",
            "job_id": job_id,
            "reason": hard_stop_reason,
            "status": "hard_stop",
            "ts": _utc_ts(),
        })

    return SpecGateResult(
        spec_id=spec_id,
        spec_hash=spec_hash,
        open_questions=filtered_questions,
        spec_version=spec_version,
        ready_for_pipeline=ready_for_pipeline,
        hard_stopped=hard_stopped,
        hard_stop_reason=hard_stop_reason,
        clarification_round=clarification_state.current_round if clarification_state else 0,
        spec_summary_markdown=spec_summary_markdown,
        spec_file_path=spec_file_path,
        db_persisted=db_persisted,
    )


async def confirm_spec_for_pipeline(
    job_id: str,
    spec_version: int,
    spec_hash: str,
    user_confirmed: bool = False,
) -> Tuple[bool, str]:
    if not _CLARIFICATION_AVAILABLE:
        return user_confirmed, "Clarification module not available"

    job_root = _get_artifact_root()
    state = load_clarification_state(job_root, job_id)

    if not state.ready_for_pipeline:
        return False, "Spec not ready (has open questions)"

    if state.hard_stopped:
        return False, f"Hard stopped: {state.hard_stop_reason}"

    if not user_confirmed:
        return False, "Awaiting user confirmation"

    confirmed = state.confirm_for_pipeline(spec_version=spec_version, spec_hash=spec_hash)

    if confirmed:
        save_clarification_state(job_root, job_id, state)
        _append_event(job_root, job_id, {
            "event": "SPEC_CONFIRMED_FOR_PIPELINE",
            "job_id": job_id,
            "spec_version": spec_version,
            "spec_hash": spec_hash,
            "confirmed_at": state.confirmed_at,
            "status": "ok",
            "ts": _utc_ts(),
        })
        return True, "Confirmed - proceeding to critical pipeline"

    return False, "Confirmation failed"


__all__ = ["SpecGateResult", "run_spec_gate_v2", "confirm_spec_for_pipeline"]
