# FILE: app/jobs/engine.py
# Purpose: Phase 4 Job Engine - Core Execution Logic
# Called-by: app.jobs.router, tests.test_job_engine
# Depends-on: app.artefacts.service, app.jobs._engine_helpers, app.jobs.models, app.jobs.schemas (+6 more)
# Last-renovated: 2026-06-11
"""
Phase 4 Job Engine - Core Execution Logic

Responsibilities:
- Convert CreateJobRequest → JobEnvelope
- Validate envelopes
- Persist job state to database
- Route to appropriate provider via registry
- Handle errors and map to ErrorType taxonomy
- Store artefacts for structured outputs
- Return JobResult with complete execution details

PHASE 4 FIXES:
- Fixed session upsert race condition using INSERT OR IGNORE pattern
- Uses SQLite-safe upsert to prevent IntegrityError on concurrent requests

v4.1 (2026-01):
- Fixed: SpecGate model selection now uses centralized runtime env lookup
- Added: Audit logging for SpecGate provider/model selection

PATCH (2025-12-26):
- Ensure all JSON columns get JSON-serializable dicts/lists:
  use model_dump(mode="json") for routing_decision_json, tool_invocations_json,
  critique_issues_json, usage_metrics_json, and error_details_json.
  This fixes: "Object of type datetime is not JSON serializable"
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.artefacts.service import ArtefactService
from app.jobs.models import Job, Session as SessionModel
from app.jobs.schemas import (
    CreateJobRequest,
    DataSensitivity,
    ErrorType,
    Importance,
    JobBudget,
    JobEnvelope,
    JobResult,
    JobState,
    JobType,
    Modality,
    ModelSelection,
    OutputContract,
    RoutingDecision,
    UsageMetrics,
    ValidationError,
    validate_job_envelope,
)
from app.jobs.stage3_locks import (
    append_stage3_ledger_event,
    parse_spec_echo_headers,
    write_stage3_artifacts,
)
from app.providers.registry import LlmCallStatus, llm_call

# Import centralized SpecGate model helpers
from app.llm.stream_utils import get_spec_gate_model, get_spec_gate_provider

logger = logging.getLogger(__name__)

# =============================================================================
# ROUTING HELPERS
# =============================================================================
# =============================================================================
# HELPERS (extracted to _engine_helpers.py)
# =============================================================================
from app.jobs._engine_helpers import (
    _determine_routing,
    _extract_user_intent,
    _build_routing_decision,
    _build_system_prompt,
    _placeholder_routing_decision,
    _get_or_create_session,
    _store_artefact,
    _request_to_envelope,
)

# =============================================================================
# JOB EXECUTION ENGINE
# =============================================================================


async def execute_job(envelope: JobEnvelope, db: Session) -> JobResult:
    job_id = envelope.job_id or str(uuid4())
    started_at = datetime.utcnow()

    provider_id, model_id, temperature = _determine_routing(envelope)
    routing_decision = _build_routing_decision(envelope, provider_id, model_id, temperature)
    routing_decision.job_id = job_id

    system_prompt = _build_system_prompt(envelope, provider_id)

    logger.info(f"[engine] Executing job {job_id}: {envelope.job_type.value} → {provider_id}/{model_id}")

    try:
        result = await llm_call(
            provider_id=provider_id,
            model_id=model_id,
            messages=envelope.messages,
            system_prompt=system_prompt,
            job_envelope=envelope.model_dump(mode="json") if hasattr(envelope, "model_dump") else None,
            temperature=temperature,
            max_tokens=envelope.budget.max_tokens,
            timeout_seconds=envelope.budget.max_wall_time_seconds,
        )
    except Exception as exc:
        completed_at = datetime.utcnow()
        duration = (completed_at - started_at).total_seconds()
        logger.exception("[engine] LLM call crashed for job %s: %s", job_id, exc)

        return JobResult(
            job_id=job_id,
            session_id=envelope.session_id,
            project_id=envelope.project_id,
            job_type=envelope.job_type.value,
            state=JobState.FAILED,
            content="",
            output_contract=envelope.output_contract,
            artefact_id=None,
            routing_decision=routing_decision,
            tools_used=[],
            was_reviewed=False,
            critique_issues=[],
            unresolved_blockers=0,
            usage_metrics=[],
            total_cost_estimate=0.0,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            error_type=ErrorType.INTERNAL_ERROR,
            error_message=str(exc),
            error_details={"exception_type": type(exc).__name__},
        )

    completed_at = datetime.utcnow()
    duration = (completed_at - started_at).total_seconds()

    if result.is_success():
        content = result.content or ""

        artefact_id: Optional[str] = None
        if envelope.output_contract in (OutputContract.ARCHITECTURE_DOC, OutputContract.CODE_PATCH_PROPOSAL):
            artefact_id = await _store_artefact(db=db, envelope=envelope, content=content, job_id=job_id)

        usage_metrics = [
            UsageMetrics(
                model_id=model_id,
                provider=provider_id,
                prompt_tokens=getattr(result.usage, "prompt_tokens", 0),
                completion_tokens=getattr(result.usage, "completion_tokens", 0),
                total_tokens=getattr(result.usage, "total_tokens", 0),
                cost_estimate=getattr(result.usage, "cost_estimate", 0.0),
            )
        ]

        return JobResult(
            job_id=job_id,
            session_id=envelope.session_id,
            project_id=envelope.project_id,
            job_type=envelope.job_type.value,
            state=JobState.SUCCEEDED,
            content=content,
            output_contract=envelope.output_contract,
            artefact_id=artefact_id,
            routing_decision=routing_decision,
            tools_used=[],
            was_reviewed=False,
            critique_issues=[],
            unresolved_blockers=0,
            usage_metrics=usage_metrics,
            total_cost_estimate=float(getattr(result.usage, "cost_estimate", 0.0) or 0.0),
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
        )

    # Error case
    err_msg = result.error_message or "Unknown provider error"
    lowered = err_msg.lower()

    # Registry currently exposes generic statuses; infer timeout if text suggests it.
    if ("timeout" in lowered) or ("timed out" in lowered):
        error_type = ErrorType.TIMEOUT
    elif result.status in (LlmCallStatus.PROVIDER_UNAVAILABLE, LlmCallStatus.INVALID_REQUEST):
        error_type = ErrorType.MODEL_ERROR
    else:
        error_type = ErrorType.MODEL_ERROR

    return JobResult(
        job_id=job_id,
        session_id=envelope.session_id,
        project_id=envelope.project_id,
        job_type=envelope.job_type.value,
        state=JobState.FAILED,
        content="",
        output_contract=envelope.output_contract,
        artefact_id=None,
        routing_decision=routing_decision,
        tools_used=[],
        was_reviewed=False,
        critique_issues=[],
        unresolved_blockers=0,
        usage_metrics=[],
        total_cost_estimate=0.0,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=duration,
        error_type=error_type,
        error_message=err_msg,
        error_details={"status": getattr(result.status, "value", str(result.status))},
    )


async def _store_artefact(db: Session, envelope: JobEnvelope, content: str, job_id: str) -> Optional[str]:
    try:
        if envelope.output_contract == OutputContract.ARCHITECTURE_DOC:
            artefact_type = "architecture_doc"
            name = (envelope.metadata or {}).get(
                "artefact_name",
                f"Architecture - {datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            )
        elif envelope.output_contract == OutputContract.CODE_PATCH_PROPOSAL:
            artefact_type = "code_patch_proposal"
            name = (envelope.metadata or {}).get(
                "artefact_name",
                f"Code Patch - {datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            )
        else:
            return None

        artefact = ArtefactService.write_artefact(
            db=db,
            project_id=envelope.project_id,
            artefact_type=artefact_type,
            name=name,
            content=content,
            metadata={
                "job_id": job_id,
                "job_type": envelope.job_type.value,
                "created_at": datetime.utcnow().isoformat(),
            },
            created_by_job_id=job_id,
        )

        logger.info(f"[engine] Stored artefact {artefact.id} for job {job_id}")
        return artefact.id
    except Exception as exc:
        logger.error(f"[engine] Failed to store artefact for job {job_id}: {exc}")
        return None


async def create_and_run_job(request: CreateJobRequest, db: Session) -> JobResult:
    job_id = str(uuid4())

    try:
        session_id = request.session_id or str(uuid4())
        session = _get_or_create_session(db, session_id, request.project_id)

        envelope = _request_to_envelope(request, job_id, session_id)

        # Validate envelope
        try:
            validate_job_envelope(envelope)
        except ValidationError as e:
            logger.warning(f"[engine] Job validation failed: {e.errors}")

            job = Job(
                id=job_id,
                session_id=session_id,
                project_id=request.project_id,
                job_spec_version=1,
                job_type=envelope.job_type.value,
                resolved_job_type=envelope.job_type.value,
                importance=envelope.importance.value,
                data_sensitivity=envelope.data_sensitivity.value,
                state=JobState.FAILED.value,
                triggered_by="user",
                envelope_json=envelope.model_dump(mode="json"),
                error_type=ErrorType.VALIDATION_ERROR.value,
                error_message=f"Validation failed: {'; '.join(e.errors)}",
                created_at=datetime.utcnow(),
            )
            db.add(job)
            db.commit()

            return JobResult(
                job_id=job_id,
                session_id=session_id,
                project_id=request.project_id,
                job_type=envelope.job_type.value,
                state=JobState.FAILED,
                content="",
                output_contract=envelope.output_contract,
                artefact_id=None,
                routing_decision=_placeholder_routing_decision(envelope),
                tools_used=[],
                was_reviewed=False,
                critique_issues=[],
                unresolved_blockers=0,
                usage_metrics=[],
                total_cost_estimate=0.0,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                duration_seconds=0.0,
                error_type=ErrorType.VALIDATION_ERROR,
                error_message=f"Validation failed: {'; '.join(e.errors)}",
                error_details={"errors": list(e.errors)},
            )

        # Create pending job entry
        job = Job(
            id=job_id,
            session_id=session_id,
            project_id=request.project_id,
            job_spec_version=1,
            job_type=envelope.job_type.value,
            resolved_job_type=envelope.job_type.value,
            importance=envelope.importance.value,
            data_sensitivity=envelope.data_sensitivity.value,
            state=JobState.PENDING.value,
            triggered_by="user",
            envelope_json=envelope.model_dump(mode="json"),
            created_at=datetime.utcnow(),
        )
        db.add(job)
        db.commit()

        logger.info(f"[engine] Created job {job_id} in state PENDING")

        # Transition to RUNNING
        job.state = JobState.RUNNING.value
        job.started_at = datetime.utcnow()
        db.commit()

        logger.info(f"[engine] Job {job_id} transitioned to RUNNING")

        # ---------------------------------------------------------------------
        # Spec Gate + Stage 3 locks variables
        # ---------------------------------------------------------------------
        spec_id: Optional[str] = None
        spec_hash: Optional[str] = None

        # Spec Gate for architecture jobs (PoT Spec)
        if envelope.job_type in (JobType.APP_ARCHITECTURE, JobType.CODE_REPO):
            from app.pot_spec.spec_gate import detect_user_questions, run_spec_gate

            # FIXED: Use centralized runtime env lookup instead of direct os.getenv with hard-coded fallback
            spec_gate_provider = get_spec_gate_provider()
            spec_gate_model = get_spec_gate_model()
            
            # AUDIT: Log the resolved model
            logger.info(f"[engine] AUDIT: SpecGate using provider={spec_gate_provider}, model={spec_gate_model}")
            
            user_intent = _extract_user_intent(request.messages)

            spec_id, spec_hash, open_questions = await run_spec_gate(
                db,
                job_id=job_id,
                user_intent=user_intent,
                provider_id=spec_gate_provider,
                model_id=spec_gate_model,
                repo_snapshot=request.metadata if isinstance(request.metadata, dict) else None,
                constraints_hint={"stability_accuracy": "high", "allowed_tools": "free_only"},
            )

            if open_questions:
                paused_payload = {
                    "pause_state": JobState.NEEDS_SPEC_CLARIFICATION.value,
                    "job_id": job_id,
                    "spec_id": spec_id,
                    "spec_hash": spec_hash,
                    "open_questions": open_questions,
                    "artifacts_written": f"jobs/{job_id}/spec/ + jobs/{job_id}/ledger/",
                }

                job.state = JobState.NEEDS_SPEC_CLARIFICATION.value
                job.output_content = json.dumps(paused_payload, ensure_ascii=False, indent=2)
                job.completed_at = datetime.utcnow()
                db.commit()

                return JobResult(
                    job_id=job_id,
                    session_id=envelope.session_id,
                    project_id=envelope.project_id,
                    job_type=envelope.job_type.value,
                    state=JobState.NEEDS_SPEC_CLARIFICATION,
                    content=job.output_content,
                    output_contract=envelope.output_contract,
                    artefact_id=None,
                    routing_decision=_placeholder_routing_decision(envelope),
                    tools_used=[],
                    was_reviewed=False,
                    critique_issues=[],
                    unresolved_blockers=len(open_questions),
                    usage_metrics=[],
                    total_cost_estimate=0.0,
                    started_at=job.started_at or datetime.utcnow(),
                    completed_at=job.completed_at or datetime.utcnow(),
                    duration_seconds=0.0,
                    error_type=None,
                    error_message=None,
                )

            # Inject spec echo requirement for downstream stages
            envelope.messages.insert(
                0,
                {
                    "role": "system",
                    "content": (
                        "You MUST echo these identifiers at the TOP of your response exactly:\n"
                        f"SPEC_ID: {spec_id}\nSPEC_HASH: {spec_hash}\n"
                        "Do not ask the user any questions."
                    ),
                },
            )

            # Stage 3: ledger STAGE_STARTED for spec-hash-locked jobs
            stage_name = (
                "architecture_job_execution"
                if envelope.job_type == JobType.APP_ARCHITECTURE
                else "code_repo_job_execution"
            )
            append_stage3_ledger_event(
                job_id,
                {
                    "event": "STAGE_STARTED",
                    "job_id": job_id,
                    "stage": stage_name,
                    "spec_id": spec_id,
                    "spec_hash": spec_hash,
                    "spec_gate_provider": spec_gate_provider,
                    "spec_gate_model": spec_gate_model,
                    "ts_utc": datetime.utcnow().isoformat() + "Z",
                },
            )

        # ---------------------------------------------------------------------
        # Execute job
        # ---------------------------------------------------------------------
        result = await execute_job(envelope, db)

        # ---------------------------------------------------------------------
        # Stage 3: Post-execution spec hash verification (if applicable)
        # ---------------------------------------------------------------------
        if spec_id and spec_hash:
            parsed_id, parsed_hash = parse_spec_echo_headers(result.content or "")

            if parsed_hash and parsed_hash != spec_hash:
                # Spec hash mismatch - verification failed
                append_stage3_ledger_event(
                    job_id,
                    {
                        "event": "STAGE_SPEC_HASH_MISMATCH",
                        "job_id": job_id,
                        "expected_spec_hash": spec_hash,
                        "actual_spec_hash": parsed_hash,
                        "ts_utc": datetime.utcnow().isoformat() + "Z",
                    },
                )

                # Store verification failure details
                parse_note = f"Expected spec_hash={spec_hash}, got {parsed_hash}"
                job.error_type = ErrorType.MODEL_ERROR.value
                job.error_message = f"Spec hash verification failed: {parse_note}"
                job.state = JobState.FAILED.value
                job.completed_at = datetime.utcnow()
                db.commit()

                return JobResult(
                    job_id=job_id,
                    session_id=envelope.session_id,
                    project_id=envelope.project_id,
                    job_type=envelope.job_type.value,
                    state=JobState.FAILED,
                    content=job.output_content or "",
                    output_contract=envelope.output_contract,
                    artefact_id=result.artefact_id,
                    routing_decision=result.routing_decision,
                    tools_used=result.tools_used,
                    was_reviewed=result.was_reviewed,
                    critique_issues=result.critique_issues,
                    unresolved_blockers=result.unresolved_blockers,
                    usage_metrics=result.usage_metrics,
                    total_cost_estimate=result.total_cost_estimate,
                    started_at=result.started_at,
                    completed_at=job.completed_at,
                    duration_seconds=result.duration_seconds,
                    error_type=ErrorType.MODEL_ERROR,
                    error_message=job.error_message,
                    error_details={"parse_note": parse_note},
                )

            # Hard enforcement: downstream stages must not ask user questions
            if detect_user_questions(result.content or ""):
                from app.pot_spec.ledger import append_event
                from app.pot_spec.service import get_job_artifact_root

                job_root = get_job_artifact_root()
                append_event(
                    job_artifact_root=job_root,
                    job_id=job_id,
                    event={
                        "event": "POLICY_VIOLATION_STAGE_ASKED_QUESTIONS",
                        "job_id": job_id,
                        "stage": "JOB_EXECUTION",
                        "status": "rejected",
                    },
                )

                # FIXED: Use centralized runtime env lookup instead of direct os.getenv with hard-coded fallback
                spec_gate_provider = get_spec_gate_provider()
                spec_gate_model = get_spec_gate_model()
                
                # AUDIT: Log the resolved model
                logger.info(f"[engine] AUDIT: SpecGate reroute using provider={spec_gate_provider}, model={spec_gate_model}")
                
                user_intent = _extract_user_intent(request.messages)

                spec_id2, spec_hash2, open_questions2 = await run_spec_gate(
                    db,
                    job_id=job_id,
                    user_intent=user_intent,
                    provider_id=spec_gate_provider,
                    model_id=spec_gate_model,
                    reroute_reason="Downstream stage asked the user questions. Only Spec Gate may ask questions.",
                    downstream_output_excerpt=(result.content or "")[:2000],
                )

                paused_payload = {
                    "pause_state": JobState.NEEDS_SPEC_CLARIFICATION.value,
                    "job_id": job_id,
                    "spec_id": spec_id2,
                    "spec_hash": spec_hash2,
                    "open_questions": open_questions2,
                    "artifacts_written": f"jobs/{job_id}/spec/ + jobs/{job_id}/ledger/",
                }

                job.state = JobState.NEEDS_SPEC_CLARIFICATION.value
                job.output_content = json.dumps(paused_payload, ensure_ascii=False, indent=2)
                job.completed_at = datetime.utcnow()
                db.commit()

                return JobResult(
                    job_id=job_id,
                    session_id=envelope.session_id,
                    project_id=envelope.project_id,
                    job_type=envelope.job_type.value,
                    state=JobState.NEEDS_SPEC_CLARIFICATION,
                    content=job.output_content,
                    output_contract=envelope.output_contract,
                    artefact_id=None,
                    routing_decision=_placeholder_routing_decision(envelope),
                    tools_used=[],
                    was_reviewed=False,
                    critique_issues=[],
                    unresolved_blockers=len(open_questions2),
                    usage_metrics=[],
                    total_cost_estimate=result.total_cost_estimate,
                    started_at=result.started_at,
                    completed_at=datetime.utcnow(),
                    duration_seconds=result.duration_seconds,
                    error_type=None,
                    error_message=None,
                )

        # ---------------------------------------------------------------------
        # Update DB job record with result (JSON-SAFE)
        # ---------------------------------------------------------------------
        job.state = result.state.value
        job.output_content = result.content
        job.output_contract = result.output_contract.value
        job.artefact_id = result.artefact_id

        # FIX: JSON-safe dumps
        job.routing_decision_json = result.routing_decision.model_dump(mode="json")
        job.tool_invocations_json = [t.model_dump(mode="json") for t in (result.tools_used or [])]
        job.critique_issues_json = [c.model_dump(mode="json") for c in (result.critique_issues or [])]
        job.was_reviewed = bool(result.was_reviewed)
        job.unresolved_blockers = int(result.unresolved_blockers or 0)
        job.usage_metrics_json = [u.model_dump(mode="json") for u in (result.usage_metrics or [])]

        job.total_tokens = sum(int(u.total_tokens or 0) for u in (result.usage_metrics or []))
        job.total_cost_estimate = float(result.total_cost_estimate or 0.0)
        job.completed_at = result.completed_at
        job.duration_seconds = float(result.duration_seconds or 0.0)

        if result.error_type:
            job.error_type = result.error_type.value
            job.error_message = result.error_message

            # FIX: error_details must be JSON-serializable
            if result.error_details is None:
                job.error_details_json = None
            elif hasattr(result.error_details, "model_dump"):
                job.error_details_json = result.error_details.model_dump(mode="json")
            else:
                job.error_details_json = result.error_details

        db.commit()

        # Update session stats
        try:
            session.job_count += 1
            session.last_activity = datetime.utcnow()
            session.total_cost_estimate += float(result.total_cost_estimate or 0.0)
            db.commit()
        except Exception:
            # Don't fail the job return just because session stats couldn't update
            db.rollback()
            logger.warning("[engine] Failed updating session stats for session_id=%s", session_id, exc_info=True)

        logger.info(
            "[engine] Job %s completed: %s (cost=$%.4f)",
            job_id,
            result.state.value,
            float(result.total_cost_estimate or 0.0),
        )

        return result

    except Exception as e:
        logger.exception(f"[engine] Unexpected error creating/running job {job_id}")

        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if job:
                job.state = JobState.FAILED.value
                job.error_type = ErrorType.INTERNAL_ERROR.value
                job.error_message = str(e)
                job.completed_at = datetime.utcnow()
                db.commit()
        except Exception:
            pass

        raise


