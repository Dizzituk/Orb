# FILE: app/jobs/engine.py
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
"""
from __future__ import annotations

import logging
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import text

from app.jobs.models import Job, Session as SessionModel
from app.jobs.schemas import (
    JobEnvelope,
    JobType,
    JobState,
    ErrorType,
    Importance,
    DataSensitivity,
    Modality,
    JobBudget,
    OutputContract,
    ValidationError,
    validate_job_envelope,
    CreateJobRequest,
    JobResult,
    RoutingDecision,
    ModelSelection,
    UsageMetrics,
)
from app.providers.registry import llm_call, LlmCallStatus
from app.artefacts.service import ArtefactService
from app.jobs.stage3_locks import (
    append_stage3_ledger_event,
    parse_spec_echo_headers,
    write_stage3_artifacts,
)

logger = logging.getLogger(__name__)

# =============================================================================
# ROUTING HELPERS
# =============================================================================

def _determine_routing(envelope: JobEnvelope) -> tuple[str, str, float]:
    """
    Determine provider, model, and temperature for a job.
    
    This is a simple routing implementation for this branch.
    Later branches will integrate with full policy routing.
    
    Returns:
        Tuple of (provider_id, model_id, temperature)
    """
    job_type = envelope.job_type
    
    # High-stakes or architecture jobs → Claude
    if job_type in (JobType.APP_ARCHITECTURE, JobType.CODE_REPO):
        return "anthropic", "claude-sonnet-4-20250514", 0.7
    
    # Code jobs → Claude
    if job_type == JobType.CODE_SMALL:
        return "anthropic", "claude-sonnet-4-20250514", 0.7
    
    # Simple chat → GPT
    if job_type == JobType.CHAT_SIMPLE:
        return "openai", "gpt-4o", 0.7
    
    # Default to GPT
    return "openai", "gpt-4o", 0.7



def _extract_user_intent(messages: list[dict]) -> str:
    for m in reversed(messages or []):
        if (m.get("role") == "user") and m.get("content"):
            return str(m.get("content")).strip()
    # Fallback: stringify full messages
    try:
        return json.dumps(messages or [], ensure_ascii=False)
    except Exception:
        return ""


def _build_routing_decision(
    envelope: JobEnvelope,
    provider_id: str,
    model_id: str,
    temperature: float,
) -> RoutingDecision:
    """
    Build a RoutingDecision object.
    
    For this branch, we use simple single-model routing.
    Multi-model critique will be added in later branches.
    """
    # Simple tier mapping
    tier_map = {
        "claude-sonnet-4-20250514": "S",
        "gpt-4o": "A",
        "gemini-2.0-flash": "B",
    }
    tier = tier_map.get(model_id, "B")
    
    return RoutingDecision(
        job_id=envelope.job_id or "unknown",
        job_type=envelope.job_type.value,
        resolved_job_type=envelope.job_type.value,
        architect=ModelSelection(
            provider=provider_id,
            model_id=model_id,
            tier=tier,
            role="architect",
        ),
        reviewers=[],  # No multi-model review in this branch
        arbiter=None,
        temperature=temperature,
        max_tokens=envelope.budget.max_tokens,
        timeout_seconds=envelope.budget.max_wall_time_seconds,
        data_sensitivity_constraint=envelope.data_sensitivity.value,
        allowed_tools=envelope.allowed_tools or [],
        forbidden_tools=envelope.forbidden_tools,
        fallback_occurred=False,
    )


def _build_system_prompt(envelope: JobEnvelope, provider_id: str) -> str:
    """Build system prompt based on job type and provider."""
    job_type = envelope.job_type
    
    # Base prompts per provider
    if provider_id == "openai":
        base = """You are Orb, a fast and helpful assistant.

Your role: Handle conversational tasks, summaries, explanations, and lightweight text work.

Be concise, clear, and direct. Get to the point quickly."""
    
    elif provider_id == "anthropic":
        if job_type == JobType.APP_ARCHITECTURE:
            base = """You are Orb's engineering brain — a senior backend architect.

Your role: Design comprehensive system architectures, explain technical decisions, and document design choices.

CRITICAL RULES:
1. Think through the entire system design before writing.
2. Document key decisions and trade-offs.
3. Consider scalability, maintainability, and security.
4. Be thorough and precise.
5. Include concrete examples where helpful."""
        
        elif job_type in (JobType.CODE_SMALL, JobType.CODE_REPO):
            base = """You are Orb's engineering brain — a senior backend architect and implementer.

Your role: Generate complete, production-ready code.

CRITICAL RULES:
1. When modifying existing files: ALWAYS ask for the full current file content first, then return the COMPLETE updated file.
2. NEVER return partial files, diffs, or snippets. Always return complete, runnable code.
3. Include all imports, all functions, all boilerplate — the user should be able to copy-paste directly.
4. Write clear comments explaining non-obvious decisions.
5. Think through edge cases before writing code.

Be precise, technical, and thorough."""
        
        else:
            base = """You are Orb's engineering brain — a senior backend architect.

Your role: Handle complex technical tasks with precision and thoroughness.

Be precise, technical, and thorough."""
    
    else:  # google
        base = """You are Orb's analyst — a reviewer and vision specialist.

Your role: Analyze content, review work, identify patterns, and provide structured feedback.

Be analytical, precise, and actionable."""
    
    # Add custom system prompt if provided
    if envelope.system_prompt:
        base += f"\n\n{envelope.system_prompt}"
    
    return base


# =============================================================================
# SESSION UPSERT (Race-condition safe)
# =============================================================================

def _get_or_create_session(
    db: Session,
    session_id: str,
    project_id: int,
) -> SessionModel:
    """
    Get or create a session in a race-condition safe manner.
    
    Uses INSERT OR IGNORE pattern to handle concurrent requests
    with the same session_id without raising IntegrityError.
    
    Args:
        db: Database session
        session_id: Session identifier
        project_id: Project identifier
    
    Returns:
        SessionModel instance (existing or newly created)
    """
    # First, try to get existing session
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if session:
        return session
    
    # Session doesn't exist - try to create it
    # Use INSERT OR IGNORE to handle race conditions
    try:
        # Create new session
        session = SessionModel(
            id=session_id,
            project_id=project_id,
            name=f"Session {session_id[:8]}",
        )
        db.add(session)
        db.commit()
        logger.info(f"[engine] Created new session: {session_id}")
        return session
    except IntegrityError:
        # Another request created the session - rollback and fetch it
        db.rollback()
        logger.debug(f"[engine] Session {session_id} created by concurrent request, fetching")
        session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if session:
            return session
        # This should never happen, but handle it gracefully
        raise RuntimeError(f"Failed to get or create session: {session_id}")


# =============================================================================
# JOB EXECUTION ENGINE
# =============================================================================

async def execute_job(
    envelope: JobEnvelope,
    db: Session,
) -> JobResult:
    """
    Execute a job using the provider registry.
    
    This is the core execution function that:
    1. Routes to appropriate provider
    2. Calls LLM via registry
    3. Handles errors and maps to ErrorType
    4. Stores artefacts for structured outputs
    5. Returns complete JobResult
    
    Args:
        envelope: Complete job specification
        db: Database session
    
    Returns:
        JobResult with execution details
    """
    job_id = envelope.job_id or str(uuid4())
    started_at = datetime.utcnow()
    
    try:
        # Determine routing
        provider_id, model_id, temperature = _determine_routing(envelope)
        routing_decision = _build_routing_decision(envelope, provider_id, model_id, temperature)
        routing_decision.job_id = job_id
        
        # Build system prompt
        system_prompt = _build_system_prompt(envelope, provider_id)
        
        logger.info(
            f"[engine] Executing job {job_id}: {envelope.job_type.value} "
            f"→ {provider_id}/{model_id}"
        )
        
        # Call provider registry
        result = await llm_call(
            provider_id=provider_id,
            model_id=model_id,
            messages=envelope.messages,
            system_prompt=system_prompt,
            job_envelope=envelope,
            temperature=temperature,
            max_tokens=envelope.budget.max_tokens,
            timeout_seconds=envelope.budget.max_wall_time_seconds,
        )
        
        completed_at = datetime.utcnow()
        duration = (completed_at - started_at).total_seconds()
        
        # Handle result based on status
        if result.is_success():
            # Success case
            content = result.content
            
            # Store artefact if needed
            artefact_id = None
            if envelope.output_contract in (
                OutputContract.ARCHITECTURE_DOC,
                OutputContract.CODE_PATCH_PROPOSAL,
            ):
                artefact_id = await _store_artefact(
                    db=db,
                    envelope=envelope,
                    content=content,
                    job_id=job_id,
                )
            
            # Build usage metrics
            usage_metrics = [
                UsageMetrics(
                    model_id=model_id,
                    provider=provider_id,
                    prompt_tokens=result.usage.prompt_tokens,
                    completion_tokens=result.usage.completion_tokens,
                    total_tokens=result.usage.total_tokens,
                    cost_estimate=result.usage.cost_estimate,
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
                tools_used=[],  # No tool invocations in this branch
                was_reviewed=False,
                critique_issues=[],
                unresolved_blockers=0,
                usage_metrics=usage_metrics,
                total_cost_estimate=result.usage.cost_estimate,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=duration,
            )
        
        else:
            # Error case - map LlmCallStatus to ErrorType
            if result.status == LlmCallStatus.TIMEOUT:
                error_type = ErrorType.TIMEOUT
            elif result.status == LlmCallStatus.RATE_LIMITED:
                error_type = ErrorType.MODEL_ERROR
            else:
                error_type = ErrorType.MODEL_ERROR
            
            logger.error(
                f"[engine] Job {job_id} failed: {result.error_code} - {result.error_message}"
            )
            
            return JobResult(
                job_id=job_id,
                session_id=envelope.session_id,
                project_id=envelope.project_id,
                job_type=envelope.job_type.value,
                state=JobState.FAILED,
                content="",
                output_contract=envelope.output_contract,
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
                error_message=result.error_message,
                error_details={"error_code": result.error_code},
            )
    
    except Exception as e:
        # Unexpected error
        completed_at = datetime.utcnow()
        duration = (completed_at - started_at).total_seconds()
        
        logger.exception(f"[engine] Unexpected error in job {job_id}")
        
        return JobResult(
            job_id=job_id,
            session_id=envelope.session_id,
            project_id=envelope.project_id,
            job_type=envelope.job_type.value,
            state=JobState.FAILED,
            content="",
            output_contract=envelope.output_contract,
            routing_decision=RoutingDecision(
                job_id=job_id,
                job_type=envelope.job_type.value,
                resolved_job_type=envelope.job_type.value,
                architect=ModelSelection(
                    provider="unknown",
                    model_id="unknown",
                    tier="B",
                    role="architect",
                ),
                temperature=0.7,
                max_tokens=envelope.budget.max_tokens,
                timeout_seconds=envelope.budget.max_wall_time_seconds,
                data_sensitivity_constraint=envelope.data_sensitivity.value,
                allowed_tools=[],
                forbidden_tools=[],
            ),
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
            error_message=str(e),
            error_details={"exception_type": type(e).__name__},
        )


async def _store_artefact(
    db: Session,
    envelope: JobEnvelope,
    content: str,
    job_id: str,
) -> Optional[str]:
    """
    Store artefact for structured outputs.
    
    Returns:
        Artefact ID if stored, None otherwise
    """
    try:
        # Determine artefact type and name
        if envelope.output_contract == OutputContract.ARCHITECTURE_DOC:
            artefact_type = "architecture_doc"
            name = envelope.metadata.get("artefact_name", f"Architecture - {datetime.utcnow().strftime('%Y%m%d-%H%M%S')}")
        
        elif envelope.output_contract == OutputContract.CODE_PATCH_PROPOSAL:
            artefact_type = "code_patch_proposal"
            name = envelope.metadata.get("artefact_name", f"Code Patch - {datetime.utcnow().strftime('%Y%m%d-%H%M%S')}")
        
        else:
            return None
        
        # Write artefact
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
    
    except Exception as e:
        logger.error(f"[engine] Failed to store artefact for job {job_id}: {e}")
        return None


async def create_and_run_job(
    request: CreateJobRequest,
    db: Session,
) -> JobResult:
    """
    Create a job from request, persist it, and execute.
    
    This is the main entry point for job execution from HTTP endpoints.
    
    Args:
        request: Job creation request
        db: Database session
    
    Returns:
        JobResult with execution details
    """
    job_id = str(uuid4())
    
    try:
        # Create or get session (race-condition safe)
        session_id = request.session_id or str(uuid4())
        session = _get_or_create_session(db, session_id, request.project_id)
        
        # Convert request to envelope
        envelope = _request_to_envelope(request, job_id, session_id)
        
        # Validate envelope
        try:
            validate_job_envelope(envelope)
        except ValidationError as e:
            # Validation failed
            logger.warning(f"[engine] Job validation failed: {e.errors}")
            
            # Create failed job entry
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
                envelope_json=envelope.model_dump(),
                error_type=ErrorType.VALIDATION_ERROR.value,
                error_message=f"Validation failed: {'; '.join(e.errors)}",
                created_at=datetime.utcnow(),
            )
            db.add(job)
            db.commit()
            
            # Return error result
            return JobResult(
                job_id=job_id,
                session_id=session_id,
                project_id=request.project_id,
                job_type=envelope.job_type.value,
                state=JobState.FAILED,
                content="",
                output_contract=envelope.output_contract,
                routing_decision=RoutingDecision(
                    job_id=job_id,
                    job_type=envelope.job_type.value,
                    resolved_job_type=envelope.job_type.value,
                    architect=ModelSelection(
                        provider="none",
                        model_id="none",
                        tier="B",
                        role="architect",
                    ),
                    temperature=0.7,
                    max_tokens=envelope.budget.max_tokens,
                    timeout_seconds=envelope.budget.max_wall_time_seconds,
                    data_sensitivity_constraint=envelope.data_sensitivity.value,
                    allowed_tools=[],
                    forbidden_tools=[],
                ),
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
            envelope_json=envelope.model_dump(),
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

        # Spec Gate for architecture jobs (PoT Spec)
        if envelope.job_type in (JobType.APP_ARCHITECTURE, JobType.CODE_REPO):
            from app.pot_spec.spec_gate import run_spec_gate, detect_user_questions

            spec_gate_provider = os.getenv("SPEC_GATE_PROVIDER", "openai")
            spec_gate_model = os.getenv(
                "OPENAI_SPEC_GATE_MODEL",
                os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5-mini"),
            )
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
                    routing_decision=routing_decision,
                    tools_used=[],
                    was_reviewed=False,
                    critique_issues=[],
                    unresolved_blockers=len(open_questions),
                    usage_metrics=[],
                    total_cost_estimate=0.0,
                    started_at=datetime.utcnow(),
                    completed_at=datetime.utcnow(),
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
        if envelope.job_type in (JobType.APP_ARCHITECTURE, JobType.CODE_REPO):
            stage_name = "architecture_job_execution" if envelope.job_type == JobType.APP_ARCHITECTURE else "code_repo_job_execution"
            append_stage3_ledger_event(
                job_id,
                {
                    "event": "STAGE_STARTED",
                    "job_id": job_id,
                    "stage": stage_name,
                    "spec_id": spec_id,
                    "spec_hash": spec_hash,
                    "ts_utc": datetime.utcnow().isoformat() + "Z",
                },
            )

        # Execute job
        result = await execute_job(envelope, db)


        # Stage 3: Spec-hash lock verification (refuse-on-mismatch)
        if envelope.job_type in (JobType.APP_ARCHITECTURE, JobType.CODE_REPO):
            stage_name = "architecture_job_execution" if envelope.job_type == JobType.APP_ARCHITECTURE else "code_repo_job_execution"
            returned_spec_id, returned_spec_hash, parse_note = parse_spec_echo_headers(result.content or "")
            verified = (returned_spec_id == spec_id and returned_spec_hash == spec_hash)

            # Best-effort artifact storage (raw output + meta)
            provider_name = getattr(getattr(result, "routing_decision", None), "provider", None)
            model_name = getattr(getattr(result, "routing_decision", None), "model", None)
            written_paths = write_stage3_artifacts(
                job_id=job_id,
                stage=stage_name,
                raw_output=result.content or "",
                expected_spec_id=spec_id,
                expected_spec_hash=spec_hash,
                returned_spec_id=returned_spec_id,
                returned_spec_hash=returned_spec_hash,
                verified=verified,
                provider=provider_name,
                model=model_name,
            )

            append_stage3_ledger_event(
                job_id,
                {
                    "event": "STAGE_SPEC_HASH_VERIFIED" if verified else "STAGE_SPEC_HASH_MISMATCH",
                    "job_id": job_id,
                    "stage": stage_name,
                    "expected_spec_id": spec_id,
                    "expected_spec_hash": spec_hash,
                    "returned_spec_id": returned_spec_id,
                    "returned_spec_hash": returned_spec_hash,
                    "parse_note": parse_note,
                    "ts_utc": datetime.utcnow().isoformat() + "Z",
                },
            )
            append_stage3_ledger_event(
                job_id,
                {
                    "event": "STAGE_OUTPUT_STORED",
                    "job_id": job_id,
                    "stage": stage_name,
                    "paths": written_paths,
                    "ts_utc": datetime.utcnow().isoformat() + "Z",
                },
            )

            if not verified:
                # Fail fast: stop pipeline, mark job failed
                job.state = JobState.FAILED.value
                job.output_content = result.content
                job.error_type = ErrorType.MODEL_ERROR.value
                job.error_message = (
                    "Stage 3 spec-hash lock failed: "
                    f"expected SPEC_ID={spec_id}, SPEC_HASH={spec_hash}; "
                    f"got SPEC_ID={returned_spec_id}, SPEC_HASH={returned_spec_hash} ({parse_note})"
                )
                job.completed_at = datetime.utcnow()
                db.commit()

                return JobResult(
                    job_id=job_id,
                    session_id=envelope.session_id,
                    project_id=envelope.project_id,
                    job_type=envelope.job_type.value,
                    state=JobState.FAILED,
                    content=job.output_content,
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
                )

        # Hard enforcement: downstream stages must not ask user questions
        if envelope.job_type in (JobType.APP_ARCHITECTURE, JobType.CODE_REPO):
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

                spec_gate_provider = os.getenv("SPEC_GATE_PROVIDER", "openai")
                spec_gate_model = os.getenv(
                    "OPENAI_SPEC_GATE_MODEL",
                    os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5-mini"),
                )
                user_intent = _extract_user_intent(request.messages)
                spec_id, spec_hash, open_questions = await run_spec_gate(
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
                    routing_decision=routing_decision,
                    tools_used=[],
                    was_reviewed=False,
                    critique_issues=[],
                    unresolved_blockers=len(open_questions),
                    usage_metrics=[],
                    total_cost_estimate=result.total_cost_estimate,
                    started_at=result.started_at,
                    completed_at=datetime.utcnow(),
                    duration_seconds=result.duration_seconds,
                    error_type=None,
                    error_message=None,
                )

        
        # Update job with result
        job.state = result.state.value
        job.output_content = result.content
        job.output_contract = result.output_contract.value
        job.artefact_id = result.artefact_id
        job.routing_decision_json = result.routing_decision.model_dump()
        job.tool_invocations_json = [t.model_dump() for t in result.tools_used]
        job.critique_issues_json = [c.model_dump() for c in result.critique_issues]
        job.was_reviewed = result.was_reviewed
        job.unresolved_blockers = result.unresolved_blockers
        job.usage_metrics_json = [u.model_dump() for u in result.usage_metrics]
        job.total_tokens = sum(u.total_tokens for u in result.usage_metrics)
        job.total_cost_estimate = result.total_cost_estimate
        job.completed_at = result.completed_at
        job.duration_seconds = result.duration_seconds
        
        if result.error_type:
            job.error_type = result.error_type.value
            job.error_message = result.error_message
            job.error_details_json = result.error_details
        
        db.commit()
        
        # Update session stats
        session.job_count += 1
        session.last_activity = datetime.utcnow()
        session.total_cost_estimate += result.total_cost_estimate
        db.commit()
        
        logger.info(
            f"[engine] Job {job_id} completed: {result.state.value} "
            f"(cost=${result.total_cost_estimate:.4f})"
        )
        
        return result
    
    except Exception as e:
        logger.exception(f"[engine] Unexpected error creating/running job {job_id}")
        
        # Try to update job if it exists
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


def _request_to_envelope(
    request: CreateJobRequest,
    job_id: str,
    session_id: str,
) -> JobEnvelope:
    """
    Convert CreateJobRequest to JobEnvelope.
    
    Fills in reasonable defaults for fields not provided.
    """
    # Parse job_type string to enum
    try:
        job_type = JobType(request.job_type)
    except ValueError:
        # Unknown job type - try to be helpful
        raise ValueError(
            f"Unknown job_type: '{request.job_type}'. "
            f"Must be one of: {[jt.value for jt in JobType]}"
        )
    
    # Determine defaults based on job type
    importance = request.importance or Importance.MEDIUM
    data_sensitivity = request.data_sensitivity or DataSensitivity.INTERNAL
    
    # Determine modalities
    modalities = [Modality.TEXT]
    if request.attachments:
        modalities.append(Modality.IMAGE)
    
    # Determine output contract
    output_contract_map = {
        JobType.CHAT_SIMPLE: OutputContract.TEXT_RESPONSE,
        JobType.CHAT_RESEARCH: OutputContract.TEXT_RESPONSE,
        JobType.CODE_SMALL: OutputContract.CODE_PATCH_PROPOSAL,
        JobType.CODE_REPO: OutputContract.CODE_PATCH_PROPOSAL,
        JobType.APP_ARCHITECTURE: OutputContract.ARCHITECTURE_DOC,
        JobType.VISION_SIMPLE: OutputContract.TEXT_RESPONSE,
        JobType.VISION_COMPLEX: OutputContract.TEXT_RESPONSE,
        JobType.CRITIQUE_REVIEW: OutputContract.CRITIQUE_REVIEW,
    }
    output_contract = output_contract_map.get(job_type, OutputContract.TEXT_RESPONSE)
    
    return JobEnvelope(
        job_id=job_id,
        session_id=session_id,
        project_id=request.project_id,
        job_type=job_type,
        importance=importance,
        data_sensitivity=data_sensitivity,
        modalities_in=modalities,
        needs_internet=request.needs_internet,
        allow_multi_model_review=request.allow_multi_model_review,
        messages=request.messages,
        system_prompt=request.system_prompt,
        attachments=request.attachments,
        budget=JobBudget(),
        output_contract=output_contract,
        metadata=request.metadata or {},
    )


__all__ = [
    "execute_job",
    "create_and_run_job",
]
