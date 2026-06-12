# FILE: app/jobs/_engine_helpers.py
# Purpose: Job engine helper functions.
# Called-by: app.jobs.engine
# Depends-on: app.artefacts.service, app.jobs.models, app.jobs.schemas, app.llm.stream_utils
# Last-renovated: 2026-06-11
"""
Job engine helper functions.

Extracted from engine.py: routing, session management, artefact storage,
and request-to-envelope conversion.
"""
from __future__ import annotations

import json
import logging
import os
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
from app.llm.stream_utils import get_spec_gate_model, get_spec_gate_provider

logger = logging.getLogger(__name__)

def _determine_routing(envelope: JobEnvelope) -> tuple[str, str, float]:
    """
    Determine provider, model, and temperature for a job.

    Returns:
        (provider_id, model_id, temperature)
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
        return "openai", os.getenv("CHAT_MODEL", "gpt-5.4-mini"), 0.7

    # Default to chat model from env
    return "openai", os.getenv("CHAT_MODEL", "gpt-5.4-mini"), 0.7


def _extract_user_intent(messages: list[dict]) -> str:
    for m in reversed(messages or []):
        if (m.get("role") == "user") and m.get("content"):
            return str(m.get("content")).strip()
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
    tier_map = {
        "claude-sonnet-4-20250514": "S",
        "gpt-5.4-mini": "A",
        "gpt-5.4": "A",
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
        reviewers=[],
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
    job_type = envelope.job_type

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
    else:
        base = """You are Orb's analyst — a reviewer and vision specialist.

Your role: Analyze content, review work, identify patterns, and provide structured feedback.

Be analytical, precise, and actionable."""

    if envelope.system_prompt:
        base += f"\n\n{envelope.system_prompt}"

    return base


def _placeholder_routing_decision(envelope: JobEnvelope) -> RoutingDecision:
    """Used for early returns (paused/validation) where we haven't executed main routing yet."""
    return RoutingDecision(
        job_id=envelope.job_id or "unknown",
        job_type=envelope.job_type.value,
        resolved_job_type=envelope.job_type.value,
        architect=ModelSelection(
            provider="unknown",
            model_id="unknown",
            tier="B",
            role="architect",
        ),
        reviewers=[],
        arbiter=None,
        temperature=0.7,
        max_tokens=envelope.budget.max_tokens,
        timeout_seconds=envelope.budget.max_wall_time_seconds,
        data_sensitivity_constraint=envelope.data_sensitivity.value,
        allowed_tools=envelope.allowed_tools or [],
        forbidden_tools=envelope.forbidden_tools,
        fallback_occurred=False,
    )


# =============================================================================
# SESSION UPSERT (Race-condition safe)
# =============================================================================


def _get_or_create_session(db: Session, session_id: str, project_id: int) -> SessionModel:
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if session:
        return session

    try:
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
        db.rollback()
        logger.debug(f"[engine] Session {session_id} created by concurrent request, fetching")
        session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if session:
            return session
        raise RuntimeError(f"Failed to get or create session: {session_id}")


# =============================================================================
# JOB EXECUTION ENGINE
# =============================================================================


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

def _request_to_envelope(request: CreateJobRequest, job_id: str, session_id: str) -> JobEnvelope:
    try:
        job_type = JobType(request.job_type)
    except ValueError:
        raise ValueError(
            f"Unknown job_type: '{request.job_type}'. Must be one of: {[jt.value for jt in JobType]}"
        )

    importance = request.importance or Importance.MEDIUM
    data_sensitivity = request.data_sensitivity or DataSensitivity.INTERNAL

    modalities = [Modality.TEXT]
    if request.attachments:
        modalities.append(Modality.IMAGE)

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
        needs_internet=bool(request.needs_internet),
        allow_multi_model_review=bool(request.allow_multi_model_review),
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
