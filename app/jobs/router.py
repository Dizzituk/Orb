# FILE: app/jobs/router.py
"""
Phase 4 Jobs Router - HTTP API Endpoints

Provides REST API for job management:
- POST /jobs/create - Create and execute job
- GET /jobs/{job_id} - Get job status and result
- GET /jobs/list - List jobs with filters
- POST /jobs/{job_id}/cancel - Cancel job (basic implementation)
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.jobs.models import Job, Session as SessionModel
from app.jobs.schemas import (
    CreateJobRequest,
    CreateJobResponse,
    GetJobResponse,
    ListJobsRequest,
    ListJobsResponse,
    JobResult,
    JobState,
    RoutingDecision,
    ModelSelection,
    ToolInvocation,
    CritiqueIssue,
    UsageMetrics,
    ErrorType,
    OutputContract,
)
from app.jobs.engine import create_and_run_job

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _reconstruct_job_result(job: Job) -> Optional[JobResult]:
    """
    Reconstruct JobResult from database Job model.
    
    Parses all JSON fields into proper Pydantic models.
    Returns None if job is not in a completed state.
    """
    if job.state not in (JobState.SUCCEEDED.value, JobState.FAILED.value, JobState.CANCELLED.value):
        return None
    
    # Parse routing decision
    routing_decision = None
    if job.routing_decision_json:
        try:
            routing_decision = RoutingDecision(**job.routing_decision_json)
        except Exception as e:
            logger.warning(f"[router] Failed to parse routing_decision_json for job {job.id}: {e}")
            routing_decision = None
    
    if not routing_decision:
        # Fallback routing decision
        routing_decision = RoutingDecision(
            job_id=job.id,
            job_type=job.job_type,
            resolved_job_type=job.resolved_job_type or job.job_type,
            architect=ModelSelection(
                provider="unknown",
                model_id="unknown",
                tier="B",
                role="architect",
            ),
            temperature=0.7,
            max_tokens=100000,
            timeout_seconds=300,
            data_sensitivity_constraint=job.data_sensitivity,
            allowed_tools=[],
            forbidden_tools=[],
        )
    
    # Parse tool invocations
    tools_used = []
    if job.tool_invocations_json:
        try:
            tools_used = [ToolInvocation(**t) for t in job.tool_invocations_json]
        except Exception as e:
            logger.warning(f"[router] Failed to parse tool_invocations_json for job {job.id}: {e}")
            tools_used = []
    
    # Parse critique issues
    critique_issues = []
    if job.critique_issues_json:
        try:
            critique_issues = [CritiqueIssue(**c) for c in job.critique_issues_json]
        except Exception as e:
            logger.warning(f"[router] Failed to parse critique_issues_json for job {job.id}: {e}")
            critique_issues = []
    
    # Parse usage metrics
    usage_metrics = []
    if job.usage_metrics_json:
        try:
            usage_metrics = [UsageMetrics(**u) for u in job.usage_metrics_json]
        except Exception as e:
            logger.warning(f"[router] Failed to parse usage_metrics_json for job {job.id}: {e}")
            usage_metrics = []
    
    # Parse error type
    error_type = None
    if job.error_type:
        try:
            error_type = ErrorType(job.error_type)
        except ValueError:
            logger.warning(f"[router] Invalid error_type '{job.error_type}' for job {job.id}")
            error_type = ErrorType.INTERNAL_ERROR
    
    # Parse output contract
    try:
        output_contract = OutputContract(job.output_contract)
    except ValueError:
        logger.warning(f"[router] Invalid output_contract '{job.output_contract}' for job {job.id}")
        output_contract = OutputContract.TEXT_RESPONSE
    
    return JobResult(
        job_id=job.id,
        session_id=job.session_id,
        project_id=job.project_id,
        job_type=job.job_type,
        state=JobState(job.state),
        content=job.output_content or "",
        output_contract=output_contract,
        artefact_id=job.artefact_id,
        routing_decision=routing_decision,
        tools_used=tools_used,
        was_reviewed=job.was_reviewed,
        critique_issues=critique_issues,
        unresolved_blockers=job.unresolved_blockers,
        usage_metrics=usage_metrics,
        total_cost_estimate=job.total_cost_estimate,
        started_at=job.started_at or job.created_at,
        completed_at=job.completed_at or job.created_at,
        duration_seconds=job.duration_seconds or 0.0,
        error_type=error_type,
        error_message=job.error_message,
        error_details=job.error_details_json,
    )


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/create", response_model=CreateJobResponse)
async def create_job(
    request: CreateJobRequest,
    db: Session = Depends(get_db),
):
    """
    Create and execute a new job.
    
    For this branch, jobs execute inline (not queued).
    
    Request body:
    - project_id: Project identifier
    - job_type: Type of job (chat_simple, code_small, app_architecture, etc.)
    - messages: List of message dicts
    - session_id: Optional session ID
    - importance, data_sensitivity, needs_internet, etc.
    
    Returns:
    - job_id: Created job identifier
    - session_id: Session identifier
    - state: Job state (should be SUCCEEDED or FAILED after execution)
    - created_at: Timestamp
    """
    try:
        logger.info(
            f"[jobs] Creating job: type={request.job_type} project={request.project_id}"
        )
        
        # Create and execute job
        result = await create_and_run_job(request, db)
        
        return CreateJobResponse(
            job_id=result.job_id,
            session_id=result.session_id,
            state=result.state,
            created_at=result.started_at,
        )
    
    except ValueError as e:
        # Validation or request errors
        logger.error(f"[jobs] Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        # Unexpected errors
        logger.exception(f"[jobs] Error creating job")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.get("/{job_id}", response_model=GetJobResponse)
async def get_job(
    job_id: str,
    db: Session = Depends(get_db),
):
    """
    Get job status and result.
    
    Path parameters:
    - job_id: Job identifier
    
    Returns:
    - job_id, session_id, project_id
    - state: Job state (pending, running, succeeded, failed, etc.)
    - result: Complete JobResult if job is finished
    - created_at, updated_at: Timestamps
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    # Reconstruct JobResult from stored data
    result = _reconstruct_job_result(job)
    
    return GetJobResponse(
        job_id=job.id,
        session_id=job.session_id,
        project_id=job.project_id,
        state=JobState(job.state),
        job_type=job.job_type,
        result=result,
        created_at=job.created_at,
        updated_at=job.completed_at or job.created_at,
    )


@router.get("/list", response_model=ListJobsResponse)
async def list_jobs(
    project_id: Optional[int] = Query(None),
    session_id: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """
    List jobs with filters.
    
    Query parameters:
    - project_id: Filter by project
    - session_id: Filter by session
    - state: Filter by state (pending, running, succeeded, failed, etc.)
    - limit: Maximum results (1-100, default 20)
    - offset: Pagination offset (default 0)
    
    Returns:
    - jobs: List of job summaries
    - total: Total count matching filters
    - limit, offset: Pagination parameters
    """
    # Build query
    query = db.query(Job)
    
    if project_id is not None:
        query = query.filter(Job.project_id == project_id)
    
    if session_id is not None:
        query = query.filter(Job.session_id == session_id)
    
    if state is not None:
        query = query.filter(Job.state == state)
    
    # Get total count BEFORE pagination
    total = query.count()
    
    # Apply pagination
    query = query.order_by(Job.created_at.desc())
    query = query.limit(limit).offset(offset)
    
    jobs_list = query.all()
    
    # Build response
    jobs_response = []
    for job in jobs_list:
        # Reconstruct JobResult from stored data
        result = _reconstruct_job_result(job)
        
        jobs_response.append(
            GetJobResponse(
                job_id=job.id,
                session_id=job.session_id,
                project_id=job.project_id,
                state=JobState(job.state),
                job_type=job.job_type,
                result=result,
                created_at=job.created_at,
                updated_at=job.completed_at or job.created_at,
            )
        )
    
    return ListJobsResponse(
        jobs=jobs_response,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    db: Session = Depends(get_db),
):
    """
    Cancel a job.
    
    For this branch, cancellation is basic:
    - If job is PENDING: Mark as CANCELLED
    - If job is RUNNING or completed: Cannot cancel (return error)
    
    Path parameters:
    - job_id: Job identifier
    
    Returns:
    - job_id: Job identifier
    - state: New state (cancelled)
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    # Check current state
    if job.state == JobState.PENDING.value:
        # Can cancel pending jobs
        job.state = JobState.CANCELLED.value
        job.completed_at = job.created_at  # Set completed time
        db.commit()
        
        logger.info(f"[jobs] Cancelled job {job_id}")
        
        return {
            "job_id": job_id,
            "state": JobState.CANCELLED.value,
        }
    
    elif job.state == JobState.RUNNING.value:
        # Cannot cancel running jobs in this branch (no worker queue)
        raise HTTPException(
            status_code=409,
            detail="Cannot cancel running job in this branch. Job execution is inline."
        )
    
    elif job.state in (JobState.SUCCEEDED.value, JobState.FAILED.value, JobState.CANCELLED.value):
        # Already completed
        raise HTTPException(
            status_code=409,
            detail=f"Job already completed with state: {job.state}"
        )
    
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job in state: {job.state}"
        )


__all__ = ["router"]