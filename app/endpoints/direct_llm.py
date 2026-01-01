# FILE: app/endpoints/direct_llm.py
"""
Direct LLM endpoint - LLM calls without project context.

Refactored from main.py.
"""

import time
from uuid import uuid4
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth import require_auth
from app.auth.middleware import AuthResult
from app.llm import call_llm, LLMTask, LLMResult, JobType
from app.llm.schemas import Provider
from app.llm.audit_logger import get_audit_logger, RoutingTrace

from app.helpers.llm_utils import (
    sync_await,
    extract_provider_value,
    extract_model_value,
    classify_job_type,
    make_session_id,
)

router = APIRouter(tags=["LLM"])


# ============================================================================
# MODELS
# ============================================================================

class DirectLLMRequest(BaseModel):
    job_type: str
    message: str
    force_provider: Optional[str] = None


class DirectLLMResponse(BaseModel):
    provider: str
    model: Optional[str] = None
    content: str
    was_reviewed: bool = False
    critic_review: Optional[str] = None


# ============================================================================
# ENDPOINT
# ============================================================================

@router.post("/llm", response_model=DirectLLMResponse)
def direct_llm(
    req: DirectLLMRequest,
    auth: AuthResult = Depends(require_auth),
) -> DirectLLMResponse:
    """Direct LLM call without project context."""
    jt = classify_job_type(req.message, req.job_type)
    print(f"[llm] Job type: {jt.value}")

    task = LLMTask(
        job_type=jt,
        messages=[{"role": "user", "content": req.message}],
        force_provider=Provider(req.force_provider) if req.force_provider else None,
    )

    try:
        audit = get_audit_logger()
        trace: Optional[RoutingTrace] = None
        request_id = str(uuid4())
        
        if getattr(audit, "enabled", False):
            trace = audit.start_trace(
                session_id=make_session_id(auth),
                project_id=0,  # No project for direct LLM
                user_text=req.message,
                request_id=request_id,
                attachments=None,
            )
            trace.log_request_start(
                job_type=jt.value,
                resolved_job_type=jt.value,
                provider="unknown",
                model="unknown",
                reason="main.py /llm",
                frontier_override=None,
                file_map_injected=False,
                attachments=None,
            )

        t0 = time.perf_counter()
        try:
            result: LLMResult = sync_await(call_llm(task))
        except Exception as e:
            if trace:
                trace.log_error(
                    where="direct_llm",
                    error_type=type(e).__name__,
                    message=str(e),
                )
                trace.finalize(success=False, error_type=type(e).__name__, message=str(e))
            raise

        dt_ms = int((time.perf_counter() - t0) * 1000)
        
        if trace:
            trace.log_routing_decision(
                job_type=jt.value,
                provider=getattr(result, "provider", "unknown"),
                model=getattr(result, "model", "unknown"),
                reason="call_llm result",
                frontier_override=None,
            )
            trace.log_model_call(
                stage="primary",
                provider=getattr(result, "provider", "unknown"),
                model=getattr(result, "model", "unknown"),
                role="primary",
                prompt_tokens=0,
                completion_tokens=0,
                duration_ms=dt_ms,
                success=True,
            )
            trace.finalize(success=True)
            
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    provider_str = extract_provider_value(result)
    model_str = extract_model_value(result)
    
    print(f"[llm] Response from: {provider_str} / {model_str}")

    return DirectLLMResponse(
        provider=provider_str,
        model=model_str,
        content=result.content,
        was_reviewed=result.was_reviewed,
        critic_review=result.critic_review,
    )
