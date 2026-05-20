# FILE: app/endpoints/ambient.py
"""
Ambient invocation endpoints.

Routes:
  POST /api/ambient/classify    - classify a transcript (no LLM call)
  POST /api/ambient/invoke      - full pipeline: classify + tier + dispatch
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException

from app.auth import require_auth
from app.invocation import (
    classify as classify_intent,
    decide as tier_decide,
    route as invocation_route,
    InvocationRequest,
    InvocationResponse,
    ClassifiedIntent,
    TierDecision,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ambient", tags=["Ambient"])


class ClassifyRequest(BaseModel):
    transcript: str
    context_bytes: int = 0
    source_count: int = 0


class ClassifyResponse(BaseModel):
    intent: ClassifiedIntent
    tier: TierDecision


@router.post("/classify", response_model=ClassifyResponse)
async def classify_endpoint(
    req: ClassifyRequest,
    _auth=Depends(require_auth),
) -> ClassifyResponse:
    """Run only the classifier + tier gate, without dispatching."""
    intent = classify_intent(req.transcript)
    invocation_req = InvocationRequest(
        transcript=req.transcript,
        context_bytes=req.context_bytes,
        source_count=req.source_count,
    )
    tier = tier_decide(invocation_req, intent)
    return ClassifyResponse(intent=intent, tier=tier)


@router.post("/invoke", response_model=InvocationResponse)
async def invoke_endpoint(
    req: InvocationRequest,
    _auth=Depends(require_auth),
) -> InvocationResponse:
    """Dispatch a full ambient invocation."""
    try:
        result = await invocation_route(req)
        return result
    except Exception as e:
        logger.exception("[ambient] invoke failed")
        raise HTTPException(status_code=500, detail=str(e))