# FILE: app/investments/chat_router.py
"""
Investments chat endpoint.

Injects live portfolio data as system context, then calls the LLM
to answer investment-related questions with real data.

Streams SSE tokens back to the frontend using the same event format
as the main chat endpoint.
"""
import os
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth
from app.investments.chat_service import build_investment_context

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/investments",
    tags=["Investments Chat"],
    dependencies=[Depends(require_auth)],
)


class InvestmentsChatRequest(BaseModel):
    message: str
    provider: Optional[str] = None
    model: Optional[str] = None


async def _stream_investment_chat(
    message: str,
    context: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
):
    """
    Generator that yields SSE events for the investments chat.
    Uses the provider registry's llm_call with portfolio context.
    """
    from app.providers._registry_utils_3 import llm_call
    from app.providers._registry_utils_4 import LlmCallStatus

    used_provider = provider or None
    used_model = model or os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5.4-mini")

    # Metadata event
    display_provider = used_provider or "auto"
    yield f"data: {json.dumps({'type': 'metadata', 'provider': display_provider, 'model': used_model})}\n\n"

    try:
        messages = [{"role": "user", "content": message}]

        result = await llm_call(
            provider_id=provider,
            model_id=used_model,
            messages=messages,
            system_prompt=context,
            temperature=0.3,
            max_tokens=1000,
        )

        if result.status == LlmCallStatus.SUCCESS:
            content = result.content or ""
            actual_provider = result.provider_id or used_provider
            actual_model = result.model_id or used_model
        else:
            content = f"Sorry, I couldn't process that: {result.error_message or 'unknown error'}"
            actual_provider = used_provider
            actual_model = used_model

        # Emit as a single token event
        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

        # Done event
        yield f"data: {json.dumps({'type': 'done', 'provider': actual_provider, 'model': actual_model, 'total_length': len(content)})}\n\n"

    except Exception as exc:
        logger.error("[investments-chat] LLM call failed: %s", exc)
        yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"


@router.post("/chat")
async def investments_chat(
    req: InvestmentsChatRequest,
    db: Session = Depends(get_db),
):
    """
    Chat with ASTRA about your investments.
    Portfolio data is automatically injected as context.
    """
    context = await build_investment_context(db, req.message)

    return StreamingResponse(
        _stream_investment_chat(
            message=req.message,
            context=context,
            provider=req.provider,
            model=req.model,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
