# FILE: app/llm/web_search_router.py
"""
FastAPI router for local-only web search.

Endpoints:
- GET  /web_search/health
- POST /web_search
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.auth import optional_auth
from app.llm.web_search import WebSearchRequest, WebSearchResponse, search_and_answer

router = APIRouter(tags=["Web Search"])


@router.get("/web_search/health")
async def health() -> dict:
    return {"ok": True, "service": "web_search_router"}


@router.post("/web_search", response_model=WebSearchResponse)
async def web_search(req: WebSearchRequest, auth=Depends(optional_auth)) -> WebSearchResponse:
    # Keep context minimal and safe. Only pass a user identifier if the auth object exposes it.
    user_id = getattr(auth, "user_id", None) or getattr(auth, "username", None) or None
    ctx = {"user_id": user_id} if user_id else {}
    return await search_and_answer(req, context=ctx)
