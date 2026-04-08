# FILE: app/bridge/schemas.py
"""
Pydantic request/response models and auth helper for the Bridge API.

Extracted from router.py during modularisation (2026-04-06).
"""

from __future__ import annotations

from typing import Optional

from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from app.auth import config as auth_config


security = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class BridgeLoginRequest(BaseModel):
    password: str


class BridgeLoginResponse(BaseModel):
    session_token: str
    message: str


class BridgeChatRequest(BaseModel):
    message: str
    project_id: Optional[int] = None


class BridgeChatResponse(BaseModel):
    reply: str
    project_id: int
    project_name: str
    domain: Optional[str] = None


class BridgeProjectOut(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    type: Optional[str] = None
    created_at: str
    updated_at: str


class BridgeMessageOut(BaseModel):
    id: int
    role: str
    content: str
    provider: Optional[str] = None
    model: Optional[str] = None
    created_at: str


class BridgeTTSRequest(BaseModel):
    text: str
    voice_name: Optional[str] = None
    speed: Optional[float] = None


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def require_bridge_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> bool:
    """Validate session token from the bridge app."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required. Use /bridge/login first.",
        )
    token = credentials.credentials
    if auth_config.validate_session(token):
        return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired session. Please log in again via /bridge/login.",
    )
