# app/auth/router.py
"""
Authentication API endpoints for password-based auth.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import Optional

from . import config
from .middleware import require_auth, AuthResult

router = APIRouter(prefix="/auth", tags=["auth"])


# ============ Request/Response Models ============

class AuthStatusResponse(BaseModel):
    configured: bool
    auth_type: str  # "none", "password", "api_key" (legacy)
    needs_migration: bool = False


class SetupPasswordRequest(BaseModel):
    password: str = Field(..., min_length=4, description="Password (min 4 characters)")


class LoginRequest(BaseModel):
    password: str


class LoginResponse(BaseModel):
    session_token: str
    message: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=4)


class MigrateRequest(BaseModel):
    api_key: str
    new_password: str = Field(..., min_length=4)


# ============ Public Endpoints (no auth required) ============

@router.get("/status", response_model=AuthStatusResponse)
async def get_auth_status():
    """Check if authentication is configured and what type."""
    if config.is_auth_configured():
        return AuthStatusResponse(
            configured=True,
            auth_type="password",
            needs_migration=False
        )
    elif config.is_legacy_api_key_auth():
        return AuthStatusResponse(
            configured=True,
            auth_type="api_key",
            needs_migration=True
        )
    else:
        return AuthStatusResponse(
            configured=False,
            auth_type="none",
            needs_migration=False
        )


@router.post("/setup", response_model=LoginResponse)
async def setup_password(request: SetupPasswordRequest):
    """
    Set up password authentication (first-time setup only).
    Returns a session token on success.
    """
    if config.is_auth_configured():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password already configured. Use /auth/change-password to change it."
        )
    
    if config.is_legacy_api_key_auth():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="API key auth exists. Use /auth/migrate to switch to password."
        )
    
    try:
        result = config.setup_password(request.password)
        return LoginResponse(
            session_token=result["session_token"],
            message=result["message"]
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Log in with password.
    Returns a session token on success.
    """
    if not config.is_auth_configured():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password not configured. Use /auth/setup first."
        )
    
    result = config.login(request.password)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password"
        )
    
    return LoginResponse(
        session_token=result["session_token"],
        message=result["message"]
    )


@router.post("/migrate", response_model=LoginResponse)
async def migrate_to_password(request: MigrateRequest):
    """
    Migrate from legacy API key auth to password auth.
    Requires valid API key and new password.
    """
    if not config.is_legacy_api_key_auth():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No API key auth to migrate from"
        )
    
    result = config.migrate_to_password(request.api_key, request.new_password)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return LoginResponse(
        session_token=result["session_token"],
        message="Migration successful. Now using password authentication."
    )


# ============ Protected Endpoints (auth required) ============

@router.post("/logout")
async def logout(auth: AuthResult = Depends(require_auth)):
    """Log out (invalidate current session)."""
    config.logout()
    return {"message": "Logged out successfully"}


@router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    auth: AuthResult = Depends(require_auth)
):
    """Change password. Requires current password and logs out after."""
    try:
        success = config.change_password(
            request.current_password,
            request.new_password
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect"
        )
    
    return {"message": "Password changed. Please log in again."}


@router.get("/check")
async def check_auth(auth: AuthResult = Depends(require_auth)):
    """Check if current session is valid."""
    return {"authenticated": True}
