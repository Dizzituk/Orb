# app/auth/middleware.py
"""
FastAPI authentication middleware using session tokens.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from dataclasses import dataclass

from . import config

security = HTTPBearer(auto_error=False)


@dataclass
class AuthResult:
    """Result of authentication check."""
    authenticated: bool
    error: Optional[str] = None


async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> AuthResult:
    """
    Dependency that requires valid authentication.
    Supports both session tokens (new) and API keys (legacy).
    
    Raises:
        HTTPException 401: If authentication fails
        HTTPException 503: If auth not configured
    """
    # Check if auth is configured at all
    if not config.is_auth_configured() and not config.is_legacy_api_key_auth():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication not configured. Please set up a password.",
            headers={"X-Auth-Status": "not_configured"}
        )
    
    # Check for credentials
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    token = credentials.credentials
    
    # Try session token first (new auth)
    if token.startswith("orb_session_"):
        if config.validate_session(token):
            return AuthResult(authenticated=True)
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session. Please log in again.",
                headers={"WWW-Authenticate": "Bearer"}
            )
    
    # Try legacy API key (for migration period)
    if token.startswith("orb_"):
        if config.validate_api_key(token):
            return AuthResult(authenticated=True)
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"}
            )
    
    # Unknown token format
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication token",
        headers={"WWW-Authenticate": "Bearer"}
    )


async def optional_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> AuthResult:
    """
    Dependency that checks auth without requiring it.
    Returns AuthResult with authenticated=True/False.
    """
    if not credentials:
        return AuthResult(authenticated=False, error="No credentials provided")
    
    token = credentials.credentials
    
    # Try session token
    if token.startswith("orb_session_"):
        if config.validate_session(token):
            return AuthResult(authenticated=True)
        return AuthResult(authenticated=False, error="Invalid session")
    
    # Try legacy API key
    if token.startswith("orb_"):
        if config.validate_api_key(token):
            return AuthResult(authenticated=True)
        return AuthResult(authenticated=False, error="Invalid API key")
    
    return AuthResult(authenticated=False, error="Unknown token format")
