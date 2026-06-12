# Purpose: Authentication module for Orb.
# Called-by: app.astra_memory.router, app.auth.middleware, app.auth.router, app.bridge.router (+41 more)
# Depends-on: app.auth.config, app.auth.middleware, app.auth.router
# Last-renovated: 2026-06-11
# app/auth/__init__.py
"""
Authentication module for Orb.
Supports password-based authentication with session tokens.
"""

from .middleware import require_auth, optional_auth, AuthResult
from .router import router as auth_router
from .config import (
    is_auth_configured,
    is_legacy_api_key_auth,
    setup_password,
    login,
    validate_session,
    logout,
    change_password,
    reset_auth,
    validate_api_key,
    migrate_to_password,
)

__all__ = [
    # Middleware
    "require_auth",
    "optional_auth", 
    "AuthResult",
    # Router
    "auth_router",
    # Config functions
    "is_auth_configured",
    "is_legacy_api_key_auth",
    "setup_password",
    "login",
    "validate_session",
    "logout",
    "change_password",
    "reset_auth",
    "validate_api_key",
    "migrate_to_password",
]
