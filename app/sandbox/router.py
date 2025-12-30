# FILE: app/sandbox/router.py
"""Sandbox Router: API endpoints for controlling the zombie Orb.

Endpoints:
- POST /sandbox/start - Start the zombie clone
- POST /sandbox/stop - Stop the zombie clone
- GET /sandbox/status - Get clone status
- GET /sandbox/health - Detailed health check
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.sandbox.manager import (
    get_sandbox_manager,
    SandboxStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sandbox", tags=["sandbox"])


# =============================================================================
# Request/Response Models
# =============================================================================

class StartRequest(BaseModel):
    """Request to start sandbox clone."""
    full_mode: bool = True  # True = backend + frontend, False = backend only


class StartResponse(BaseModel):
    """Response from start operation."""
    success: bool
    message: str


class StopResponse(BaseModel):
    """Response from stop operation."""
    success: bool
    message: str


class StatusResponse(BaseModel):
    """Status response."""
    status: str
    message: str
    controller_connected: bool
    backend_running: bool


class HealthResponse(BaseModel):
    """Detailed health response."""
    status: str
    controller_ok: bool
    backend_ok: bool
    controller_url: str
    backend_url: str
    message: str
    details: Optional[dict] = None


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/start", response_model=StartResponse)
def start_sandbox(request: StartRequest = StartRequest()):
    """Start the zombie Orb clone in sandbox.
    
    This will:
    1. Write a startup script to the sandbox
    2. Launch it in a visible PowerShell window
    3. Wait for the backend to respond
    
    Args:
        request: Start configuration (full_mode)
    
    Returns:
        Success status and message
    """
    logger.info(f"Starting sandbox clone (full_mode={request.full_mode})")
    
    manager = get_sandbox_manager()
    success, message = manager.start_clone(full=request.full_mode)
    
    return StartResponse(success=success, message=message)


@router.post("/stop", response_model=StopResponse)
def stop_sandbox():
    """Stop the zombie Orb clone.
    
    This will kill the Orb processes in the sandbox but leave
    the sandbox controller running.
    
    Returns:
        Success status and message
    """
    logger.info("Stopping sandbox clone")
    
    manager = get_sandbox_manager()
    success, message = manager.stop_clone()
    
    return StopResponse(success=success, message=message)


@router.get("/status", response_model=StatusResponse)
def get_status():
    """Get the current status of the sandbox clone.
    
    Returns:
        Status information including connection and running state
    """
    manager = get_sandbox_manager()
    health = manager.check_health()
    
    return StatusResponse(
        status=health.status.value,
        message=health.message,
        controller_connected=health.controller_ok,
        backend_running=health.backend_ok,
    )


@router.get("/health", response_model=HealthResponse)
def get_health():
    """Get detailed health information about the sandbox.
    
    Returns:
        Comprehensive health check results
    """
    manager = get_sandbox_manager()
    health = manager.check_health()
    
    return HealthResponse(
        status=health.status.value,
        controller_ok=health.controller_ok,
        backend_ok=health.backend_ok,
        controller_url=health.controller_url,
        backend_url=health.backend_url,
        message=health.message,
        details=health.details,
    )


__all__ = ["router"]
