# FILE: app/settings/router.py
"""
Settings API Router.

Endpoints for managing API keys and system configuration
via the UI instead of editing .env files.
"""
import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/settings",
    tags=["Settings"],
    dependencies=[Depends(require_auth)],
)


# ─── REQUEST SCHEMAS ───

class SetApiKeyRequest(BaseModel):
    key_name: str
    key_value: str


class ToggleApiKeyRequest(BaseModel):
    key_name: str
    active: bool


# ═══════════════════════════════════════════════════
# API KEY MANAGEMENT
# ═══════════════════════════════════════════════════

@router.get("/api-keys", response_model=list)
def list_api_keys(db: Session = Depends(get_db)):
    """
    Get status of all API keys.
    Shows which are configured, their source (settings vs .env),
    and masked values. Never returns actual key values.
    """
    from app.settings.service import get_all_keys_status
    return get_all_keys_status(db)


@router.post("/api-keys", response_model=dict)
def set_api_key(
    body: SetApiKeyRequest,
    db: Session = Depends(get_db),
):
    """
    Store or update an API key.
    Encrypts at rest and syncs to environment immediately.
    """
    from app.settings.service import set_api_key as svc_set, mask_key

    try:
        entry = svc_set(db, body.key_name, body.key_value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "key_name": entry.key_name,
        "masked_value": mask_key(entry.key_value),
        "active": entry.active,
        "message": f"API key '{body.key_name}' saved and active",
    }


@router.delete("/api-keys/{key_name}", response_model=dict)
def remove_api_key(
    key_name: str,
    db: Session = Depends(get_db),
):
    """Remove an API key from the database and environment."""
    from app.settings.service import remove_api_key as svc_remove

    removed = svc_remove(db, key_name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Key '{key_name}' not found")

    return {
        "key_name": key_name,
        "message": f"API key '{key_name}' removed",
    }


@router.post("/api-keys/toggle", response_model=dict)
def toggle_api_key(
    body: ToggleApiKeyRequest,
    db: Session = Depends(get_db),
):
    """Enable or disable an API key without deleting it."""
    from app.settings.service import toggle_api_key as svc_toggle

    entry = svc_toggle(db, body.key_name, body.active)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Key '{body.key_name}' not found")

    return {
        "key_name": entry.key_name,
        "active": entry.active,
        "message": f"API key '{body.key_name}' {'enabled' if body.active else 'disabled'}",
    }


@router.post("/api-keys/{key_name}/verify", response_model=dict)
async def verify_api_key(
    key_name: str,
    db: Session = Depends(get_db),
):
    """
    Test an API key by making a lightweight call to the provider.
    Returns verification status (valid/invalid/error).
    """
    from app.settings.service import verify_api_key as svc_verify

    result = await svc_verify(db, key_name)
    return result


@router.post("/api-keys/verify-all", response_model=dict)
async def verify_all_keys(
    db: Session = Depends(get_db),
):
    """Verify all configured API keys."""
    from app.settings.service import get_all_keys_status, verify_api_key as svc_verify

    keys = get_all_keys_status(db)
    results = []
    for key_info in keys:
        if key_info["is_set"]:
            result = await svc_verify(db, key_info["key_name"])
            results.append(result)

    return {
        "verified": len(results),
        "results": results,
    }


# ═══════════════════════════════════════════════════
# SYSTEM INFO
# ═══════════════════════════════════════════════════

@router.get("/system", response_model=dict)
def system_info():
    """Get system configuration summary."""
    import shutil
    import os

    return {
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "encryption_active": _check_encryption(),
        "database": os.getenv("ORB_DATABASE_URL", "sqlite:///./data/orb_memory.db"),
        "phase4_enabled": os.getenv("ORB_ENABLE_PHASE4", "false").lower() == "true",
    }


def _check_encryption() -> bool:
    """Check if encryption is initialized."""
    try:
        from app.crypto import is_encryption_ready
        return is_encryption_ready()
    except Exception:
        return False
