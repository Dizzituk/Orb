# FILE: app/cloud/router.py
# Purpose: Cloud API endpoints — Proton Drive via rclone.
# Called-by: main
# Depends-on: app.auth, app.cloud.build_and_deploy, app.cloud.gdrive_auth, app.cloud.gdrive_service (+1 more)
# Last-renovated: 2026-06-11
"""
Cloud API endpoints — Proton Drive via rclone.

Endpoints:
  GET  /cloud/status           — Check rclone + Proton config
  GET  /cloud/files             — List files at a path
  GET  /cloud/storage           — Storage usage info
  POST /cloud/upload            — Upload a local file to cloud
  POST /cloud/download          — Download a cloud file locally
  POST /cloud/mkdir             — Create a cloud directory
  DELETE /cloud/files            — Delete a cloud file
  POST /cloud/sync              — Sync a local directory to cloud

v1.0 (2026-03-14): Initial implementation.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.auth import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cloud", tags=["Cloud"], dependencies=[Depends(require_auth)])


# ── Request models ──

class UploadRequest(BaseModel):
    local_path: str
    cloud_path: str


class DownloadRequest(BaseModel):
    cloud_path: str
    local_dest: Optional[str] = None


class MkdirRequest(BaseModel):
    path: str


class DeleteRequest(BaseModel):
    path: str


class SyncRequest(BaseModel):
    local_dir: str
    cloud_dir: str
    dry_run: bool = False


# ── Endpoints ──

@router.get("/status")
async def cloud_status():
    """Check all cloud provider statuses."""
    from app.cloud.rclone_service import is_configured as proton_check
    proton = await proton_check()

    gdrive_configured = False
    try:
        from app.cloud.gdrive_service import is_configured as gdrive_check
        gdrive_configured = gdrive_check()
    except Exception:
        pass

    return {
        "proton": proton,
        "google_drive": {
            "configured": gdrive_configured,
            "message": "Google Drive connected" if gdrive_configured else "Google Drive not authenticated",
        },
        # Legacy compatibility
        "configured": proton.get("configured", False) or gdrive_configured,
        "message": proton.get("message", ""),
    }


@router.get("/files")
async def cloud_list_files(
    path: str = Query("", description="Cloud directory path"),
    recursive: bool = Query(False, description="List recursively"),
):
    """List files and directories at a cloud path."""
    from app.cloud.rclone_service import list_files
    items = await list_files(path, recursive=recursive)
    return {
        "path": path or "/",
        "count": len(items),
        "items": items,
    }


@router.get("/storage")
async def cloud_storage_info():
    """Get cloud storage usage."""
    from app.cloud.rclone_service import get_storage_info
    return await get_storage_info()


@router.post("/upload")
async def cloud_upload(req: UploadRequest):
    """Upload a local file to Proton Drive."""
    from app.cloud.rclone_service import upload_file
    return await upload_file(req.local_path, req.cloud_path)


@router.post("/download")
async def cloud_download(req: DownloadRequest):
    """Download a file from Proton Drive to local filesystem."""
    from app.cloud.rclone_service import download_file
    return await download_file(req.cloud_path, req.local_dest)


@router.post("/mkdir")
async def cloud_mkdir(req: MkdirRequest):
    """Create a directory on Proton Drive."""
    from app.cloud.rclone_service import mkdir
    return await mkdir(req.path)


@router.delete("/files")
async def cloud_delete(req: DeleteRequest):
    """Delete a file from Proton Drive."""
    from app.cloud.rclone_service import delete_file
    return await delete_file(req.path)


@router.post("/sync")
async def cloud_sync(req: SyncRequest):
    """Sync a local directory to Proton Drive."""
    from app.cloud.rclone_service import sync_directory
    return await sync_directory(req.local_dir, req.cloud_dir, req.dry_run)


@router.get("/gdrive/auth-status")
def gdrive_auth_status():
    """Check Google Drive authentication status."""
    from app.cloud.gdrive_auth import check_auth
    return check_auth()


@router.post("/gdrive/auth-start")
def gdrive_auth_start():
    """Start Google Drive OAuth flow (opens browser)."""
    from app.cloud.gdrive_auth import run_auth_flow
    success = run_auth_flow()
    if success:
        return {"status": "ok", "message": "Google Drive authenticated"}
    return {"status": "error", "message": "Auth flow failed"}


class BuildDeployRequest(BaseModel):
    text: str = ""  # Natural language (e.g. "build the bridge app")
    target_id: Optional[str] = None  # Explicit target override


@router.post("/build-deploy")
async def cloud_build_deploy(req: BuildDeployRequest):
    """Build an Android APK and deploy to Proton Drive."""
    from app.cloud.build_and_deploy import build_and_deploy
    return await build_and_deploy(text=req.text, target_id=req.target_id)
