# FILE: app/bridge/log_uploads.py
# Purpose: Receive AstraBridge debug-log exports from the phone so bug logs
#          land on the PC's disk (data/bridge_logs/) where Claude, Claude Code
#          and ASTRA's own tooling can read them directly. No third-party hop.
# Called-by: main.py (mounted beside the bridge router)
# Depends-on: app.bridge.schemas (require_bridge_auth)
"""POST /bridge/logs/upload  — multipart file field 'file' (.jsonl/.zip/.txt)
GET  /bridge/logs/recent     — newest stored logs, for sanity checks.

Retention: newest 40 files kept; 20MB per-file cap. Filenames are
timestamped server-side and sanitised — the phone can call it whatever it
likes."""
from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from .schemas import require_bridge_auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bridge/logs", tags=["bridge-logs"])

LOG_DIR = Path("data/bridge_logs")
MAX_BYTES = 20 * 1024 * 1024
KEEP_NEWEST = 40
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _retention() -> None:
    files = sorted(LOG_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[KEEP_NEWEST:]:
        try:
            old.unlink()
        except OSError:  # pragma: no cover - defensive
            pass


@router.post("/upload")
async def upload_bridge_log(
    file: UploadFile = File(...),
    _auth: None = Depends(require_bridge_auth),
) -> dict:
    raw = await file.read()
    if not raw:
        raise HTTPException(422, "empty file")
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, f"log too large ({len(raw)} bytes, cap {MAX_BYTES})")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    safe = _SAFE.sub("_", file.filename or "bridge_log")[-80:]
    name = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{safe}"
    path = LOG_DIR / name
    path.write_bytes(raw)
    _retention()
    logger.info("[bridge_logs] stored %s (%d bytes)", name, len(raw))
    return {"ok": True, "stored_as": name, "bytes": len(raw),
            "path": str(path.resolve())}


@router.get("/recent")
async def recent_bridge_logs(_auth: None = Depends(require_bridge_auth)) -> dict:
    if not LOG_DIR.exists():
        return {"files": []}
    files = sorted(LOG_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {"files": [
        {"name": p.name, "bytes": p.stat().st_size,
         "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")}
        for p in files[:15]
    ]}
