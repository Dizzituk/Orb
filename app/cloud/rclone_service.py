# FILE: app/cloud/rclone_service.py
# Purpose: Cloud storage service via rclone.
# Called-by: app.cloud.build_and_deploy, app.cloud.router, app.debug.executors.user_files
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Cloud storage service via rclone.

Wraps rclone CLI commands to provide read/write access to cloud storage.
Requires: rclone installed + configured with a remote (default: gdrive).

Setup: run `rclone config` and add a remote. Set ASTRA_CLOUD_REMOTE env var to override.

v1.0 (2026-03-14): Initial — list, download, upload, delete, mkdir, info.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# The rclone remote name (default: Google Drive; override with ASTRA_CLOUD_REMOTE)
REMOTE_NAME = os.getenv("ASTRA_CLOUD_REMOTE", "gdrive")
RCLONE_BIN = os.getenv("RCLONE_PATH", "rclone")


def _transfer_flags() -> List[str]:
    """Reliability flags for large transfers (LANE E fix, 2026-07-03).

    A 107MB APK upload to gdrive stalled for 27 min at near-zero CPU: the bare
    `copyto` had no connection/idle timeouts, so a half-open connection just
    hung until the caller's outer timeout (and one such stall even escaped the
    kill as an orphan). These make rclone fail-fast on a stall and retry
    cleanly, so the caller's timeout is a backstop, not the primary path.
    --drive-chunk-size is a gdrive backend flag (registered globally; ignored
    by other remotes), fewer chunks = fewer stall points for a 100MB+ file.
    """
    return [
        "--contimeout", os.getenv("RCLONE_CONTIMEOUT", "30s"),
        "--timeout", os.getenv("RCLONE_IO_TIMEOUT", "120s"),
        "--retries", os.getenv("RCLONE_RETRIES", "3"),
        "--low-level-retries", os.getenv("RCLONE_LOW_LEVEL_RETRIES", "10"),
        "--drive-chunk-size", os.getenv("RCLONE_DRIVE_CHUNK_SIZE", "16M"),
    ]

# Local cache directory for downloaded files
CACHE_DIR = Path(os.getenv("ASTRA_CLOUD_CACHE", "D:/Orb/data/cloud_cache"))


async def _run_rclone(args: List[str], timeout: int = 60) -> Dict[str, Any]:
    """Run an rclone command and return stdout/stderr/exit_code."""
    cmd = [RCLONE_BIN] + args
    logger.debug("[cloud] Running: %s", " ".join(cmd))
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return {
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
            "exit_code": proc.returncode,
        }
    except asyncio.TimeoutError:
        # Kill the child on timeout — otherwise rclone keeps running (and
        # uploading) in the background after we've reported failure. That's
        # exactly what happened on 2026-07-02: a 107MB APK upload was declared
        # failed at 300s, completed anyway as an orphan, and the caller's
        # fallback re-uploaded the same file in parallel.
        try:
            proc.kill()
            await proc.communicate()
        except Exception:
            pass
        return {"stdout": "", "stderr": f"Command timed out after {timeout}s", "exit_code": -1}
    except FileNotFoundError:
        return {"stdout": "", "stderr": "rclone not found. Install with: winget install Rclone.Rclone", "exit_code": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "exit_code": -1}


def _remote_path(cloud_path: str = "") -> str:
    """Build a full remote path like 'proton:path/to/file'."""
    clean = cloud_path.strip("/")
    if clean:
        return f"{REMOTE_NAME}:{clean}"
    return f"{REMOTE_NAME}:"


async def is_configured() -> Dict[str, Any]:
    """Check if the rclone proton remote is configured."""
    result = await _run_rclone(["listremotes"], timeout=10)
    if result["exit_code"] != 0:
        return {"configured": False, "error": result["stderr"]}
    remotes = [r.strip().rstrip(":") for r in result["stdout"].strip().split("\n") if r.strip()]
    configured = REMOTE_NAME in remotes
    return {
        "configured": configured,
        "remote_name": REMOTE_NAME,
        "remotes_found": remotes,
        "message": "Proton Drive connected" if configured else (
            f"Remote '{REMOTE_NAME}' not found. Run: rclone config"
        ),
    }


async def list_files(cloud_path: str = "", recursive: bool = False) -> List[Dict[str, Any]]:
    """List files and directories at a cloud path.
    
    Returns list of dicts with: name, path, size, mod_time, is_dir, mime_type.
    """
    args = ["lsjson", _remote_path(cloud_path)]
    if not recursive:
        args.append("--no-modtime")  # faster for non-recursive
    else:
        args.append("-R")

    result = await _run_rclone(args, timeout=30)
    if result["exit_code"] != 0:
        logger.warning("[cloud] list_files failed: %s", result["stderr"])
        return []

    try:
        items = json.loads(result["stdout"])
        return [
            {
                "name": item.get("Name", ""),
                "path": (cloud_path.rstrip("/") + "/" + item["Path"]).lstrip("/") if cloud_path else item["Path"],
                "size": item.get("Size", 0),
                "mod_time": item.get("ModTime", ""),
                "is_dir": item.get("IsDir", False),
                "mime_type": item.get("MimeType", ""),
            }
            for item in items
        ]
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("[cloud] Failed to parse lsjson output: %s", e)
        return []


async def download_file(cloud_path: str, local_dest: Optional[str] = None) -> Dict[str, Any]:
    """Download a file from Proton Drive to local filesystem.
    
    If no local_dest, downloads to the cloud cache directory.
    Returns: local_path, size, success.
    """
    if not local_dest:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        filename = Path(cloud_path).name
        local_dest = str(CACHE_DIR / filename)

    # Ensure parent directory exists
    Path(local_dest).parent.mkdir(parents=True, exist_ok=True)

    result = await _run_rclone(
        ["copyto", _remote_path(cloud_path), local_dest] + _transfer_flags(),
        timeout=120,
    )

    if result["exit_code"] == 0 and Path(local_dest).exists():
        size = Path(local_dest).stat().st_size
        logger.info("[cloud] Downloaded %s -> %s (%d bytes)", cloud_path, local_dest, size)
        return {"success": True, "local_path": local_dest, "size": size}
    else:
        logger.warning("[cloud] Download failed: %s", result["stderr"])
        return {"success": False, "error": result["stderr"]}


async def upload_file(local_path: str, cloud_dest: str, timeout: int = 300) -> Dict[str, Any]:
    """Upload a local file to the cloud remote.
    
    cloud_dest should be the full cloud path including filename.
    """
    if not Path(local_path).exists():
        return {"success": False, "error": f"Local file not found: {local_path}"}

    result = await _run_rclone(
        ["copyto", local_path, _remote_path(cloud_dest)] + _transfer_flags(),
        timeout=timeout,  # callers moving big files (APKs) pass a bigger value
    )

    if result["exit_code"] == 0:
        size = Path(local_path).stat().st_size
        logger.info("[cloud] Uploaded %s -> %s (%d bytes)", local_path, cloud_dest, size)
        return {"success": True, "cloud_path": cloud_dest, "size": size}
    else:
        logger.warning("[cloud] Upload failed: %s", result["stderr"])
        return {"success": False, "error": result["stderr"]}


async def delete_file(cloud_path: str) -> Dict[str, Any]:
    """Delete a file from Proton Drive."""
    result = await _run_rclone(
        ["deletefile", _remote_path(cloud_path)],
        timeout=30,
    )
    if result["exit_code"] == 0:
        logger.info("[cloud] Deleted: %s", cloud_path)
        return {"success": True, "deleted": cloud_path}
    else:
        return {"success": False, "error": result["stderr"]}


async def mkdir(cloud_path: str) -> Dict[str, Any]:
    """Create a directory on Proton Drive."""
    result = await _run_rclone(
        ["mkdir", _remote_path(cloud_path)],
        timeout=15,
    )
    if result["exit_code"] == 0:
        logger.info("[cloud] Created directory: %s", cloud_path)
        return {"success": True, "path": cloud_path}
    else:
        return {"success": False, "error": result["stderr"]}


async def get_storage_info() -> Dict[str, Any]:
    """Get storage usage information from Proton Drive."""
    result = await _run_rclone(
        ["about", f"{REMOTE_NAME}:", "--json"],
        timeout=15,
    )
    if result["exit_code"] == 0:
        try:
            info = json.loads(result["stdout"])
            return {
                "total": info.get("total"),
                "used": info.get("used"),
                "free": info.get("free"),
                "trashed": info.get("trashed"),
            }
        except json.JSONDecodeError:
            return {"error": "Failed to parse storage info"}
    return {"error": result["stderr"]}


async def sync_directory(local_dir: str, cloud_dir: str, dry_run: bool = False) -> Dict[str, Any]:
    """Sync a local directory to Proton Drive (one-way: local -> cloud).
    
    Use dry_run=True to preview what would be synced.
    """
    args = ["sync", local_dir, _remote_path(cloud_dir), "--progress"]
    if dry_run:
        args.append("--dry-run")

    result = await _run_rclone(args, timeout=600)
    return {
        "success": result["exit_code"] == 0,
        "dry_run": dry_run,
        "output": result["stdout"][-2000:] if result["stdout"] else "",
        "errors": result["stderr"][-1000:] if result["stderr"] else "",
    }
