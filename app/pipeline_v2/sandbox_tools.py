# FILE: app/pipeline_v2/sandbox_tools.py
"""
Sandbox Tools — clean interface for pipeline stages to interact with the sandbox.

All sandbox operations go through here. No direct httpx calls from stages.
The Builder uses read/write/shell. The Verifier uses screenshot/boot.
"""
from __future__ import annotations

import base64
import logging
from typing import Optional

from app.pipeline_v2.config import SANDBOX_URL

logger = logging.getLogger(__name__)


async def _sandbox_post(endpoint: str, json_body: dict, timeout: float = 15.0) -> Optional[dict]:
    """POST to sandbox API. Returns parsed JSON or None on failure."""
    import httpx
    url = f"{SANDBOX_URL}{endpoint}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=json_body)
            if resp.status_code == 200:
                return resp.json()
            logger.error("[sandbox] POST %s → HTTP %d", endpoint, resp.status_code)
            return None
    except Exception as e:
        logger.error("[sandbox] POST %s failed: %s", endpoint, e)
        return None


async def _sandbox_get(endpoint: str, timeout: float = 10.0) -> Optional[dict]:
    """GET from sandbox API."""
    import httpx
    url = f"{SANDBOX_URL}{endpoint}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json()
            return None
    except Exception as e:
        logger.error("[sandbox] GET %s failed: %s", endpoint, e)
        return None


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------

async def read_file(path: str) -> Optional[str]:
    """Read a file from the sandbox. Returns content or None.

    Uses /repo/file GET endpoint which accepts relative paths.
    Falls back to /fs/contents POST with absolute path.
    """
    import urllib.parse

    norm = path.replace("\\", "/")

    # Try /repo/file first (accepts relative paths like app/debug/models.py)
    # Strip absolute prefixes to get relative
    rel = norm
    for prefix in ("D:/Orb/", "D:/orb-desktop/"):
        if rel.lower().startswith(prefix.lower()):
            rel = rel[len(prefix):]
            break

    encoded = urllib.parse.quote(rel, safe="")
    data = await _sandbox_get(f"/repo/file?path={encoded}")
    if data and "content" in data:
        return data["content"]

    # Fallback: ensure absolute path for /fs/contents
    if not (len(norm) > 1 and norm[1] == ":"):
        if norm.startswith("src/"):
            norm = f"D:/orb-desktop/{norm}"
        else:
            norm = f"D:/Orb/{norm}"

    data = await _sandbox_post("/fs/contents", {"paths": [norm], "max_file_size": 500000})
    if data:
        files = data.get("files", [])
        if files and not files[0].get("error") and files[0].get("content") is not None:
            return files[0]["content"]

    return None


async def write_file(path: str, content: str) -> bool:
    """Write a file to the sandbox. Returns True on success.

    The sandbox /fs/write endpoint expects:
        path: absolute path (e.g. D:/Orb/app/debug/models.py)
        content: raw text content
    """
    norm = path.replace("\\", "/")

    # Ensure absolute path — resolve relative paths to D:/Orb or D:/orb-desktop
    if not (len(norm) > 1 and norm[1] == ":"):
        if norm.startswith("src/") or norm.startswith("orb-desktop/"):
            if norm.startswith("orb-desktop/"):
                norm = norm[len("orb-desktop/"):]
            norm = f"D:/orb-desktop/{norm}"
        else:
            norm = f"D:/Orb/{norm}"

    data = await _sandbox_post("/fs/write", {"path": norm, "content": content})
    return data is not None and data.get("status") == "ok"


async def file_exists(path: str) -> bool:
    """Check if a file exists in the sandbox."""
    norm = path.replace("\\", "/")
    if not (len(norm) > 1 and norm[1] == ":"):
        if norm.startswith("src/"):
            norm = f"D:/orb-desktop/{norm}"
        else:
            norm = f"D:/Orb/{norm}"
    data = await _sandbox_post("/fs/contents", {"paths": [norm], "max_file_size": 500000})
    if data:
        files = data.get("files", [])
        return bool(files and not files[0].get("error"))
    return False


async def list_dir(path: str) -> Optional[list]:
    """List directory contents in the sandbox."""
    norm = path.replace("\\", "/")
    if not (len(norm) > 1 and norm[1] == ":"):
        if norm.startswith("src/"):
            norm = f"D:/orb-desktop/{norm}"
        else:
            norm = f"D:/Orb/{norm}"
    data = await _sandbox_post("/fs/tree", {"roots": [norm], "max_files": 500, "include_size": True})
    if data:
        return data.get("files", [])
    return None


# ---------------------------------------------------------------------------
# Shell execution
# ---------------------------------------------------------------------------

async def run_shell(cmd: str, timeout_sec: int = 30) -> dict:
    """Run a shell command in the sandbox.

    Returns dict with: stdout, stderr, returncode.
    """
    data = await _sandbox_post(
        "/shell/run",
        {"cmd": ["powershell", "-Command", cmd], "timeout_sec": timeout_sec},
        timeout=float(timeout_sec + 5),
    )
    if data:
        return {
            "stdout": data.get("stdout", ""),
            "stderr": data.get("stderr", ""),
            "returncode": data.get("returncode", -1),
        }
    return {"stdout": "", "stderr": "sandbox unreachable", "returncode": -1}


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

async def check_python_syntax(file_path: str) -> tuple:
    """Check Python file syntax via sandbox. Returns (ok, error_msg)."""
    sandbox_path = file_path.replace("/", "\\")
    if not sandbox_path.startswith("D:"):
        sandbox_path = f"D:\\Orb\\{sandbox_path}"
    cmd = (
        f'& "D:\\Orb\\.venv\\Scripts\\python.exe" -c '
        f'"import py_compile; py_compile.compile(r\'{sandbox_path}\', doraise=True); print(\'OK\')"'
    )
    result = await run_shell(cmd, timeout_sec=10)
    if "OK" in result["stdout"]:
        return True, ""
    return False, result["stderr"][:300]


async def boot_check() -> tuple:
    """Run backend boot check. Returns (ok, output)."""
    cmd = (
        'cd "D:\\Orb" ; '
        '& "D:\\Orb\\.venv\\Scripts\\python.exe" -c '
        '"import sys; sys.path.insert(0, r\'D:\\Orb\'); '
        'from app.db import init_db; init_db(); '
        'from main import app; print(\'BOOT_CHECK_PASS\')"'
    )
    result = await run_shell(cmd, timeout_sec=30)
    ok = "BOOT_CHECK_PASS" in result["stdout"]
    return ok, result["stdout"][:500]


async def build_check() -> tuple:
    """Run frontend TypeScript build check. Returns (ok, output)."""
    cmd = 'cd "D:\\orb-desktop" ; npx tsc --noEmit 2>&1'
    result = await run_shell(cmd, timeout_sec=55)
    ok = result["returncode"] == 0
    return ok, result["stdout"][:500]


async def is_sandbox_alive() -> bool:
    """Quick health check."""
    data = await _sandbox_get("/health", timeout=5.0)
    return data is not None
