# FILE: app/pipeline_v2/sandbox_tools.py
"""
Sandbox Tools — clean interface for pipeline stages to interact with the sandbox.

All sandbox operations go through here. No direct httpx calls from stages.
The Builder uses read/write/shell. The Verifier uses screenshot/boot.

v2.2 (2026-03-10): Profile-aware path resolution. All functions accept an
    optional BuildTargetProfile. When provided, relative paths resolve against
    the profile's project_root instead of hardcoded D:/Orb or D:/orb-desktop.
    Backward-compatible: without a profile, old behaviour is preserved.
"""
from __future__ import annotations

import base64
import logging
from typing import Optional, TYPE_CHECKING

from app.pipeline_v2.config import SANDBOX_URL

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# HTTP helpers
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
# Path resolution
# ═══════════════════════════════════════════════════════════════════

def _resolve_path(
    path: str,
    profile: Optional["BuildTargetProfile"] = None,
) -> str:
    """Resolve a relative path to an absolute path.

    If a profile is provided, resolves against the profile's project_root.
    Otherwise, falls back to legacy logic (D:/Orb or D:/orb-desktop).
    """
    norm = path.replace("\\", "/")

    # Already absolute
    if len(norm) > 1 and norm[1] == ":":
        return norm

    # Profile-aware resolution
    if profile:
        return profile.resolve_path(norm)

    # Legacy fallback — keep old behaviour for backward compat
    if norm.startswith("src/") or norm.startswith("orb-desktop/"):
        if norm.startswith("orb-desktop/"):
            norm = norm[len("orb-desktop/"):]
        return f"D:/orb-desktop/{norm}"
    return f"D:/Orb/{norm}"


# ═══════════════════════════════════════════════════════════════════
# Host-mode file operations (for Android / external builds)
# ═══════════════════════════════════════════════════════════════════

async def _host_read_file(
    path: str,
    profile: Optional["BuildTargetProfile"] = None,
) -> Optional[str]:
    """Read a file directly from the host filesystem (no sandbox)."""
    import os
    abs_path = _resolve_path(path, profile)
    norm = abs_path.replace("/", os.sep)
    try:
        if os.path.isfile(norm):
            with open(norm, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        logger.debug("[host_read] File not found: %s", norm)
        return None
    except Exception as e:
        logger.error("[host_read] Failed to read %s: %s", norm, e)
        return None


async def _host_write_file(abs_path: str, content: str) -> bool:
    """Write a file directly to the host filesystem (no sandbox).

    Creates parent directories as needed.
    """
    import os
    norm = abs_path.replace("/", os.sep)
    try:
        os.makedirs(os.path.dirname(norm), exist_ok=True)
        with open(norm, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        logger.info("[host_write] Wrote %d chars to %s", len(content), norm)
        return True
    except Exception as e:
        logger.error("[host_write] Failed to write %s: %s", norm, e)
        return False


async def _host_file_exists(
    path: str,
    profile: Optional["BuildTargetProfile"] = None,
) -> bool:
    """Check if a file exists on the host filesystem."""
    import os
    abs_path = _resolve_path(path, profile)
    return os.path.exists(abs_path.replace("/", os.sep))


async def _host_list_dir(
    path: str,
    profile: Optional["BuildTargetProfile"] = None,
) -> Optional[list]:
    """List directory contents on the host filesystem."""
    import os
    abs_path = _resolve_path(path, profile)
    norm = abs_path.replace("/", os.sep)
    try:
        if not os.path.isdir(norm):
            return None
        entries = []
        for name in os.listdir(norm):
            full = os.path.join(norm, name)
            entries.append({
                "path": full,
                "name": name,
                "is_dir": os.path.isdir(full),
                "size": os.path.getsize(full) if os.path.isfile(full) else 0,
            })
        return entries
    except Exception as e:
        logger.error("[host_list] Failed to list %s: %s", norm, e)
        return None


# ═══════════════════════════════════════════════════════════════════
# Sandbox file operations (for ASTRA self-modification)
# ═══════════════════════════════════════════════════════════════════

async def read_file(
    path: str,
    profile: Optional["BuildTargetProfile"] = None,
) -> Optional[str]:
    """Read a file. For Android builds, reads directly from host.
    For ASTRA builds, reads from the sandbox."""
    import urllib.parse

    # v2.3: Host-mode read for Android builds
    if profile is not None:
        from app.pipeline_v2.android_sandbox import is_android_build
        if is_android_build(profile):
            return await _host_read_file(path, profile)

    norm = path.replace("\\", "/")

    # Try /repo/file first (accepts relative paths)
    rel = norm
    for prefix in ("D:/Orb/", "D:/orb-desktop/"):
        if rel.lower().startswith(prefix.lower()):
            rel = rel[len(prefix):]
            break
    # Also strip profile root if present
    if profile:
        proot = profile.project_root.replace("\\", "/").rstrip("/") + "/"
        if rel.lower().startswith(proot.lower()):
            rel = rel[len(proot):]

    encoded = urllib.parse.quote(rel, safe="")
    data = await _sandbox_get(f"/repo/file?path={encoded}")
    if data and "content" in data:
        return data["content"]

    # Fallback: ensure absolute path for /fs/contents
    abs_path = _resolve_path(norm, profile)
    data = await _sandbox_post("/fs/contents", {"paths": [abs_path], "max_file_size": 500000})
    if data:
        files = data.get("files", [])
        if files and not files[0].get("error") and files[0].get("content") is not None:
            return files[0]["content"]

    return None


async def write_file(
    path: str,
    content: str,
    profile: Optional["BuildTargetProfile"] = None,
) -> bool:
    """Write a file. For Android builds, writes directly to host filesystem.
    For ASTRA builds, writes through the sandbox.

    Android writes are path-validated to stay within the project root.
    """
    abs_path = _resolve_path(path, profile)

    # v2.3: Host-mode write for Android builds
    if profile is not None:
        from app.pipeline_v2.android_sandbox import is_android_build, validate_android_write_path
        if is_android_build(profile):
            abs_path = validate_android_write_path(abs_path, profile)
            return await _host_write_file(abs_path, content)

    data = await _sandbox_post("/fs/write", {"path": abs_path, "content": content})
    return data is not None and data.get("status") == "ok"


async def file_exists(
    path: str,
    profile: Optional["BuildTargetProfile"] = None,
) -> bool:
    """Check if a file exists. Host-mode for Android, sandbox for ASTRA."""
    if profile is not None:
        from app.pipeline_v2.android_sandbox import is_android_build
        if is_android_build(profile):
            return await _host_file_exists(path, profile)

    abs_path = _resolve_path(path, profile)
    data = await _sandbox_post("/fs/contents", {"paths": [abs_path], "max_file_size": 500000})
    if data:
        files = data.get("files", [])
        return bool(files and not files[0].get("error"))
    return False


async def list_dir(
    path: str,
    profile: Optional["BuildTargetProfile"] = None,
) -> Optional[list]:
    """List directory contents. Host-mode for Android, sandbox for ASTRA."""
    if profile is not None:
        from app.pipeline_v2.android_sandbox import is_android_build
        if is_android_build(profile):
            return await _host_list_dir(path, profile)

    abs_path = _resolve_path(path, profile)
    data = await _sandbox_post("/fs/tree", {"roots": [abs_path], "max_files": 500, "include_size": True})
    if data:
        return data.get("files", [])
    return None


# ═══════════════════════════════════════════════════════════════════
# Shell execution
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
# Verification helpers — profile-aware
# ═══════════════════════════════════════════════════════════════════

async def check_python_syntax(
    file_path: str,
    profile: Optional["BuildTargetProfile"] = None,
) -> tuple:
    """Check Python file syntax via sandbox. Returns (ok, error_msg)."""
    sandbox_path = _resolve_path(file_path, profile).replace("/", "\\")
    cmd = (
        f'& "D:\\Orb\\.venv\\Scripts\\python.exe" -c '
        f'"import py_compile; py_compile.compile(r\'{sandbox_path}\', doraise=True); print(\'OK\')"'
    )
    result = await run_shell(cmd, timeout_sec=10)
    if "OK" in result["stdout"]:
        return True, ""
    return False, result["stderr"][:300]


async def boot_check(
    profile: Optional["BuildTargetProfile"] = None,
) -> tuple:
    """Run boot check appropriate for the target. Returns (ok, output)."""
    if profile and profile.boot_cmd:
        result = await run_shell(profile.boot_cmd, timeout_sec=30)
        ok = "BOOT_CHECK_PASS" in result["stdout"] or "BOOT_OK" in result["stdout"]
        return ok, result["stdout"][:500]

    if profile and profile.boot_cmd is None:
        # No boot cmd (e.g. Android) — skip, return OK
        return True, "No boot check required for this target"

    # Legacy fallback
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


async def build_check(
    profile: Optional["BuildTargetProfile"] = None,
) -> tuple:
    """Run build/compilation check for the target. Returns (ok, output)."""
    if profile and profile.build_cmd:
        timeout = 120 if profile.build_system == "gradle" else 55
        result = await run_shell(profile.build_cmd, timeout_sec=timeout)
        ok = result["returncode"] == 0
        combined = result["stdout"][:500]
        if result["stderr"]:
            combined += "\n" + result["stderr"][:300]
        return ok, combined

    # Legacy fallback — TypeScript
    cmd = 'cd "D:\\orb-desktop" ; npx tsc --noEmit 2>&1'
    result = await run_shell(cmd, timeout_sec=55)
    ok = result["returncode"] == 0
    return ok, result["stdout"][:500]


async def syntax_check(
    file_path: str,
    profile: Optional["BuildTargetProfile"] = None,
) -> tuple:
    """Language-aware syntax check. Returns (ok, error_msg).

    For Python: per-file py_compile.
    For Kotlin: full module compilation via Gradle (no per-file check).
    For TypeScript: npx tsc --noEmit.
    """
    if profile:
        if profile.language == "python":
            return await check_python_syntax(file_path, profile)
        elif profile.language == "kotlin":
            # Gradle compiles the whole module — use compileDebugKotlin
            result = await run_shell(profile.syntax_check_cmd, timeout_sec=120)
            ok = result["returncode"] == 0
            combined = result["stdout"][:500]
            if result["stderr"]:
                combined += "\n" + result["stderr"][:300]
            return ok, combined
        elif profile.language == "typescript":
            result = await run_shell(profile.syntax_check_cmd, timeout_sec=55)
            ok = result["returncode"] == 0
            return ok, result["stdout"][:500]

    # Default to Python
    return await check_python_syntax(file_path)


async def is_sandbox_alive() -> bool:
    """Quick health check."""
    data = await _sandbox_get("/health", timeout=5.0)
    return data is not None
