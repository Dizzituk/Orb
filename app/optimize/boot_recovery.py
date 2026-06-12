# Purpose: Boot Recovery — Snapshot, smart repair, and hard revert for optimisation passes.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.pipeline_v2.llm_compat
# Last-renovated: 2026-06-11
"""
Boot Recovery — Snapshot, smart repair, and hard revert for optimisation passes.

When the optimizer runs a pass that breaks the sandbox boot:
1. Try smart recovery: feed the boot error to the agentic builder and let it fix
2. If smart recovery fails: hard revert all touched files from pre-pass snapshots
3. Either way: guarantee the sandbox is bootable when control returns

Usage:
    from app.optimize.boot_recovery import (
        snapshot_files_before_pass,
        attempt_recovery,
    )

v1.0 (2026-03-27): Initial implementation.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

SANDBOX_CONTROLLER_URL = "http://192.168.250.2:8765"
MAX_REPAIR_ATTEMPTS = 2


# ══════════════════════════════════════════════════════════
# SNAPSHOT — capture file contents before a pass starts
# ══════════════════════════════════════════════════════════

async def snapshot_files_before_pass(
    file_paths: List[str],
    target_root: str,
) -> Dict[str, Optional[str]]:
    """Read current content of files that will be touched during a pass.

    Returns a dict of {relative_path: content_string_or_None}.
    Files that don't exist yet map to None (they were created by the pass).
    """
    import httpx

    snapshots: Dict[str, Optional[str]] = {}
    if not file_paths:
        return snapshots

    # Batch-read via sandbox /fs/contents API
    abs_paths = [
        f"{target_root}/{p}".replace("/", "\\") if not p.startswith(target_root) else p
        for p in file_paths
    ]

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{SANDBOX_CONTROLLER_URL}/fs/contents",
                json={"paths": abs_paths},
            )
            data = resp.json()
        for file_info in data.get("files", []):
            rel = file_info.get("path", "").replace("\\", "/")
            # Normalise to relative path
            root_norm = target_root.replace("\\", "/").rstrip("/")
            if rel.startswith(root_norm + "/"):
                rel = rel[len(root_norm) + 1:]
            snapshots[rel] = file_info.get("content")
    except Exception as exc:
        logger.warning("[boot_recovery] Snapshot batch read failed: %s", exc)

    # Mark any requested paths we didn't get back as None (new files)
    for p in file_paths:
        if p not in snapshots:
            snapshots[p] = None

    logger.info(
        "[boot_recovery] Snapshot captured: %d files (%d existing, %d new)",
        len(snapshots),
        sum(1 for v in snapshots.values() if v is not None),
        sum(1 for v in snapshots.values() if v is None),
    )
    return snapshots


def collect_target_paths_from_proposals(proposals: List[Any]) -> List[str]:
    """Extract all file paths that proposals intend to modify."""
    paths: Set[str] = set()
    for proposal in proposals:
        for chunk_path in getattr(proposal, "target_chunks", []):
            paths.add(chunk_path)
    return sorted(paths)


# ══════════════════════════════════════════════════════════
# SMART RECOVERY — feed the error to the builder to fix it
# ══════════════════════════════════════════════════════════

async def attempt_recovery(
    boot_error: str,
    target_root: str,
    snapshots: Dict[str, Optional[str]],
    emit: Callable[[str], None],
) -> bool:
    """Try to recover from a broken boot.

    Strategy:
    1. Attempt smart repair via the agentic builder (up to MAX_REPAIR_ATTEMPTS)
    2. If smart repair fails, hard revert from snapshots
    3. Verify boot after each attempt

    Returns True if the sandbox is bootable after recovery.
    """
    emit("\n🔧 RECOVERY: Attempting smart repair...")

    # Try smart repair first
    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
        emit(f"   🤖 Smart repair attempt {attempt}/{MAX_REPAIR_ATTEMPTS}...")
        repaired = await _try_smart_repair(boot_error, target_root, emit)
        if repaired:
            # Verify the fix actually worked
            boot_ok = await _boot_check(target_root)
            if boot_ok:
                emit(f"   ✅ Smart repair succeeded on attempt {attempt}")
                return True
            else:
                emit(f"   ⚠️ Smart repair attempt {attempt} didn't fix the boot")
                # Get updated error for next attempt
                boot_error = await _get_boot_error(target_root)

    # Smart repair failed — hard revert
    emit("\n🔄 RECOVERY: Smart repair failed. Reverting to pre-pass state...")
    reverted = await _hard_revert(snapshots, target_root, emit)
    if not reverted:
        emit("   ❌ Hard revert failed — sandbox may be in a broken state")
        return False

    boot_ok = await _boot_check(target_root)
    if boot_ok:
        emit("   ✅ Hard revert succeeded — sandbox is bootable")
        return True

    emit("   ❌ Hard revert completed but boot still fails — manual intervention needed")
    return False


# ══════════════════════════════════════════════════════════
# SMART REPAIR — send error to builder
# ══════════════════════════════════════════════════════════

async def _try_smart_repair(
    boot_error: str,
    target_root: str,
    emit: Callable[[str], None],
) -> bool:
    """Ask the agentic builder to fix the boot error."""
    try:
        # JOB 16 (2026-06-10): run_agentic_loop never existed in llm_tools
        # (phantom API from an older design) — this import was failing
        # silently inside the except below, so smart boot-repair never ran.
        # The legacy signature now lives in llm_compat as a real adapter.
        from app.pipeline_v2.llm_compat import run_agentic_loop

        repair_prompt = _build_repair_prompt(boot_error)
        result = await run_agentic_loop(
            prompt=repair_prompt,
            root_path=target_root,
            language="python",
            model="openai/gpt-5.4",
            max_tool_calls=20,
            emit=emit,
        )
        files_written = result.get("files_written", 0) if result else 0
        emit(f"   Builder repair: {files_written} files modified")
        return files_written > 0
    except Exception as exc:
        emit(f"   Smart repair error: {exc}")
        logger.warning("[boot_recovery] Smart repair failed: %s", exc)
        return False


def _build_repair_prompt(boot_error: str) -> str:
    """Build a focused repair prompt from the boot error."""
    # Extract the last meaningful error line
    err_lines = [l.strip() for l in boot_error.strip().split("\n") if l.strip()]
    # Find the actual error (ImportError, NameError, etc.)
    error_line = ""
    traceback_lines = []
    for line in err_lines:
        if any(line.startswith(prefix) for prefix in (
            "ImportError:", "ModuleNotFoundError:", "NameError:",
            "AttributeError:", "SyntaxError:", "TypeError:",
        )):
            error_line = line
        if "File " in line:
            traceback_lines.append(line)

    if not error_line and err_lines:
        error_line = err_lines[-1]

    context = "\n".join(traceback_lines[-5:]) if traceback_lines else ""

    return f"""URGENT FIX: The application fails to boot after optimisation.

Error: {error_line}

Traceback (last 5 frames):
{context}

Instructions:
1. Read the file(s) mentioned in the traceback
2. Fix the import or reference error
3. Do NOT rewrite the whole file — make the minimum change needed
4. Run py_compile to verify syntax
5. Run 'from main import app' to verify boot

The goal is to fix ONLY this boot error, nothing else."""


# ══════════════════════════════════════════════════════════
# HARD REVERT — restore files from pre-pass snapshots
# ══════════════════════════════════════════════════════════

async def _hard_revert(
    snapshots: Dict[str, Optional[str]],
    target_root: str,
    emit: Callable[[str], None],
) -> bool:
    """Restore all files to their pre-pass state."""
    import httpx

    restored = 0
    deleted = 0
    failed = 0

    async with httpx.AsyncClient(timeout=10.0) as client:
        for rel_path, original_content in snapshots.items():
            abs_path = f"{target_root}/{rel_path}".replace("/", "\\")
            try:
                if original_content is None:
                    # File was created during the pass — delete it
                    # (sandbox API doesn't have delete, write empty stub)
                    await client.post(
                        f"{SANDBOX_CONTROLLER_URL}/fs/write",
                        json={
                            "path": abs_path,
                            "content": f'"""Removed by optimizer recovery."""\n',
                        },
                    )
                    deleted += 1
                else:
                    # Restore original content
                    await client.post(
                        f"{SANDBOX_CONTROLLER_URL}/fs/write",
                        json={"path": abs_path, "content": original_content},
                    )
                    restored += 1
            except Exception as exc:
                logger.warning("[boot_recovery] Failed to revert %s: %s", rel_path, exc)
                failed += 1

    emit(f"   Reverted: {restored} restored, {deleted} removed, {failed} failed")
    return failed == 0


# ══════════════════════════════════════════════════════════
# BOOT CHECK HELPERS
# ══════════════════════════════════════════════════════════

async def _boot_check(target_root: str) -> bool:
    """Quick boot check — returns True/False, no emit."""
    import httpx

    venv_python = f"{target_root}\\.venv\\Scripts\\python.exe"
    payload = {
        "cmd": [venv_python, "-c", "from main import app; print('BOOT_OK')"],
        "cwd": target_root,
        "timeout_ms": 25000,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{SANDBOX_CONTROLLER_URL}/shell/run", json=payload
            )
            data = resp.json()
        return data.get("returncode", -1) == 0 and "BOOT_OK" in data.get("stdout", "")
    except Exception:
        return False


async def _get_boot_error(target_root: str) -> str:
    """Run the boot check and return the stderr output."""
    import httpx

    venv_python = f"{target_root}\\.venv\\Scripts\\python.exe"
    payload = {
        "cmd": [venv_python, "-c", "from main import app; print('BOOT_OK')"],
        "cwd": target_root,
        "timeout_ms": 25000,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{SANDBOX_CONTROLLER_URL}/shell/run", json=payload
            )
            data = resp.json()
        return data.get("stderr", "Unknown boot error")
    except Exception as exc:
        return f"Failed to get boot error: {exc}"