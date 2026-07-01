# FILE: app/debug/executors/filesystem.py
# Purpose: Filesystem executors: read, write, edit, list, search, run_command.
# Called-by: app.debug.executors
# Depends-on: app.debug.executors._paths, app.debug.executors.filesystem_guards, app.debug.size_warning
# Last-renovated: 2026-06-11
"""
Filesystem executors: read, write, edit, list, search, run_command.

Routing rules:
  - Host-only paths (architecture, logs, Android, user home) -> direct host
  - Protected paths (Orb codebase) -> sandbox controller
  - Hard-blocked paths (Windows system) -> always refused
  - Everything else writable on host

run_command runs ON THE HOST via PowerShell, with extensive blocklists
to prevent file-write escape hatches that would bypass sandbox routing.

If this file approaches 30 KB, split run_command + its blocklists into
filesystem_run_command.py.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from app.debug.executors._paths import (
    SANDBOX_CONTROLLER_URL,
    is_android_repo_path,
    is_host_only,
    is_host_write_blocked,
    read_host_file,
    resolve_sandbox_path,
    sandbox_health_status,
)
from app.debug.run_context import is_phone_initiated
from app.debug.executors.filesystem_guards import (
    check_syntax,
    describe_line_numbered_corruption,
    looks_like_line_numbered_content,
    strip_line_number_prefixes,
)
from app.debug.size_warning import add_size_warning as _size_warn, strip_size_warning

logger = logging.getLogger(__name__)


# =============================================================================
# READ
# =============================================================================

async def execute_read_file(params: Dict[str, Any]) -> str:
    """Read a file from the sandbox (or host for architecture/logs)."""
    path = params.get("path", "")
    head = params.get("head")
    tail = params.get("tail")

    if not path:
        return "Error: path is required."

    path = resolve_sandbox_path(path)

    # Host-only data (architecture maps, logs, user home)
    if is_host_only(path):
        result = read_host_file(path, head, tail)
        if result is not None:
            return _size_warn(result, path) if not head and not tail else result
        return f"File not found on host: {path}"

    # Everything else: sandbox controller.
    # Note: include_line_numbers is False by default so the raw content
    # ASTRA sees matches what would be written back. Decorated output is
    # a foot-gun: it has caused real corruption when read-then-written.
    try:
        import httpx
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{SANDBOX_CONTROLLER_URL}/fs/contents",
                json={"paths": [path], "max_file_size": 500000, "include_line_numbers": False},
            )
            if resp.status_code == 200:
                data = resp.json()
                files = data.get("files", [])
                if files:
                    if files[0].get("error"):
                        return f"Error: {files[0]['error']}"
                    content = files[0].get("content", "")
                    # Defence in depth: even if a future controller change
                    # re-introduces decoration, strip it here so writes are safe.
                    cleaned, was_stripped = strip_line_number_prefixes(content)
                    if was_stripped:
                        logger.warning(
                            "[executors.filesystem] sandbox returned line-numbered content for %s; stripped %d -> %d chars",
                            path, len(content), len(cleaned),
                        )
                        content = cleaned
                    if head:
                        content = "\n".join(content.splitlines()[:head])
                    elif tail:
                        content = "\n".join(content.splitlines()[-tail:])
                    return _size_warn(content, path) if not head and not tail else content
                return f"File not found: {path}"
            return f"Sandbox error ({resp.status_code}): {resp.text}"
    except Exception as e:
        return f"Sandbox read failed for {path}: {e}"


async def execute_list_files(params: Dict[str, Any]) -> str:
    """List directory contents via sandbox controller, or host iterdir for host-only paths."""
    path = params.get("path", "")
    if not path:
        return "Error: path is required."

    path = resolve_sandbox_path(path)

    # Host-only directories (architecture, logs, user home)
    if is_host_only(path):
        p = Path(path)
        if p.exists() and p.is_dir():
            try:
                entries = []
                for item in sorted(p.iterdir()):
                    prefix = "[DIR]" if item.is_dir() else "[FILE]"
                    entries.append(f"{prefix} {item.name}")
                return "\n".join(entries) if entries else "(empty directory)"
            except Exception as e:
                return f"Error listing directory: {e}"
        return f"Directory not found: {path}"

    # Sandbox controller
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{SANDBOX_CONTROLLER_URL}/fs/tree",
                json={"roots": [path], "max_depth": 1},
            )
            if resp.status_code == 200:
                data = resp.json()
                files = data.get("files", [])
                if files:
                    lines = []
                    for f in files:
                        name = f.get("name", "?")
                        ext = f.get("ext", "")
                        is_dir = not ext and f.get("size_bytes") is None
                        prefix = "[DIR]" if is_dir else "[FILE]"
                        lines.append(f"{prefix} {name}")
                    return "\n".join(lines)
                return f"(empty directory or not found: {path})"
            return f"Sandbox error ({resp.status_code}): {resp.text}"
    except Exception as e:
        return f"Sandbox list failed for {path}: {e}"


async def execute_search_files(params: Dict[str, Any]) -> str:
    """Search for files matching a pattern via sandbox /fs/tree + client-side filter."""
    import fnmatch

    root = params.get("path", "D:/Orb")
    pattern = params.get("pattern", "**/*")

    root = resolve_sandbox_path(root)

    # Host-only directories (architecture)
    if is_host_only(root):
        try:
            p = Path(root)
            if not p.exists():
                return f"Directory not found: {root}"
            matches = list(p.glob(pattern))
            skip_dirs = {".git", "node_modules", ".venv", "__pycache__", "dist", "build"}
            filtered = [m for m in matches if not any(sd in m.parts for sd in skip_dirs)]
            if not filtered:
                return f"No files matching '{pattern}' in {root}"
            result_lines = [str(m) for m in filtered[:100]]
            suffix = f"\n... ({len(filtered)} total)" if len(filtered) > 100 else ""
            return "\n".join(result_lines) + suffix
        except Exception as e:
            return f"Search error: {e}"

    # Sandbox controller
    try:
        import httpx
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{SANDBOX_CONTROLLER_URL}/fs/tree",
                json={"roots": [root], "max_depth": 10},
            )
            if resp.status_code == 200:
                data = resp.json()
                files = data.get("files", [])
                skip_dirs = {".git", "node_modules", ".venv", "__pycache__", "dist", "build"}
                match_pattern = pattern.lstrip("*/") if pattern.startswith("**/") else pattern

                matched = []
                for f in files:
                    fpath = f.get("path", "")
                    fname = f.get("name", "")
                    if any(sd in fpath for sd in skip_dirs):
                        continue
                    if fnmatch.fnmatch(fname, match_pattern) or fnmatch.fnmatch(fpath, pattern):
                        matched.append(fpath)

                if not matched:
                    return f"No files matching '{pattern}' in {root}"
                result_lines = matched[:100]
                suffix = f"\n... ({len(matched)} total)" if len(matched) > 100 else ""
                return "\n".join(result_lines) + suffix
            return f"Sandbox error ({resp.status_code}): {resp.text}"
    except Exception as e:
        return f"Sandbox search failed for {root}: {e}"


# =============================================================================
# WRITE / EDIT
# =============================================================================

async def execute_write_file(params: Dict[str, Any]) -> str:
    """Write a file. Host-only paths write directly; everything else via sandbox.

    Two corruption guards apply before the actual write:
      1. Line-numbered prefix detection: rejects writes whose content looks
         like the output of a line-numbered reader (e.g. "  1: foo"), which
         is almost always the result of read-then-write contamination.
      2. Post-write syntax check (.py / .json): if the written content fails
         to parse, the previous file content is restored and an error is
         returned to the caller.
    """
    path = params.get("path", "")
    content = params.get("content", "")
    if not path:
        return "Error: path is required."

    # GUARD: reject obviously corrupted content (line-numbered read output
    # being written back). Catches multi-level decoration too.
    if looks_like_line_numbered_content(content):
        preview = describe_line_numbered_corruption(content)
        logger.warning(
            "[executors.filesystem] WRITE GUARD: content appears line-numbered for %s. Refusing write.",
            path,
        )
        return (
            "BLOCKED - LINE-NUMBER GUARD: write_file content has line-number "
            "prefixes (e.g. '  1: foo'). This usually means you wrote back "
            "the output of a previous read instead of raw file content. The "
            "read tool no longer adds line numbers by default, so this should "
            "not happen. If you DO need the line-numbered view for reasoning, "
            "do not paste it into write_file -- strip the prefixes first or "
            "call edit_file with the original (un-decorated) text.\n\n"
            f"First lines of rejected content:\n{preview}"
        )

    # GUARD: syntax check for languages with cheap in-process parsers
    # (.py, .json). Catches: brace mismatches, accidental indentation,
    # half-applied edits, etc. Non-parseable extensions (.ts, .css, .md,
    # .txt) are skipped -- check_syntax returns None for them.
    syntax_error = check_syntax(path, content)
    if syntax_error:
        logger.warning(
            "[executors.filesystem] SYNTAX GUARD: %s would have syntax error: %s",
            path, syntax_error,
        )
        return (
            f"BLOCKED - SYNTAX GUARD: writing this content to {path} would "
            f"produce a syntax error:\n  {syntax_error}\n\n"
            "Fix the syntax in the content you're sending, then retry. "
            "This guard prevents broken files from being committed -- it "
            "runs the same parser Python / json.loads would use at import time."
        )

    path = resolve_sandbox_path(path)

    # TRUNCATION GUARD
    _existing_size = 0
    try:
        _check_path = Path(path)
        if _check_path.is_file():
            _existing_size = _check_path.stat().st_size
    except Exception:
        pass
    _new_size = len(content)
    if _existing_size > 500 and _new_size < _existing_size * 0.5:
        _pct = int((_new_size / _existing_size) * 100)
        logger.warning(
            "[executors.filesystem] TRUNCATION GUARD: write_file would shrink %s "
            "from %d to %d chars (%d%%). Blocked.",
            path, _existing_size, _new_size, _pct,
        )
        return (
            f"BLOCKED - TRUNCATION GUARD: write_file would reduce {path} from "
            f"{_existing_size} to {_new_size} chars ({_pct}% of original). "
            f"This would destroy content. Use edit_file for targeted changes, "
            f"or read the COMPLETE file first before using write_file."
        )

    # Hard-block Windows system paths (no sandbox route exists)
    _HARD_BLOCKED = ["C:/Windows", "C:\\Windows", "C:/Program Files",
                     "C:\\Program Files", "C:/ProgramData", "C:\\ProgramData"]
    norm_check = path.replace("\\", "/").rstrip("/")
    for hb in _HARD_BLOCKED:
        hb_norm = hb.replace("\\", "/").rstrip("/")
        if norm_check == hb_norm or norm_check.startswith(hb_norm + "/"):
            logger.warning("[executors.filesystem] HARD BLOCKED system write: %s", path)
            return f"BLOCKED: Cannot write to {path} - system directory."

    if is_host_write_blocked(path):
        # Route through sandbox instead of blocking
        logger.info("[executors.filesystem] Protected path %s - routing to sandbox", path)
        sandbox_ok, sandbox_message = sandbox_health_status()
        if not sandbox_ok:
            return (
                f"Cannot write to {path} on host (protected). "
                f"Sandbox is also unavailable: {sandbox_message}\n"
                "Start the sandbox to modify ASTRA's own code."
            )
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{SANDBOX_CONTROLLER_URL}/fs/write",
                    json={"path": path, "content": content, "overwrite": True},
                )
                if resp.status_code == 200:
                    return (
                        f"Successfully wrote {len(content)} chars to {path} (SANDBOX).\n"
                        "This change is in the sandbox clone, not the live host. "
                        "Say 'promote' or use git to push changes to the host repo."
                    )
                return f"Sandbox write failed ({resp.status_code}): {resp.text}"
        except Exception as e:
            return f"Sandbox write error: {e}"

    # Host-only paths: write directly to host filesystem
    if is_host_only(path):
        if is_android_repo_path(path) and is_phone_initiated():
            # Confirmation gate (live-session plan §2.4/4.1): a phone-initiated
            # driving session must not silently edit the Android app on the host
            # (writes here bypass the sandbox entirely -- see is_host_only above).
            # There is no interactive confirmation UI while driving, so the safe
            # default is a hard block; the change must be made/confirmed on desktop.
            logger.warning(
                "[executors.filesystem] BLOCKED phone-initiated Android write: %s", path,
            )
            return (
                f"BLOCKED: {path} is part of the Android app. Writes to the Android "
                "app are disabled for phone-initiated debug sessions and need to be "
                "confirmed from the desktop Debug tab. Skip this change for now and "
                "say so in your reply; the user can pick it up on desktop."
            )
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            size_kb = len(content.encode("utf-8")) / 1024
            logger.info("[executors.filesystem] Host write: %s (%.1f KB)", path, size_kb)
            return f"Successfully wrote {len(content)} chars ({size_kb:.1f} KB) to {path} (host)"
        except Exception as e:
            return f"Host write error for {path}: {e}"

    # Sandbox paths
    sandbox_ok, sandbox_message = sandbox_health_status()
    if not sandbox_ok:
        return sandbox_message

    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{SANDBOX_CONTROLLER_URL}/fs/write",
                json={"path": path, "content": content, "overwrite": True},
            )
            if resp.status_code == 200:
                return f"Successfully wrote {len(content)} chars to {path} (sandbox)"
            return f"Write failed ({resp.status_code}): {resp.text}"
    except Exception as e:
        return f"Write error: {e}"


async def execute_edit_file(params: Dict[str, Any]) -> str:
    """Edit a file via read-modify-write. Host-only paths bypass sandbox."""
    path = params.get("path", "")
    old_text = params.get("old_text", "")
    new_text = params.get("new_text", "")

    if not path or not old_text:
        return "Error: path and old_text are required."

    resolved = resolve_sandbox_path(path)

    _HARD_BLOCKED_EDIT = ["C:/Windows", "C:\\Windows", "C:/Program Files",
                          "C:\\Program Files", "C:/ProgramData", "C:\\ProgramData"]
    norm_edit = resolved.replace("\\", "/").rstrip("/")
    for hb in _HARD_BLOCKED_EDIT:
        hb_norm = hb.replace("\\", "/").rstrip("/")
        if norm_edit == hb_norm or norm_edit.startswith(hb_norm + "/"):
            return f"BLOCKED: Cannot edit {resolved} - system directory."

    if is_host_write_blocked(resolved):
        # Route edit through sandbox
        logger.info("[executors.filesystem] Protected edit %s - routing to sandbox", resolved)
        current = await execute_read_file({"path": path})
        if current and current.startswith("Error"):
            return current
        if not current:
            return f"Cannot read {resolved} from sandbox for editing."
        # execute_read_file prepends a SIZE ALERT banner for >30KB code files;
        # strip it so the decoration never round-trips into the written file.
        current = strip_size_warning(current)
        if old_text not in current:
            return f"Error: old_text not found in {resolved} (sandbox). The text must exist exactly."
        count = current.count(old_text)
        if count > 1:
            return f"Error: old_text found {count} times in {resolved}. Must be unique."
        updated = current.replace(old_text, new_text, 1)
        return await execute_write_file({"path": resolved, "content": updated})

    # Host-only paths
    if is_host_only(resolved):
        current = read_host_file(resolved)
        if current is None:
            return f"File not found on host: {resolved}"
        if old_text not in current:
            return f"Error: old_text not found in {resolved}. The text to replace must exist exactly."
        count = current.count(old_text)
        if count > 1:
            return f"Error: old_text found {count} times in {resolved}. Must be unique."
        updated = current.replace(old_text, new_text, 1)
        return await execute_write_file({"path": resolved, "content": updated})

    # Sandbox paths
    sandbox_ok, sandbox_message = sandbox_health_status()
    if not sandbox_ok:
        return sandbox_message

    current = await execute_read_file({"path": path})
    if current.startswith("Error") or current.startswith("Sandbox read failed"):
        return current
    current = strip_size_warning(current)

    if old_text not in current:
        return f"Error: old_text not found in {path}. The text to replace must exist exactly."

    count = current.count(old_text)
    if count > 1:
        return f"Error: old_text found {count} times in {path}. Must be unique."

    updated = current.replace(old_text, new_text, 1)
    return await execute_write_file({"path": resolved, "content": updated})


# ---------------------------------------------------------------------------
# Host command execution lives in filesystem_run_command.py (split 2026-06-20).
# Re-exported so executors/__init__, host_console, and tests resolve it from
# this module unchanged (execute_run_command stays a real, patchable attribute).
# ---------------------------------------------------------------------------
from app.debug.executors.filesystem_run_command import execute_run_command  # noqa: F401
