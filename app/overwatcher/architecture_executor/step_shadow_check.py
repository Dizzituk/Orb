"""
Step 3b: Module shadowing pre-flight check.

Auto-creates __init__.py for new Python packages, then checks for
module shadowing conflicts (e.g. creating stream_utils/__init__.py
when stream_utils.py already exists would break all imports).

Extracted from orchestrator.py monolith (v2.8–v3.3 logic).
"""
from __future__ import annotations

import logging
from typing import List

from ..sandbox_client import SandboxClient
from .execution_state import ExecutionContext
from .path_resolution import _resolve_multi_root_path, _ensure_python_init_files

logger = logging.getLogger(__name__)


def run_init_file_creation(ctx: ExecutionContext, client: SandboxClient) -> None:
    """v2.6: Auto-create __init__.py for new Python packages.

    Prepends any generated init files to ctx.new_files so they are
    created BEFORE the files that need them.  Updates ctx.total_operations.
    """
    try:
        init_files = _ensure_python_init_files(
            ctx.new_files, ctx.modified_files, ctx.sandbox_base, client,
        )
        if init_files:
            ctx.new_files = init_files + ctx.new_files
            ctx.total_operations = len(ctx.new_files) + len(ctx.modified_files)
            ctx.add_trace("AUTO_INIT_PY", "success", {
                "init_files_added": [f["path"] for f in init_files],
                "new_total_operations": ctx.total_operations,
            })
            logger.info(
                "[arch_exec] v2.6 Added %d __init__.py files, total ops now %d",
                len(init_files), ctx.total_operations,
            )
    except Exception as e:
        logger.warning("[arch_exec] v2.6 _ensure_python_init_files failed: %s", e)
        ctx.add_trace("AUTO_INIT_PY", "failed", {"error": str(e)})


# ---------------------------------------------------------------------------
# Shadow detection helpers
# ---------------------------------------------------------------------------

def _detect_refactor_package_dirs(
    ctx: ExecutionContext,
    client: SandboxClient,
) -> set:
    """Detect file→package refactors.

    v3.2: If the segment creates an __init__.py inside a directory that
    shadows an existing .py module, the conversion is intentional.
    v3.3: Also check on-disk __init__.py from prior segments.
    """
    refactor_dirs: set = set()
    all_new_paths = {f["path"].replace("\\", "/") for f in ctx.new_files}

    for np in all_new_paths:
        if np.endswith("/__init__.py"):
            pkg_dir = np.rsplit("/", 1)[0]
            refactor_dirs.add(pkg_dir)

    # v3.3: Check on-disk __init__.py for packages prior segments created
    if not refactor_dirs:
        for file_info in ctx.new_files:
            fp = file_info["path"].replace("\\", "/")
            parts = fp.split("/")
            for depth in range(1, len(parts)):
                dir_seg = "/".join(parts[:depth])
                init_path = dir_seg + "/__init__.py"
                shadow_py = dir_seg + ".py"
                try:
                    check_init = client.shell_run(
                        f'if (Test-Path -Path "{_resolve_multi_root_path(init_path, ctx.sandbox_base)}") '
                        f'{{ "EXISTS" }} else {{ "NONE" }}',
                        timeout_seconds=10,
                    )
                    check_shadow = client.shell_run(
                        f'if (Test-Path -Path "{_resolve_multi_root_path(shadow_py, ctx.sandbox_base)}") '
                        f'{{ "EXISTS" }} else {{ "NONE" }}',
                        timeout_seconds=10,
                    )
                    if (check_init.stdout and "EXISTS" in check_init.stdout
                            and check_shadow.stdout and "EXISTS" in check_shadow.stdout):
                        refactor_dirs.add(dir_seg)
                except Exception:
                    pass

    if refactor_dirs:
        logger.info(
            "[arch_exec] v3.2 File->package refactor detected: %s "
            "— shadow check skipped for package contents",
            refactor_dirs,
        )
    return refactor_dirs


def _check_shadows_for_file(
    new_path: str,
    refactor_dirs: set,
    ctx: ExecutionContext,
    client: SandboxClient,
) -> List[dict]:
    """Check a single new file for module shadowing conflicts.

    Returns list of blocked-item dicts (empty if no conflicts).
    """
    blocked: List[dict] = []
    parts = new_path.replace("\\", "/").split("/")
    new_path_norm = new_path.replace("\\", "/")

    # Skip if inside a deliberate file→package refactor
    if any(new_path_norm.startswith(pkg + "/") for pkg in refactor_dirs):
        return blocked

    for depth in range(1, len(parts)):
        dir_segment = "/".join(parts[:depth])
        existing_py = dir_segment + ".py"
        try:
            check_cmd = (
                f'if (Test-Path -Path '
                f'"{_resolve_multi_root_path(existing_py, ctx.sandbox_base)}") '
                f'{{ "EXISTS" }} else {{ "NONE" }}'
            )
            result = client.shell_run(check_cmd, timeout_seconds=10)
            if result.stdout and "EXISTS" in result.stdout:
                blocked.append({
                    "new_path": new_path,
                    "shadows": existing_py,
                    "dir_segment": dir_segment,
                    "reason": (
                        f"Creating '{new_path}' would create a package directory "
                        f"that shadows existing module '{existing_py}'. "
                        f"Python resolves packages before modules, so all "
                        f"existing 'import {dir_segment.replace('/', '.')}' "
                        f"statements would break."
                    ),
                })
        except Exception as e:
            logger.warning(
                "[arch_exec] v2.8 Shadow check failed for %s: %s", new_path, e,
            )
    return blocked


def _handle_blocked_shadows(
    shadowing_blocked: List[dict],
    ctx: ExecutionContext,
) -> None:
    """Log shadow conflicts and remove blocked files from ctx.new_files."""
    new_paths_set = {f["path"].replace("\\", "/") for f in ctx.new_files}

    for blocked in shadowing_blocked:
        dir_seg = blocked["dir_segment"]
        init_path = dir_seg + "/__init__.py"
        if init_path in new_paths_set:
            logger.error(
                "[arch_exec] v2.9 Shadow still exists after quarantine for %s "
                "— package_quarantine may have failed. Check quarantine logs.",
                blocked["shadows"],
            )
            print(
                f"[ARCH_EXEC] ⚠ Shadow conflict: {blocked['shadows']} still exists. "
                f"Expected package_quarantine to have moved it."
            )

    for blocked in shadowing_blocked:
        logger.error(
            "[arch_exec] v2.8 MODULE SHADOW BLOCKED: %s shadows %s",
            blocked["new_path"], blocked["shadows"],
        )
        print(f"[ARCH_EXEC] ✗ BLOCKED: {blocked['reason']}")
        ctx.add_trace("MODULE_SHADOW_BLOCKED", "fatal", blocked)

    shadow_paths = {b["new_path"] for b in shadowing_blocked}
    original_count = len(ctx.new_files)
    ctx.new_files = [f for f in ctx.new_files if f["path"] not in shadow_paths]
    if original_count != len(ctx.new_files):
        logger.info(
            "[arch_exec] v2.8 Removed %d shadowing files from task list",
            original_count - len(ctx.new_files),
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_shadow_preflight(ctx: ExecutionContext, client: SandboxClient) -> None:
    """Run the full module-shadowing pre-flight check.

    Detects file→package refactors, checks each new file for shadows,
    logs and removes any blocked files from ctx.new_files.
    """
    refactor_dirs = _detect_refactor_package_dirs(ctx, client)
    shadowing_blocked: List[dict] = []

    for file_info in ctx.new_files:
        blocked = _check_shadows_for_file(
            file_info["path"], refactor_dirs, ctx, client,
        )
        shadowing_blocked.extend(blocked)

    if shadowing_blocked:
        _handle_blocked_shadows(shadowing_blocked, ctx)
