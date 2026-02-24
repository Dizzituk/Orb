"""
Step 4 preamble: Sandbox file scanning and import evidence.

Scans the sandbox filesystem to discover existing .py files (siblings
and parents) so the Implementer LLM knows which modules are available
for import.  Builds the available-modules evidence string.

Extracted from orchestrator.py monolith (v5.11–v5.15 logic).
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Set

from ..sandbox_client import SandboxClient
from .execution_state import ExecutionContext

logger = logging.getLogger(__name__)


def scan_sandbox_files(
    ctx: ExecutionContext,
    client: SandboxClient,
    all_tasks: List[Dict],
) -> None:
    """Populate ctx.existing_sandbox_files, ctx.parent_module_files,
    and ctx.available_modules_evidence from the sandbox filesystem.
    """
    # v5.32: Seed with manifest files
    if ctx.manifest_all_files:
        ctx.existing_sandbox_files.update(ctx.manifest_all_files)
        logger.info(
            "[arch_exec] v5.32 Seeded %d manifest file(s) as expected imports",
            len(ctx.manifest_all_files),
        )

    pkg_dirs = _collect_package_dirs(all_tasks)

    try:
        _scan_sibling_modules(ctx, client, pkg_dirs)
        _scan_parent_modules(ctx, client, pkg_dirs)
    except Exception as scan_err:
        logger.warning("[arch_exec] v5.11 Sandbox file scan failed: %s", scan_err)

    # v5.12: Add ALL planned task files as "known"
    for task in all_tasks:
        task_path = task["info"]["path"].replace("\\", "/")
        ctx.existing_sandbox_files.add(task_path)
    logger.info(
        "[arch_exec] v5.12 Total known files for import validation: %d (sandbox + planned)",
        len(ctx.existing_sandbox_files),
    )

    # v5.15: Build evidence string
    ctx.available_modules_evidence = _build_modules_evidence(
        ctx.existing_sandbox_files, ctx.parent_module_files,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_package_dirs(all_tasks: List[Dict]) -> Set[str]:
    """Extract unique package directory paths from task list."""
    pkg_dirs: Set[str] = set()
    for task in all_tasks:
        tp = task["info"]["path"].replace("\\", "/")
        parts = tp.rsplit("/", 1)
        if len(parts) == 2:
            pkg_dirs.add(parts[0])
    return pkg_dirs


def _scan_sibling_modules(
    ctx: ExecutionContext,
    client: SandboxClient,
    pkg_dirs: Set[str],
) -> None:
    """Scan each package directory for existing .py files."""
    for pkg_dir in pkg_dirs:
        abs_pkg = os.path.join(ctx.sandbox_base, pkg_dir.replace("/", os.sep))
        scan_cmd = (
            f'if (Test-Path "{abs_pkg}" -PathType Container) {{ '
            f'Get-ChildItem -Path "{abs_pkg}" -Filter "*.py" -File | '
            f'ForEach-Object {{ $_.Name }} '
            f'}} else {{ "" }}'
        )
        result = client.shell_run(scan_cmd, timeout_seconds=10)
        if result.stdout:
            for fname in result.stdout.strip().split("\n"):
                fname = fname.strip()
                if fname:
                    ctx.existing_sandbox_files.add(f"{pkg_dir}/{fname}")

    if ctx.existing_sandbox_files:
        logger.info(
            "[arch_exec] v5.11 Found %d existing .py files on sandbox for import validation: %s",
            len(ctx.existing_sandbox_files),
            sorted(ctx.existing_sandbox_files),
        )


def _scan_parent_modules(
    ctx: ExecutionContext,
    client: SandboxClient,
    pkg_dirs: Set[str],
) -> None:
    """Scan parent directories so LLM knows about `..` imports."""
    for pkg_dir in pkg_dirs:
        pkg_norm = pkg_dir.replace("\\", "/")
        parent_parts = pkg_norm.rsplit("/", 1)
        parent_dir = parent_parts[0] if len(parent_parts) == 2 else "."
        abs_parent = os.path.join(ctx.sandbox_base, parent_dir.replace("/", os.sep))
        scan_cmd = (
            f'if (Test-Path "{abs_parent}" -PathType Container) {{ '
            f'Get-ChildItem -Path "{abs_parent}" -Filter "*.py" -File | '
            f'ForEach-Object {{ $_.Name }} '
            f'}} else {{ "" }}'
        )
        result = client.shell_run(scan_cmd, timeout_seconds=10)
        if result.stdout:
            for fname in result.stdout.strip().split("\n"):
                fname = fname.strip()
                if fname:
                    ctx.parent_module_files.add(f"{parent_dir}/{fname}")

    if ctx.parent_module_files:
        logger.info(
            "[arch_exec] v5.15 Found %d parent-level .py modules for `..` import evidence: %s",
            len(ctx.parent_module_files),
            sorted(ctx.parent_module_files),
        )


def _build_modules_evidence(
    existing_files: Set[str],
    parent_files: Set[str],
) -> str:
    """Build the available-modules evidence string for Implementer prompts."""
    if not existing_files and not parent_files:
        return ""

    parts = [
        "\n\n## Available Modules (DO NOT invent imports outside this list)\n",
    ]

    if existing_files:
        sib_lines = [f"  - `{m}`" for m in sorted(existing_files)]
        parts.append(
            "### Sibling modules (use `from .module import ...`)\n"
            "These are in the same package. Import with a single dot.\n\n"
            + "\n".join(sib_lines) + "\n"
        )

    if parent_files:
        par_lines = [f"  - `{m}`" for m in sorted(parent_files)]
        parts.append(
            "\n### Parent modules (use `from ..module import ...`)\n"
            "These are in the parent package directory. Import with double dot `..`.\n"
            "Do NOT use absolute imports like `from app.x.y import Z`. "
            "Use RELATIVE imports: `from ..module_name import ClassName`.\n\n"
            + "\n".join(par_lines) + "\n"
        )

    parts.append(
        "\n**CRITICAL**: Do NOT invent imports to files not listed above. "
        "Do NOT use absolute imports (e.g. `from app.models.X`) when a relative "
        "import from the parent package exists (e.g. `from ..X import Y`). "
        "Every import MUST resolve to a file in one of these lists.\n"
    )
    return "\n".join(parts)
