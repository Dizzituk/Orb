"""
Path Resolution Utilities

Provides path resolution, __init__.py auto-creation, and language detection
for the architecture executor pipeline.

Restored to match original monolith signatures expected by orchestrator.py.
"""

import logging
from typing import Dict, List, Optional

from .constants import FRONTEND_PREFIX, FRONTEND_ROOT
from ..sandbox_client import SandboxClient

logger = logging.getLogger(__name__)


def _resolve_multi_root_path(rel_path: str, sandbox_base: str) -> str:
    r"""Resolve a relative path to its correct absolute path.

    v2.2: The project has two separate root directories:
    - Backend (D:\Orb): paths like app/routers/voice.py, main.py
    - Frontend (D:\orb-desktop): paths like orb-desktop/src/components/VoiceInput.tsx

    Architecture map and prompt both use orb-desktop/ prefix for frontend files.
    This function strips the prefix and resolves to the correct root.

    Args:
        rel_path: Relative path from architecture document
        sandbox_base: Resolved backend base (e.g. D:\Orb)

    Returns:
        Absolute path with correct root
    """
    normalized = rel_path.replace("\\", "/")

    if normalized.startswith(FRONTEND_PREFIX):
        # Strip the orb-desktop/ prefix and resolve against frontend root
        frontend_rel = normalized[len(FRONTEND_PREFIX):]
        abs_path = f"{FRONTEND_ROOT}\\{frontend_rel.replace('/', '\\')}"
        logger.info("[arch_exec] v2.2 Frontend path: %s -> %s", rel_path, abs_path)
        return abs_path
    else:
        # Backend path — resolve against sandbox_base as before
        abs_path = f"{sandbox_base}\\{normalized.replace('/', '\\')}"
        return abs_path


def _ensure_python_init_files(
    new_files: List[Dict[str, str]],
    modified_files: List[Dict[str, str]],
    sandbox_base: str,
    client: SandboxClient,
) -> List[Dict[str, str]]:
    """Auto-create __init__.py files for new Python package directories.

    v2.6: When the architecture creates Python files in new directories
    (e.g. app/services/transcription_service.py), those directories need
    __init__.py to be importable as Python packages. The architecture
    rarely includes these, and the Implementer doesn't know to create them.

    This function:
    1. Collects all directories that will contain new .py files
    2. For each directory, walks up to the project root checking for __init__.py
    3. Skips directories that already have __init__.py (in manifest or on disk)
    4. Returns a list of __init__.py file entries to prepend to new_files

    Only applies to backend Python paths (not orb-desktop/ frontend paths).

    Args:
        new_files: List of new file dicts from parse_file_inventory
        modified_files: List of modified file dicts (for manifest awareness)
        sandbox_base: Resolved backend root (e.g. D:\\Orb)
        client: SandboxClient for checking existing files on disk

    Returns:
        List of __init__.py file dicts to prepend to new_files
    """
    # Collect all paths already in the manifest (new + modified)
    manifest_paths = set()
    for f in new_files:
        manifest_paths.add(f["path"].replace("\\", "/"))
    for f in modified_files:
        manifest_paths.add(f["path"].replace("\\", "/"))

    # Collect directories that need __init__.py checking
    dirs_needing_init: set = set()

    for f in new_files:
        rel_path = f["path"].replace("\\", "/")

        # Skip non-Python files
        if not rel_path.endswith(".py"):
            continue

        # Skip frontend paths
        if rel_path.startswith(FRONTEND_PREFIX):
            continue

        # Skip if this IS an __init__.py (already being created)
        if rel_path.endswith("__init__.py"):
            continue

        # Walk up directory tree from the file's parent to the project root
        parts = rel_path.split("/")
        for depth in range(1, len(parts)):  # depth=1 is immediate parent dir
            dir_path = "/".join(parts[:depth])
            init_path = f"{dir_path}/__init__.py"

            # Skip if __init__.py already in manifest
            if init_path in manifest_paths:
                continue

            # Skip top-level (no __init__.py needed at project root)
            if "/" not in dir_path:
                # e.g. "app" — this IS a package dir, check it
                # But if dir_path is just a filename component, skip
                pass

            dirs_needing_init.add(init_path)

    if not dirs_needing_init:
        return []

    # Check which of these __init__.py files already exist on disk
    init_files_to_create: List[Dict[str, str]] = []

    for init_path in sorted(dirs_needing_init):
        abs_path = f"{sandbox_base}\\{init_path.replace('/', '\\')}"

        # Check if file exists in sandbox
        try:
            cmd = f'Test-Path -Path "{abs_path}" -PathType Leaf'
            result = client.shell_run(cmd, timeout_seconds=10)
            if result.stdout and result.stdout.strip().lower() == "true":
                logger.info(
                    "[arch_exec] v2.6 __init__.py already exists: %s",
                    init_path,
                )
                continue
        except Exception as e:
            logger.warning(
                "[arch_exec] v2.6 Could not check %s: %s — will create anyway",
                init_path, e,
            )

        init_files_to_create.append({
            "path": init_path,
            "description": f"v2.6 auto-created: Python package init for {init_path.rsplit('/', 1)[0]}/",
        })
        logger.info(
            "[arch_exec] v2.6 Auto-creating __init__.py: %s",
            init_path,
        )

    if init_files_to_create:
        print(
            f"[ARCH_EXEC] v2.6 Auto-creating {len(init_files_to_create)} __init__.py file(s): "
            + ", ".join(f["path"] for f in init_files_to_create)
        )

    return init_files_to_create


def _infer_lang_from_path(path: str) -> Optional[str]:
    """
    Infer programming language from file extension.

    Args:
        path: File path (relative or absolute)

    Returns:
        Language string ("python", "typescript", "javascript") or None if unknown
    """
    path_lower = path.lower()

    if path_lower.endswith(".py"):
        return "python"
    elif path_lower.endswith((".ts", ".tsx")):
        return "typescript"
    elif path_lower.endswith((".js", ".jsx")):
        return "javascript"

    return None
