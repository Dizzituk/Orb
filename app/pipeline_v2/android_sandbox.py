# FILE: app/pipeline_v2/android_sandbox.py
"""
Android Build Folder Sandboxing.

Ensures that when the pipeline targets an Android build (language == "kotlin"),
file write operations are constrained to the project's project_root.

Android builds happen on the host (not in Windows Sandbox VM) because Android
Studio and the emulator need direct access. The sandboxing here is path-based:
validate that every write targets the correct project folder.

v1.0 (2026-03-11): Initial implementation.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile

logger = logging.getLogger(__name__)


# Directories that must NEVER be written to during Android builds
_BLOCKED_ROOTS = [
    os.path.normpath("D:/Orb"),
    os.path.normpath("D:/orb-desktop"),
]


class AndroidPathViolation(Exception):
    """Raised when an Android build attempts to write outside its project root."""
    pass


def validate_android_write_path(
    path: str,
    profile: "BuildTargetProfile",
) -> str:
    """Validate that a write path is within the Android project root.

    Args:
        path: The absolute or relative path to validate.
        profile: The active build target profile.

    Returns:
        The normalised absolute path if valid.

    Raises:
        AndroidPathViolation: If the path is outside the project root.
    """
    # Resolve to absolute
    abs_path = profile.resolve_path(path)
    norm_path = os.path.normpath(abs_path)
    norm_root = os.path.normpath(profile.project_root)

    # Check: path must start with project root
    if not norm_path.lower().startswith(norm_root.lower() + os.sep) and \
       norm_path.lower() != norm_root.lower():
        logger.error(
            "[android_sandbox] BLOCKED write outside project root: %s (root: %s)",
            norm_path, norm_root,
        )
        raise AndroidPathViolation(
            f"Android build cannot write to '{norm_path}'. "
            f"Writes are restricted to '{norm_root}'."
        )

    # Check: path must NOT be in any blocked root
    for blocked in _BLOCKED_ROOTS:
        if norm_path.lower().startswith(blocked.lower() + os.sep) or \
           norm_path.lower() == blocked.lower():
            logger.error(
                "[android_sandbox] BLOCKED write to protected directory: %s",
                norm_path,
            )
            raise AndroidPathViolation(
                f"Android build cannot write to '{norm_path}'. "
                f"'{blocked}' is a protected directory."
            )

    return norm_path


def is_android_build(profile: "BuildTargetProfile") -> bool:
    """Check if the current profile targets an Android build."""
    return profile.language == "kotlin"
