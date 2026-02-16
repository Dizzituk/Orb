"""
Sandbox Read-Only Operations

Provides foundational sandbox verification and file reading operations.
These are imported by multiple architecture executor segments and must maintain
stable signatures and semantics.

v1.1 (2026-02-16): Fixed execute_script -> shell_run (SandboxClient API).
"""

import logging
from typing import Any, Dict, Optional

from ..sandbox_client import SandboxClient
from .constants import VERIFY_READ_TIMEOUT

logger = logging.getLogger(__name__)

SANDBOX_OPS_BUILD_ID = "2026-02-16-v1.1-shell-run-fix"
print(f"[SANDBOX_OPS_LOADED] BUILD_ID={SANDBOX_OPS_BUILD_ID}")

__all__ = ["_verify_file_via_sandbox", "_resolve_sandbox_base", "_read_existing_file"]


def _verify_file_via_sandbox(
    client: SandboxClient,
    path: str,
    expected_min_chars: int = 10
) -> Dict[str, Any]:
    """
    Verify a file exists in the sandbox and meets minimum character requirements.

    Uses PowerShell Get-Content with UTF-8 encoding to read the file and validate
    its existence and content length.

    Args:
        client: Sandbox client instance
        path: Path to file in sandbox
        expected_min_chars: Minimum expected character count (default: 10)

    Returns:
        Dict with keys:
            - exists (bool): Whether file exists
            - chars (int): Character count (0 if file doesn't exist)
            - valid (bool): Whether file meets minimum requirements
            - error (str or None): Error message if validation failed
    """
    cmd = (
        f'if (Test-Path "{path}") {{ '
        f'$content = Get-Content -Path "{path}" -Raw -Encoding UTF8; '
        f'Write-Output "EXISTS:$($content.Length)" '
        f'}} else {{ Write-Output "MISSING" }}'
    )

    try:
        result = client.shell_run(cmd, timeout_seconds=VERIFY_READ_TIMEOUT)
        output = (result.stdout or "").strip()

        if output.startswith("EXISTS:"):
            char_count_str = output.split(":", 1)[1].strip()
            char_count = int(char_count_str)

            if char_count < expected_min_chars:
                return {
                    "exists": True,
                    "chars": char_count,
                    "valid": False,
                    "error": f"File too short: {char_count} chars < {expected_min_chars}"
                }

            return {
                "exists": True,
                "chars": char_count,
                "valid": True,
                "error": None
            }
        elif "MISSING" in output:
            return {
                "exists": False,
                "chars": 0,
                "valid": False,
                "error": "File does not exist"
            }
        else:
            return {
                "exists": False,
                "chars": 0,
                "valid": False,
                "error": f"Unexpected output: {output[:200]}"
            }

    except Exception as e:
        logger.error("[sandbox_ops] Verification failed for %s: %s", path, e)
        return {
            "exists": False,
            "chars": 0,
            "valid": False,
            "error": str(e)
        }


def _resolve_sandbox_base(client: SandboxClient) -> str:
    """
    Resolve the sandbox base directory by checking candidate locations.

    Probes multiple candidate directories and checks for the existence of main.py
    to determine the correct sandbox base path.

    Args:
        client: Sandbox client instance

    Returns:
        str: Resolved sandbox base directory path (defaults to D:\\Orb if not found)
    """
    candidates = [
        r"C:\Orb\Orb",
        r"C:\Orb",
        r"D:\Orb",
    ]

    for candidate in candidates:
        cmd = f'if (Test-Path "{candidate}\\main.py") {{ "TRUE" }} else {{ "FALSE" }}'
        try:
            result = client.shell_run(cmd, timeout_seconds=10)
            output = (result.stdout or "").strip()

            if "TRUE" in output:
                logger.info("[sandbox_ops] Resolved sandbox base: %s", candidate)
                return candidate
        except Exception as e:
            logger.debug("[sandbox_ops] Could not check %s: %s", candidate, e)
            continue

    default = r"D:\Orb"
    logger.warning("[sandbox_ops] Could not resolve sandbox base, defaulting to %s", default)
    return default


def _read_existing_file(client: SandboxClient, path: str) -> Optional[str]:
    """
    Read an existing file from the sandbox.

    Uses PowerShell Get-Content with UTF-8 encoding to read file content.

    Args:
        client: Sandbox client instance
        path: Path to file in sandbox

    Returns:
        str: File content if successful, None on error
    """
    cmd = f'Get-Content -Path "{path}" -Raw -Encoding UTF8'

    try:
        result = client.shell_run(cmd, timeout_seconds=30)

        if result.exit_code == 0:
            return result.stdout or ""
        else:
            error_msg = result.stderr or "Unknown error"
            logger.error("[sandbox_ops] Failed to read %s: %s", path, error_msg)
            return None

    except Exception as e:
        logger.error("[sandbox_ops] Exception reading %s: %s", path, e)
        return None
