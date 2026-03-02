from __future__ import annotations
import base64
import logging
from app.overwatcher.sandbox_client import SandboxClient
from typing import Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


POT_EXECUTOR_BUILD_ID = "2026-02-03-v2.0-idempotent"

MAX_CONSECUTIVE_ERRORS = 5  # Abort after N consecutive file failures

READ_TIMEOUT = 30           # Seconds per file read

WRITE_TIMEOUT = 60          # Seconds per file write

def _encode_base64(content: str) -> str:
    """Encode content as Base64 for PowerShell transmission."""
    return base64.b64encode(content.encode('utf-8')).decode('ascii')

def _build_write_command(path: str, content: str) -> str:
    """Build PowerShell command to write file via Base64 → raw bytes.

    CRITICAL: Do NOT use 'Set-Content -Encoding UTF8' — PowerShell 5.1
    writes a UTF-8 BOM (EF BB BF) which corrupts JSON files and breaks
    Vite/Node.js parsers. Instead, decode Base64 to raw bytes and write
    via [System.IO.File]::WriteAllBytes() which is BOM-free.

    v1.1 (2026-02-03): Fixed UTF-8 BOM corruption bug.
    """
    encoded = _encode_base64(content)
    return (
        f'$bytes = [System.Convert]::FromBase64String("{encoded}"); '
        f'[System.IO.File]::WriteAllBytes("{path}", $bytes)'
    )

def _read_file_via_sandbox(client: SandboxClient, path: str) -> Optional[str]:
    """Read file content from sandbox via PowerShell."""
    try:
        cmd = f'Get-Content -Path "{path}" -Raw -Encoding UTF8'
        result = client.shell_run(cmd, timeout_seconds=READ_TIMEOUT)
        
        if result.stdout is not None:
            return result.stdout
        
        logger.warning(
            "[pot_executor] Read failed for %s: stderr=%s",
            path, (result.stderr or "")[:100]
        )
        return None
    except Exception as e:
        logger.error("[pot_executor] Read exception for %s: %s", path, e)
        return None

def _write_file_via_sandbox(client: SandboxClient, path: str, content: str) -> bool:
    """Write file content to sandbox via Base64 PowerShell command."""
    try:
        # v3.4-fix: Strip surviving scaffold markers before write
        from app.overwatcher._implementer_utils_6 import _strip_scaffold_markers
        content = _strip_scaffold_markers(content, path)

        cmd = _build_write_command(path, content)
        result = client.shell_run(cmd, timeout_seconds=WRITE_TIMEOUT)
        
        # Check for errors
        if result.stderr and result.stderr.strip():
            logger.warning(
                "[pot_executor] Write stderr for %s: %s",
                path, result.stderr[:200]
            )
            return False
        
        return True
    except Exception as e:
        logger.error("[pot_executor] Write exception for %s: %s", path, e)
        return False
