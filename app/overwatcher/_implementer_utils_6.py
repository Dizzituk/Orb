from __future__ import annotations
import base64
import logging
import re
from app.overwatcher._implementer_utils_2 import INLINE_BASE64_CHAR_LIMIT
from app.overwatcher.sandbox_client import SandboxClient
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
logger = logging.getLogger(__name__)


# v3.4-fix: Scaffold marker pattern - matches [LLM_FILL: ...] placeholders
_SCAFFOLD_MARKER_LINE = re.compile(r"^[ \t]*\[LLM_FILL[:\s][^\]]*\][ \t]*\r?\n?$", re.MULTILINE)
_SCAFFOLD_MARKER_INLINE = re.compile(r"\[LLM_FILL[:\s][^\]]*\]")


def _strip_scaffold_markers(content: str, path: str) -> str:
    """Remove any surviving [LLM_FILL: ...] scaffold markers from content."""
    cleaned, n_lines = _SCAFFOLD_MARKER_LINE.subn("", content)
    cleaned, n_inline = _SCAFFOLD_MARKER_INLINE.subn("", cleaned)
    total = n_lines + n_inline
    if total > 0:
        logger.warning(
            "[implementer] v3.4 Stripped %d scaffold marker(s) from %s "
            "(%d full-line, %d inline)",
            total, path, n_lines, n_inline,
        )
    return cleaned


def _write_content_to_sandbox(
    client: 'SandboxClient',
    path: str,
    content: str,
    timeout_seconds: int = 60,
) -> 'ShellResult':
    """
    v1.13: Write content to sandbox, automatically choosing inline or temp-file method.

    If the Base64-encoded content fits within INLINE_BASE64_CHAR_LIMIT, uses the
    fast inline method (existing behaviour). If it exceeds the limit, writes the
    Base64 to a temp file in the sandbox first, then has PowerShell read and
    decode from the temp file. This avoids WinError 206 (command line too long).

    Args:
        client: SandboxClient instance
        path: Absolute path in sandbox to write to
        content: File content (UTF-8 string)
        timeout_seconds: Timeout for the write command

    Returns:
        ShellResult from the sandbox client
    """
    # v3.4-fix: Strip any surviving scaffold markers before write
    content = _strip_scaffold_markers(content, path)

    # v3.4-fix: Deduplicate imports (scaffold + LLM both add imports)
    from app.overwatcher.import_dedup import deduplicate_imports
    content = deduplicate_imports(content, path)

    encoded = base64.b64encode(content.encode('utf-8')).decode('ascii')

    if len(encoded) <= INLINE_BASE64_CHAR_LIMIT:
        # Fast path: inline Base64 (existing behaviour)
        write_cmd = (
            f'$bytes = [System.Convert]::FromBase64String("{encoded}"); '
            f'[System.IO.File]::WriteAllBytes("{path}", $bytes)'
        )
        logger.info(
            "[implementer] v1.13 Writing %d chars inline (b64=%d) to %s",
            len(content), len(encoded), path,
        )
        return client.shell_run(write_cmd, timeout_seconds=timeout_seconds)
    else:
        # v1.15: Temp-file path — avoids WinError 206 (command line too long)
        # CRITICAL: Use the TARGET FILE'S PARENT DIRECTORY for temp storage.
        # This is guaranteed to exist because we're about to write there.
        # NEVER use C:\Users\WDAGUtilityAccount\AppData\Local\Temp\ — that
        # path does not exist in Windows Sandbox (WDAG clones the host session,
        # it does NOT create a WDAGUtilityAccount profile with temp dirs).
        import uuid as _uuid
        from pathlib import PureWindowsPath
        temp_name = f"_orb_impl_{_uuid.uuid4().hex[:12]}.b64"
        parent_dir = str(PureWindowsPath(path).parent)
        temp_path = f"{parent_dir}\\{temp_name}"

        logger.info(
            "[implementer] v1.15 LARGE FILE: %d chars, b64=%d chars — "
            "temp-file write to %s for target %s",
            len(content), len(encoded), temp_path, path,
        )
        print(
            f"[IMPLEMENTER] v1.15 LARGE FILE ({len(content)} chars, "
            f"b64={len(encoded)}) — temp-file in parent dir"
        )

        # Step 0: Ensure parent directory exists
        ensure_dir_cmd = (
            f'if (-not (Test-Path -Path "{parent_dir}")) {{ '
            f'New-Item -Path "{parent_dir}" -ItemType Directory -Force | Out-Null }}; '
            f'"DIR_OK"'
        )
        dir_result = client.shell_run(ensure_dir_cmd, timeout_seconds=15)
        if not (dir_result.stdout and "DIR_OK" in dir_result.stdout):
            logger.error(
                "[implementer] v1.15 INFRASTRUCTURE_ERROR: Cannot ensure parent dir %s: %s",
                parent_dir, (dir_result.stderr or "")[:200],
            )
            return dir_result

        # Step A: Write Base64 to temp file in chunks
        # Chunk size 8000 chars — well under the 32,767 cmd line limit even
        # with the PowerShell wrapper overhead. Each chunk is written via
        # Set-Content/Add-Content (NOT embedded in .NET method args) and
        # verified with a confirmation token.
        chunk_size = 8000
        chunks = [encoded[i:i+chunk_size] for i in range(0, len(encoded), chunk_size)]

        # First chunk: create/overwrite the temp file
        first_chunk_cmd = (
            f'Set-Content -Path "{temp_path}" -Value "{chunks[0]}" '
            f'-NoNewline -Encoding ASCII; "CHUNK_OK"'
        )
        result = client.shell_run(first_chunk_cmd, timeout_seconds=30)
        if not (result.stdout and "CHUNK_OK" in result.stdout):
            logger.error(
                "[implementer] v1.15 INFRASTRUCTURE_ERROR: First chunk write failed: %s",
                (result.stderr or "")[:200],
            )
            client.shell_run(f'Remove-Item -Path "{temp_path}" -Force -ErrorAction SilentlyContinue', timeout_seconds=5)
            return result

        # Remaining chunks: append
        for i, chunk in enumerate(chunks[1:], 2):
            append_cmd = (
                f'Add-Content -Path "{temp_path}" -Value "{chunk}" '
                f'-NoNewline -Encoding ASCII; "CHUNK_OK"'
            )
            result = client.shell_run(append_cmd, timeout_seconds=30)
            if not (result.stdout and "CHUNK_OK" in result.stdout):
                logger.error(
                    "[implementer] v1.15 INFRASTRUCTURE_ERROR: Chunk %d/%d failed: %s",
                    i, len(chunks), (result.stderr or "")[:200],
                )
                client.shell_run(f'Remove-Item -Path "{temp_path}" -Force -ErrorAction SilentlyContinue', timeout_seconds=5)
                return result

        logger.info(
            "[implementer] v1.15 All %d chunks written to temp file (%d total b64 chars)",
            len(chunks), len(encoded),
        )

        # Step B: Read temp file, decode Base64, write to target path, clean up
        decode_cmd = (
            f'$b64 = [System.IO.File]::ReadAllText("{temp_path}"); '
            f'$bytes = [System.Convert]::FromBase64String($b64); '
            f'[System.IO.File]::WriteAllBytes("{path}", $bytes); '
            f'Remove-Item -Path "{temp_path}" -Force -ErrorAction SilentlyContinue; '
            f'"WRITE_OK"'
        )
        result = client.shell_run(decode_cmd, timeout_seconds=timeout_seconds)

        if result.stdout and "WRITE_OK" in result.stdout:
            logger.info("[implementer] v1.15 Temp-file write succeeded for %s", path)
        else:
            logger.error(
                "[implementer] v1.15 INFRASTRUCTURE_ERROR: Decode/write failed for %s: "
                "stderr=%s, stdout=%s",
                path, (result.stderr or "")[:200],
                (result.stdout or "")[:200],
            )
            # Clean up on failure
            client.shell_run(f'Remove-Item -Path "{temp_path}" -Force -ErrorAction SilentlyContinue', timeout_seconds=5)

        return result

@dataclass
class ImplementerResult:
    """Result from Implementer execution."""
    success: bool
    output_path: Optional[str] = None
    sha256: Optional[str] = None
    error: Optional[str] = None
    duration_ms: int = 0
    sandbox_used: bool = False
    filename: Optional[str] = None
    content_written: Optional[str] = None
    action_taken: Optional[str] = None
    write_method: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output_path": self.output_path,
            "sha256": self.sha256,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "sandbox_used": self.sandbox_used,
            "filename": self.filename,
            "content_written": self.content_written,
            "action_taken": self.action_taken,
            "write_method": self.write_method,
        }

async def run_multi_file_operation(
    *,
    multi_file: Dict[str, Any],
    client: Optional[SandboxClient] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> MultiFileResult:
    """
    v1.11: Dispatch to appropriate multi-file handler based on operation type.
    
    This is the main entry point for multi-file operations.
    
    Args:
        multi_file: Dict with multi_file data from spec
        client: Optional sandbox client
        progress_callback: Optional callback for progress updates
        
    Returns:
        MultiFileResult from appropriate handler
    """
    from .implementer import MultiFileResult, run_multi_file_refactor, run_multi_file_search
    operation_type = multi_file.get("operation_type", "search")
    
    logger.info(
        "[implementer] v1.11 run_multi_file_operation: type=%s",
        operation_type
    )
    
    if operation_type == "refactor":
        return await run_multi_file_refactor(
            multi_file=multi_file,
            client=client,
            progress_callback=progress_callback,
        )
    else:
        return await run_multi_file_search(
            multi_file=multi_file,
            client=client,
            progress_callback=progress_callback,
        )
