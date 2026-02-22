from __future__ import annotations
import logging
import re
from app.overwatcher.sandbox_client import SandboxClient, get_sandbox_client
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


MULTI_FILE_VERIFY_TIMEOUT = 30  # Seconds per file verification

def _parse_corrections(generated_reply: str) -> Dict[int, str]:
    """
    v1.10: Parse SpecGate's correction output format.
    
    Input formats supported:
        Q1: [INCORRECT] Correct answer: 2. The sum of 1+1 is 2.
        Q5: [MISSING] Answer: const. The keyword for constants is const.
        Q3: [INCORRECT] The correct answer is O(log n) because...
        Q7: [TRICK] This is a trick question because 1/0 raises ZeroDivisionError.
    
    Returns:
        Dict mapping question number to corrected answer text
        {1: "2", 5: "const", 3: "O(log n)", ...}
    """
    corrections = {}
    
    if not generated_reply:
        return corrections
    
    # Split by Q# markers to process each correction
    # Pattern: Q followed by number, colon, then status in brackets
    q_pattern = r'Q(\d+):\s*\[([A-Z]+)\]\s*(.*?)(?=Q\d+:\s*\[|$)'
    
    for match in re.finditer(q_pattern, generated_reply, re.DOTALL | re.IGNORECASE):
        q_num = int(match.group(1))
        status = match.group(2).upper()
        explanation = match.group(3).strip()
        
        # Skip CORRECT entries - no change needed
        if status == "CORRECT":
            continue
        
        # Extract the answer from the explanation
        answer = None
        
        # Try various answer extraction patterns
        answer_patterns = [
            # "Correct answer: X" or "Answer: X"
            r'(?:Correct answer|Answer)[:\s]+([^.]+)',
            # "The correct answer is X"
            r'(?:The )?correct answer (?:is|should be)[:\s]+([^.]+)',
            # "should be X" 
            r'should be[:\s]+([^.]+)',
            # "is X" at start after status
            r'^(?:is\s+)?([^.]{1,100})',
        ]
        
        for pattern in answer_patterns:
            ans_match = re.search(pattern, explanation, re.IGNORECASE)
            if ans_match:
                answer = ans_match.group(1).strip()
                # Clean up common trailing content
                answer = re.sub(r'\s*\(.*$', '', answer)  # Remove parenthetical
                answer = re.sub(r'\s*because.*$', '', answer, flags=re.IGNORECASE)  # Remove "because..."
                answer = answer.rstrip('.,;:')
                if answer:
                    break
        
        # For TRICK questions, use the full explanation as the "answer"
        if status == "TRICK" and not answer:
            answer = f"TRICK QUESTION: {explanation[:200]}"
        
        if answer:
            corrections[q_num] = answer
            logger.info("[implementer] v1.10 Parsed correction Q%d [%s]: '%s'", q_num, status, answer[:50])
        else:
            logger.warning("[implementer] v1.10 Could not extract answer for Q%d from: %s", q_num, explanation[:100])
    
    logger.info("[implementer] v1.10 _parse_corrections: parsed %d corrections", len(corrections))
    return corrections

def _parse_answers_from_reply(reply_text: str) -> Dict[int, str]:
    """Parse SpecGate's combined reply to extract individual answers."""
    answers: Dict[int, str] = {}
    
    if not reply_text:
        return answers
    
    # Try "Question N:" pattern first
    pattern = r'Question\s*(\d+)\s*[:\.]?\s*(.*?)(?=Question\s*\d+|$)'
    matches = re.findall(pattern, reply_text, re.IGNORECASE | re.DOTALL)
    
    if matches:
        for q_num_str, answer_text in matches:
            q_num = int(q_num_str)
            answer = answer_text.strip()
            if answer:
                answers[q_num] = answer
        
        if answers:
            logger.info("[implementer] Parsed %d answers from 'Question N:' format", len(answers))
            return answers
    
    # Fallback: Split by double newlines
    parts = re.split(r'\n\n+', reply_text.strip())
    if len(parts) > 1:
        for i, part in enumerate(parts, start=1):
            part = part.strip()
            if part:
                answers[i] = part
        
        logger.info("[implementer] Parsed %d answers from double-newline split", len(answers))
        return answers
    
    # Last resort: entire reply as answer to question 1
    answers[1] = reply_text.strip()
    logger.info("[implementer] Using entire reply as single answer")
    
    return answers

@dataclass
class VerificationResult:
    """Result from verification step."""
    passed: bool
    file_exists: bool = False
    content_matches: bool = False
    filename_matches: bool = False
    actual_content: Optional[str] = None
    expected_content: Optional[str] = None
    expected_filename: Optional[str] = None
    actual_filename: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "file_exists": self.file_exists,
            "content_matches": self.content_matches,
            "filename_matches": self.filename_matches,
            "actual_content": self.actual_content,
            "expected_content": self.expected_content,
            "expected_filename": self.expected_filename,
            "actual_filename": self.actual_filename,
            "error": self.error,
        }

@dataclass
class AtomicTaskResult:
    """v1.12: Result from a single atomic task execution.
    
    Used by architecture_executor and future task-based callers.
    The Implementer is the ONLY writer — this interface enforces that.
    """
    success: bool
    path: str
    action: str  # "create" or "modify"
    chars_written: int = 0
    error: Optional[str] = None
    verified: bool = False
    duration_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "path": self.path,
            "action": self.action,
            "chars_written": self.chars_written,
            "error": self.error,
            "verified": self.verified,
            "duration_ms": self.duration_ms,
        }

async def run_implementer_task(
    *,
    path: str,
    content: str,
    action: str = "create",
    ensure_parents: bool = True,
    client: Optional[SandboxClient] = None,
) -> AtomicTaskResult:
    """v1.12: Execute a single atomic write task in the sandbox.
    
    This is the task-level interface for the Implementer.
    The architecture_executor (Overwatcher) calls this for each file.
    
    IMPLEMENTER IS THE ONLY WRITER.
    
    Flow:
        1. Ensure parent directory exists (if ensure_parents=True)
        2. Write file to sandbox via Base64 encoding
        3. Read back to verify the write
        4. Return result with verification status
    
    Args:
        path: Absolute sandbox path to write to
        content: Complete file content to write
        action: "create" for new files, "modify" for modifications
        ensure_parents: Create parent directories if needed
        client: Optional sandbox client (uses default if None)
    
    Returns:
        AtomicTaskResult with success/failure and verification
    """
    from .implementer import _write_content_to_sandbox
    import time
    start_time = time.time()
    
    def elapsed() -> int:
        return int((time.time() - start_time) * 1000)
    
    logger.info(
        "[implementer] v1.12 Atomic task: action=%s, path=%s, content=%d chars",
        action, path, len(content),
    )
    print(f"[IMPLEMENTER_TASK] {action.upper()}: {path} ({len(content)} chars)")
    
    if client is None:
        client = get_sandbox_client()
    
    if not client.is_connected():
        return AtomicTaskResult(
            success=False,
            path=path,
            action=action,
            error="SAFETY: Sandbox not available",
            duration_ms=elapsed(),
        )
    
    # Step 1: Ensure parent directory exists
    if ensure_parents:
        parent_dir = str(Path(path).parent)
        try:
            mkdir_cmd = (
                f'if (-not (Test-Path -Path "{parent_dir}")) '
                f'{{ New-Item -ItemType Directory -Path "{parent_dir}" -Force | Out-Null; '
                f'"CREATED" }} else {{ "EXISTS" }}'
            )
            mkdir_result = client.shell_run(mkdir_cmd, timeout_seconds=10)
            if mkdir_result.stdout and ("CREATED" in mkdir_result.stdout or "EXISTS" in mkdir_result.stdout):
                logger.debug("[implementer] v1.12 Parent dir: %s", mkdir_result.stdout.strip())
            else:
                logger.warning(
                    "[implementer] v1.12 mkdir uncertain for %s: %s",
                    parent_dir, mkdir_result.stderr or ""
                )
        except Exception as e:
            return AtomicTaskResult(
                success=False,
                path=path,
                action=action,
                error=f"Failed to create parent directory {parent_dir}: {e}",
                duration_ms=elapsed(),
            )
    
    # Step 2: Write file via _write_content_to_sandbox (v1.13: auto temp-file for large files)
    try:
        write_result = _write_content_to_sandbox(client, path, content, timeout_seconds=60)
        
        if write_result.stderr and write_result.stderr.strip():
            return AtomicTaskResult(
                success=False,
                path=path,
                action=action,
                error=f"Write failed: {write_result.stderr[:200]}",
                duration_ms=elapsed(),
            )
        
        logger.info("[implementer] v1.13 Wrote %d chars to %s", len(content), path)
        
    except Exception as e:
        return AtomicTaskResult(
            success=False,
            path=path,
            action=action,
            error=f"Write exception: {e}",
            duration_ms=elapsed(),
        )
    
    # Step 3: Read back to verify
    # v1.16: Compare stripped content only. PowerShell's Get-Content -Raw
    # appends a trailing newline, causing a consistent +1 byte mismatch.
    # The .strip() comparison is the authoritative check — if stripped
    # content matches, the write succeeded. No length-based warning needed.
    verified = False
    try:
        read_cmd = f'Get-Content -Path "{path}" -Raw -Encoding UTF8'
        read_result = client.shell_run(read_cmd, timeout_seconds=30)
        
        if read_result.stdout is not None:
            _written_stripped = content.strip()
            _readback_stripped = read_result.stdout.strip()
            if _readback_stripped == _written_stripped:
                verified = True
                logger.info("[implementer] v1.16 Verified: %s (%d chars)", path, len(content))
            else:
                # Genuine mismatch — log both stripped lengths for debugging
                logger.warning(
                    "[implementer] v1.16 Verify MISMATCH for %s "
                    "(wrote_stripped=%d, read_stripped=%d)",
                    path, len(_written_stripped), len(_readback_stripped),
                )
        else:
            logger.warning("[implementer] v1.16 Verify read returned None for %s", path)
    except Exception as e:
        logger.warning("[implementer] v1.16 Verify exception for %s: %s", path, e)
    
    print(
        f"[IMPLEMENTER_TASK] {'✓' if verified else '⚠'} "
        f"{action.upper()} {path} ({len(content)} chars, verified={verified})"
    )
    
    return AtomicTaskResult(
        success=True,
        path=path,
        action=action,
        chars_written=len(content),
        verified=verified,
        duration_ms=elapsed(),
    )

@dataclass
class EditTaskResult:
    """v1.13: Result from a targeted edit task.
    
    Used when MODIFY operations can be expressed as {old_text, new_text} pairs
    instead of requiring the LLM to regenerate the entire file.
    """
    success: bool
    path: str
    edits_applied: int = 0
    edits_failed: int = 0
    chars_before: int = 0
    chars_after: int = 0
    error: Optional[str] = None
    failed_edits: List[Dict[str, str]] = field(default_factory=list)
    verified: bool = False
    duration_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "path": self.path,
            "edits_applied": self.edits_applied,
            "edits_failed": self.edits_failed,
            "chars_before": self.chars_before,
            "chars_after": self.chars_after,
            "error": self.error,
            "failed_edits": self.failed_edits,
            "verified": self.verified,
            "duration_ms": self.duration_ms,
        }
