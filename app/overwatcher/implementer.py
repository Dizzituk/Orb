# FILE: app/overwatcher/implementer.py
"""Implementer: Executes approved work and verifies results.

Handles:
- Writing files to sandbox based on spec
- Enforcing must_exist constraint for modify actions
- Verifying output matches spec requirements
- APPEND_IN_PLACE mode for appending to existing files (v1.2)
- REWRITE_IN_PLACE mode for multi-question file edits (v1.3)
- CHAT_ONLY mode safety: no file writes at all (v1.4)

v1.5 (2026-01-25): REWRITE_IN_PLACE improvements for Q&A file tasks
    - SpecGate v1.17 added "fill in the missing", "fill blank", etc. triggers
    - These trigger REWRITE_IN_PLACE (intelligent Q&A block insertion)
    - IMPROVED: _block_has_answer() now checks for actual content, not just marker
    - IMPROVED: _insert_answers_under_questions() handles blank "Answer:" sections
    - If Answer: exists but is empty → insert content AFTER the Answer: line
    - If Answer: has content → skip (preserve existing answer)
    - If no Answer: marker → insert full "Answer:\n{content}" at end of block
    - Better logging: shows skipped_filled and skipped_no_answer lists
    - Updated BUILD_ID to v1.5 for verification
v1.4.1 (2026-01-25): CHAT_ONLY safety fix - BULLETPROOF EDITION
    - Added BUILD_ID for verifying correct code is running
    - Added repr() logging of raw output_mode for debugging
    - Added strip() to normalize whitespace in output_mode
    - CHAT_ONLY early return is now the FIRST check after reading spec
    - Aggressive logging to prove code path execution
v1.4 (2026-01-24): CHAT_ONLY safety fix - CRITICAL BUG FIX
    - CHAT_ONLY mode is a true no-op: absolutely no file writes
    - Early return BEFORE any path resolution or sandbox operations
    - Unknown/empty output_mode now fails safely instead of defaulting to overwrite
    - Verification passes immediately for CHAT_ONLY without file checks
    - Valid modes: chat_only, rewrite_in_place, append_in_place, separate_reply_file
v1.3 (2026-01-24): Added REWRITE_IN_PLACE support for multi-question file edits
    - Reads entire file, parses question blocks, inserts answers, writes back
    - Question detection: "Question N:" headers and lines ending with "?"
    - Skips insertion if block already contains "Answer:" (avoids duplicates)
    - Answers inserted at END of each question block, before next question
v1.2 (2026-01-24): Added APPEND_IN_PLACE support with insertion_format
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.overwatcher.overwatcher import OverwatcherOutput, Decision
from app.overwatcher.sandbox_client import (
    SandboxClient,
    SandboxError,
    get_sandbox_client,
)

from .spec_resolution import ResolvedSpec, SpecMissingDeliverableError

logger = logging.getLogger(__name__)

# =============================================================================
# v1.5 BUILD VERIFICATION - Proves correct code is running
# v1.5: Added support for REWRITE_IN_PLACE "fill in" triggers from SpecGate v1.17
# =============================================================================
IMPLEMENTER_BUILD_ID = "2026-01-25-v1.5-rewrite-in-place-fill"
print(f"[IMPLEMENTER_LOADED] BUILD_ID={IMPLEMENTER_BUILD_ID}")
logger.info(f"[implementer] Module loaded: BUILD_ID={IMPLEMENTER_BUILD_ID}")


def _is_absolute_windows_path(path: str) -> bool:
    """Check if path is an absolute Windows path (e.g., C:\\..., D:\\...)."""
    if len(path) >= 3:
        return path[1] == ':' and path[2] in ('\\', '/')
    return False


def _escape_powershell_string(s: str) -> str:
    """Escape a string for use in PowerShell double-quoted strings."""
    # Escape backticks first, then quotes, then dollar signs
    return s.replace('`', '``').replace('"', '`"').replace('$', '`$')


def _generate_sandbox_path_candidates(path: str) -> List[str]:
    """Generate candidate paths for sandbox resolution.
    
    For Desktop paths, tries:
    1. Original path as-is
    2. Same user, non-OneDrive Desktop (if original has OneDrive)
    3. WDAGUtilityAccount OneDrive Desktop (if original has OneDrive)
    4. WDAGUtilityAccount Desktop
    
    Args:
        path: Original absolute Windows path
        
    Returns:
        List of candidate paths to try (in priority order)
    """
    candidates = [path]
    
    # Match: C:\Users\<username>\OneDrive\Desktop\<rest>
    onedrive_match = re.match(
        r'^([A-Za-z]):\\Users\\([^\\]+)\\OneDrive\\Desktop\\(.*)$',
        path,
        re.IGNORECASE
    )
    if onedrive_match:
        drive = onedrive_match.group(1)
        username = onedrive_match.group(2)
        rest = onedrive_match.group(3)
        
        # Candidate 2: Same user, non-OneDrive Desktop
        non_onedrive = f"{drive}:\\Users\\{username}\\Desktop\\{rest}"
        if non_onedrive not in candidates:
            candidates.append(non_onedrive)
        
        # Candidate 3: WDAGUtilityAccount OneDrive Desktop
        wdag_onedrive = f"{drive}:\\Users\\WDAGUtilityAccount\\OneDrive\\Desktop\\{rest}"
        if wdag_onedrive not in candidates:
            candidates.append(wdag_onedrive)
        
        # Candidate 4: WDAGUtilityAccount Desktop
        wdag = f"{drive}:\\Users\\WDAGUtilityAccount\\Desktop\\{rest}"
        if wdag not in candidates:
            candidates.append(wdag)
        
        return candidates
    
    # Match: C:\Users\<username>\Desktop\<rest> (non-OneDrive)
    desktop_match = re.match(
        r'^([A-Za-z]):\\Users\\([^\\]+)\\Desktop\\(.*)$',
        path,
        re.IGNORECASE
    )
    if desktop_match:
        drive = desktop_match.group(1)
        username = desktop_match.group(2)
        rest = desktop_match.group(3)
        
        # Candidate 2: WDAGUtilityAccount Desktop
        wdag = f"{drive}:\\Users\\WDAGUtilityAccount\\Desktop\\{rest}"
        if wdag not in candidates:
            candidates.append(wdag)
        
        return candidates
    
    return candidates


# =============================================================================
# REWRITE_IN_PLACE HELPERS (v1.3)
# =============================================================================

def _find_question_block_starts(text: str) -> List[Tuple[int, int, str]]:
    """
    Find all question block start positions in text.
    
    Returns list of (line_number, char_position, question_identifier) tuples.
    
    Question detection (per user spec):
    - Lines starting with "Question N:" (case-insensitive)
    - Lines ending with "?" that also start with "N." or "N)" pattern
    
    Examples:
        "Question 1: What is Python?" -> (line_num, pos, "1")
        "1. How does X work?" -> (line_num, pos, "1")
        "2) Why is Y?" -> (line_num, pos, "2")
    """
    blocks: List[Tuple[int, int, str]] = []
    lines = text.split('\n')
    char_pos = 0
    
    for line_num, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Pattern 1: "Question N:" (case-insensitive)
        question_match = re.match(r'^question\s*(\d+)\s*[:\.]', line_stripped, re.IGNORECASE)
        if question_match:
            q_num = question_match.group(1)
            blocks.append((line_num, char_pos, q_num))
            char_pos += len(line) + 1  # +1 for newline
            continue
        
        # Pattern 2: Numbered line ending with "?" (e.g., "1. How?" or "2) Why?")
        numbered_question_match = re.match(r'^(\d+)[.\)]\s*', line_stripped)
        if numbered_question_match and line_stripped.rstrip().endswith('?'):
            q_num = numbered_question_match.group(1)
            blocks.append((line_num, char_pos, q_num))
        
        char_pos += len(line) + 1  # +1 for newline
    
    return blocks


def _block_has_answer(block_text: str) -> bool:
    """
    Check if a question block already has a non-empty answer.
    
    v1.5: IMPROVED - Now checks if Answer: has actual content, not just existence.
    
    Returns True if "Answer:" is found AND has non-whitespace content after it.
    Returns False if:
    - No "Answer:" exists in the block
    - "Answer:" exists but is followed only by whitespace/blank lines
    
    This allows us to fill in blank "Answer:" sections while preserving
    already-filled answers.
    """
    text_lower = block_text.lower()
    
    # Find the position of "Answer:" (case-insensitive)
    answer_pos = text_lower.find('answer:')
    if answer_pos == -1:
        # No Answer: marker at all
        return False
    
    # Get everything after "Answer:"
    after_answer = block_text[answer_pos + len('answer:'):]
    
    # Check if there's non-whitespace content
    # Strip all whitespace and see if anything remains
    content = after_answer.strip()
    
    has_content = len(content) > 0
    
    logger.debug(
        "[implementer] v1.5 _block_has_answer: answer_pos=%d, after_answer='%s', has_content=%s",
        answer_pos, after_answer[:50] if after_answer else "", has_content
    )
    
    return has_content


def _parse_answers_from_reply(reply_text: str) -> Dict[int, str]:
    """
    Parse SpecGate's combined reply to extract individual answers.
    
    Handles two formats:
    1. "Question N: <answer>" pattern (multiline, until next Question N or end)
    2. Double-newline separated answers (fallback)
    
    Returns:
        Dict mapping question number (1-indexed) to answer text
    """
    answers: Dict[int, str] = {}
    
    if not reply_text:
        return answers
    
    # Try to parse "Question N:" pattern first
    # This regex captures: Question 1: <answer until next Question or end>
    pattern = r'Question\s*(\d+)\s*[:\.]?\s*(.*?)(?=Question\s*\d+|$)'
    matches = re.findall(pattern, reply_text, re.IGNORECASE | re.DOTALL)
    
    if matches:
        for q_num_str, answer_text in matches:
            q_num = int(q_num_str)
            answer = answer_text.strip()
            if answer:  # Only add non-empty answers
                answers[q_num] = answer
        
        if answers:
            logger.info(
                "[implementer] v1.3 Parsed %d answers from 'Question N:' format",
                len(answers)
            )
            return answers
    
    # Fallback: Split by double newlines
    parts = re.split(r'\n\n+', reply_text.strip())
    if len(parts) > 1:
        for i, part in enumerate(parts, start=1):
            part = part.strip()
            if part:
                answers[i] = part
        
        logger.info(
            "[implementer] v1.3 Parsed %d answers from double-newline split (fallback)",
            len(answers)
        )
        return answers
    
    # Last resort: treat entire reply as answer to question 1
    answers[1] = reply_text.strip()
    logger.info("[implementer] v1.3 Using entire reply as single answer (last resort)")
    
    return answers


def _insert_answers_under_questions(
    original_text: str,
    answers: Dict[int, str],
    insertion_format: str,
) -> str:
    """
    Insert answers at the appropriate position in each question block.
    
    v1.5: IMPROVED ALGORITHM - Handles blank "Answer:" sections correctly.
    
    Algorithm:
    1. Find all question block start positions
    2. For each block, determine its end (next question start or EOF)
    3. Check if block already has non-empty "Answer:" - if so, skip
    4. If block has empty "Answer:", insert content AFTER the Answer: line
    5. If block has no "Answer:", insert full formatted answer at end of block
    
    Args:
        original_text: The original file content
        answers: Dict mapping question number to answer text
        insertion_format: Format string with {reply} placeholder
        
    Returns:
        Modified text with answers inserted
    """
    if not answers:
        logger.warning("[implementer] v1.5 No answers to insert")
        return original_text
    
    # Find all question block starts
    block_starts = _find_question_block_starts(original_text)
    
    if not block_starts:
        logger.warning("[implementer] v1.5 No question blocks found in file")
        return original_text
    
    logger.info(
        "[implementer] v1.5 Found %d question blocks: %s",
        len(block_starts),
        [(bs[0], bs[2]) for bs in block_starts]  # (line_num, q_num)
    )
    
    lines = original_text.split('\n')
    
    # Build list of (line_number, question_number, block_end_line)
    # block_end_line is the line BEFORE the next question, or the last line
    blocks_with_ends: List[Tuple[int, int, int]] = []
    
    for i, (start_line, _, q_num_str) in enumerate(block_starts):
        q_num = int(q_num_str)
        
        # Determine block end
        if i + 1 < len(block_starts):
            # Next question starts at block_starts[i+1][0]
            end_line = block_starts[i + 1][0] - 1
        else:
            # Last block - ends at last line
            end_line = len(lines) - 1
        
        blocks_with_ends.append((start_line, q_num, end_line))
    
    # Process blocks in REVERSE order to preserve line numbers
    insertions_made = 0
    skipped_filled = []
    skipped_no_answer = []
    
    for start_line, q_num, end_line in reversed(blocks_with_ends):
        # Check if we have an answer for this question
        if q_num not in answers:
            logger.info(
                "[implementer] v1.5 No answer for question %d, skipping",
                q_num
            )
            skipped_no_answer.append(q_num)
            continue
        
        # Extract block text to check for existing answer
        block_lines = lines[start_line:end_line + 1]
        block_text = '\n'.join(block_lines)
        
        if _block_has_answer(block_text):
            logger.info(
                "[implementer] v1.5 Question %d already has non-empty Answer:, skipping",
                q_num
            )
            skipped_filled.append(q_num)
            continue
        
        # Get the answer text
        answer_text = answers[q_num]
        
        # v1.5: Check if block has an empty "Answer:" marker
        # If so, insert content AFTER that line instead of at end
        answer_line_idx = None
        for i, line in enumerate(block_lines):
            if line.strip().lower().startswith('answer:'):
                answer_line_idx = start_line + i
                break
        
        if answer_line_idx is not None:
            # Block has empty "Answer:" - insert content right after it
            # The answer content goes on the next line
            insert_position = answer_line_idx + 1
            
            # Just insert the answer content (Answer: header already exists)
            answer_lines = [answer_text]
            
            logger.info(
                "[implementer] v1.5 Inserting answer for Q%d AFTER existing Answer: at line %d",
                q_num, answer_line_idx
            )
        else:
            # No "Answer:" marker - insert full formatted answer at end of block
            insert_position = end_line + 1
            
            try:
                formatted_answer = insertion_format.format(reply=answer_text)
            except KeyError as e:
                logger.warning(
                    "[implementer] v1.5 insertion_format KeyError: %s, using default",
                    e
                )
                formatted_answer = f"\n\nAnswer:\n{answer_text}\n"
            
            # Ensure the formatted answer starts fresh (has leading newlines)
            if not formatted_answer.startswith('\n'):
                formatted_answer = '\n' + formatted_answer
            
            answer_lines = formatted_answer.split('\n')
            
            logger.info(
                "[implementer] v1.5 Inserting full Answer block for Q%d at end of block (line %d)",
                q_num, end_line
            )
        
        # Insert the answer lines at the correct position
        lines[insert_position:insert_position] = answer_lines
        
        insertions_made += 1
    
    logger.info(
        "[implementer] v1.5 REWRITE_IN_PLACE complete: %d insertions, skipped_filled=%s, skipped_no_answer=%s",
        insertions_made, skipped_filled, skipped_no_answer
    )
    
    return '\n'.join(lines)


# =============================================================================
# DATA CLASSES
# =============================================================================

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
    write_method: Optional[str] = None  # v1.2: "append", "overwrite", or "rewrite"
    
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


async def run_implementer(
    *,
    spec: ResolvedSpec,
    output: OverwatcherOutput,
    client: Optional[SandboxClient] = None,
) -> ImplementerResult:
    """Execute approved work via Sandbox.
    
    For action="modify" with must_exist=True:
    - Checks file exists BEFORE writing
    - Fails if file doesn't exist (does NOT create it)
    
    v1.3: Added REWRITE_IN_PLACE support
    - Reads entire file, parses question blocks, inserts answers, writes back
    - Used when output_mode="rewrite_in_place"
    
    v1.2: Added APPEND_IN_PLACE support
    - When output_mode="append_in_place", uses Add-Content instead of Set-Content
    - Respects insertion_format for formatting the appended text
    """
    import time
    start_time = time.time()
    
    def elapsed() -> int:
        return int((time.time() - start_time) * 1000)
    
    if output.decision != Decision.PASS:
        return ImplementerResult(
            success=False,
            error=f"Overwatcher decision was {output.decision.value}",
            duration_ms=elapsed(),
        )
    
    # Get spec-driven file details
    try:
        filename, content, action = spec.get_target_file()
        target = spec.get_target()
        must_exist = spec.get_must_exist()
        # v1.2/v1.3: Get output mode and insertion format
        output_mode = spec.get_output_mode()
        insertion_format = spec.get_insertion_format()
    except SpecMissingDeliverableError as e:
        return ImplementerResult(
            success=False,
            error=str(e),
            duration_ms=elapsed(),
        )
    
    # =========================================================================
    # v1.4.1: BULLETPROOF CHAT_ONLY CHECK - MUST BE FIRST!
    # This check happens IMMEDIATELY after reading spec, BEFORE any other ops
    # =========================================================================
    
    # v1.4.1: Aggressive logging to prove code path
    logger.info(f"[implementer] v1.4.1 BUILD_ID={IMPLEMENTER_BUILD_ID}")
    logger.info(f"[implementer] v1.4.1 RAW output_mode={repr(output_mode)}")
    print(f"\n>>> [IMPLEMENTER v1.4.1] BUILD={IMPLEMENTER_BUILD_ID} <<<")
    print(f">>> [IMPLEMENTER v1.4.1] RAW output_mode={repr(output_mode)} <<<\n")
    
    # v1.4.1: Normalize with strip() to handle any whitespace
    mode_lower = (output_mode or "").strip().lower()
    
    logger.info(f"[implementer] v1.4.1 NORMALIZED mode_lower={repr(mode_lower)}")
    print(f">>> [IMPLEMENTER v1.4.1] NORMALIZED mode_lower={repr(mode_lower)} <<<\n")
    
    # v1.4.1: CHAT_ONLY HARD STOP - Return IMMEDIATELY, no sandbox, no writes
    if mode_lower == "chat_only":
        logger.info("[implementer] v1.4.1 CHAT_ONLY DETECTED - RETURNING EARLY (NO FILE OPS)")
        print(f"\n>>> [IMPLEMENTER v1.4.1] CHAT_ONLY SAFETY: EXITING NOW - NO FILE WRITES <<<\n")
        return ImplementerResult(
            success=True,
            output_path=None,  # No file written
            sha256=None,
            duration_ms=elapsed(),
            sandbox_used=False,  # No sandbox write occurred
            filename=filename,
            content_written=None,  # Nothing written to disk
            action_taken="chat_only_noop",
            write_method="none",  # Explicit: no write method used
        )
    
    # =========================================================================
    # Past this point: NOT chat_only, proceed with normal logging and execution
    # =========================================================================
    
    logger.info(f"[implementer] === SPEC-DRIVEN TASK ===  [v1.4.1 MODE: {mode_lower}]")
    logger.info(f"[implementer] Action: {action}")
    logger.info(f"[implementer] Filename: {filename}")
    logger.info(f"[implementer] Target: {target}")
    logger.info(f"[implementer] Content: '{content[:100]}...' ({len(content)} chars)" if len(content) > 100 else f"[implementer] Content: '{content}'")
    logger.info(f"[implementer] Must exist: {must_exist}")
    logger.info(f"[implementer] Output mode: {output_mode}")
    logger.info(f"[implementer] Insertion format: {repr(insertion_format)}")
    
    # Get sandbox client
    if client is None:
        client = get_sandbox_client()
    
    try:
        if not client.is_connected():
            return ImplementerResult(
                success=False,
                error="SAFETY: Sandbox not available",
                duration_ms=elapsed(),
                sandbox_used=False,
            )
        
        # Build expected path
        if _is_absolute_windows_path(filename):
            expected_path = filename
            base_filename = Path(filename).name
            is_absolute = True
            logger.info(f"[implementer] Using absolute path as-is: {expected_path}")
        else:
            base_filename = filename
            is_absolute = False
            if target == "DESKTOP":
                expected_path = f"C:\\Users\\WDAGUtilityAccount\\Desktop\\{base_filename}"
            else:
                expected_path = f"{target}\\{base_filename}"
        
        # For "modify" action with must_exist: verify file exists first
        if action == "modify" and must_exist:
            sandbox_whoami = "<unknown>"
            sandbox_userprofile = "<unknown>"
            try:
                whoami_result = client.shell_run("whoami", timeout_seconds=5)
                sandbox_whoami = whoami_result.stdout.strip() if whoami_result.stdout else f"<error: {whoami_result.stderr}>"
                userprofile_result = client.shell_run("echo $env:USERPROFILE", timeout_seconds=5)
                sandbox_userprofile = userprofile_result.stdout.strip() if userprofile_result.stdout else f"<error: {userprofile_result.stderr}>"
                logger.info(f"[implementer] Sandbox env: whoami={sandbox_whoami}")
                logger.info(f"[implementer] Sandbox env: USERPROFILE={sandbox_userprofile}")
            except Exception as e:
                logger.warning(f"[implementer] Could not log sandbox env: {e}")
            
            candidates = _generate_sandbox_path_candidates(expected_path)
            
            resolved_path = None
            candidate_results = []
            for candidate in candidates:
                logger.info(f"[implementer] Checking existence: {candidate}")
                exists_cmd = f'Test-Path -Path "{candidate}"'
                exists_result = client.shell_run(exists_cmd, timeout_seconds=10)
                
                file_exists = "True" in exists_result.stdout
                candidate_results.append((candidate, file_exists))
                logger.info(f"[implementer] Exists? {candidate} -> {file_exists}")
                
                if file_exists:
                    resolved_path = candidate
                    break
            
            if resolved_path is None:
                candidate_lines = "\n".join(f"  - {c} -> {r}" for c, r in candidate_results)
                error_msg = (
                    f"SPEC VIOLATION: File '{filename}' does not exist at any candidate path.\n"
                    f"Tried:\n{candidate_lines}\n"
                    f"Sandbox env:\n"
                    f"  - whoami: {sandbox_whoami}\n"
                    f"  - USERPROFILE: {sandbox_userprofile}\n"
                    f"Spec requires modifying an existing file (action=modify, must_exist=True). "
                    f"Cannot create a new file."
                )
                return ImplementerResult(
                    success=False,
                    error=error_msg,
                    duration_ms=elapsed(),
                    sandbox_used=True,
                    filename=filename,
                    action_taken="existence_check_failed",
                )
            
            expected_path = resolved_path
            logger.info(f"[implementer] File exists at: {expected_path}, proceeding with modify")
        
        # =====================================================================
        # WRITE FILE VIA SANDBOX
        # v1.3: Branch on output_mode for REWRITE vs APPEND vs OVERWRITE
        # =====================================================================
        
        if is_absolute:
            logger.info(f"[implementer] Writing via PowerShell to absolute path: {expected_path}")
            
            # Normalize output_mode
            mode_lower = (output_mode or "").lower()
            
            # v1.3: REWRITE_IN_PLACE mode - read, parse, insert, write back
            if mode_lower == "rewrite_in_place":
                logger.info("[implementer] v1.3 REWRITE_IN_PLACE mode: multi-question file edit")
                
                # Step 1: Read entire file
                read_cmd = f'Get-Content -Path "{expected_path}" -Raw'
                read_result = client.shell_run(read_cmd, timeout_seconds=30)
                
                if read_result.stderr and read_result.stderr.strip():
                    return ImplementerResult(
                        success=False,
                        error=f"Failed to read file for rewrite: {read_result.stderr}",
                        duration_ms=elapsed(),
                        sandbox_used=True,
                        write_method="rewrite",
                    )
                
                original_text = read_result.stdout or ""
                logger.info(f"[implementer] v1.3 Read {len(original_text)} chars from file")
                
                # Step 2: Parse answers from SpecGate's reply (content)
                answers = _parse_answers_from_reply(content)
                logger.info(f"[implementer] v1.3 Parsed {len(answers)} answers: {list(answers.keys())}")
                
                # Step 3: Insert answers under questions
                fmt = insertion_format or "\n\nAnswer:\n{reply}\n"
                updated_text = _insert_answers_under_questions(original_text, answers, fmt)
                
                # Step 4: Write back entire file
                escaped_content = _escape_powershell_string(updated_text)
                write_cmd = f'Set-Content -Path "{expected_path}" -Value "{escaped_content}" -NoNewline -Encoding UTF8'
                
                logger.info(f"[implementer] v1.3 Writing {len(updated_text)} chars back to file")
                write_result = client.shell_run(write_cmd, timeout_seconds=30)
                
                write_success = not write_result.stderr or write_result.stderr.strip() == ""
                if write_success:
                    logger.info(f"[implementer] v1.3 SUCCESS: REWRITE_IN_PLACE completed for {expected_path}")
                    return ImplementerResult(
                        success=True,
                        output_path=expected_path,
                        sha256=None,
                        duration_ms=elapsed(),
                        sandbox_used=True,
                        filename=filename,
                        content_written=updated_text,
                        action_taken=action,
                        write_method="rewrite",
                    )
                else:
                    return ImplementerResult(
                        success=False,
                        error=f"PowerShell write failed (rewrite): {write_result.stderr or write_result.stdout}",
                        duration_ms=elapsed(),
                        sandbox_used=True,
                        write_method="rewrite",
                    )
            
            # v1.2: APPEND_IN_PLACE mode
            elif mode_lower == "append_in_place":
                if insertion_format:
                    try:
                        append_text = insertion_format.format(reply=content)
                    except KeyError as e:
                        logger.warning(f"[implementer] insertion_format KeyError: {e}, using simple format")
                        append_text = f"\n\nAnswer:\n{content}\n"
                else:
                    append_text = f"\n\nAnswer:\n{content}\n"
                
                logger.info(f"[implementer] APPEND_IN_PLACE mode: appending {len(append_text)} chars")
                logger.info(f"[implementer] Append text preview: {repr(append_text[:100])}")
                
                escaped_append = _escape_powershell_string(append_text)
                write_cmd = f'Add-Content -Path "{expected_path}" -Value "{escaped_append}" -Encoding UTF8 -NoNewline'
                write_method = "append"
                
                logger.info(f"[implementer] Write method: {write_method}")
                write_result = client.shell_run(write_cmd, timeout_seconds=30)
                
                write_success = not write_result.stderr or write_result.stderr.strip() == ""
                if write_success:
                    logger.info(f"[implementer] SUCCESS: {expected_path} ({write_method})")
                    return ImplementerResult(
                        success=True,
                        output_path=expected_path,
                        sha256=None,
                        duration_ms=elapsed(),
                        sandbox_used=True,
                        filename=filename,
                        content_written=append_text,
                        action_taken=action,
                        write_method=write_method,
                    )
                else:
                    return ImplementerResult(
                        success=False,
                        error=f"PowerShell write failed ({write_method}): {write_result.stderr or write_result.stdout}",
                        duration_ms=elapsed(),
                        sandbox_used=True,
                        write_method=write_method,
                    )
            
            # v1.4: SEPARATE_REPLY_FILE mode - explicit overwrite to output path
            elif mode_lower == "separate_reply_file":
                logger.info(f"[implementer] v1.4 SEPARATE_REPLY_FILE mode: writing {len(content)} chars to output file")
                escaped_content = _escape_powershell_string(content)
                write_cmd = f'Set-Content -Path "{expected_path}" -Value "{escaped_content}" -NoNewline'
                write_method = "overwrite"
                
                logger.info(f"[implementer] Write method: {write_method}")
                write_result = client.shell_run(write_cmd, timeout_seconds=30)
                
                write_success = not write_result.stderr or write_result.stderr.strip() == ""
                if write_success:
                    logger.info(f"[implementer] SUCCESS: {expected_path} ({write_method})")
                    return ImplementerResult(
                        success=True,
                        output_path=expected_path,
                        sha256=None,
                        duration_ms=elapsed(),
                        sandbox_used=True,
                        filename=filename,
                        content_written=content,
                        action_taken=action,
                        write_method=write_method,
                    )
                else:
                    return ImplementerResult(
                        success=False,
                        error=f"PowerShell write failed ({write_method}): {write_result.stderr or write_result.stdout}",
                        duration_ms=elapsed(),
                        sandbox_used=True,
                        write_method=write_method,
                    )
            
            # v1.4: UNKNOWN MODE - FAIL SAFE (prevents silent destructive writes)
            else:
                logger.error(
                    "[implementer] v1.4 SAFETY STOP: Unknown output_mode='%s' - refusing to write",
                    output_mode
                )
                print(f"\n>>> [IMPLEMENTER v1.4] SAFETY STOP: Unknown mode '{output_mode}' <<<\n")
                return ImplementerResult(
                    success=False,
                    error=(
                        f"SAFETY: Unknown output_mode '{output_mode}' - cannot determine safe write method. "
                        f"Valid modes: chat_only, rewrite_in_place, append_in_place, separate_reply_file. "
                        f"Refusing to write to prevent accidental data loss."
                    ),
                    duration_ms=elapsed(),
                    sandbox_used=False,
                    filename=filename,
                    write_method=None,
                )
        else:
            logger.info(f"[implementer] Writing via sandbox API: {base_filename} -> {target}")
            result = client.write_file(
                target=target,
                filename=base_filename,
                content=content,
                overwrite=True,
            )
            
            if result.ok:
                logger.info(f"[implementer] SUCCESS: {result.path}")
                return ImplementerResult(
                    success=True,
                    output_path=result.path,
                    sha256=result.sha256,
                    duration_ms=elapsed(),
                    sandbox_used=True,
                    filename=filename,
                    content_written=content,
                    action_taken=action,
                    write_method="overwrite",
                )
            else:
                return ImplementerResult(
                    success=False,
                    error=f"Sandbox write failed: {getattr(result, 'error', 'unknown')}",
                    duration_ms=elapsed(),
                    sandbox_used=True,
                )
            
    except SandboxError as e:
        return ImplementerResult(
            success=False,
            error=f"Sandbox error: {e}",
            duration_ms=elapsed(),
            sandbox_used=True,
        )
    except Exception as e:
        logger.exception(f"[implementer] Failed: {e}")
        return ImplementerResult(
            success=False,
            error=str(e),
            duration_ms=elapsed(),
        )


async def run_verification(
    *,
    impl_result: ImplementerResult,
    spec: ResolvedSpec,
    client: Optional[SandboxClient] = None,
) -> VerificationResult:
    """Verify Implementer output against spec requirements.
    
    Checks:
    1. Correct filename
    2. Content matches spec (for overwrite) or contains content (for append/rewrite)
    3. File exists at expected path
    
    v1.3: Updated to handle REWRITE_IN_PLACE verification
    """
    try:
        expected_filename, expected_content, expected_action = spec.get_target_file()
        output_mode = spec.get_output_mode()
        insertion_format = spec.get_insertion_format()
    except SpecMissingDeliverableError as e:
        return VerificationResult(
            passed=False,
            error=str(e),
        )
    
    logger.info(f"[verification] === SPEC VERIFICATION ===")
    logger.info(f"[verification] Expected filename: {expected_filename}")
    logger.info(f"[verification] Expected content: '{expected_content[:100]}...'" if len(expected_content) > 100 else f"[verification] Expected content: '{expected_content}'")
    logger.info(f"[verification] Expected action: {expected_action}")
    logger.info(f"[verification] Output mode: {output_mode}")
    logger.info(f"[verification] Write method used: {impl_result.write_method}")
    
    # =========================================================================
    # v1.4: CHAT_ONLY VERIFICATION BYPASS
    # No file to verify - implementer returned success with no writes
    # =========================================================================
    mode_lower = (output_mode or "").lower()
    
    if mode_lower == "chat_only":
        logger.info("[verification] v1.4 CHAT_ONLY mode: no file verification needed")
        # For CHAT_ONLY, implementer should have returned success with write_method="none"
        if impl_result.write_method == "none" and impl_result.success:
            return VerificationResult(
                passed=True,
                file_exists=False,  # No file was written - this is expected
                content_matches=True,  # N/A for chat_only
                filename_matches=True,  # N/A for chat_only
                actual_content=None,
                expected_content=expected_content,
                expected_filename=expected_filename,
                actual_filename=None,
                error=None,
            )
        else:
            # Something unexpected happened - implementer should not have written anything
            logger.warning(
                "[verification] v1.4 CHAT_ONLY unexpected: write_method=%s, success=%s",
                impl_result.write_method, impl_result.success
            )
            return VerificationResult(
                passed=False,
                error=(
                    f"CHAT_ONLY mode verification failed: expected write_method='none' and success=True, "
                    f"but got write_method='{impl_result.write_method}' and success={impl_result.success}"
                ),
            )
    
    if not impl_result.success:
        return VerificationResult(
            passed=False,
            expected_filename=expected_filename,
            expected_content=expected_content,
            error=f"Implementation failed: {impl_result.error}",
        )
    
    if not impl_result.output_path:
        return VerificationResult(
            passed=False,
            expected_filename=expected_filename,
            expected_content=expected_content,
            error="No output path from Implementer",
        )
    
    # Check filename matches spec
    actual_filename = Path(impl_result.output_path).name
    
    if _is_absolute_windows_path(expected_filename):
        expected_basename = Path(expected_filename).name
        filename_matches = actual_filename == expected_basename
    else:
        filename_matches = actual_filename == expected_filename
    
    logger.info(f"[verification] Actual filename: {actual_filename}")
    logger.info(f"[verification] Filename matches: {filename_matches}")
    
    if not filename_matches:
        return VerificationResult(
            passed=False,
            file_exists=True,
            content_matches=False,
            filename_matches=False,
            expected_filename=expected_filename,
            actual_filename=actual_filename,
            error=f"WRONG FILE: Spec requires '{expected_filename}' but got '{actual_filename}'.",
        )
    
    # Verify content via sandbox
    if client is None:
        client = get_sandbox_client()
    
    try:
        if not client.is_connected():
            return VerificationResult(
                passed=False,
                expected_filename=expected_filename,
                expected_content=expected_content,
                error="Sandbox not available for verification",
            )
        
        ps_path = impl_result.output_path.replace("/", "\\")
        
        # Check exists
        exists_result = client.shell_run(f'Test-Path -Path "{ps_path}"', timeout_seconds=10)
        file_exists = "True" in exists_result.stdout
        
        if not file_exists:
            return VerificationResult(
                passed=False,
                file_exists=False,
                filename_matches=filename_matches,
                expected_filename=expected_filename,
                expected_content=expected_content,
                error=f"File not found at {impl_result.output_path}",
            )
        
        # Read content
        read_result = client.shell_run(f'Get-Content -Path "{ps_path}" -Raw', timeout_seconds=10)
        
        if read_result.stderr and read_result.stderr.strip():
            return VerificationResult(
                passed=False,
                file_exists=True,
                filename_matches=filename_matches,
                expected_filename=expected_filename,
                expected_content=expected_content,
                error=f"Failed to read file: {read_result.stderr}",
            )
        
        actual_content = read_result.stdout.strip() if read_result.stdout else ""
        
        # Content verification depends on write method
        mode_lower = (output_mode or "").lower()
        
        if mode_lower == "rewrite_in_place":
            # For rewrite mode: verify that answers are present in the file
            # Parse answers from expected content and check each is present
            answers = _parse_answers_from_reply(expected_content)
            all_present = True
            missing = []
            
            for q_num, answer in answers.items():
                # Check if answer text is in the file (fuzzy match - strip whitespace)
                if answer.strip() not in actual_content:
                    all_present = False
                    missing.append(q_num)
            
            content_matches = all_present
            
            logger.info(f"[verification] REWRITE mode: checking if answers are present")
            logger.info(f"[verification] Answers present: {content_matches}, missing: {missing}")
            
        elif mode_lower == "append_in_place":
            # For append mode: verify that the content is in the file
            content_matches = expected_content.strip() in actual_content
            
            logger.info(f"[verification] APPEND mode: checking if reply content is present")
            logger.info(f"[verification] Reply found in file: {content_matches}")
        else:
            # For overwrite mode: exact match
            content_matches = actual_content == expected_content
        
        logger.info(f"[verification] Actual content length: {len(actual_content)} chars")
        logger.info(f"[verification] Content matches: {content_matches}")
        
        passed = content_matches and filename_matches
        
        return VerificationResult(
            passed=passed,
            file_exists=True,
            content_matches=content_matches,
            filename_matches=filename_matches,
            actual_content=actual_content,
            expected_content=expected_content,
            expected_filename=expected_filename,
            actual_filename=actual_filename,
            error=None if passed else f"Content verification failed",
        )
        
    except SandboxError as e:
        return VerificationResult(
            passed=False,
            expected_filename=expected_filename,
            expected_content=expected_content,
            error=f"Sandbox error: {e}",
        )
    except Exception as e:
        return VerificationResult(
            passed=False,
            expected_filename=expected_filename,
            expected_content=expected_content,
            error=str(e),
        )


__all__ = [
    "ImplementerResult",
    "VerificationResult",
    "run_implementer",
    "run_verification",
]
