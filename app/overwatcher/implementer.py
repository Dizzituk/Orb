# FILE: app/overwatcher/implementer.py
"""Implementer: Executes approved work and verifies results.

Handles:
- Writing files to sandbox based on spec
- Enforcing must_exist constraint for modify actions
- Verifying output matches spec requirements
- APPEND_IN_PLACE mode for appending to existing files (v1.2)
- REWRITE_IN_PLACE mode for multi-question file edits (v1.3)
- CHAT_ONLY mode safety: no file writes at all (v1.4)
- OVERWRITE_FULL mode for complete file replacement (v1.6)
- Multi-file batch operations (search and refactor) (v1.11)

v1.11 (2026-01-28): Multi-file batch operations (Level 3 - Phase 5)
    - Added run_multi_file_search(): read-only search across multiple files
    - Added run_multi_file_refactor(): batch search/replace with verification
    - Added _multi_file_read_content(): helper for reading files via PowerShell
    - Added _multi_file_write_content(): helper for Base64-safe writes
    - Added MULTI_FILE_MAX_ERRORS constant (10) for consecutive error limit
    - Added MULTI_FILE_VERIFY_TIMEOUT constant (30s) per file
    - Progress callbacks supported for streaming updates
v1.10 (2026-01-28): Intelligent Q&A correction for REWRITE_IN_PLACE
    - Added _find_question_answer_pairs(): flexible detection of any Q&A format
    - Added _parse_corrections(): parse SpecGate's Q#: [STATUS] format
    - Added _apply_qa_corrections(): apply corrections to file in-place
    - REWRITE_IN_PLACE now tries intelligent correction first
    - Works with unnumbered, mixed format Q&A files
v1.9 (2026-01-27): Fix Answer marker detection (with or without colon)
    - Detects both "Answer" and "Answer:" patterns in _block_has_answer()
    - Detects both patterns in _insert_answers_under_questions()
    - Fixes duplicate "Answer:" sections being added to files
v1.8 (2026-01-27): Base64 encoding for PowerShell writes
    - Fixes escaping issues with embedded quotes (e.g., "works on my machine")
    - Uses Base64 encoding to safely transmit complex content
    - Completely avoids shell escaping problems
v1.7 (2026-01-27): Pattern 3 for standalone numbered lines
    - Added detection of "1)" or "2." format question headers
    - Fixes parsing for files with format: "1)\nQuestion\n..."
v1.6 (2026-01-27): OVERWRITE_FULL mode for complete file replacement
v1.5 (2026-01-25): REWRITE_IN_PLACE improvements for Q&A file tasks
v1.4.1 (2026-01-25): CHAT_ONLY safety fix - BULLETPROOF EDITION
v1.4 (2026-01-24): CHAT_ONLY safety fix - CRITICAL BUG FIX
v1.3 (2026-01-24): Added REWRITE_IN_PLACE support for multi-question file edits
v1.2 (2026-01-24): Added APPEND_IN_PLACE support with insertion_format
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.overwatcher.overwatcher import OverwatcherOutput, Decision
from app.overwatcher.sandbox_client import (
    SandboxClient,
    SandboxError,
    get_sandbox_client,
)

from .spec_resolution import ResolvedSpec, SpecMissingDeliverableError

logger = logging.getLogger(__name__)

# =============================================================================
# v1.11 BUILD VERIFICATION - Proves correct code is running
# v1.11: Multi-file batch operations (Level 3 - Phase 5)
# =============================================================================
IMPLEMENTER_BUILD_ID = "2026-01-28-v1.11-multi-file-batch-ops"
print(f"[IMPLEMENTER_LOADED] BUILD_ID={IMPLEMENTER_BUILD_ID}")
logger.info(f"[implementer] Module loaded: BUILD_ID={IMPLEMENTER_BUILD_ID}")

# =============================================================================
# v1.11: MULTI-FILE OPERATION CONSTANTS
# =============================================================================
MULTI_FILE_MAX_ERRORS = 10  # Stop after N consecutive errors
MULTI_FILE_VERIFY_TIMEOUT = 30  # Seconds per file verification


def _is_absolute_windows_path(path: str) -> bool:
    """Check if path is an absolute Windows path (e.g., C:\\..., D:\\...)."""
    if len(path) >= 3:
        return path[1] == ':' and path[2] in ('\\', '/')
    return False


def _escape_powershell_string(s: str) -> str:
    """Escape a string for use in PowerShell double-quoted strings.
    
    NOTE: For complex content with embedded quotes/newlines, use
    _build_powershell_write_command_base64() instead - it's more reliable.
    """
    return s.replace('`', '``').replace('"', '`"').replace('$', '`$')


def _encode_for_powershell_base64(content: str) -> str:
    """
    Encode content as Base64 for safe PowerShell transmission.
    
    v1.8: This is more robust than escaping for complex content with
    embedded quotes, newlines, and special characters.
    """
    encoded = base64.b64encode(content.encode('utf-8')).decode('ascii')
    return encoded


def _build_powershell_write_command_base64(path: str, content: str) -> str:
    """
    Build a PowerShell command that writes content using Base64 encoding.
    
    v1.8: Uses Base64 to safely transmit complex content with quotes/newlines.
    This avoids all escaping issues that occur with embedded quotes like
    "works on my machine".
    """
    encoded = _encode_for_powershell_base64(content)
    # Decode Base64 in PowerShell, then write to file
    return (
        f'[System.Text.Encoding]::UTF8.GetString('
        f'[System.Convert]::FromBase64String("{encoded}"))'
        f' | Set-Content -Path "{path}" -NoNewline -Encoding UTF8'
    )


def _generate_sandbox_path_candidates(path: str) -> List[str]:
    """Generate candidate paths for sandbox resolution."""
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
        
        non_onedrive = f"{drive}:\\Users\\{username}\\Desktop\\{rest}"
        if non_onedrive not in candidates:
            candidates.append(non_onedrive)
        
        wdag_onedrive = f"{drive}:\\Users\\WDAGUtilityAccount\\OneDrive\\Desktop\\{rest}"
        if wdag_onedrive not in candidates:
            candidates.append(wdag_onedrive)
        
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
        
        wdag = f"{drive}:\\Users\\WDAGUtilityAccount\\Desktop\\{rest}"
        if wdag not in candidates:
            candidates.append(wdag)
        
        return candidates
    
    return candidates


# =============================================================================
# v1.10: INTELLIGENT Q&A CORRECTION
# =============================================================================

def _find_question_answer_pairs(content: str) -> List[Dict[str, Any]]:
    """
    v1.10: Find ALL question/answer pairs regardless of format.
    
    Supports:
    - "Question\n<text>\n...\nanswer\n<text>" (unnumbered)
    - "Question 1:\n<text>\n...\nAnswer:\n<text>" (numbered)
    - "Q1.\n<text>" (abbreviated)
    - Mixed formats in same file
    
    Returns list of dicts with:
        - index: sequential position (1-based)
        - question_start: char position of question start
        - question_end: char position before answer marker
        - answer_start: char position after answer marker
        - answer_end: char position of answer end
        - answer_text: current answer text (may be empty)
        - full_match: the entire Q&A block
    """
    pairs = []
    
    # Pattern matches: Question (optional number) ... Answer (optional colon) ... (until next Question or EOF)
    # This is flexible - matches "Question\n", "Question 1:", "Question:", etc.
    pattern = r'(?i)(question(?:\s*\d+)?[:\.]?\s*\n)(.*?)(answer[:\s]*\n)(.*?)(?=question(?:\s*\d+)?[:\.]?\s*\n|$)'
    
    for i, match in enumerate(re.finditer(pattern, content, re.DOTALL | re.IGNORECASE), 1):
        answer_text = match.group(4).strip()
        
        pairs.append({
            "index": i,
            "question_start": match.start(),
            "question_header_end": match.end(1),
            "question_text": match.group(2).strip(),
            "answer_marker_start": match.start(3),
            "answer_start": match.end(3),
            "answer_end": match.end(4),
            "answer_text": answer_text,
            "full_match": match.group(0),
        })
        
        logger.debug(
            "[implementer] v1.10 Found Q%d: answer_start=%d, answer_end=%d, answer='%s'",
            i, match.end(3), match.end(4), answer_text[:50] if answer_text else "(empty)"
        )
    
    logger.info("[implementer] v1.10 _find_question_answer_pairs: found %d pairs", len(pairs))
    return pairs


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


def _apply_qa_corrections(
    content: str,
    generated_reply: str,
) -> Tuple[str, int]:
    """
    v1.10: Apply SpecGate corrections to Q&A file.
    
    1. Find all question/answer pairs in content (flexible detection)
    2. Parse corrections from generated_reply (Q#: [STATUS] format)
    3. Replace each corrected answer in-place
    4. Return (modified_content, corrections_applied_count)
    
    Works backwards through the file to preserve character positions.
    """
    pairs = _find_question_answer_pairs(content)
    corrections = _parse_corrections(generated_reply)
    
    if not pairs:
        logger.warning("[implementer] v1.10 No question/answer pairs found in file")
        return content, 0
    
    if not corrections:
        logger.warning("[implementer] v1.10 No corrections to apply from SpecGate reply")
        return content, 0
    
    logger.info(
        "[implementer] v1.10 Applying up to %d corrections to %d question/answer pairs",
        len(corrections), len(pairs)
    )
    
    # Work backwards to preserve character positions
    modified = content
    corrections_applied = 0
    
    for pair in reversed(pairs):
        q_idx = pair["index"]
        
        if q_idx not in corrections:
            logger.debug("[implementer] v1.10 Q%d: no correction needed", q_idx)
            continue
        
        new_answer = corrections[q_idx]
        old_answer = pair["answer_text"]
        
        # Replace the answer section
        # Keep everything before answer_start, insert new answer, skip to answer_end
        before = modified[:pair["answer_start"]]
        after = modified[pair["answer_end"]:]
        
        # Ensure proper formatting
        if not new_answer.endswith('\n'):
            new_answer = new_answer + '\n'
        
        modified = before + new_answer + after
        corrections_applied += 1
        
        logger.info(
            "[implementer] v1.10 Q%d: '%s' -> '%s'",
            q_idx,
            old_answer[:30] if old_answer else "(empty)",
            new_answer[:30].strip()
        )
    
    logger.info("[implementer] v1.10 Applied %d corrections", corrections_applied)
    return modified, corrections_applied


def _is_specgate_correction_format(reply: str) -> bool:
    """
    v1.10: Check if the reply is in SpecGate's correction format.
    
    Returns True if reply contains Q#: [STATUS] patterns.
    """
    if not reply:
        return False
    
    # Look for Q#: [STATUS] pattern
    pattern = r'Q\d+:\s*\[(INCORRECT|MISSING|CORRECT|TRICK|ANSWER)\]'
    matches = re.findall(pattern, reply, re.IGNORECASE)
    
    is_correction = len(matches) >= 1
    logger.debug("[implementer] v1.10 _is_specgate_correction_format: %s (found %d matches)", is_correction, len(matches))
    return is_correction


# =============================================================================
# REWRITE_IN_PLACE HELPERS (v1.3, updated v1.7, v1.9)
# =============================================================================

def _find_question_block_starts(text: str) -> List[Tuple[int, int, str]]:
    """
    Find all question block start positions in text.
    
    Returns list of (line_number, char_position, question_identifier) tuples.
    
    Question detection patterns:
    - Pattern 1: Lines starting with "Question N:" (case-insensitive)
    - Pattern 2: Lines ending with "?" that start with "N." or "N)" 
    - Pattern 3 (v1.7): Standalone numbered lines like "1)" or "2."
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
            char_pos += len(line) + 1
            continue
        
        # Pattern 2: Numbered line ending with "?" (e.g., "1. How?" or "2) Why?")
        numbered_question_match = re.match(r'^(\d+)[.\)]\s*', line_stripped)
        if numbered_question_match and line_stripped.rstrip().endswith('?'):
            q_num = numbered_question_match.group(1)
            blocks.append((line_num, char_pos, q_num))
            char_pos += len(line) + 1
            continue
        
        # Pattern 3 (v1.7): Standalone numbered line (e.g., "1)" or "2.")
        standalone_match = re.match(r'^(\d+)[.\)]\s*$', line_stripped)
        if standalone_match:
            q_num = standalone_match.group(1)
            blocks.append((line_num, char_pos, q_num))
            logger.debug("[implementer] v1.7 Pattern 3 matched: line %d = %r -> Q%s", line_num, line_stripped, q_num)
        
        char_pos += len(line) + 1
    
    return blocks


def _block_has_answer(block_text: str) -> bool:
    """Check if a question block already has a non-empty answer.
    
    v1.9: Detect both "Answer" and "Answer:" patterns (with or without colon).
    """
    lines = block_text.split('\n')
    
    for i, line in enumerate(lines):
        line_stripped = line.strip().lower()
        
        # Match "answer" or "answer:" (with or without colon)
        if line_stripped == 'answer' or line_stripped.startswith('answer:'):
            # Check content AFTER this line
            remaining_lines = lines[i + 1:]
            remaining_text = '\n'.join(remaining_lines).strip()
            
            if remaining_text:
                # There's content after the Answer line
                return True
    
    return False


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


def _insert_answers_under_questions(
    original_text: str,
    answers: Dict[int, str],
    insertion_format: str,
) -> str:
    """Insert answers at the appropriate position in each question block."""
    if not answers:
        logger.warning("[implementer] No answers to insert")
        return original_text
    
    block_starts = _find_question_block_starts(original_text)
    
    if not block_starts:
        logger.warning("[implementer] No question blocks found in file")
        return original_text
    
    logger.info(
        "[implementer] Found %d question blocks: %s",
        len(block_starts),
        [(bs[0], bs[2]) for bs in block_starts]
    )
    
    lines = original_text.split('\n')
    
    blocks_with_ends: List[Tuple[int, int, int]] = []
    
    for i, (start_line, _, q_num_str) in enumerate(block_starts):
        q_num = int(q_num_str)
        
        if i + 1 < len(block_starts):
            end_line = block_starts[i + 1][0] - 1
        else:
            end_line = len(lines) - 1
        
        blocks_with_ends.append((start_line, q_num, end_line))
    
    insertions_made = 0
    skipped_filled = []
    skipped_no_answer = []
    
    for start_line, q_num, end_line in reversed(blocks_with_ends):
        if q_num not in answers:
            skipped_no_answer.append(q_num)
            continue
        
        block_lines = lines[start_line:end_line + 1]
        block_text = '\n'.join(block_lines)
        
        if _block_has_answer(block_text):
            skipped_filled.append(q_num)
            continue
        
        answer_text = answers[q_num]
        
        # v1.9: Find "Answer" or "Answer:" line (with or without colon)
        answer_line_idx = None
        for i, line in enumerate(block_lines):
            line_stripped = line.strip().lower()
            # Match "answer" or "answer:" (with or without colon)
            if line_stripped == 'answer' or line_stripped.startswith('answer:'):
                answer_line_idx = start_line + i
                logger.debug("[implementer] v1.9 Found answer marker at line %d: %r", answer_line_idx, line.strip())
                break
        
        if answer_line_idx is not None:
            # Insert answer on the next line after "Answer"/"Answer:"
            insert_position = answer_line_idx + 1
            # Just insert the answer text without "Answer:" prefix since it's already there
            answer_lines = [answer_text]
        else:
            # No "Answer" marker found - use full insertion format
            insert_position = end_line + 1
            
            try:
                formatted_answer = insertion_format.format(reply=answer_text)
            except KeyError:
                formatted_answer = f"\n\nAnswer:\n{answer_text}\n"
            
            if not formatted_answer.startswith('\n'):
                formatted_answer = '\n' + formatted_answer
            
            answer_lines = formatted_answer.split('\n')
        
        lines[insert_position:insert_position] = answer_lines
        insertions_made += 1
    
    logger.info(
        "[implementer] REWRITE complete: %d insertions, skipped_filled=%s, skipped_no_answer=%s",
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
    """Execute approved work via Sandbox."""
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
    
    try:
        filename, content, action = spec.get_target_file()
        target = spec.get_target()
        must_exist = spec.get_must_exist()
        output_mode = spec.get_output_mode()
        insertion_format = spec.get_insertion_format()
    except SpecMissingDeliverableError as e:
        return ImplementerResult(
            success=False,
            error=str(e),
            duration_ms=elapsed(),
        )
    
    # v1.10: Logging
    logger.info(f"[implementer] BUILD_ID={IMPLEMENTER_BUILD_ID}")
    logger.info(f"[implementer] RAW output_mode={repr(output_mode)}")
    print(f"\n>>> [IMPLEMENTER v1.10] BUILD={IMPLEMENTER_BUILD_ID} <<<")
    print(f">>> [IMPLEMENTER v1.10] RAW output_mode={repr(output_mode)} <<<\n")
    
    mode_lower = (output_mode or "").strip().lower()
    
    # CHAT_ONLY check
    if mode_lower == "chat_only":
        logger.info("[implementer] CHAT_ONLY DETECTED - RETURNING EARLY")
        return ImplementerResult(
            success=True,
            output_path=None,
            sha256=None,
            duration_ms=elapsed(),
            sandbox_used=False,
            filename=filename,
            content_written=None,
            action_taken="chat_only_noop",
            write_method="none",
        )
    
    logger.info(f"[implementer] === SPEC-DRIVEN TASK === MODE: {mode_lower}")
    logger.info(f"[implementer] Action: {action}, Filename: {filename}")
    
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
        else:
            base_filename = filename
            is_absolute = False
            if target == "DESKTOP":
                expected_path = f"C:\\Users\\WDAGUtilityAccount\\Desktop\\{base_filename}"
            else:
                expected_path = f"{target}\\{base_filename}"
        
        # For "modify" action with must_exist: verify file exists first
        if action == "modify" and must_exist:
            candidates = _generate_sandbox_path_candidates(expected_path)
            
            resolved_path = None
            for candidate in candidates:
                exists_cmd = f'Test-Path -Path "{candidate}"'
                exists_result = client.shell_run(exists_cmd, timeout_seconds=10)
                
                if "True" in exists_result.stdout:
                    resolved_path = candidate
                    break
            
            if resolved_path is None:
                return ImplementerResult(
                    success=False,
                    error=f"SPEC VIOLATION: File '{filename}' does not exist",
                    duration_ms=elapsed(),
                    sandbox_used=True,
                    filename=filename,
                    action_taken="existence_check_failed",
                )
            
            expected_path = resolved_path
        
        # WRITE FILE VIA SANDBOX
        if is_absolute:
            logger.info(f"[implementer] Writing via PowerShell to: {expected_path}")
            
            # REWRITE_IN_PLACE mode
            if mode_lower == "rewrite_in_place":
                logger.info("[implementer] REWRITE_IN_PLACE mode: multi-question file edit")
                
                # Read file
                read_cmd = f'Get-Content -Path "{expected_path}" -Raw'
                read_result = client.shell_run(read_cmd, timeout_seconds=30)
                
                if read_result.stderr and read_result.stderr.strip():
                    return ImplementerResult(
                        success=False,
                        error=f"Failed to read file: {read_result.stderr}",
                        duration_ms=elapsed(),
                        sandbox_used=True,
                        write_method="rewrite",
                    )
                
                original_text = read_result.stdout or ""
                logger.info(f"[implementer] Read {len(original_text)} chars from file")
                
                # =============================================================
                # v1.10: Try intelligent Q&A correction FIRST
                # =============================================================
                if _is_specgate_correction_format(content):
                    logger.info("[implementer] v1.10 Detected SpecGate correction format (Q#: [STATUS])")
                    
                    updated_text, corrections_count = _apply_qa_corrections(original_text, content)
                    
                    if corrections_count > 0:
                        logger.info("[implementer] v1.10 Applied %d intelligent corrections", corrections_count)
                        
                        # Write corrected content
                        write_cmd = _build_powershell_write_command_base64(expected_path, updated_text)
                        write_result = client.shell_run(write_cmd, timeout_seconds=60)
                        
                        write_success = not write_result.stderr or write_result.stderr.strip() == ""
                        if write_success:
                            logger.info("[implementer] v1.10 SUCCESS: Intelligent Q&A correction completed")
                            return ImplementerResult(
                                success=True,
                                output_path=expected_path,
                                sha256=None,
                                duration_ms=elapsed(),
                                sandbox_used=True,
                                filename=filename,
                                content_written=updated_text,
                                action_taken=action,
                                write_method="rewrite_intelligent",
                            )
                        else:
                            return ImplementerResult(
                                success=False,
                                error=f"PowerShell write failed: {write_result.stderr or write_result.stdout}",
                                duration_ms=elapsed(),
                                sandbox_used=True,
                                write_method="rewrite_intelligent",
                            )
                    else:
                        logger.warning("[implementer] v1.10 No corrections applied - falling back to legacy method")
                
                # =============================================================
                # FALLBACK: Legacy answer insertion method (v1.9)
                # =============================================================
                logger.info("[implementer] Using legacy answer insertion method")
                
                # Parse and insert answers
                answers = _parse_answers_from_reply(content)
                logger.info(f"[implementer] Parsed {len(answers)} answers: {list(answers.keys())}")
                
                fmt = insertion_format or "\n\nAnswer:\n{reply}\n"
                updated_text = _insert_answers_under_questions(original_text, answers, fmt)
                
                # v1.8: Write using Base64 encoding to avoid escaping issues
                write_cmd = _build_powershell_write_command_base64(expected_path, updated_text)
                
                logger.info(f"[implementer] v1.8 Writing {len(updated_text)} chars via Base64")
                write_result = client.shell_run(write_cmd, timeout_seconds=60)
                
                write_success = not write_result.stderr or write_result.stderr.strip() == ""
                if write_success:
                    logger.info(f"[implementer] SUCCESS: REWRITE_IN_PLACE completed")
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
                        error=f"PowerShell write failed: {write_result.stderr or write_result.stdout}",
                        duration_ms=elapsed(),
                        sandbox_used=True,
                        write_method="rewrite",
                    )
            
            # APPEND_IN_PLACE mode
            elif mode_lower == "append_in_place":
                if insertion_format:
                    try:
                        append_text = insertion_format.format(reply=content)
                    except KeyError:
                        append_text = f"\n\nAnswer:\n{content}\n"
                else:
                    append_text = f"\n\nAnswer:\n{content}\n"
                
                # v1.8: Use Base64 for append too
                # Read existing content, append, write back
                read_cmd = f'Get-Content -Path "{expected_path}" -Raw'
                read_result = client.shell_run(read_cmd, timeout_seconds=30)
                existing_content = read_result.stdout or ""
                
                combined_content = existing_content + append_text
                write_cmd = _build_powershell_write_command_base64(expected_path, combined_content)
                
                write_result = client.shell_run(write_cmd, timeout_seconds=60)
                
                write_success = not write_result.stderr or write_result.stderr.strip() == ""
                if write_success:
                    return ImplementerResult(
                        success=True,
                        output_path=expected_path,
                        sha256=None,
                        duration_ms=elapsed(),
                        sandbox_used=True,
                        filename=filename,
                        content_written=append_text,
                        action_taken=action,
                        write_method="append",
                    )
                else:
                    return ImplementerResult(
                        success=False,
                        error=f"PowerShell write failed: {write_result.stderr}",
                        duration_ms=elapsed(),
                        sandbox_used=True,
                        write_method="append",
                    )
            
            # SEPARATE_REPLY_FILE or OVERWRITE_FULL mode
            elif mode_lower in ("separate_reply_file", "overwrite_full"):
                write_cmd = _build_powershell_write_command_base64(expected_path, content)
                write_method = "overwrite_full" if mode_lower == "overwrite_full" else "overwrite"
                
                write_result = client.shell_run(write_cmd, timeout_seconds=60)
                
                write_success = not write_result.stderr or write_result.stderr.strip() == ""
                if write_success:
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
                        error=f"PowerShell write failed: {write_result.stderr}",
                        duration_ms=elapsed(),
                        sandbox_used=True,
                        write_method=write_method,
                    )
            
            # UNKNOWN MODE - FAIL SAFE
            else:
                logger.error(f"[implementer] SAFETY STOP: Unknown output_mode='{output_mode}'")
                return ImplementerResult(
                    success=False,
                    error=f"SAFETY: Unknown output_mode '{output_mode}'",
                    duration_ms=elapsed(),
                    sandbox_used=False,
                    filename=filename,
                    write_method=None,
                )
        else:
            # Use sandbox API for non-absolute paths
            result = client.write_file(
                target=target,
                filename=base_filename,
                content=content,
                overwrite=True,
            )
            
            if result.ok:
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
    """Verify Implementer output against spec requirements."""
    try:
        expected_filename, expected_content, expected_action = spec.get_target_file()
        output_mode = spec.get_output_mode()
        insertion_format = spec.get_insertion_format()
    except SpecMissingDeliverableError as e:
        return VerificationResult(passed=False, error=str(e))
    
    mode_lower = (output_mode or "").lower()
    
    # CHAT_ONLY verification
    if mode_lower == "chat_only":
        if impl_result.write_method == "none" and impl_result.success:
            return VerificationResult(
                passed=True,
                file_exists=False,
                content_matches=True,
                filename_matches=True,
                expected_content=expected_content,
                expected_filename=expected_filename,
            )
        else:
            return VerificationResult(
                passed=False,
                error=f"CHAT_ONLY verification failed: write_method={impl_result.write_method}",
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
    
    actual_filename = Path(impl_result.output_path).name
    
    if _is_absolute_windows_path(expected_filename):
        expected_basename = Path(expected_filename).name
        filename_matches = actual_filename == expected_basename
    else:
        filename_matches = actual_filename == expected_filename
    
    if not filename_matches:
        return VerificationResult(
            passed=False,
            file_exists=True,
            content_matches=False,
            filename_matches=False,
            expected_filename=expected_filename,
            actual_filename=actual_filename,
            error=f"WRONG FILE: Expected '{expected_filename}' but got '{actual_filename}'",
        )
    
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
        
        # Content verification depends on mode
        if mode_lower == "rewrite_in_place":
            # v1.10: Enhanced verification for intelligent corrections
            if _is_specgate_correction_format(expected_content):
                # Parse corrections and verify each was applied
                corrections = _parse_corrections(expected_content)
                all_present = True
                missing = []
                
                for q_num, answer in corrections.items():
                    # Check if the correction appears in the file
                    # Be flexible - check if the answer text is present
                    if answer.strip() not in actual_content:
                        all_present = False
                        missing.append(q_num)
                
                content_matches = all_present
                if not all_present:
                    logger.warning("[implementer] v1.10 Verification: missing corrections for Q%s", missing)
            else:
                # Legacy verification
                answers = _parse_answers_from_reply(expected_content)
                all_present = True
                missing = []
                
                for q_num, answer in answers.items():
                    if answer.strip() not in actual_content:
                        all_present = False
                        missing.append(q_num)
                
                content_matches = all_present
        
        elif mode_lower == "append_in_place":
            content_matches = expected_content.strip() in actual_content
        
        elif mode_lower == "overwrite_full":
            content_matches = actual_content.strip() == expected_content.strip()
        
        else:
            content_matches = actual_content == expected_content
        
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
            error=None if passed else "Content verification failed",
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


# =============================================================================
# v1.11: MULTI-FILE OPERATIONS (Level 3 - Phase 5)
# =============================================================================

@dataclass
class MultiFileResult:
    """v1.11: Result from multi-file batch operations."""
    success: bool
    operation: str  # "search" or "refactor"
    search_pattern: str = ""
    replacement_pattern: str = ""
    total_files: int = 0
    total_occurrences: int = 0  # v1.11: For search operations
    files_processed: int = 0
    files_modified: int = 0
    files_unchanged: int = 0
    files_failed: int = 0
    total_replacements: int = 0
    file_preview: str = ""
    target_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    details: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    awaiting_confirmation: bool = False
    duration_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "operation": self.operation,
            "search_pattern": self.search_pattern,
            "replacement_pattern": self.replacement_pattern,
            "total_files": self.total_files,
            "total_occurrences": self.total_occurrences,
            "files_processed": self.files_processed,
            "files_modified": self.files_modified,
            "files_unchanged": self.files_unchanged,
            "files_failed": self.files_failed,
            "total_replacements": self.total_replacements,
            "file_preview": self.file_preview,
            "target_files": self.target_files,
            "errors": self.errors,
            "details": self.details,
            "error": self.error,
            "awaiting_confirmation": self.awaiting_confirmation,
            "duration_ms": self.duration_ms,
        }


async def _multi_file_read_content(
    client: SandboxClient,
    file_path: str,
) -> Optional[str]:
    """
    v1.11: Read file content from sandbox.
    
    Returns file content as string, or None if read fails.
    """
    try:
        read_cmd = f'Get-Content -Path "{file_path}" -Raw -Encoding UTF8'
        result = client.shell_run(read_cmd, timeout_seconds=MULTI_FILE_VERIFY_TIMEOUT)
        
        if result.exit_code == 0 and result.stdout is not None:
            return result.stdout
        
        logger.warning(
            "[implementer] v1.11 Read failed for %s: exit=%s, stderr=%s",
            file_path, result.exit_code, result.stderr[:100] if result.stderr else ""
        )
        return None
        
    except Exception as e:
        logger.error("[implementer] v1.11 Read exception for %s: %s", file_path, e)
        return None


async def _multi_file_write_content(
    client: SandboxClient,
    file_path: str,
    content: str,
) -> bool:
    """
    v1.11: Write file content to sandbox using Base64 encoding.
    
    Returns True if write succeeded, False otherwise.
    """
    try:
        # Encode content as Base64 for safe PowerShell transfer
        content_bytes = content.encode('utf-8')
        content_b64 = base64.b64encode(content_bytes).decode('ascii')
        
        # PowerShell command to decode and write
        ps_command = (
            f'[System.Text.Encoding]::UTF8.GetString('
            f'[System.Convert]::FromBase64String("{content_b64}"))'
            f' | Set-Content -Path "{file_path}" -NoNewline -Encoding UTF8'
        )
        
        result = client.shell_run(ps_command, timeout_seconds=MULTI_FILE_VERIFY_TIMEOUT)
        
        if result.exit_code == 0:
            return True
        
        logger.warning(
            "[implementer] v1.11 Write failed for %s: exit=%s, stderr=%s",
            file_path, result.exit_code, result.stderr[:100] if result.stderr else ""
        )
        return False
        
    except Exception as e:
        logger.error("[implementer] v1.11 Write exception for %s: %s", file_path, e)
        return False


async def run_multi_file_search(
    *,
    multi_file: Dict[str, Any],
    client: Optional[SandboxClient] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> MultiFileResult:
    """
    v1.11: Execute multi-file search (read-only).
    
    For search operations, the discovery results are already in the spec.
    This method formats them for display and returns the summary.
    
    Args:
        multi_file: Dict with multi_file data from spec
        client: Sandbox client (optional, for verification)
        progress_callback: Optional callback for progress updates
        
    Returns:
        MultiFileResult with search results summary
    """
    import time
    start_time = time.time()
    
    if not multi_file.get("is_multi_file"):
        return MultiFileResult(
            success=False,
            operation="search",
            error="Not a multi-file operation",
            duration_ms=int((time.time() - start_time) * 1000),
        )
    
    logger.info(
        "[implementer] v1.11 Multi-file SEARCH: pattern='%s', files=%d, occurrences=%d",
        multi_file.get("search_pattern", ""),
        multi_file.get("total_files", 0),
        multi_file.get("total_occurrences", 0),
    )
    
    # For search, results are already computed by SpecGate discovery
    # Just format and return
    result = MultiFileResult(
        success=True,
        operation="search",
        search_pattern=multi_file.get("search_pattern", ""),
        total_files=multi_file.get("total_files", 0),
        total_occurrences=multi_file.get("total_occurrences", 0),
        file_preview=multi_file.get("file_preview", ""),
        target_files=multi_file.get("target_files", []),
        files_processed=multi_file.get("total_files", 0),
        files_modified=0,  # Search is read-only
        files_failed=0,
        duration_ms=int((time.time() - start_time) * 1000),
    )
    
    # Send completion callback
    if progress_callback:
        try:
            callback_data = {
                "type": "complete",
                "operation": "search",
                "total_files": result.total_files,
                "total_occurrences": result.total_replacements,
                "success": True,
            }
            # Handle both sync and async callbacks
            import asyncio
            if asyncio.iscoroutinefunction(progress_callback):
                await progress_callback(callback_data)
            else:
                progress_callback(callback_data)
        except Exception as e:
            logger.warning("[implementer] v1.11 Progress callback error: %s", e)
    
    return result


async def run_multi_file_refactor(
    *,
    multi_file: Dict[str, Any],
    client: Optional[SandboxClient] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> MultiFileResult:
    """
    v1.11: Execute multi-file refactor (search and replace).
    
    Processes each file in target_files:
    1. Read current content
    2. Apply search/replace
    3. Write updated content
    4. Verify write succeeded
    5. Report progress
    
    Args:
        multi_file: Dict with multi_file data from spec
        client: Sandbox client for file operations
        progress_callback: Optional callback for progress updates
        
    Returns:
        MultiFileResult with aggregate results
    """
    import time
    import asyncio
    start_time = time.time()
    
    def elapsed_ms() -> int:
        return int((time.time() - start_time) * 1000)
    
    async def call_progress(data: Dict[str, Any]) -> None:
        """Helper to call progress callback (handles sync/async)."""
        if not progress_callback:
            return
        try:
            if asyncio.iscoroutinefunction(progress_callback):
                await progress_callback(data)
            else:
                progress_callback(data)
        except Exception as e:
            logger.warning("[implementer] v1.11 Progress callback error: %s", e)
    
    if not multi_file.get("is_multi_file"):
        return MultiFileResult(
            success=False,
            operation="refactor",
            error="Not a multi-file operation",
            duration_ms=elapsed_ms(),
        )
    
    # Check confirmation for refactor operations
    if multi_file.get("requires_confirmation") and not multi_file.get("confirmed"):
        return MultiFileResult(
            success=False,
            operation="refactor",
            error="Refactor operation requires confirmation",
            awaiting_confirmation=True,
            duration_ms=elapsed_ms(),
        )
    
    target_files = multi_file.get("target_files", [])
    search_pattern = multi_file.get("search_pattern", "")
    replacement_pattern = multi_file.get("replacement_pattern", "")
    
    if not target_files:
        return MultiFileResult(
            success=False,
            operation="refactor",
            error="No target files specified",
            search_pattern=search_pattern,
            replacement_pattern=replacement_pattern,
            duration_ms=elapsed_ms(),
        )
    
    if not search_pattern:
        return MultiFileResult(
            success=False,
            operation="refactor",
            error="No search pattern specified",
            duration_ms=elapsed_ms(),
        )
    
    logger.info(
        "[implementer] v1.11 Multi-file REFACTOR: '%s' -> '%s', files=%d",
        search_pattern,
        replacement_pattern or "(remove)",
        len(target_files),
    )
    
    # Get sandbox client
    if client is None:
        client = get_sandbox_client()
    
    if not client.is_connected():
        return MultiFileResult(
            success=False,
            operation="refactor",
            error="Sandbox not available",
            search_pattern=search_pattern,
            replacement_pattern=replacement_pattern,
            total_files=len(target_files),
            duration_ms=elapsed_ms(),
        )
    
    # Initialize results tracking
    files_modified = 0
    files_unchanged = 0
    files_failed = 0
    files_processed = 0
    total_replacements = 0
    errors: List[str] = []
    details: List[Dict[str, Any]] = []
    consecutive_errors = 0
    abort_error = None
    
    # Process each file
    for i, file_path in enumerate(target_files, 1):
        file_result: Dict[str, Any] = {
            "path": file_path,
            "status": "pending",
            "replacements": 0,
            "error": None,
        }
        
        try:
            # Progress callback: starting file
            await call_progress({
                "type": "progress",
                "current": i,
                "total": len(target_files),
                "file": file_path,
                "status": "processing",
            })
            
            # Step 1: Read file
            content = await _multi_file_read_content(client, file_path)
            
            if content is None:
                file_result["status"] = "error"
                file_result["error"] = "Could not read file"
                files_failed += 1
                errors.append(f"{file_path}: Could not read file")
                consecutive_errors += 1
                
                if consecutive_errors >= MULTI_FILE_MAX_ERRORS:
                    logger.error(
                        "[implementer] v1.11 Aborting: %d consecutive errors",
                        consecutive_errors
                    )
                    abort_error = f"Aborted after {consecutive_errors} consecutive errors"
                    details.append(file_result)
                    break
                
                details.append(file_result)
                continue
            
            # Step 2: Check if pattern exists in file
            if search_pattern not in content:
                file_result["status"] = "unchanged"
                file_result["replacements"] = 0
                files_unchanged += 1
                files_processed += 1
                consecutive_errors = 0  # Reset on success
                
                await call_progress({
                    "type": "progress",
                    "current": i,
                    "total": len(target_files),
                    "file": file_path,
                    "status": "unchanged",
                    "replacements": 0,
                })
                
                details.append(file_result)
                continue
            
            # Step 3: Count replacements and apply
            replacement_count = content.count(search_pattern)
            new_content = content.replace(search_pattern, replacement_pattern)
            
            # Step 4: Write file
            write_success = await _multi_file_write_content(client, file_path, new_content)
            
            if not write_success:
                file_result["status"] = "error"
                file_result["error"] = "Write failed"
                files_failed += 1
                errors.append(f"{file_path}: Write failed")
                consecutive_errors += 1
                
                if consecutive_errors >= MULTI_FILE_MAX_ERRORS:
                    logger.error(
                        "[implementer] v1.11 Aborting: %d consecutive errors",
                        consecutive_errors
                    )
                    abort_error = f"Aborted after {consecutive_errors} consecutive errors"
                    details.append(file_result)
                    break
                
                details.append(file_result)
                continue
            
            # Step 5: Verify write
            verify_content = await _multi_file_read_content(client, file_path)
            
            if verify_content != new_content:
                file_result["status"] = "verify_failed"
                file_result["error"] = "Verification failed - content mismatch"
                files_failed += 1
                errors.append(f"{file_path}: Verification failed")
                consecutive_errors += 1
                
                if consecutive_errors >= MULTI_FILE_MAX_ERRORS:
                    abort_error = f"Aborted after {consecutive_errors} consecutive errors"
                    details.append(file_result)
                    break
                
                details.append(file_result)
                continue
            
            # Success!
            file_result["status"] = "success"
            file_result["replacements"] = replacement_count
            files_modified += 1
            files_processed += 1
            total_replacements += replacement_count
            consecutive_errors = 0  # Reset on success
            
            logger.info(
                "[implementer] v1.11 Modified %s: %d replacements",
                file_path, replacement_count
            )
            
            # Progress callback: file complete
            await call_progress({
                "type": "progress",
                "current": i,
                "total": len(target_files),
                "file": file_path,
                "status": "success",
                "replacements": replacement_count,
            })
            
            details.append(file_result)
                
        except Exception as e:
            file_result["status"] = "error"
            file_result["error"] = str(e)[:200]
            files_failed += 1
            errors.append(f"{file_path}: {str(e)[:100]}")
            consecutive_errors += 1
            
            logger.error(
                "[implementer] v1.11 Error processing %s: %s",
                file_path, e
            )
            
            if consecutive_errors >= MULTI_FILE_MAX_ERRORS:
                abort_error = f"Aborted after {consecutive_errors} consecutive errors"
                details.append(file_result)
                break
            
            details.append(file_result)
    
    # Final success determination
    success = files_modified > 0 or (files_failed == 0 and files_unchanged > 0)
    if abort_error:
        success = False
    
    # Completion callback
    await call_progress({
        "type": "complete",
        "operation": "refactor",
        "total_files": len(target_files),
        "files_modified": files_modified,
        "files_unchanged": files_unchanged,
        "files_failed": files_failed,
        "total_replacements": total_replacements,
        "success": success,
    })
    
    logger.info(
        "[implementer] v1.11 Multi-file REFACTOR complete: "
        "modified=%d, unchanged=%d, failed=%d, replacements=%d",
        files_modified,
        files_unchanged,
        files_failed,
        total_replacements,
    )
    
    return MultiFileResult(
        success=success,
        operation="refactor",
        search_pattern=search_pattern,
        replacement_pattern=replacement_pattern,
        total_files=len(target_files),
        files_processed=files_processed,
        files_modified=files_modified,
        files_unchanged=files_unchanged,
        files_failed=files_failed,
        total_replacements=total_replacements,
        errors=errors,
        details=details,
        error=abort_error,
        duration_ms=elapsed_ms(),
    )


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


__all__ = [
    "ImplementerResult",
    "VerificationResult",
    "MultiFileResult",
    "run_implementer",
    "run_verification",
    "run_multi_file_search",
    "run_multi_file_refactor",
    "run_multi_file_operation",
    "MULTI_FILE_MAX_ERRORS",
    "MULTI_FILE_VERIFY_TIMEOUT",
]
