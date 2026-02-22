from __future__ import annotations
import logging
import re
from app.overwatcher._implementer_utils_2 import _block_has_answer, _find_question_block_starts
from app.overwatcher.sandbox_client import SandboxClient
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
MULTI_FILE_VERIFY_TIMEOUT = 30  # Seconds per file verification


MULTI_FILE_MAX_ERRORS = 10  # Stop after N consecutive errors

def _is_absolute_windows_path(path: str) -> bool:
    """Check if path is an absolute Windows path (e.g., C:\\..., D:\\...)."""
    if len(path) >= 3:
        return path[1] == ':' and path[2] in ('\\', '/')
    return False

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
        # v1.13: Use shared write helper (auto temp-file for large files)
        result = _write_content_to_sandbox(client, file_path, content, timeout_seconds=MULTI_FILE_VERIFY_TIMEOUT)
        
        if result.stderr and result.stderr.strip():
            logger.warning(
                "[implementer] v1.13 Write failed for %s: stderr=%s",
                file_path, result.stderr[:100] if result.stderr else ""
            )
            return False
        
        return True
        
    except Exception as e:
        logger.error("[implementer] v1.13 Write exception for %s: %s", file_path, e)
        return False
