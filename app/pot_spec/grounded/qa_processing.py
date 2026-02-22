# FILE: app/pot_spec/grounded/qa_processing.py
"""
Q&A File Analysis and Reply Generation (v1.30)

Handles Q&A file structure detection and LLM-powered reply generation.

Version Notes:
-------------
v1.30 (2026-01-28): Intelligent document analysis - removed rigid Q&A parsing
    - Simplified analyze_qa_file() to detection-only (no regex parsing)
    - LLM now receives ENTIRE document for intelligent analysis
    - LLM identifies ALL questions regardless of format/numbering
    - Removed _process_question_block() and _process_numbered_question_block()
    - Works with unnumbered, mixed format, code questions, trick questions
    - More robust - LLM is smarter than regex patterns
v1.29 (2026-01-28): Massively increased token limits for larger jobs
    - Deep analysis: max 32000 tokens (was 6000)
    - Standard Q&A: max 16000 tokens (was 4000)
    - Non-Q&A files: max 8000 tokens (was 500)
    - Better scaling formula for large question counts
    - Added detailed token budget logging
v1.28 (2026-01-28): Deep Q&A analysis with correctness checking
    - Added _needs_deep_analysis() to detect when user wants answer verification
    - Enhanced LLM prompt for checking existing answers for correctness
    - Identifies trick questions when user requests
    - Only activates when user explicitly asks for correction checking
v1.27 (2026-01-27): Multi-format Q&A detection
    - Added support for numbered format: "1)\nQuestion\n...\nAnswer"
    - Better detection of Q&A files regardless of exact format
    - Improved trick question detection
v1.23 (2026-01): Dynamic max_tokens calculation
v1.22 (2026-01): Debug + hard-fail for Q&A files
v1.16 (2026-01): Q&A file structure detection
v1.8 (2026-01): LLM-powered reply generation
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# v1.31 BUILD VERIFICATION
# =============================================================================
QA_PROCESSING_BUILD_ID = "2026-01-30-v1.31-multi-file-synthesis"
print(f"[QA_PROCESSING_LOADED] BUILD_ID={QA_PROCESSING_BUILD_ID}")
logger.info(f"[qa_processing] Module loaded: BUILD_ID={QA_PROCESSING_BUILD_ID}")

# =============================================================================
# TOKEN BUDGET CONFIGURATION (v1.29)
# =============================================================================

# Deep analysis mode - checking ALL questions for correctness
DEEP_ANALYSIS_MIN_TOKENS = 4000
DEEP_ANALYSIS_TOKENS_PER_QUESTION = 400
DEEP_ANALYSIS_BASE_TOKENS = 2000  # v1.30: Increased for full doc analysis
DEEP_ANALYSIS_MAX_TOKENS = 32000

# Standard Q&A mode - only answering unanswered questions
STANDARD_QA_MIN_TOKENS = 2000
STANDARD_QA_TOKENS_PER_QUESTION = 300
STANDARD_QA_BASE_TOKENS = 1000  # v1.30: Increased for full doc analysis
STANDARD_QA_MAX_TOKENS = 16000

# Non-Q&A file analysis
NON_QA_MAX_TOKENS = 8000


def _calculate_token_budget(
    question_count: int,
    is_deep_analysis: bool,
    content_length: int = 0,
) -> int:
    """
    v1.30: Calculate token budget based on question count, analysis type, and content size.
    
    Deep analysis needs more tokens because it:
    - Analyzes ALL questions (not just unanswered)
    - Checks correctness of existing answers
    - Identifies trick questions with explanations
    - v1.30: Now includes full document context
    
    Args:
        question_count: Estimated number of questions to process
        is_deep_analysis: True if checking correctness, False if just filling blanks
        content_length: v1.30: Length of document content (for sizing)
        
    Returns:
        Token budget for LLM call
    """
    # v1.30: Account for document content in token budget
    content_tokens = content_length // 4  # Rough estimate: 4 chars per token
    
    if is_deep_analysis:
        calculated = (
            DEEP_ANALYSIS_BASE_TOKENS + 
            (question_count * DEEP_ANALYSIS_TOKENS_PER_QUESTION) +
            content_tokens
        )
        budget = max(DEEP_ANALYSIS_MIN_TOKENS, min(calculated, DEEP_ANALYSIS_MAX_TOKENS))
        logger.info(
            "[qa_processing] v1.30 TOKEN BUDGET (deep): questions=%d, content_tokens=%d, calculated=%d, final=%d",
            question_count, content_tokens, calculated, budget
        )
    else:
        calculated = (
            STANDARD_QA_BASE_TOKENS + 
            (question_count * STANDARD_QA_TOKENS_PER_QUESTION) +
            content_tokens
        )
        budget = max(STANDARD_QA_MIN_TOKENS, min(calculated, STANDARD_QA_MAX_TOKENS))
        logger.info(
            "[qa_processing] v1.30 TOKEN BUDGET (standard): questions=%d, content_tokens=%d, calculated=%d, final=%d",
            question_count, content_tokens, calculated, budget
        )
    
    return budget


# =============================================================================
# Q&A FILE DETECTION (v1.30 - Simplified, no rigid parsing)
# =============================================================================

def analyze_qa_file(content: str) -> Dict[str, Any]:
    """
    v1.30: Simplified - just detect IF it's a Q&A file, not parse structure.
    
    Let the LLM do the actual parsing since it can handle any format.
    The rigid regex parsing in v1.27-v1.29 was missing questions due to format issues.
    
    Returns:
        Dict with is_qa_file, estimated_questions, detection_method
    """
    if not content:
        return {
            "is_qa_file": False,
            "estimated_questions": 0,
            "detection_method": None,
        }
    
    content_lower = content.lower()
    
    # =================================================================
    # Simple heuristics to detect Q&A-like content
    # We're NOT parsing - just detecting presence of Q&A structure
    # =================================================================
    
    # Count indicators of Q&A structure
    question_keyword_count = len(re.findall(r'\bquestion\b', content_lower))
    answer_keyword_count = len(re.findall(r'\banswer\b', content_lower))
    q_markers = len(re.findall(r'\bq\d+\b', content_lower))
    numbered_markers = len(re.findall(r'(?:^|\n)\s*\d+[\)\.\:]', content))
    question_marks = content.count('?')
    
    # Estimate question count (rough - LLM will find actual count)
    estimated = max(
        question_keyword_count,
        answer_keyword_count,
        q_markers,
        numbered_markers,
        question_marks // 2,  # Halve question marks (some are in answers)
    )
    
    # v1.30: More permissive detection - let LLM verify
    is_qa = (
        # Traditional Q&A format indicators
        (question_keyword_count >= 2 and answer_keyword_count >= 2) or
        (q_markers >= 2) or
        (question_keyword_count >= 3) or
        (answer_keyword_count >= 3) or
        # v1.30: Also detect by question mark density
        (question_marks >= 3 and len(content) < 5000) or
        # v1.30: Numbered list with reasonable density
        (numbered_markers >= 3 and answer_keyword_count >= 1) or
        # v1.30: Just lots of "Question" or "Answer" keywords
        (question_keyword_count + answer_keyword_count >= 4)
    )
    
    detection_method = None
    if is_qa:
        if question_keyword_count >= 2 and answer_keyword_count >= 2:
            detection_method = "keyword_pairs"
        elif q_markers >= 2:
            detection_method = "q_markers"
        elif numbered_markers >= 3:
            detection_method = "numbered_list"
        elif question_marks >= 3:
            detection_method = "question_marks"
        else:
            detection_method = "heuristic"
    
    result = {
        "is_qa_file": is_qa,
        "estimated_questions": estimated,
        "detection_method": detection_method,
        # v1.30: Keep some stats for debugging
        "_stats": {
            "question_keywords": question_keyword_count,
            "answer_keywords": answer_keyword_count,
            "q_markers": q_markers,
            "numbered_markers": numbered_markers,
            "question_marks": question_marks,
        }
    }
    
    logger.info(
        "[qa_processing] v1.30 analyze_qa_file: is_qa=%s, estimated=%d, method=%s, stats=%s",
        is_qa, estimated, detection_method, result["_stats"]
    )
    
    return result


# =============================================================================
# SIMPLE INSTRUCTION DETECTION
# =============================================================================

def detect_simple_instruction(content: str) -> Optional[str]:
    """
    Detect simple instructions like "Say X" that don't need LLM.
    
    Returns the simple reply if detected, None otherwise.
    """
    if not content:
        return None
    
    content_lower = content.lower().strip()
    
    # "Say X" / "Reply X" / "Answer X" patterns
    simple_patterns = [
        r'^(?:just\s+)?say\s+["\']?([^"\']+)["\']?[.!?]?$',
        r'^(?:just\s+)?reply\s+(?:with\s+)?["\']?([^"\']+)["\']?[.!?]?$',
        r'^(?:your\s+)?answer\s+(?:is\s+)?["\']?([^"\']+)["\']?[.!?]?$',
        r'^(?:respond\s+(?:with\s+)?)["\']?([^"\']+)["\']?[.!?]?$',
    ]
    
    for pattern in simple_patterns:
        match = re.match(pattern, content_lower)
        if match:
            reply = match.group(1).strip()
            if reply:
                return reply
    
    # "Say OK" / "Reply OK" without quotes
    if re.match(r'^(?:just\s+)?(?:say|reply|respond)\s+(ok|okay|yes|no|done|understood)[.!?]?$', content_lower):
        match = re.search(r'\b(ok|okay|yes|no|done|understood)\b', content_lower)
        if match:
            return match.group(1).capitalize()
    
    return None


# =============================================================================
# DEEP ANALYSIS DETECTION (v1.28)
# =============================================================================

def _needs_deep_analysis(user_text: str) -> bool:
    """
    v1.28: Detect if user wants deep Q&A analysis beyond simple fill-in.
    
    Deep analysis includes:
    - Checking if existing answers are correct
    - Identifying trick questions
    - Fixing wrong answers
    
    Args:
        user_text: The user's request text (from Weaver or direct input)
        
    Returns:
        True if user wants correctness checking, False for simple fill-in
    """
    if not user_text:
        return False
    
    text_lower = user_text.lower()
    
    # Triggers that indicate user wants correctness checking
    deep_analysis_triggers = [
        # Fix/correct existing answers
        "fix wrong",
        "fix incorrect",
        "fix the wrong",
        "fix any wrong",
        "correct wrong",
        "correct incorrect",
        "correct the wrong",
        "correct the answers",
        "correct any wrong",
        
        # Check/verify existing answers
        "check the answers",
        "check answers",
        "verify answers",
        "verify the answers",
        "review answers",
        "review the answers",
        
        # Trick questions
        "identify trick",
        "spot the trick",
        "find trick",
        "trick question",
        "watch for trick",
        "look for trick",
        
        # Answers might be wrong
        "some are wrong",
        "might be wrong",
        "may be wrong",
        "may be incorrect",
        "could be wrong",
        "answers wrong",
        "wrong answers",
        "incorrect answers",
        "answered wrong",
        
        # General deep analysis signals
        "all of the above",
        "check all",
        "verify all",
        "fix all",
        "correct all",
        
        # v1.30: Additional triggers for fill-in scenarios
        "fill in the missing",
        "fill in missing",
        "fill the missing",
        "missing ones",
        "missing answers",
    ]
    
    for trigger in deep_analysis_triggers:
        if trigger in text_lower:
            logger.info(
                "[qa_processing] v1.30 _needs_deep_analysis: TRIGGERED by '%s'",
                trigger
            )
            return True
    
    return False


# =============================================================================
# LLM REPLY GENERATION (v1.30 - Full Document Analysis)
# =============================================================================

async def generate_reply_from_content(
    content: str,
    content_type: Optional[str] = None,
    provider_id: str = "openai",
    model_id: str = "gpt-5-mini",
    llm_call_func: Optional[Any] = None,
    output_mode: Optional[str] = None,
    user_request: Optional[str] = None,
) -> str:
    """
    v1.30: Generate an intelligent reply based on file content using LLM.
    
    KEY CHANGE in v1.30: The ENTIRE document is sent to the LLM.
    The LLM identifies ALL questions regardless of format - no regex parsing.
    
    This fixes issues where rigid regex patterns missed questions due to:
    - Unnumbered questions
    - Unusual formatting (Question\\n<text>\\nanswer\\n<text>)
    - Questions about code formatting or structure
    - Any format not explicitly programmed
    
    Args:
        content: The full text content of the input file
        content_type: Classification type (MESSAGE, CODE, etc.)
        provider_id: LLM provider to use
        model_id: LLM model to use
        llm_call_func: The llm_call function (passed to avoid import issues)
        output_mode: The detected output mode (for OVERWRITE_FULL handling)
        user_request: The user's request text (for deep analysis detection)
    
    Returns:
        Generated reply text
        
    Raises:
        RuntimeError: For Q&A files if LLM is unavailable or fails
    """
    if not content:
        return "(No content to reply to - file was empty)"
    
    content = content.strip()
    
    # Handle OVERWRITE_FULL mode - bypass Q&A analysis
    if output_mode == "overwrite_full":
        from .sandbox_discovery import extract_replacement_text
        replacement = extract_replacement_text(content)
        if replacement:
            logger.info("[qa_processing] v1.30 OVERWRITE_FULL: Using extracted replacement text")
            return replacement
        return "(Could not extract replacement text for OVERWRITE_FULL operation)"
    
    # v1.30: Simplified Q&A detection (no parsing)
    qa_analysis = analyze_qa_file(content)
    
    if qa_analysis["is_qa_file"]:
        estimated_questions = qa_analysis["estimated_questions"]
        
        logger.info(
            "[qa_processing] v1.30 Q&A file detected: estimated_questions=%d, method=%s",
            estimated_questions, qa_analysis.get("detection_method")
        )
        
        # v1.28: Check if user wants deep analysis (correctness checking)
        needs_deep = _needs_deep_analysis(user_request or "")
        
        if needs_deep:
            logger.info(
                "[qa_processing] v1.30 DEEP ANALYSIS MODE: LLM will check all questions for correctness"
            )
        
        # Check LLM availability
        logger.info(
            "[qa_processing] v1.30 Q&A LLM PRE-CHECK: llm_call=%s, provider=%s, model=%s, deep_analysis=%s",
            "available" if llm_call_func else "unavailable",
            provider_id, model_id, needs_deep
        )
        
        if not llm_call_func:
            error_msg = (
                f"[SPECGATE_LLM_UNAVAILABLE] Cannot process Q&A document. "
                f"llm_call function not provided. estimated_questions={estimated_questions}"
            )
            logger.error("[qa_processing] v1.30 HARD-FAIL: %s", error_msg)
            raise RuntimeError(error_msg)
        
        # =================================================================
        # v1.30: Send ENTIRE document to LLM - let it identify all questions
        # =================================================================
        
        user_request_display = user_request or "(answer questions in file)"
        
        if needs_deep:
            # =============================================================
            # DEEP ANALYSIS MODE - Check ALL questions for correctness
            # =============================================================
            
            system_prompt = f"""You are analyzing a document that contains questions and answers.

YOUR TASK:
1. Read the ENTIRE document carefully
2. Identify ALL questions (regardless of format/numbering)
3. For each question, find its corresponding answer (if any)
4. Evaluate each answer:
   - Is it CORRECT? (skip these in output)
   - Is it INCORRECT? (provide the right answer)
   - Is it MISSING? (provide the answer)
   - Is it a TRICK QUESTION? (explain why)

IMPORTANT:
- Questions may or may not be numbered
- Questions may be about code, formatting, structure - don't get confused
- "Question" and "Answer" are section markers, not part of the content itself
- Be thorough - check EVERY question in the document
- For code questions: explain what the code does and why, not just raw output

USER REQUEST: {user_request_display}

OUTPUT FORMAT:
For each question that needs action, output:
Q<number>: [STATUS] <your response>

Where STATUS is:
- [CORRECT] - answer is right (SKIP these to keep output clean)
- [INCORRECT] - answer is wrong. Correct answer: ...
- [MISSING] - no answer provided. Answer: ...
- [TRICK] - trick question because: ...

Number questions sequentially as you find them (Q1, Q2, Q3...).
Only output questions that need fixes - do NOT output [CORRECT] entries."""

            user_prompt = f"""Here is the COMPLETE document to analyze:

---START OF DOCUMENT---
{content}
---END OF DOCUMENT---

Analyze every question in this document. 
Identify wrong answers, missing answers, and trick questions.
Number them Q1, Q2, Q3... as you find them."""

            # v1.30: Token budget accounts for full document
            max_tokens_to_use = _calculate_token_budget(
                question_count=max(estimated_questions, 10),  # Minimum 10 for safety
                is_deep_analysis=True,
                content_length=len(content),
            )
            
            logger.info(
                "[qa_processing] v1.30 DEEP ANALYSIS LLM CALL: estimated_questions=%d, content_len=%d, max_tokens=%d",
                estimated_questions, len(content), max_tokens_to_use
            )
        
        else:
            # =============================================================
            # STANDARD MODE - Just answer unanswered questions
            # =============================================================
            
            system_prompt = f"""You are answering questions in a document.

YOUR TASK:
1. Read the ENTIRE document carefully
2. Identify ALL questions (regardless of format/numbering)
3. For questions that already have answers, leave them alone
4. For questions WITHOUT answers (or with blank/empty answers), provide answers

IMPORTANT:
- Questions may or may not be numbered
- Questions may be about code, formatting, structure - don't get confused
- "Question" and "Answer" are section markers, not part of the content itself
- Only answer questions that are UNANSWERED or have blank answers
- For code questions: explain what the code does and why, not just raw output

USER REQUEST: {user_request_display}

OUTPUT FORMAT:
For each question you answer, output:
Q<number>: [ANSWER] <your response>

Number questions sequentially as you find them (Q1, Q2, Q3...).
Only output questions that needed answers - skip already-answered ones."""

            user_prompt = f"""Here is the COMPLETE document to analyze:

---START OF DOCUMENT---
{content}
---END OF DOCUMENT---

Find all questions that need answers (unanswered or blank).
Provide clear, helpful answers for each.
Number them Q1, Q2, Q3... as you find them."""

            # v1.30: Token budget accounts for full document
            max_tokens_to_use = _calculate_token_budget(
                question_count=max(estimated_questions, 5),  # Minimum 5 for safety
                is_deep_analysis=False,
                content_length=len(content),
            )
            
            logger.info(
                "[qa_processing] v1.30 STANDARD Q&A LLM CALL: estimated_questions=%d, content_len=%d, max_tokens=%d",
                estimated_questions, len(content), max_tokens_to_use
            )
        
        # =================================================================
        # Execute LLM call
        # =================================================================
        try:
            result = await llm_call_func(
                provider_id=provider_id,
                model_id=model_id,
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=max_tokens_to_use,
                timeout_seconds=180,  # v1.30: Increased timeout for full document analysis
            )
            
            if result.is_success() and result.content:
                reply = result.content.strip()
                mode_str = "DEEP ANALYSIS" if needs_deep else "STANDARD"
                logger.info("[qa_processing] v1.30 Q&A LLM SUCCESS (%s): reply_len=%d", mode_str, len(reply))
                return reply
            
            if result.is_success():
                error_msg = (
                    f"[SPECGATE_LLM_EMPTY_RESPONSE] Cannot answer questions. "
                    f"status=SUCCESS but content is empty. max_tokens={max_tokens_to_use}"
                )
                logger.error("[qa_processing] v1.30 HARD-FAIL: %s", error_msg)
                raise RuntimeError(error_msg)
            
            status_str = str(getattr(result, 'status', 'UNKNOWN'))
            error_str = getattr(result, 'error_message', None) or 'No error message'
            sanitized_error = error_str[:200] if error_str else 'None'
            
            error_msg = (
                f"[SPECGATE_LLM_CALL_FAILED] Cannot answer questions. "
                f"status={status_str}, error={sanitized_error}"
            )
            logger.error("[qa_processing] v1.30 HARD-FAIL: %s", error_msg)
            raise RuntimeError(error_msg)
            
        except RuntimeError:
            raise
        except Exception as e:
            exc_type = type(e).__name__
            exc_msg = str(e)[:200]
            
            error_msg = (
                f"[SPECGATE_LLM_EXCEPTION] Cannot answer questions. "
                f"exception={exc_type}: {exc_msg}"
            )
            logger.error("[qa_processing] v1.30 HARD-FAIL: %s", error_msg)
            raise RuntimeError(error_msg) from e
    
    # =================================================================
    # Non-Q&A file handling (v1.29: increased token budget)
    # =================================================================
    simple_reply = detect_simple_instruction(content)
    if simple_reply:
        logger.info("[qa_processing] v1.30 Simple instruction detected")
        return simple_reply
    
    # Use LLM for non-Q&A files (with fallback)
    if llm_call_func:
        try:
            system_prompt = """You are a helpful assistant answering questions found in files.
Your job is to provide clear, concise, accurate answers.
If the file contains code, explain what it does.
If the file contains a question, answer it directly.
Keep your response brief and to the point (1-3 sentences for simple questions).
Do NOT include any preamble like "The answer is" - just give the answer directly."""

            user_prompt = f"""The following content was found in a file. Please provide an appropriate response:

---
{content}
---

Your response:"""

            logger.info(
                "[qa_processing] v1.30 NON-Q&A LLM CALL: content_len=%d, max_tokens=%d",
                len(content), NON_QA_MAX_TOKENS
            )

            result = await llm_call_func(
                provider_id=provider_id,
                model_id=model_id,
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=NON_QA_MAX_TOKENS,
                timeout_seconds=60,
            )
            
            if result.is_success() and result.content:
                logger.info("[qa_processing] v1.30 NON-Q&A LLM SUCCESS: reply_len=%d", len(result.content))
                return result.content.strip()
                
        except Exception as e:
            logger.warning("[qa_processing] v1.30 Non-Q&A LLM failed, using fallback: %s", e)
    
    # Fallback for non-Q&A files
    return _generate_reply_fallback(content, content_type)


def _generate_reply_fallback(content: str, content_type: Optional[str] = None) -> str:
    """Fallback heuristic reply generation for non-Q&A files."""
    if not content:
        return "(No content to process)"
    
    content_preview = content[:500] if len(content) > 500 else content
    
    # Detect questions
    question_patterns = [
        r'\?',
        r'\bwhat\b',
        r'\bhow\b',
        r'\bwhy\b',
        r'\bwhen\b',
        r'\bwhere\b',
        r'\bwhich\b',
    ]
    
    has_question = any(re.search(p, content_preview, re.IGNORECASE) for p in question_patterns)
    
    if has_question:
        return "(Response to question in file - LLM unavailable for intelligent answer)"
    
    # Detect code
    code_patterns = [
        r'\bdef\s+\w+',
        r'\bclass\s+\w+',
        r'\bimport\s+',
        r'\bfunction\s+',
        r'\bconst\s+',
        r'\blet\s+',
        r'\bvar\s+',
    ]
    
    has_code = any(re.search(p, content_preview) for p in code_patterns)
    
    if has_code:
        return "(Code analysis requested - LLM unavailable for intelligent explanation)"
    
    return "(Content acknowledged - LLM unavailable for intelligent response)"


# =============================================================================
# MULTI-FILE SYNTHESIS (v1.31 - Level 2.5)
# =============================================================================

# Token budget for multi-file synthesis
MULTI_FILE_SYNTHESIS_MIN_TOKENS = 2000
MULTI_FILE_SYNTHESIS_MAX_TOKENS = 16000
MULTI_FILE_SYNTHESIS_TOKENS_PER_FILE = 500


async def generate_synthesized_reply_from_files(
    file_contents: list[dict],
    provider_id: str = "openai",
    model_id: str = "gpt-5-mini",
    llm_call_func: Optional[Any] = None,
    user_request: Optional[str] = None,
) -> str:
    """
    v1.31: Generate a SYNTHESIZED reply from MULTIPLE files.
    
    Unlike generate_reply_from_content() which handles one file at a time,
    this function takes ALL files and generates a unified conceptual understanding.
    
    Example:
        File 1: "My name is Astra"
        File 2: "I'm going to be an assistant"
        File 3: "I'm going to help you every day"
        File 4: "Where shall we begin?"
        
        Synthesis: "The files together describe Astra introducing itself as an 
                   assistant that will help the user daily, and is ready to start."
    
    Args:
        file_contents: List of dicts with keys:
            - 'path': file path (str)
            - 'name': file name (str) 
            - 'content': file content (str)
            - 'content_type': optional content type (str)
        provider_id: LLM provider to use
        model_id: LLM model to use
        llm_call_func: The llm_call function
        user_request: The user's original request for context
    
    Returns:
        Synthesized reply combining understanding from all files
    """
    if not file_contents:
        return "(No files to synthesize)"
    
    # Filter to files that have content
    files_with_content = [
        f for f in file_contents 
        if f.get('content') and f['content'].strip()
    ]
    
    if not files_with_content:
        return "(All files were empty or had no content)"
    
    # Build combined context
    file_count = len(files_with_content)
    combined_parts = []
    total_content_length = 0
    
    for i, f in enumerate(files_with_content, 1):
        name = f.get('name', f.get('path', f'File {i}'))
        content = f['content'].strip()
        total_content_length += len(content)
        
        combined_parts.append(f"### File {i}: {name}\n{content}")
    
    combined_context = "\n\n".join(combined_parts)
    
    logger.info(
        "[qa_processing] v1.31 generate_synthesized_reply_from_files: "
        "files=%d, total_content_len=%d",
        file_count, total_content_length
    )
    
    # Check LLM availability
    if not llm_call_func:
        # Fallback: just list what we found
        logger.warning(
            "[qa_processing] v1.31 LLM unavailable for synthesis, using fallback"
        )
        fallback_lines = [f"Read {file_count} files:"]
        for i, f in enumerate(files_with_content, 1):
            name = f.get('name', f'File {i}')
            content = f['content'].strip()[:100]
            fallback_lines.append(f"  {i}. {name}: {content}...")
        return "\n".join(fallback_lines)
    
    # Build synthesis prompt
    user_context = user_request or "read and understand the files"
    
    system_prompt = """You are an intelligent assistant that synthesizes information from multiple files.

YOUR TASK:
1. Read ALL the files provided below
2. Understand what each file contains
3. Synthesize the combined meaning - what do these files TOGETHER tell us?
4. Provide a unified, conceptual understanding

IMPORTANT:
- Look for connections and patterns across files
- The files may form a larger message, story, or concept when combined
- Provide a clear, synthesized summary that captures the COMBINED meaning
- Don't just list what each file says - SYNTHESIZE them into a unified understanding
- If the files contain questions, answer them
- If the files form a narrative or message, explain what it conveys

OUTPUT:
Provide a synthesized understanding that combines the meaning from all files.
Be concise but comprehensive - capture the essence of what the files together communicate."""

    user_prompt = f"""The user wants to: {user_context}

Here are {file_count} files to synthesize:

{combined_context}

---

Please provide a synthesized understanding of what these files together communicate.
What is the combined meaning or message?"""

    # Calculate token budget
    content_tokens = total_content_length // 4
    calculated = (
        MULTI_FILE_SYNTHESIS_MIN_TOKENS + 
        (file_count * MULTI_FILE_SYNTHESIS_TOKENS_PER_FILE) +
        (content_tokens // 2)  # Allow some room for response
    )
    max_tokens = min(calculated, MULTI_FILE_SYNTHESIS_MAX_TOKENS)
    
    logger.info(
        "[qa_processing] v1.31 MULTI-FILE SYNTHESIS LLM CALL: "
        "files=%d, content_tokens=%d, max_tokens=%d",
        file_count, content_tokens, max_tokens
    )
    
    try:
        result = await llm_call_func(
            provider_id=provider_id,
            model_id=model_id,
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=max_tokens,
            timeout_seconds=120,
        )
        
        if result.is_success() and result.content:
            reply = result.content.strip()
            logger.info(
                "[qa_processing] v1.31 MULTI-FILE SYNTHESIS SUCCESS: reply_len=%d",
                len(reply)
            )
            return reply
        
        if result.is_success():
            logger.error(
                "[qa_processing] v1.31 MULTI-FILE SYNTHESIS: LLM returned empty content"
            )
            return "(LLM returned empty response for synthesis)"
        
        status_str = str(getattr(result, 'status', 'UNKNOWN'))
        error_str = getattr(result, 'error_message', None) or 'No error message'
        logger.error(
            "[qa_processing] v1.31 MULTI-FILE SYNTHESIS FAILED: status=%s, error=%s",
            status_str, error_str[:200]
        )
        return f"(Synthesis failed: {error_str[:100]})"
        
    except Exception as e:
        logger.exception(
            "[qa_processing] v1.31 MULTI-FILE SYNTHESIS EXCEPTION: %s", e
        )
        return f"(Synthesis error: {str(e)[:100]})"
