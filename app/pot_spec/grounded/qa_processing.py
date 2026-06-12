# FILE: app/pot_spec/grounded/qa_processing.py
# Purpose: Q&A File Analysis and Reply Generation (v1.30)
# Called-by: app.pot_spec.grounded, app.pot_spec.grounded._evidence_gathering_utils_8
# Depends-on: app.pot_spec.grounded._qa_processing_utils_6, app.pot_spec.grounded._qa_processing_utils_7, app.pot_spec.grounded._qa_processing_utils_8, app.pot_spec.grounded.sandbox_discovery
# Last-renovated: 2026-06-11
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
from typing import Any, Dict, List, Optional
from app.pot_spec.grounded._qa_processing_utils_6 import DEEP_ANALYSIS_MAX_TOKENS, DEEP_ANALYSIS_MIN_TOKENS, NON_QA_MAX_TOKENS, QA_PROCESSING_BUILD_ID, STANDARD_QA_MAX_TOKENS, STANDARD_QA_MIN_TOKENS, STANDARD_QA_TOKENS_PER_QUESTION, _generate_reply_fallback
from app.pot_spec.grounded._qa_processing_utils_7 import DEEP_ANALYSIS_BASE_TOKENS, DEEP_ANALYSIS_TOKENS_PER_QUESTION, MULTI_FILE_SYNTHESIS_MAX_TOKENS, MULTI_FILE_SYNTHESIS_MIN_TOKENS, MULTI_FILE_SYNTHESIS_TOKENS_PER_FILE, STANDARD_QA_BASE_TOKENS, _needs_deep_analysis, detect_simple_instruction
from app.pot_spec.grounded._qa_processing_utils_8 import _calculate_token_budget, analyze_qa_file, generate_synthesized_reply_from_files

logger = logging.getLogger(__name__)

# =============================================================================
# v1.31 BUILD VERIFICATION
# =============================================================================
print(f"[QA_PROCESSING_LOADED] BUILD_ID={QA_PROCESSING_BUILD_ID}")
logger.info(f"[qa_processing] Module loaded: BUILD_ID={QA_PROCESSING_BUILD_ID}")

# =============================================================================
# TOKEN BUDGET CONFIGURATION (v1.29)
# =============================================================================

# Deep analysis mode - checking ALL questions for correctness

# Standard Q&A mode - only answering unanswered questions

# Non-Q&A file analysis


# =============================================================================
# Q&A FILE DETECTION (v1.30 - Simplified, no rigid parsing)
# =============================================================================


# =============================================================================
# SIMPLE INSTRUCTION DETECTION
# =============================================================================


# =============================================================================
# DEEP ANALYSIS DETECTION (v1.28)
# =============================================================================


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


# =============================================================================
# MULTI-FILE SYNTHESIS (v1.31 - Level 2.5)
# =============================================================================

# Token budget for multi-file synthesis
