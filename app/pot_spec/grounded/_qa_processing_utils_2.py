from __future__ import annotations
import logging
import re
from app.pot_spec.grounded._qa_processing_utils import DEEP_ANALYSIS_BASE_TOKENS, DEEP_ANALYSIS_TOKENS_PER_QUESTION, MULTI_FILE_SYNTHESIS_MAX_TOKENS, MULTI_FILE_SYNTHESIS_MIN_TOKENS, MULTI_FILE_SYNTHESIS_TOKENS_PER_FILE, STANDARD_QA_BASE_TOKENS
from app.pot_spec.grounded._qa_processing_utils import DEEP_ANALYSIS_MAX_TOKENS, DEEP_ANALYSIS_MIN_TOKENS, STANDARD_QA_MAX_TOKENS, STANDARD_QA_MIN_TOKENS, STANDARD_QA_TOKENS_PER_QUESTION
from typing import Any, Dict, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


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
