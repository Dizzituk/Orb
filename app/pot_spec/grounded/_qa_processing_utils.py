from __future__ import annotations
import re
from typing import Optional


QA_PROCESSING_BUILD_ID = "2026-01-30-v1.31-multi-file-synthesis"

DEEP_ANALYSIS_MIN_TOKENS = 4000

DEEP_ANALYSIS_MAX_TOKENS = 32000

STANDARD_QA_MIN_TOKENS = 2000

STANDARD_QA_TOKENS_PER_QUESTION = 300

STANDARD_QA_MAX_TOKENS = 16000

NON_QA_MAX_TOKENS = 8000

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


# Re-exports: symbols extracted to _utils_1/_2 by refactor loop
_REEXPORT_MAP = {
    "DEEP_ANALYSIS_BASE_TOKENS": "_qa_processing_utils_1",
    "DEEP_ANALYSIS_TOKENS_PER_QUESTION": "_qa_processing_utils_1",
    "MULTI_FILE_SYNTHESIS_MAX_TOKENS": "_qa_processing_utils_1",
    "MULTI_FILE_SYNTHESIS_MIN_TOKENS": "_qa_processing_utils_1",
    "MULTI_FILE_SYNTHESIS_TOKENS_PER_FILE": "_qa_processing_utils_1",
    "STANDARD_QA_BASE_TOKENS": "_qa_processing_utils_1",
    "_needs_deep_analysis": "_qa_processing_utils_1",
    "detect_simple_instruction": "_qa_processing_utils_1",
    "_calculate_token_budget": "_qa_processing_utils_2",
    "analyze_qa_file": "_qa_processing_utils_2",
    "generate_synthesized_reply_from_files": "_qa_processing_utils_2",
}

def __getattr__(name):
    if name in _REEXPORT_MAP:
        import importlib
        mod = importlib.import_module(f"app.pot_spec.grounded.{_REEXPORT_MAP[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
