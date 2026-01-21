# FILE: app/llm/local_tools/zobie/content_classifier.py
"""
Content Classifier for Sandbox File Discovery.

Classifies file content by type using cheap heuristics (no LLM needed).
Used by sandbox_inspector.py to select the right file for a job.

RULES:
- Pattern matching only - no API calls
- Fast (operates on first ~800 chars)
- Wide score gaps to prevent wrong-type selection

v1.0 (2026-01): Initial implementation for SpecGate sandbox discovery
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, Tuple


class ContentType(str, Enum):
    """File content type classification."""
    CODE = "code"
    MESSAGE = "message"  # Natural text a human wrote (email, letter, note)
    STRUCTURED_DATA = "structured_data"  # JSON/YAML/CSV
    MIXED = "mixed"  # Can't confidently classify as one type
    UNKNOWN = "unknown"


# =============================================================================
# CONTENT CLASSIFICATION (Cheap Heuristics)
# =============================================================================

def classify_content(content: str, max_chars: int = 800) -> Tuple[ContentType, float]:
    """
    Classify content type and return confidence score (0-1).
    
    Uses first max_chars only for speed.
    No LLM needed - pattern matching is sufficient.
    
    Args:
        content: File content to classify
        max_chars: Maximum characters to analyze (default 800)
        
    Returns:
        (ContentType, confidence) tuple
    """
    if not content or not content.strip():
        return ContentType.UNKNOWN, 0.0
    
    sample = content[:max_chars]
    lines = sample.split('\n')
    
    # Count indicators
    code_score = 0.0
    message_score = 0.0
    structured_score = 0.0
    
    # === CODE INDICATORS ===
    code_symbols = len(re.findall(r'[{}()\[\];:<>=]', sample))
    code_keywords = len(re.findall(
        r'\b(def|class|import|from|function|const|let|var|return|if|else|for|while|#include|async|await|try|except|catch)\b',
        sample
    ))
    indented_lines = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))
    file_header = 1 if re.search(r'^#\s*FILE:', sample, re.MULTILINE) else 0
    
    code_score = (
        min(code_symbols / 50, 1) * 0.3 +
        min(code_keywords / 5, 1) * 0.4 +
        min(indented_lines / 10, 1) * 0.2 +
        file_header * 0.1
    )
    
    # === MESSAGE INDICATORS ===
    # Sentence patterns (capital start, punctuation end)
    sentences = len(re.findall(r'[A-Z][^.!?]*[.!?]', sample))
    # Question marks (human asking something)
    questions = sample.count('?')
    # Greeting/letter patterns
    greetings = len(re.findall(
        r'\b(hello|hi|hey|dear|please|thanks|thank you|sorry|sincerely|regards|best|cheers)\b',
        sample.lower()
    ))
    # Personal pronouns (I, you, we, my, your)
    pronouns = len(re.findall(r'\b(i|you|we|my|your|our|me|us)\b', sample.lower()))
    # Word density (words per line) - prose has higher density
    words = len(sample.split())
    word_density = words / max(len(lines), 1)
    
    message_score = (
        min(sentences / 5, 1) * 0.25 +
        min(questions / 2, 1) * 0.15 +
        min(greetings / 2, 1) * 0.25 +
        min(pronouns / 5, 1) * 0.15 +
        min(word_density / 10, 1) * 0.2
    )
    
    # === STRUCTURED DATA INDICATORS ===
    json_braces = sample.count('{') + sample.count('}')
    json_brackets = sample.count('[') + sample.count(']')
    json_colons = sample.count(':')
    yaml_colons = len(re.findall(r'^\s*\w+:', sample, re.MULTILINE))
    csv_commas = sample.count(',')
    
    # JSON: has braces/brackets AND colons (key-value pairs)
    looks_like_json = (json_braces >= 2 or json_brackets >= 2) and json_colons >= 2
    # YAML: has indented key: patterns
    looks_like_yaml = yaml_colons >= 3
    # CSV: has commas with consistent line structure
    lines_with_commas = sum(1 for line in lines if ',' in line)
    looks_like_csv = lines_with_commas >= 3 and csv_commas >= 10
    
    if looks_like_json:
        structured_score = max(structured_score, 0.8)
    if looks_like_yaml:
        structured_score = max(structured_score, 0.7)
    if looks_like_csv:
        structured_score = max(structured_score, 0.6)
    
    # === DETERMINE TYPE ===
    scores: Dict[ContentType, float] = {
        ContentType.CODE: code_score,
        ContentType.MESSAGE: message_score,
        ContentType.STRUCTURED_DATA: structured_score,
    }
    
    best_type = max(scores, key=lambda k: scores[k])
    best_score = scores[best_type]
    
    # Check for unknown (nothing scores well)
    if best_score < 0.3:
        return ContentType.UNKNOWN, best_score
    
    # Check if multiple types score similarly (mixed)
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) >= 2 and sorted_scores[0] - sorted_scores[1] < 0.15:
        return ContentType.MIXED, best_score
    
    return best_type, best_score


# =============================================================================
# INTENT SCORING (Wide Gaps for Type Separation)
# =============================================================================

def score_file_for_intent(content_type: ContentType, job_intent: str) -> float:
    """
    Score how well a content type matches the job intent.
    
    CRITICAL: Uses wide score gaps to ensure type separation.
    - MESSAGE jobs → CODE gets 0.05 (effectively impossible to select)
    - CODE jobs → MESSAGE gets 0.05 (effectively impossible to select)
    
    This prevents selecting a script.py when user wants to "reply to message".
    
    Args:
        content_type: Classified content type
        job_intent: User's intent text (from Weaver)
        
    Returns:
        Score 0.0-1.0 (higher = better match)
    """
    if not job_intent:
        # No intent - neutral scores, slight preference for message
        return {
            ContentType.MESSAGE: 0.6,
            ContentType.CODE: 0.5,
            ContentType.STRUCTURED_DATA: 0.4,
            ContentType.MIXED: 0.4,
            ContentType.UNKNOWN: 0.3,
        }.get(content_type, 0.3)
    
    intent_lower = job_intent.lower()
    
    # STRONG MESSAGE SIGNALS (reply/respond jobs)
    message_jobs = [
        "reply", "respond", "answer", "message", "email", "letter",
        "write back", "draft response", "compose", "write to"
    ]
    if any(w in intent_lower for w in message_jobs):
        return {
            ContentType.MESSAGE: 1.0,
            ContentType.MIXED: 0.4,
            ContentType.STRUCTURED_DATA: 0.1,
            ContentType.CODE: 0.05,  # Wide gap - never wins
            ContentType.UNKNOWN: 0.2,
        }.get(content_type, 0.2)
    
    # STRONG CODE SIGNALS (fix/debug jobs)
    code_jobs = [
        "fix", "debug", "refactor", "edit code", "function", "class", "script",
        "bug", "error", "implement", "add feature", "modify code", "update code",
        "syntax", "compile", "run", "execute"
    ]
    if any(w in intent_lower for w in code_jobs):
        return {
            ContentType.CODE: 1.0,
            ContentType.MIXED: 0.4,
            ContentType.MESSAGE: 0.05,  # Wide gap - never wins
            ContentType.STRUCTURED_DATA: 0.2,
            ContentType.UNKNOWN: 0.2,
        }.get(content_type, 0.2)
    
    # STRONG DATA SIGNALS
    data_jobs = [
        "parse", "config", "json", "yaml", "csv", "data", "settings",
        "configuration", "schema", "format"
    ]
    if any(w in intent_lower for w in data_jobs):
        return {
            ContentType.STRUCTURED_DATA: 1.0,
            ContentType.MIXED: 0.4,
            ContentType.CODE: 0.3,
            ContentType.MESSAGE: 0.1,
            ContentType.UNKNOWN: 0.2,
        }.get(content_type, 0.2)
    
    # AMBIGUOUS INTENT - similar scores → triggers ask
    ambiguous_jobs = ["review", "look", "check", "examine", "see", "read", "open"]
    if any(w in intent_lower for w in ambiguous_jobs):
        return {
            ContentType.MESSAGE: 0.5,
            ContentType.CODE: 0.5,
            ContentType.STRUCTURED_DATA: 0.4,
            ContentType.MIXED: 0.4,
            ContentType.UNKNOWN: 0.3,
        }.get(content_type, 0.3)
    
    # DEFAULT: Slight preference for message (common case)
    return {
        ContentType.MESSAGE: 0.6,
        ContentType.CODE: 0.5,
        ContentType.STRUCTURED_DATA: 0.4,
        ContentType.MIXED: 0.4,
        ContentType.UNKNOWN: 0.3,
    }.get(content_type, 0.3)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ContentType",
    "classify_content",
    "score_file_for_intent",
]
