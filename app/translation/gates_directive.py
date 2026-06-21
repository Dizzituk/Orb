# FILE: app/translation/gates_directive.py
# Purpose: Directive-vs-story gate (split from gates.py).
# Called-by: app.translation.gates
# Depends-on: app.translation.schemas
# Last-renovated: 2026-06-21
"""Directive vs Story gate: blocks past tense, questions, future planning, meta-discussion."""
from __future__ import annotations
import re
from typing import Optional, Tuple
from .schemas import DirectiveGateResult


# =============================================================================
# DIRECTIVE VS STORY GATE (VERY IMPORTANT)
# =============================================================================
# Inside Command-Capable mode, ONLY true imperatives may become commands.
# Block the following from ever triggering commands:
# - past tense: "that time you mapped..."
# - future planning: "next week we'll map..."
# - questions: "how do you map...?"
# - talking about commands: "when we run start your zombie..."

# Patterns that indicate NON-DIRECTIVE speech
NON_DIRECTIVE_PATTERNS = {
    # Past tense indicators
    "past_tense": [
        r"\bthat time\b",
        r"\bwhen you\b.*\bed\b",          # "when you mapped", "when you started"
        r"\byou\s+\w+ed\b",                # "you mapped", "you created"
        r"\bi\s+\w+ed\b",                  # "I asked", "I ran"
        r"\bwe\s+\w+ed\b",                 # "we mapped", "we discussed"
        r"\blast\s+(?:time|week|month)\b",
        r"\byesterday\b",
        r"\bpreviously\b",
        r"\bearlier\b",
        r"\bbefore\b",
        r"\bremember when\b",
        r"\brecall when\b",
    ],
    
    # Future planning indicators (not commands)
    "future_planning": [
        r"\bnext\s+(?:time|week|month)\b",
        r"\bwe(?:'ll| will)\b",           # "we'll map", "we will run"
        r"\bi(?:'ll| will)\b",            # "I'll do", "I will start"
        r"\bgoing to\b",
        r"\bplan(?:ning)? to\b",
        r"\bshould we\b",
        r"\bcould we\b",
        r"\bwould be nice to\b",
        r"\bmaybe\s+(?:we|i)\b",
        r"\beventually\b",
        r"\blater\b",
        r"\bsomeday\b",
    ],
    
    # Question indicators
    "question": [
        r"\?$",                            # Ends with question mark
        r"^(?:how|what|when|where|why|who|which|can|could|would|should|is|are|do|does|did)\b",
        r"\bhow do(?:es)?\b",
        r"\bwhat (?:is|are|does|do)\b",
        r"\bcan you\b",
        r"\bcould you\b",
        r"\bwould you\b",
        r"\btell me about\b",
        r"\bexplain\b",
        r"\bdescribe\b",
        r"\bwhat happens when\b",
    ],
    
    # Talking ABOUT commands (meta-discussion)
    "meta_discussion": [
        r"\bwhen (?:we|you|i) (?:run|start|create|update)\b",
        r"\bif (?:we|you|i) (?:run|start|create|update)\b",
        r"\babout the\b.*\bcommand\b",
        r"\babout\b.*\bpipeline\b",
        r"\bhow does the\b",
        r"\bwhat does\b.*\bdo\b",
        r"\bthe\b.*\bsystem\b",
        r"\byour\b.*\bsubsystem\b",        # "your Overwatch subsystem"
        r"\bthe\b.*\bsubsystem\b",
        r"\btell me\b",                    # "tell me about"
        r"\bshow me\b",                    # "show me the"
    ],
    
    # Hypothetical/conditional
    "hypothetical": [
        r"\bif\s+(?:we|you|i)\b",
        r"\bwhat if\b",
        r"\bsuppose\b",
        r"\bimagine\b",
        r"\bhypothetically\b",
        r"\bin theory\b",
    ],
    
    # Storytelling
    "storytelling": [
        r"\bonce upon\b",
        r"\bthere was\b",
        r"\blong ago\b",
        r"\bback when\b",
    ],
}

# Compiled patterns for efficiency
_COMPILED_NON_DIRECTIVE = {
    category: [re.compile(p, re.IGNORECASE) for p in patterns]
    for category, patterns in NON_DIRECTIVE_PATTERNS.items()
}


def check_directive_gate(text: str) -> DirectiveGateResult:
    """
    Check if text is a true directive (imperative command) vs story/question/planning.
    
    Returns:
        DirectiveGateResult with:
        - passed=True if this looks like a genuine imperative command
        - passed=False if this looks like chat (question, past tense, planning, etc.)
    """
    text_lower = text.lower().strip()
    
    # Check each category of non-directive patterns
    for category, patterns in _COMPILED_NON_DIRECTIVE.items():
        for pattern in patterns:
            if pattern.search(text_lower):
                return DirectiveGateResult(
                    passed=False,
                    gate_name="directive_vs_story",
                    reason=f"Detected {category} pattern - not a command",
                    blocked_by=category,
                    detected_pattern=category,
                    original_text_snippet=text[:100],
                )
    
    # If no non-directive patterns found, it passes
    return DirectiveGateResult(
        passed=True,
        gate_name="directive_vs_story",
        reason="No non-directive patterns detected",
    )


def is_obvious_chat(text: str) -> Tuple[bool, Optional[str]]:
    """
    Quick check for messages that are OBVIOUSLY chat.
    Used for Tier 0 short-circuit to avoid classifier entirely.
    
    Returns:
        (is_chat, reason)
    """
    # Check directive gate
    result = check_directive_gate(text)
    if not result.passed:
        return True, result.detected_pattern
    
    # Additional quick checks
    text_lower = text.lower().strip()
    
    # Very short messages without command keywords are chat
    if len(text_lower) < 10:
        command_keywords = ["create", "start", "run", "update", "execute", "launch"]
        if not any(kw in text_lower for kw in command_keywords):
            return True, "short_non_command"
    
    # Messages starting with certain words are chat
    chat_starters = [
        "i think", "i'm", "i am", "it's", "it is", "that's", "that is",
        "well", "so", "hmm", "huh", "ok", "okay", "sure", "yeah", "yes",
        "no", "nope", "thanks", "thank you", "please", "hey", "hi", "hello",
        "good", "great", "nice", "cool", "interesting", "actually", "basically",
    ]
    for starter in chat_starters:
        if text_lower.startswith(starter):
            return True, "chat_starter"
    
    return False, None
