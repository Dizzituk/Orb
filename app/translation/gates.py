# FILE: app/translation/gates.py
"""
Safety gates for ASTRA Translation Layer.
- Directive vs Story Gate: Blocks past tense, questions, future planning
- Context Gate: Ensures required context is present
- Confirmation Gate: Requires explicit Yes for high-stakes operations
"""
from __future__ import annotations
import re
from typing import Dict, Any, List, Optional, Tuple
from .schemas import (
    GateResult,
    DirectiveGateResult,
    ContextGateResult,
    ConfirmationGateResult,
    CanonicalIntent,
)
from .intents import get_intent_definition


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


# =============================================================================
# CONTEXT GATE
# =============================================================================

def check_context_gate(
    intent: CanonicalIntent,
    provided_context: Dict[str, Any],
) -> ContextGateResult:
    """
    Check if required context is present for the given intent.
    
    Args:
        intent: The resolved canonical intent
        provided_context: Context provided (from UI, previous messages, etc.)
        
    Returns:
        ContextGateResult indicating if context requirements are met
    """
    defn = get_intent_definition(intent)
    required = defn.requires_context
    
    if not required:
        return ContextGateResult(
            passed=True,
            gate_name="context",
            reason="No context required for this intent",
            provided_context=provided_context,
        )
    
    missing = []
    for key in required:
        if key not in provided_context or provided_context[key] is None:
            missing.append(key)
    
    if missing:
        return ContextGateResult(
            passed=False,
            gate_name="context",
            reason=f"Missing required context: {', '.join(missing)}",
            missing_context=missing,
            provided_context=provided_context,
        )
    
    return ContextGateResult(
        passed=True,
        gate_name="context",
        reason="All required context present",
        missing_context=[],
        provided_context=provided_context,
    )


def extract_context_from_text(
    text: str,
    intent: CanonicalIntent,
) -> Dict[str, Any]:
    """
    Attempt to extract context from the message text itself.
    E.g., "Run critical pipeline for job abc123" -> {"job_id": "abc123"}
    
    This is a best-effort extraction. Missing context will still
    require clarification.
    """
    context = {}
    text_lower = text.lower()
    
    # Extract job_id
    job_patterns = [
        r"for job\s+([a-zA-Z0-9\-_]+)",
        r"job[:\s]+([a-zA-Z0-9\-_]+)",
        r"job_id[:\s]+([a-zA-Z0-9\-_]+)",
    ]
    for pattern in job_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            context["job_id"] = match.group(1)
            break
    
    # Extract sandbox_id
    sandbox_patterns = [
        r"for sandbox\s+([a-zA-Z0-9\-_]+)",
        r"sandbox[:\s]+([a-zA-Z0-9\-_]+)",
        r"sandbox_id[:\s]+([a-zA-Z0-9\-_]+)",
    ]
    for pattern in sandbox_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            context["sandbox_id"] = match.group(1)
            break
    
    # Extract change_set_id
    changeset_patterns = [
        r"change(?:_?set)?[:\s]+([a-zA-Z0-9\-_]+)",
        r"changes[:\s]+([a-zA-Z0-9\-_]+)",
    ]
    for pattern in changeset_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            context["change_set_id"] = match.group(1)
            break
    
    return context


# =============================================================================
# CONFIRMATION GATE (High-Stakes)
# =============================================================================

class ConfirmationState:
    """
    Tracks pending confirmations for high-stakes operations.
    This would typically be stored in session/conversation state.
    """
    
    def __init__(self):
        self._pending: Dict[str, Dict[str, Any]] = {}
    
    def request_confirmation(
        self,
        confirmation_id: str,
        intent: CanonicalIntent,
        context: Dict[str, Any],
    ) -> str:
        """
        Request confirmation for a high-stakes operation.
        Returns the confirmation prompt to show the user.
        """
        defn = get_intent_definition(intent)
        prompt = defn.confirmation_prompt or (
            f"⚠️ HIGH-STAKES OPERATION\n"
            f"You are about to execute: {intent.value}\n"
            f"Type 'Yes' to confirm."
        )
        
        # Format with context
        try:
            prompt = prompt.format(**context)
        except KeyError:
            pass  # Keep unformatted if context missing
        
        self._pending[confirmation_id] = {
            "intent": intent,
            "context": context,
            "prompt": prompt,
        }
        
        return prompt
    
    def check_confirmation(
        self,
        confirmation_id: str,
        user_response: str,
    ) -> Tuple[bool, Optional[CanonicalIntent], Optional[Dict[str, Any]]]:
        """
        Check if user response confirms the pending operation.
        
        Returns:
            (confirmed, intent, context) if confirmed
            (False, None, None) if not confirmed or no pending
        """
        if confirmation_id not in self._pending:
            return False, None, None
        
        pending = self._pending[confirmation_id]
        
        # Check for explicit "Yes" confirmation
        response = user_response.strip().lower()
        if response in ("yes", "y", "confirm", "confirmed"):
            # Remove from pending and return confirmed
            del self._pending[confirmation_id]
            return True, pending["intent"], pending["context"]
        
        # Not confirmed - remove pending
        del self._pending[confirmation_id]
        return False, None, None
    
    def clear_pending(self, confirmation_id: str) -> None:
        """Clear a pending confirmation."""
        self._pending.pop(confirmation_id, None)
    
    def has_pending(self, confirmation_id: str) -> bool:
        """Check if there's a pending confirmation."""
        return confirmation_id in self._pending


def check_confirmation_gate(
    intent: CanonicalIntent,
    context: Dict[str, Any],
    confirmation_state: Optional[ConfirmationState] = None,
    confirmation_id: Optional[str] = None,
) -> ConfirmationGateResult:
    """
    Check if high-stakes confirmation is required/provided.
    
    Args:
        intent: The resolved intent
        context: Provided context
        confirmation_state: State tracker for pending confirmations
        confirmation_id: ID for this confirmation (e.g., conversation_id)
        
    Returns:
        ConfirmationGateResult indicating confirmation status
    """
    defn = get_intent_definition(intent)
    
    if not defn.requires_confirmation:
        return ConfirmationGateResult(
            passed=True,
            gate_name="confirmation",
            reason="No confirmation required for this intent",
            requires_confirmation=False,
        )
    
    # Requires confirmation
    prompt = defn.confirmation_prompt or f"Confirm execution of {intent.value}?"
    try:
        prompt = prompt.format(**context)
    except KeyError:
        pass
    
    return ConfirmationGateResult(
        passed=False,
        gate_name="confirmation",
        reason="High-stakes operation requires explicit confirmation",
        requires_confirmation=True,
        confirmation_prompt=prompt,
        awaiting_confirmation=True,
    )
