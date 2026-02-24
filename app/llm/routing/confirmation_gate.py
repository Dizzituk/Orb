# FILE: app/llm/routing/confirmation_gate.py
"""
User Confirmation Gate — Yes/No safety gate for costly routing decisions.

When the system wants to:
  1. Escalate to a more expensive model (e.g., GPT-5 Mini → Claude Opus)
  2. Route to a non-chat action (web search, deep research, memory ingest)

...this gate fires a quick yes/no confirmation in the chat UI.

Every decision (yes or no) is logged as training data so the system
learns the user's intent patterns over time and can eventually suppress
confirmations for high-confidence matches.

Suppression logic:
  - If the pattern has been confirmed >= 5 times with 0 rejections → auto-approve
  - If the pattern has mixed signals → always confirm
  - If confidence >= 0.95 from Tier 0 rules → auto-approve (deterministic match)

v2.1 (2026-02): Initial implementation.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# =========================================================================
# Configuration
# =========================================================================

# Minimum confirms before auto-approval kicks in
AUTO_APPROVE_THRESHOLD = int(os.getenv("CONFIRM_GATE_AUTO_THRESHOLD", "5"))

# Confidence floor — Tier 0 matches at or above this skip confirmation
CONFIDENCE_AUTO_APPROVE = float(os.getenv("CONFIRM_GATE_CONFIDENCE_FLOOR", "0.95"))

# Enable/disable the gate entirely (for testing)
GATE_ENABLED = os.getenv("CONFIRM_GATE_ENABLED", "true").lower() in ("true", "1", "yes")

# Decisions that always require confirmation regardless of history
ALWAYS_CONFIRM_INTENTS = {
    # Pipeline operations are already gated by "Astra command:" prefix
    # so they don't need this gate. Add any future high-cost intents here.
}

# Model tiers that are considered "expensive" (escalation from cheap)
EXPENSIVE_TIERS = {"deep", "reasoning"}
CHEAP_TIERS = {"ping_pong", "lookup"}

# Persistence file for decision history
_DECISION_LOG_PATH = Path(
    os.getenv("CONFIRM_GATE_LOG_PATH", "D:/Orb/data/confirmation_decisions.jsonl")
)


# =========================================================================
# Data models
# =========================================================================

@dataclass
class ConfirmationRequest:
    """A pending confirmation for the user."""
    gate_type: str              # "model_escalation" | "intent_routing" | "web_search"
    description: str            # Human-readable: "Promote to Claude Opus?"
    detail: str                 # Why: "Your message appears complex (architecture + multi-file)"
    original_message: str       # The user's original text
    proposed_action: str        # What would happen: "route to Claude Opus 4.6"
    proposed_intent: Optional[str] = None
    proposed_provider: Optional[str] = None
    proposed_model: Optional[str] = None
    confidence: float = 0.0     # How confident the system is
    pattern_key: str = ""       # Key for decision history lookup


@dataclass
class ConfirmationDecision:
    """A logged decision (yes/no) for training."""
    timestamp: float
    pattern_key: str            # Normalised pattern for grouping
    gate_type: str
    proposed_action: str
    approved: bool
    original_message: str
    confidence: float
    user_correction: Optional[str] = None  # If user said "no", what did they want?


@dataclass
class PatternHistory:
    """Aggregated history for a specific pattern."""
    pattern_key: str
    total_asks: int = 0
    approvals: int = 0
    rejections: int = 0

    @property
    def approval_rate(self) -> float:
        return self.approvals / self.total_asks if self.total_asks > 0 else 0.0

    @property
    def should_auto_approve(self) -> bool:
        """Auto-approve if enough consistent approvals and no rejections."""
        return (
            self.approvals >= AUTO_APPROVE_THRESHOLD
            and self.rejections == 0
        )

    @property
    def should_always_ask(self) -> bool:
        """Always ask if there are mixed signals."""
        return self.approvals > 0 and self.rejections > 0


# =========================================================================
# Pattern key generation
# =========================================================================

def _make_pattern_key(gate_type: str, proposed_action: str, message: str) -> str:
    """
    Generate a normalised pattern key for decision history.

    Groups similar decisions together so "search the web for X"
    and "search the web for Y" both map to the same pattern.
    """
    # For intent routing, the key is the gate_type + intent
    if gate_type == "intent_routing":
        return f"intent:{proposed_action}"

    # For model escalation, key is the tier transition
    if gate_type == "model_escalation":
        return f"model:{proposed_action}"

    return f"{gate_type}:{proposed_action}"


# =========================================================================
# Decision history (file-based JSONL)
# =========================================================================

def _load_pattern_history(pattern_key: str) -> PatternHistory:
    """Load decision history for a specific pattern."""
    history = PatternHistory(pattern_key=pattern_key)

    if not _DECISION_LOG_PATH.exists():
        return history

    try:
        with open(_DECISION_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("pattern_key") == pattern_key:
                        history.total_asks += 1
                        if record.get("approved"):
                            history.approvals += 1
                        else:
                            history.rejections += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning("[confirm_gate] Failed to load history: %s", e)

    return history


def log_decision(decision: ConfirmationDecision) -> None:
    """Append a decision to the JSONL log."""
    try:
        _DECISION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_DECISION_LOG_PATH, "a", encoding="utf-8") as f:
            record = {
                "timestamp": decision.timestamp,
                "pattern_key": decision.pattern_key,
                "gate_type": decision.gate_type,
                "proposed_action": decision.proposed_action,
                "approved": decision.approved,
                "original_message": decision.original_message[:200],
                "confidence": decision.confidence,
            }
            if decision.user_correction:
                record["user_correction"] = decision.user_correction
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning("[confirm_gate] Failed to log decision: %s", e)


# =========================================================================
# Core gate logic
# =========================================================================

def should_confirm_model_escalation(
    from_tier: str,
    to_tier: str,
    confidence: float,
    message: str,
) -> Optional[ConfirmationRequest]:
    """
    Check if a model escalation needs user confirmation.

    Returns ConfirmationRequest if confirmation needed, None if auto-approved.
    """
    if not GATE_ENABLED:
        return None

    # No confirmation needed if staying in same tier or going cheaper
    if from_tier == to_tier:
        return None
    if to_tier in CHEAP_TIERS:
        return None
    if from_tier not in CHEAP_TIERS:
        return None  # Already on an expensive tier

    # High-confidence Tier 0 matches skip confirmation
    if confidence >= CONFIDENCE_AUTO_APPROVE:
        return None

    # Build the proposed action description
    tier_model_names = {
        "deep": "Claude Opus 4.6",
        "reasoning": "GPT-5.2",
        "multimodal": "Gemini 3.1 Pro",
    }
    model_name = tier_model_names.get(to_tier, to_tier)
    action = f"{from_tier}_to_{to_tier}"
    pattern_key = _make_pattern_key("model_escalation", action, message)

    # Check history
    history = _load_pattern_history(pattern_key)
    if history.should_auto_approve:
        logger.info("[confirm_gate] Auto-approving model escalation %s (history: %d/%d)",
                     action, history.approvals, history.total_asks)
        return None

    return ConfirmationRequest(
        gate_type="model_escalation",
        description=f"Promote to {model_name}?",
        detail=f"Your message looks like it needs a more capable model ({to_tier} tier).",
        original_message=message,
        proposed_action=action,
        proposed_model=model_name,
        confidence=confidence,
        pattern_key=pattern_key,
    )


def should_confirm_intent_routing(
    intent: str,
    confidence: float,
    message: str,
) -> Optional[ConfirmationRequest]:
    """
    Check if an intent routing decision needs user confirmation.

    Returns ConfirmationRequest if confirmation needed, None if auto-approved.
    """
    if not GATE_ENABLED:
        return None

    # Pipeline intents already gated by "Astra command:" prefix
    pipeline_intents = {
        "RUN_PIPELINE", "RUN_CRITICAL_PIPELINE_FOR_JOB",
        "RUN_SEGMENT_LOOP", "IMPLEMENT_SEGMENTS",
        "OVERWATCHER_EXECUTE_CHANGES", "REFACTOR_CODEBASE",
        "WEAVER_BUILD_SPEC", "SEND_TO_SPEC_GATE",
    }
    if intent in pipeline_intents:
        return None  # Already has wake phrase gate

    # Chat doesn't need confirmation
    if intent in ("CHAT_ONLY", "USER_BEHAVIOR_FEEDBACK"):
        return None

    # Read-only codebase queries don't need confirmation
    read_only_intents = {
        "RAG_CODEBASE_QUERY", "EMBEDDING_STATUS", "FILESYSTEM_QUERY",
        "CODEBASE_REPORT", "LATEST_ARCHITECTURE_MAP",
        "LATEST_CODEBASE_REPORT_FULL", "MULTI_FILE_SEARCH",
        "QUARANTINE_STATUS",
    }
    if intent in read_only_intents:
        return None

    # High-confidence Tier 0 matches skip confirmation
    if confidence >= CONFIDENCE_AUTO_APPROVE:
        return None

    # Build description based on intent
    intent_descriptions = {
        "WEB_SEARCH": ("Search the web?", "I'll search the web for relevant results."),
        "DEEP_RESEARCH": ("Run deep research?", "I'll do a thorough multi-round web research."),
        "MEMORY_INGEST": ("Ingest into memory?", "I'll process this data into ASTRA's memory."),
        "MEMORY_STORE": ("Save to memory?", "I'll remember this fact/preference."),
        "GENERATE_EMBEDDINGS": ("Generate embeddings?", "I'll run the embedding pipeline."),
        "MULTI_FILE_REFACTOR": ("Refactor across files?", "I'll modify multiple files."),
    }

    desc, detail = intent_descriptions.get(intent, (f"Run {intent}?", f"Routing to {intent} handler."))
    pattern_key = _make_pattern_key("intent_routing", intent, message)

    # Check history
    history = _load_pattern_history(pattern_key)
    if history.should_auto_approve:
        logger.info("[confirm_gate] Auto-approving intent %s (history: %d/%d)",
                     intent, history.approvals, history.total_asks)
        return None

    return ConfirmationRequest(
        gate_type="intent_routing",
        description=desc,
        detail=detail,
        original_message=message,
        proposed_action=intent,
        proposed_intent=intent,
        confidence=confidence,
        pattern_key=pattern_key,
    )


# =========================================================================
# SSE helpers for chat UI
# =========================================================================

def format_confirmation_sse(req: ConfirmationRequest) -> str:
    """Format a confirmation request as an SSE event for the chat UI."""
    payload = {
        "type": "confirmation_required",
        "gate_type": req.gate_type,
        "description": req.description,
        "detail": req.detail,
        "proposed_action": req.proposed_action,
        "pattern_key": req.pattern_key,
        "confidence": req.confidence,
    }
    if req.proposed_model:
        payload["proposed_model"] = req.proposed_model
    if req.proposed_intent:
        payload["proposed_intent"] = req.proposed_intent

    return f"data: {json.dumps(payload)}\n\n"


def process_confirmation_response(
    pattern_key: str,
    gate_type: str,
    proposed_action: str,
    approved: bool,
    original_message: str,
    confidence: float,
    user_correction: Optional[str] = None,
) -> None:
    """Process a user's yes/no response and log it for training."""
    decision = ConfirmationDecision(
        timestamp=time.time(),
        pattern_key=pattern_key,
        gate_type=gate_type,
        proposed_action=proposed_action,
        approved=approved,
        original_message=original_message,
        confidence=confidence,
        user_correction=user_correction,
    )
    log_decision(decision)

    logger.info(
        "[confirm_gate] Decision logged: %s %s=%s (approved=%s, confidence=%.2f)",
        gate_type, proposed_action,
        "YES" if approved else "NO",
        approved, confidence,
    )


# =========================================================================
# Public API
# =========================================================================

__all__ = [
    # Configuration
    "GATE_ENABLED",
    "AUTO_APPROVE_THRESHOLD",
    "CONFIDENCE_AUTO_APPROVE",
    # Data models
    "ConfirmationRequest",
    "ConfirmationDecision",
    "PatternHistory",
    # Gate functions
    "should_confirm_model_escalation",
    "should_confirm_intent_routing",
    # SSE helpers
    "format_confirmation_sse",
    "process_confirmation_response",
    # History
    "log_decision",
]
