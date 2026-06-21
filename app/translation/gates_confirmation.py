# FILE: app/translation/gates_confirmation.py
# Purpose: Confirmation gate + ConfirmationState for high-stakes operations — split from gates.py.
# Called-by: app.translation.gates
# Depends-on: app.translation.confirmation_log, app.translation.intents, app.translation.schemas
# Last-renovated: 2026-06-21
"""Confirmation gate (high-stakes): explicit-Yes confirmation with graduated-confidence bypass."""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Tuple
from .schemas import CanonicalIntent, ConfirmationGateResult
from .intents import get_intent_definition

logger = logging.getLogger(__name__)


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
        original_message: str = "",
        confidence: float = 0.0,
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
        
        # Stash original message + confidence for confirmation logging
        context["_original_message"] = original_message
        context["_confidence"] = confidence
        
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
        intent = pending["intent"]
        context = pending["context"]
        original_excerpt = context.get("_original_message", "")[:200]
        confidence = context.get("_confidence", 0.0)
        
        # Check for explicit "Yes" confirmation
        response = user_response.strip().lower()
        confirmed = response in (
            "yes", "y", "confirm", "confirmed", "go",
            "do it", "proceed", "ok", "okay",
        )
        
        # Log the confirmation event
        try:
            from app.translation.confirmation_log import log_confirmation_event
            log_confirmation_event(
                intent=intent,
                user_message_excerpt=original_excerpt,
                confirmed=confirmed,
                confidence=confidence,
                conversation_id=confirmation_id,
            )
        except Exception:
            pass  # Don't break flow if logging fails
        
        # Remove from pending
        del self._pending[confirmation_id]
        
        if confirmed:
            return True, intent, context
        
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
    
    v2.2: Auto-execute intents that have reached 95%+ confirmation rate
    over 20+ samples. This is the graduated confidence bypass.
    
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
    
    # v2.2: Check graduated confidence — auto-execute if 95%+ over 20+ samples
    try:
        from app.translation.confirmation_log import get_confirmation_rate
        rate = get_confirmation_rate(intent, min_samples=20)
        if rate is not None and rate >= 0.95:
            logger.info(
                "[confirmation_gate] Auto-execute: %s has %.1f%% confirmation rate",
                intent.value, rate * 100,
            )
            return ConfirmationGateResult(
                passed=True,
                gate_name="confirmation",
                reason=f"Auto-execute: {rate:.0%} confirmation rate (graduated)",
                requires_confirmation=False,
            )
    except Exception:
        pass  # Fall through to normal gating
    
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
