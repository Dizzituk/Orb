# FILE: app/astra_memory/learning.py
"""
Preference Learning Module for ASTRA Memory System.

Captures user feedback and learns preferences from:
1. Explicit feedback (thumbs up/down on messages)
2. Implicit signals (re-use, edits, corrections)
3. Behavioral patterns (repeated choices)

Spec §6: Preference learning from approvals
"""

from __future__ import annotations

import logging
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from sqlalchemy.orm import Session

from app.astra_memory.preference_models import (
    PreferenceRecord,
    PreferenceEvidence,
    SignalType,
    PreferenceStrength,
    RecordStatus,
)
from app.astra_memory.confidence_scoring import (
    append_preference_evidence,
    record_contradiction,
    recompute_preference_confidence,
)
from app.astra_memory.preference_service import (
    create_preference,
    get_preference,
    reinforce_preference,
    learn_from_behavior,
)

logger = logging.getLogger(__name__)


# =============================================================================
# FEEDBACK RECORDING
# =============================================================================

def record_message_feedback(
    db: Session,
    message_id: int,
    feedback_type: str,  # "positive", "negative", "correction"
    user_comment: Optional[str] = None,
    correction_text: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Record user feedback on a message and trigger preference learning.
    
    Args:
        db: Database session
        message_id: ID of the message receiving feedback
        feedback_type: Type of feedback
        user_comment: Optional user comment explaining feedback
        correction_text: For corrections, the corrected text
        metadata: Additional context (provider, model, etc.)
        
    Returns:
        Dict with learning results
    """
    from app.memory.models import Message
    
    # Get the message
    message = db.query(Message).filter(Message.id == message_id).first()
    if not message:
        logger.warning(f"[learning] Message {message_id} not found")
        return {"status": "error", "reason": "message_not_found"}
    
    results = {
        "message_id": message_id,
        "feedback_type": feedback_type,
        "preferences_updated": [],
        "preferences_created": [],
    }
    
    # Extract learning signals based on feedback type
    if feedback_type == "positive":
        results.update(_learn_from_positive(db, message, metadata or {}))
    elif feedback_type == "negative":
        results.update(_learn_from_negative(db, message, user_comment, metadata or {}))
    elif feedback_type == "correction":
        results.update(_learn_from_correction(db, message, correction_text, metadata or {}))
    
    logger.info(f"[learning] Recorded {feedback_type} feedback for message {message_id}: {results}")
    return results


def _learn_from_positive(
    db: Session,
    message,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Learn from positive feedback (thumbs up)."""
    results = {"preferences_updated": [], "preferences_created": []}
    
    # 1. Reinforce provider/model preference if present
    provider = metadata.get("provider")
    model = metadata.get("model")
    
    if provider:
        pref = _get_or_create_preference(
            db,
            namespace="llm.routing",
            key=f"preferred_provider",
            default_value=provider,
        )
        if pref:
            append_preference_evidence(
                db, pref.id,
                signal_type=SignalType.APPROVAL,
                signal_value=provider,
                context={"message_id": message.id, "model": model},
            )
            recompute_preference_confidence(db, pref.id)
            results["preferences_updated"].append(f"llm.routing.preferred_provider")
    
    # 2. Learn response style preferences from message content
    style_signals = _extract_style_signals(message.content)
    for style_key, style_value in style_signals.items():
        pref = _get_or_create_preference(
            db,
            namespace="response.style",
            key=style_key,
            default_value=style_value,
        )
        if pref:
            append_preference_evidence(
                db, pref.id,
                signal_type=SignalType.IMPLICIT,
                signal_value=style_value,
                context={"message_id": message.id, "inferred_from": "positive_feedback"},
            )
            recompute_preference_confidence(db, pref.id)
            results["preferences_updated"].append(f"response.style.{style_key}")
    
    return results


def _learn_from_negative(
    db: Session,
    message,
    user_comment: Optional[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Learn from negative feedback (thumbs down)."""
    results = {"preferences_updated": [], "contradictions_recorded": []}
    
    provider = metadata.get("provider")
    model = metadata.get("model")
    
    # Record contradiction for provider if there's a preferred one
    if provider:
        pref = get_preference(db, "llm.routing", "preferred_provider")
        if pref and pref.value == provider:
            record_contradiction(
                db, pref.id,
                contradicting_value=f"negative_feedback_on_{provider}",
                context={"message_id": message.id, "comment": user_comment},
            )
            results["contradictions_recorded"].append("llm.routing.preferred_provider")
    
    # If user provided comment, try to extract preference hints
    if user_comment:
        hints = _extract_preference_hints(user_comment)
        for namespace, key, value in hints:
            pref = _get_or_create_preference(db, namespace, key, value)
            if pref:
                append_preference_evidence(
                    db, pref.id,
                    signal_type=SignalType.EXPLICIT,
                    signal_value=value,
                    context={"message_id": message.id, "from_negative_comment": user_comment},
                )
                recompute_preference_confidence(db, pref.id)
                results["preferences_updated"].append(f"{namespace}.{key}")
    
    return results


def _learn_from_correction(
    db: Session,
    message,
    correction_text: Optional[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Learn from user corrections."""
    results = {"preferences_updated": [], "corrections_analyzed": 0}
    
    if not correction_text:
        return results
    
    original = message.content or ""
    
    # Analyze what changed
    changes = _analyze_correction(original, correction_text)
    results["corrections_analyzed"] = len(changes)
    
    for change_type, change_detail in changes:
        # Map changes to preferences
        if change_type == "tone_shift":
            pref = _get_or_create_preference(
                db, "response.style", "tone", change_detail
            )
        elif change_type == "length_preference":
            pref = _get_or_create_preference(
                db, "response.style", "length", change_detail
            )
        elif change_type == "format_change":
            pref = _get_or_create_preference(
                db, "response.format", change_detail, "preferred"
            )
        else:
            pref = None
        
        if pref:
            append_preference_evidence(
                db, pref.id,
                signal_type=SignalType.CORRECTION,
                signal_value=change_detail,
                context={"message_id": message.id, "change_type": change_type},
            )
            recompute_preference_confidence(db, pref.id)
            results["preferences_updated"].append(f"{pref.namespace}.{pref.key}")
    
    return results


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_or_create_preference(
    db: Session,
    namespace: str,
    key: str,
    default_value: Any,
) -> Optional[PreferenceRecord]:
    """Get existing preference or create new one."""
    pref = get_preference(db, namespace, key)
    if pref:
        return pref
    
    try:
        pref = create_preference(
            db,
            namespace=namespace,
            key=key,
            value=default_value,
            strength=PreferenceStrength.SUGGESTED,
            source="learned",
        )
        return pref
    except Exception as e:
        logger.warning(f"[learning] Failed to create preference {namespace}.{key}: {e}")
        return None


def _extract_style_signals(content: str) -> Dict[str, str]:
    """Extract style signals from message content."""
    signals = {}
    
    if not content:
        return signals
    
    # Length preference
    word_count = len(content.split())
    if word_count < 50:
        signals["length"] = "concise"
    elif word_count > 300:
        signals["length"] = "detailed"
    
    # Format signals
    if "```" in content:
        signals["uses_code_blocks"] = "yes"
    if content.count("•") > 3 or content.count("-") > 5:
        signals["uses_bullets"] = "yes"
    if content.count("#") > 2:
        signals["uses_headers"] = "yes"
    
    return signals


def _extract_preference_hints(comment: str) -> List[tuple]:
    """Extract preference hints from user comment."""
    hints = []
    comment_lower = comment.lower()
    
    # Tone hints
    if "too formal" in comment_lower:
        hints.append(("response.style", "tone", "casual"))
    elif "too casual" in comment_lower:
        hints.append(("response.style", "tone", "formal"))
    
    # Length hints
    if "too long" in comment_lower or "too verbose" in comment_lower:
        hints.append(("response.style", "length", "concise"))
    elif "too short" in comment_lower or "more detail" in comment_lower:
        hints.append(("response.style", "length", "detailed"))
    
    # Format hints
    if "no bullets" in comment_lower or "no lists" in comment_lower:
        hints.append(("response.format", "bullets", "avoid"))
    if "use bullets" in comment_lower or "use lists" in comment_lower:
        hints.append(("response.format", "bullets", "preferred"))
    
    return hints


def _analyze_correction(original: str, correction: str) -> List[tuple]:
    """Analyze differences between original and correction."""
    changes = []
    
    orig_words = len(original.split())
    corr_words = len(correction.split())
    
    # Length change
    if corr_words < orig_words * 0.7:
        changes.append(("length_preference", "concise"))
    elif corr_words > orig_words * 1.3:
        changes.append(("length_preference", "detailed"))
    
    # Format changes
    orig_bullets = original.count("•") + original.count("- ")
    corr_bullets = correction.count("•") + correction.count("- ")
    
    if orig_bullets > 3 and corr_bullets < 2:
        changes.append(("format_change", "no_bullets"))
    elif orig_bullets < 2 and corr_bullets > 3:
        changes.append(("format_change", "use_bullets"))
    
    return changes


# =============================================================================
# BATCH LEARNING
# =============================================================================

def analyze_conversation_patterns(
    db: Session,
    project_id: Optional[int] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """
    Analyze conversation patterns to infer preferences.
    
    Looks for:
    - Repeated provider/model usage
    - Response length patterns
    - Format preferences
    """
    from app.memory.models import Message
    from sqlalchemy import func
    
    results = {
        "messages_analyzed": 0,
        "patterns_found": [],
        "preferences_suggested": [],
    }
    
    query = db.query(Message).filter(Message.role == "assistant")
    if project_id:
        query = query.filter(Message.project_id == project_id)
    
    messages = query.order_by(Message.id.desc()).limit(limit).all()
    results["messages_analyzed"] = len(messages)
    
    # Analyze provider distribution
    providers = {}
    for msg in messages:
        # Try to get provider from metadata if stored
        provider = getattr(msg, 'provider', None)
        if provider:
            providers[provider] = providers.get(provider, 0) + 1
    
    if providers:
        top_provider = max(providers, key=providers.get)
        if providers[top_provider] > len(messages) * 0.6:
            results["patterns_found"].append({
                "type": "provider_preference",
                "value": top_provider,
                "confidence": providers[top_provider] / len(messages),
            })
    
    # Analyze response lengths
    lengths = [len(m.content.split()) if m.content else 0 for m in messages]
    if lengths:
        avg_length = sum(lengths) / len(lengths)
        if avg_length < 100:
            results["patterns_found"].append({
                "type": "length_preference",
                "value": "concise",
                "avg_words": avg_length,
            })
        elif avg_length > 300:
            results["patterns_found"].append({
                "type": "length_preference",
                "value": "detailed",
                "avg_words": avg_length,
            })
    
    return results


__all__ = [
    "record_message_feedback",
    "analyze_conversation_patterns",
]
