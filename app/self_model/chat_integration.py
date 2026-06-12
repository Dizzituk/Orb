# FILE: app/self_model/chat_integration.py
# Purpose: Chat Integration for the Self-Model
# Called-by: app.self_model.hooks
# Depends-on: app.self_model.capability_map, app.self_model.journal, app.self_model.observer, app.self_model.suggestions (+1 more)
# Last-renovated: 2026-06-11
"""
Chat Integration for the Self-Model

Provides hooks that allow ASTRA to reference its self-model during
conversations. Detects when the user is asking about capabilities,
requesting what ASTRA knows about them, or when ASTRA should
surface a suggestion naturally.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from app.self_model.capability_map import get_capability_map
from app.self_model.journal import get_journal
from app.self_model.observer import get_observer
from app.self_model.suggestions import get_suggestion_engine
from app.self_model.user_model import get_user_model

logger = logging.getLogger(__name__)


def detect_self_model_intent(user_message: str) -> Optional[str]:
    """Detect if the user is asking about ASTRA's self-awareness."""
    msg = user_message.lower().strip()

    capability_patterns = [
        r"\bcan you\b.*\b(do|handle|manage|help with)\b",
        r"\bwhat can you\b",
        r"\bwhat are your capabilities\b",
        r"\bare you able to\b",
        r"\bdo you have\b.*\b(module|feature|ability)\b",
        r"\bwhat do you know how to do\b",
    ]
    for pattern in capability_patterns:
        if re.search(pattern, msg):
            return "capability_query"

    knowledge_patterns = [
        r"\bwhat do you know about me\b",
        r"\bwhat have you learned\b",
        r"\bwhat do you remember\b",
        r"\bwhat.*preferences\b.*\b(do you|have you)\b",
        r"\btell me what you know\b",
        r"\bwhat facts.*about me\b",
    ]
    for pattern in knowledge_patterns:
        if re.search(pattern, msg):
            return "user_knowledge_query"

    suggestion_patterns = [
        r"\bany suggestions\b",
        r"\bwhat.*could.*improve\b",
        r"\bwhat.*should.*work on\b",
        r"\bwhat have you noticed\b",
        r"\bany patterns\b",
        r"\bwhat.*needs.*fixing\b",
        r"\bhow.*getting better\b",
    ]
    for pattern in suggestion_patterns:
        if re.search(pattern, msg):
            return "suggestion_query"

    journal_patterns = [
        r"\bevolution journal\b",
        r"\bwhat.*learned.*recently\b",
        r"\bhow.*evolved\b",
        r"\bshow.*journal\b",
        r"\bwhat.*changed\b.*\b(recently|lately)\b",
    ]
    for pattern in journal_patterns:
        if re.search(pattern, msg):
            return "journal_query"

    return None


def handle_self_model_query(intent: str, user_message: str = "") -> Dict[str, Any]:
    """Handle a self-model query and return a response."""
    if intent == "capability_query":
        cap_map = get_capability_map()
        if user_message:
            result = cap_map.can_do(user_message)
            return {"type": "capability_check", "response": result["answer"], "detail": result}
        return {"type": "capability_overview", "response": cap_map.to_plain_language()}

    if intent == "user_knowledge_query":
        return {"type": "user_knowledge", "response": get_user_model().to_plain_language()}

    if intent == "suggestion_query":
        engine = get_suggestion_engine()
        pending = engine.get_pending()
        observer = get_observer()
        actionable = observer.get_actionable_patterns()

        if not pending and actionable:
            for pattern in actionable:
                suggestion = engine.from_pattern(pattern)
                if suggestion:
                    observer.mark_suggested(pattern.pattern_id)
                    journal = get_journal()
                    journal.record_suggestion(suggestion.suggestion_id, suggestion.title, suggestion.description)

        return {"type": "suggestions", "response": engine.to_plain_language()}

    if intent == "journal_query":
        return {"type": "journal", "response": get_journal().to_plain_language()}

    return {"type": "unknown", "response": "I'm not sure what you're asking about my self-model."}


def get_session_greeting_context() -> Optional[str]:
    """Get context for a session greeting if there are pending suggestions."""
    engine = get_suggestion_engine()
    pending = engine.get_pending()
    if not pending:
        return None

    count = len(pending)
    if count == 1:
        return f"I have 1 observation from recent sessions. Want to hear it?"
    return f"I have {count} observations from recent sessions. Want to hear them?"
