# FILE: app/translation/recent_decisions.py
# Purpose: Recent classifier decisions store.
# Called-by: app.llm.routing.prompt_builders, app.llm.routing.reject_intent_handler, app.llm.stream_router
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Recent classifier decisions store.

Records the last few classifier decisions per conversation so the chat
LLM can answer "why did you think that was a web search?" accurately
instead of confabulating an explanation.

The chat LLM does not see the translation layer's reasoning. It only
sees the user's messages and the gate UI events. Without this store,
when the user asks why a classification happened, the chat LLM has no
ground truth and tends to invent a plausible-sounding cause. This
module gives prompt_builders a small, fresh block of classifier
metadata to inject into the system prompt.

Storage is process-local and in-memory:
- Keyed by conversation_id (str)
- Each entry is a list of the last N decisions (default 3)
- Decisions older than the TTL (default 1 hour) are pruned on read
- No persistence — restart wipes the store, which is fine because
  older context is rarely relevant by then anyway

v1.0 (2026-05-01): Initial implementation.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Default cap and TTL — see module docstring for rationale.
_MAX_PER_CONVERSATION = 3
_TTL_SECONDS = 60 * 60  # 1 hour

# Decision status values
STATUS_PENDING = "pending"      # gate fired, awaiting user response
STATUS_CONFIRMED = "confirmed"  # user confirmed, intent executed
STATUS_REJECTED = "rejected"    # user rejected
STATUS_AUTO = "auto_executed"   # high-confidence, no gate, ran directly


@dataclass
class ClassifierDecision:
    """A single classifier decision recorded for chat-LLM context."""
    intent: str
    rule_name: str
    reason: str
    message_excerpt: str
    confidence: float
    status: str = STATUS_PENDING
    recorded_at: float = field(default_factory=time.time)


# Process-local store, guarded by a lock since async handlers may
# write concurrently from different requests.
_store: Dict[str, List[ClassifierDecision]] = {}
_lock = threading.Lock()


def _prune_expired(decisions: List[ClassifierDecision]) -> List[ClassifierDecision]:
    """Drop decisions older than the TTL."""
    cutoff = time.time() - _TTL_SECONDS
    return [d for d in decisions if d.recorded_at >= cutoff]


def record_decision(
    conversation_id: Optional[str],
    intent: str,
    rule_name: str,
    reason: str,
    message_excerpt: str,
    confidence: float,
    status: str = STATUS_PENDING,
) -> None:
    """Record a classifier decision for later context injection.

    Idempotent on identical message_excerpt — if the most recent decision
    has the same excerpt, its fields are updated rather than a new entry
    being added. This prevents double-recording when the same message
    is re-evaluated.
    """
    if not conversation_id:
        return

    key = str(conversation_id)
    excerpt = (message_excerpt or "")[:200]

    with _lock:
        decisions = _store.setdefault(key, [])
        decisions = _prune_expired(decisions)

        # Idempotent: update the most-recent entry if it's the same message.
        if decisions and decisions[-1].message_excerpt == excerpt:
            d = decisions[-1]
            d.intent = intent
            d.rule_name = rule_name
            d.reason = reason
            d.confidence = confidence
            d.status = status
            d.recorded_at = time.time()
        else:
            decisions.append(ClassifierDecision(
                intent=intent,
                rule_name=rule_name,
                reason=reason,
                message_excerpt=excerpt,
                confidence=confidence,
                status=status,
            ))

        # Cap to most recent N
        if len(decisions) > _MAX_PER_CONVERSATION:
            decisions = decisions[-_MAX_PER_CONVERSATION:]

        _store[key] = decisions


def mark_status(
    conversation_id: Optional[str],
    status: str,
    message_excerpt: Optional[str] = None,
) -> None:
    """Update the status of an existing decision.

    If `message_excerpt` is given, finds the matching decision and updates
    it. Otherwise updates the most recent decision for that conversation.
    """
    if not conversation_id:
        return

    key = str(conversation_id)
    with _lock:
        decisions = _store.get(key)
        if not decisions:
            return

        if message_excerpt:
            target_excerpt = message_excerpt[:200]
            for d in reversed(decisions):
                if d.message_excerpt == target_excerpt:
                    d.status = status
                    return
        else:
            decisions[-1].status = status


def get_recent_decisions(
    conversation_id: Optional[str],
    limit: int = _MAX_PER_CONVERSATION,
) -> List[ClassifierDecision]:
    """Return the recent classifier decisions for a conversation.

    Auto-prunes expired entries on read.
    """
    if not conversation_id:
        return []

    key = str(conversation_id)
    with _lock:
        decisions = _store.get(key, [])
        decisions = _prune_expired(decisions)
        _store[key] = decisions
        return list(decisions[-limit:])


def build_decisions_block(conversation_id: Optional[str]) -> str:
    """Build a [CLASSIFIER DECISIONS] block for system prompt injection.

    Returns an empty string if there are no recent decisions, so the
    caller can append unconditionally without a None-check.
    """
    decisions = get_recent_decisions(conversation_id)
    if not decisions:
        return ""

    lines = [
        "[CLASSIFIER DECISIONS — recent translation-layer routing]",
        "These are the last classifier decisions made for this conversation.",
        "If the user asks WHY a message was classified a certain way (e.g.",
        "'why did you think that was a web search?'), refer to these. Do NOT",
        "invent an explanation. If a decision was rejected by the user, the",
        "classifier was wrong about that intent — own that, don't justify it.",
        "",
    ]

    now = time.time()
    for d in decisions:
        age_sec = max(0, int(now - d.recorded_at))
        if age_sec < 60:
            age = f"{age_sec}s ago"
        elif age_sec < 3600:
            age = f"{age_sec // 60}m ago"
        else:
            age = f"{age_sec // 3600}h ago"

        lines.append(
            f"- [{age}] {d.intent} via {d.rule_name} "
            f"(conf={d.confidence:.2f}, status={d.status})"
        )
        if d.reason:
            lines.append(f"    reason: {d.reason}")
        if d.message_excerpt:
            excerpt = d.message_excerpt
            if len(excerpt) > 100:
                excerpt = excerpt[:100] + "..."
            lines.append(f"    message: \"{excerpt}\"")

    lines.append("[/CLASSIFIER DECISIONS]")
    return "\n".join(lines)


def clear_conversation(conversation_id: Optional[str]) -> None:
    """Clear all decisions for a conversation. Mainly for tests / resets."""
    if not conversation_id:
        return
    with _lock:
        _store.pop(str(conversation_id), None)


__all__ = [
    "ClassifierDecision",
    "STATUS_PENDING",
    "STATUS_CONFIRMED",
    "STATUS_REJECTED",
    "STATUS_AUTO",
    "record_decision",
    "mark_status",
    "get_recent_decisions",
    "build_decisions_block",
    "clear_conversation",
]
