# FILE: app/debug/orchestrator/router_classifier.py
"""
Classifier that routes an incoming debug chat message to either:
  - "chat"       -> the existing debug chat tool loop
  - "orchestrate" -> the parallel investigate/plan/execute pipeline

Uses a small, cheap reasoning call (gpt-5.4-nano default) that returns a
single JSON verdict. Falls back to deterministic heuristics if the LLM is
unavailable so the UI still works.

v1.0 (2026-04-13): initial implementation for chat-box auto-routing.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, Literal, Optional

logger = logging.getLogger(__name__)

Decision = Literal["chat", "orchestrate"]

_CLASSIFIER_MODEL_ENV = "DEBUG_ROUTER_CLASSIFIER_MODEL"
_DEFAULT_CLASSIFIER_MODEL = "gpt-5.4-nano"
_CLASSIFIER_TIMEOUT = 15.0  # seconds — keep it snappy

# --- Heuristic fallback signals -------------------------------------------

# Strong verbs that almost always mean "do it now, fully agentic"
_GO_VERBS = [
    r"\bgo\s+and\s+do\s+it\b",
    r"\boff\s+you\s+go\b",
    r"\bon\s+you\s+go\b",
    r"\bcrack\s+on\b",
    r"\bget\s+on\s+with\s+it\b",
    r"\bmake\s+it\s+so\b",
    r"\bfix\s+(it|them|these|those|this)\b",
    r"\bimplement\s+(it|them|the\s+(plan|fixes|changes))\b",
    r"\bexecute\s+(the\s+)?(plan|list|fixes|tasks|jobs)\b",
    r"\bdebug\s+(this|these|the\s+(bugs|issues|list))\b",
    r"\bwork\s+through\s+(the\s+)?(list|bugs|issues)\b",
    r"\bsort\s+(it|them|this)\s+out\b",
    r"\btake\s+care\s+of\s+(it|them|these)\b",
    r"\bgo\s+to\s+town\s+on\b",
    r"\brun\s+(the\s+)?orchestration\b",
    r"\borchestrate\b",
    r"\brun\s+(it|them|the\s+(plan|list))\b",
    r"\bgo\s+ahead\s+and\s+(fix|run|implement|execute|do)\b",
    r"\bproceed\s+with\s+(the\s+)?(plan|fix|implementation)\b",
]

# Signals the message is *informational* — user wants analysis or chat
_CHAT_SIGNALS = [
    r"\bwhat\s+do\s+you\s+think\b",
    r"\bcan\s+you\s+explain\b",
    r"\bcould\s+you\s+explain\b",
    r"\btell\s+me\s+about\b",
    r"\bshow\s+me\b",
    r"^\s*(hi|hey|hello|morning|afternoon|evening)\b",
    r"\?\s*$",  # ends in a question mark
]

_GO_RX = [re.compile(p, re.IGNORECASE) for p in _GO_VERBS]
_CHAT_RX = [re.compile(p, re.IGNORECASE) for p in _CHAT_SIGNALS]


def _heuristic_decision(message: str) -> Decision:
    """Deterministic fallback classifier."""
    msg = message.strip()
    if not msg:
        return "chat"

    # Very short messages (< 40 chars) are almost always chat
    if len(msg) < 40 and not any(rx.search(msg) for rx in _GO_RX):
        return "chat"

    # Strong action verbs win over everything else
    if any(rx.search(msg) for rx in _GO_RX):
        return "orchestrate"

    # Questions and casual greetings stay in chat
    if any(rx.search(msg) for rx in _CHAT_RX):
        return "chat"

    # Large bug-list shaped payloads (many list items or numbered entries)
    bullet_lines = len(re.findall(r"^\s*(?:[-*•]|\d+\.)\s+\S", msg, re.MULTILINE))
    if bullet_lines >= 4:
        return "orchestrate"

    return "chat"


# --- LLM classifier --------------------------------------------------------

_CLASSIFIER_SYSTEM = (
    "You are a strict router for the ASTRA Debug assistant. You decide whether "
    "an incoming user message should be handled by:\n"
    "  - \"chat\"       : normal conversational/investigative reply\n"
    "  - \"orchestrate\": kick off the full parallel debug pipeline "
    "(decompose -> investigate -> plan -> execute -> verify)\n"
    "\n"
    "Return \"orchestrate\" ONLY when the user is clearly giving a go-ahead "
    "to actually DO the work — apply fixes, run the plan, work through a bug "
    "list, sort it out, execute, etc.\n"
    "\n"
    "Return \"chat\" for questions, analysis, clarifications, greetings, "
    "brainstorming, or any message that is primarily informational.\n"
    "\n"
    "When in doubt, return \"chat\" — orchestration writes code and must not "
    "be triggered accidentally.\n"
    "\n"
    "Respond with a single JSON object: {\"decision\": \"chat\"|\"orchestrate\", "
    "\"reason\": \"short explanation\", \"confidence\": 0.0-1.0}."
)


async def _call_classifier_llm(message: str, model: str) -> Optional[Dict]:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    client = AsyncOpenAI(api_key=api_key, timeout=_CLASSIFIER_TIMEOUT)
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CLASSIFIER_SYSTEM},
                {"role": "user", "content": f"USER MESSAGE:\n{message[:4000]}"},
            ],
            max_completion_tokens=200,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.info("[router_classifier] LLM call failed: %s", e)
        return None

    raw = resp.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except Exception:
        return None


async def classify_message(
    message: str,
    history_has_plan: bool = False,
    model: Optional[str] = None,
) -> Dict:
    """Return a routing decision for an incoming chat message.

    Args:
        message: The user's latest message.
        history_has_plan: True if the assistant's prior turn contained a
            written plan/task-list — raises the prior on "orchestrate" for
            short affirmative replies like "go", "off you go", "do it".
        model: Classifier model override; defaults to env or nano.

    Returns:
        {"decision": "chat"|"orchestrate", "reason": str, "confidence": float,
         "source": "llm"|"heuristic"}
    """
    msg = (message or "").strip()
    if not msg:
        return {"decision": "chat", "reason": "empty message", "confidence": 1.0, "source": "heuristic"}

    # Short affirmative replies RIGHT AFTER a plan message => orchestrate
    if history_has_plan:
        short = msg.lower()
        if len(short) <= 30 and any(
            phrase in short
            for phrase in (
                "go", "yes", "yep", "do it", "run it", "off you go",
                "crack on", "go ahead", "execute", "sort it", "make it so",
                "ok go", "proceed",
            )
        ):
            return {
                "decision": "orchestrate",
                "reason": "short affirmative after a plan was surfaced",
                "confidence": 0.9,
                "source": "heuristic",
            }

    classifier_model = model or os.getenv(_CLASSIFIER_MODEL_ENV) or _DEFAULT_CLASSIFIER_MODEL
    raw = await _call_classifier_llm(msg, classifier_model)

    if raw and isinstance(raw, dict) and raw.get("decision") in ("chat", "orchestrate"):
        return {
            "decision": raw["decision"],
            "reason": str(raw.get("reason", ""))[:300],
            "confidence": float(raw.get("confidence", 0.5) or 0.5),
            "source": "llm",
        }

    # Heuristic fallback
    decision = _heuristic_decision(msg)
    return {
        "decision": decision,
        "reason": "heuristic fallback (LLM classifier unavailable)",
        "confidence": 0.5,
        "source": "heuristic",
    }
