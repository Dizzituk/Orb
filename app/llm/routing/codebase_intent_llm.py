# FILE: app/llm/routing/codebase_intent_llm.py
# Purpose: LLM safety-net — decide if a chat message is a question about ASTRA's own code/features.
# Called-by: app.llm.routing.chat_routing
# Depends-on: app.translation.tier1_classifier (lightweight client)
# Last-renovated: 2026-06-14
"""
LLM safety-net for codebase/self questions (2026-06-14).

Tier-0 regex + guard keywords route obvious codebase questions to RAG, but miss
natural phrasings whose feature/UI names aren't in the guard list (e.g. "what
data does the chat window have", "how does the finance panel work"). Broadening
the regex risks false positives, so instead — ONLY for messages that already
LOOK like a feature question but lacked a guard keyword (see
rag_fallback.needs_llm_codebase_check, which is deliberately narrow / "the
<feature>"-anchored to keep call volume low) — we ask a lightweight model a
single yes/no:

    "Is this about ASTRA's OWN source code / architecture / data / app features?"

Execution safety (hardened after adversarial review, 2026-06-14):
  - A MODULE-LEVEL executor runs the async call off the event-loop thread; we
    never create/tear down a ThreadPoolExecutor per call (a per-call `with`
    pool.shutdown(wait=True) on the event-loop thread could block all requests).
  - A short timeout is FORWARDED to the underlying LLM call, so a hung worker
    self-terminates fast instead of riding the registry's 180s default.
  - On timeout we fail-closed and abandon the worker without waiting.
Fail-closed: any error, timeout, missing key, or non-"yes" answer returns False
(→ normal chat). Never raises.
"""
from __future__ import annotations

import concurrent.futures
import logging
import os

logger = logging.getLogger(__name__)

# Lightweight model only — intent routing must never need a frontier model.
_CLASSIFIER_MODEL = os.getenv("ORB_CODEBASE_INTENT_MODEL", "gpt-5-mini")
# Hard per-call cap (forwarded to the LLM call AND used as the wait bound).
_LLM_CALL_TIMEOUT = float(os.getenv("ORB_CODEBASE_INTENT_TIMEOUT", "5"))

# Module-level executor: created once, reused. No per-call shutdown, so a
# timed-out classification never blocks the caller (the event-loop thread).
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="codebase_intent"
)

_SYSTEM_PROMPT = (
    "You decide whether a user's message is a question about the ASTRA "
    "application ITSELF — its own source code, internal architecture, modules, "
    "data model/tables, or its app features and UI panels (e.g. the chat window, "
    "finance panel, sidebar, build pipeline, memory system).\n"
    "Answer 'yes' ONLY for questions about ASTRA's own system / code / data / features.\n"
    "Answer 'no' for general knowledge, world facts, personal or life questions, "
    "small talk, or requests to perform a task.\n"
    "Reply with exactly one word: yes or no."
)


async def _aclassify(message: str) -> str:
    """Run the one-word classification via the lightweight client, time-bounded."""
    import asyncio

    from app.translation.tier1_classifier import get_lightweight_client

    client = get_lightweight_client()
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f'Message: "{message}"'},
    ]
    # asyncio.wait_for is a backstop; the real bound is timeout_seconds forwarded
    # into the LLM call so the worker dies fast even if cancellation is ignored.
    return await asyncio.wait_for(
        client.call_async(
            model=_CLASSIFIER_MODEL,
            messages=messages,
            max_tokens=3,
            temperature=0.0,
            timeout_seconds=int(_LLM_CALL_TIMEOUT),
        ),
        timeout=_LLM_CALL_TIMEOUT,
    )


def is_codebase_question_llm(message: str) -> bool:
    """Lightweight yes/no: is this about ASTRA's own codebase/features? Fail-closed."""
    text = (message or "").strip()
    if not text:
        return False
    import asyncio

    try:
        # Build the coroutine INSIDE the worker (lambda) so a submit failure
        # never leaves an un-awaited coroutine.
        fut = _EXECUTOR.submit(lambda: asyncio.run(_aclassify(text)))
        answer = fut.result(timeout=_LLM_CALL_TIMEOUT + 1)
        verdict = (answer or "").strip().lower()
        is_yes = verdict.startswith("yes")
        logger.info(
            "[codebase_intent_llm] %r -> %s (%r)", text[:60], is_yes, verdict[:20]
        )
        return is_yes
    except concurrent.futures.TimeoutError:
        # Abandon the worker (do NOT wait) — it self-terminates via the forwarded
        # LLM timeout. The module-level executor means no shutdown(wait=True) here.
        logger.warning("[codebase_intent_llm] timed out (fail-closed): %r", text[:60])
        return False
    except Exception as e:  # noqa: BLE001 — fail-closed by design
        logger.warning(
            "[codebase_intent_llm] classification failed (fail-closed): %s", e
        )
        return False


async def is_codebase_question_llm_async(message: str) -> bool:
    """Async variant for callers already on the event loop (e.g. the bridge).

    Awaits the (time-bounded) classification directly — no thread executor — so
    it never blocks the loop. Fail-closed: returns False on any error/timeout.
    """
    text = (message or "").strip()
    if not text:
        return False
    try:
        answer = await _aclassify(text)
        verdict = (answer or "").strip().lower()
        is_yes = verdict.startswith("yes")
        logger.info(
            "[codebase_intent_llm] (async) %r -> %s (%r)", text[:60], is_yes, verdict[:20]
        )
        return is_yes
    except Exception as e:  # noqa: BLE001 — fail-closed by design
        logger.warning(
            "[codebase_intent_llm] (async) classification failed (fail-closed): %s", e
        )
        return False


__all__ = ["is_codebase_question_llm", "is_codebase_question_llm_async"]
