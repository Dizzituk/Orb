# FILE: app/llm/routing/codebase_context_bridge.py
# Purpose: Guarded sync bridge to app.rag.answerer.build_codebase_context —
#          RAG-as-context so the chat model answers codebase questions in
#          Astra's voice (Lane A side of the A<->C contract, 2026-07-01).
# Called-by: app.llm.routing.chat_routing
# Depends-on: app.rag.answerer (optional at runtime; fully guarded)
"""
RAG-as-context bridge (Lane A, 2026-07-01).

Contract (written into both Lane A and Lane C briefs verbatim):

    # app/rag/answerer.py  (Lane C)
    async def build_codebase_context(question: str, db) -> str

    Returns a grounded context block (files/findings, <=3000 chars) for
    prompt injection, or "" on no results/any error. Never raises.

This module is the guarded caller. It works whether or not Lane C has
merged: any ImportError / AttributeError / Exception / timeout resolves
to "" and the caller falls back to the legacy divert (or plain chat).
Raw RAG output must never again be the user-facing reply — when this
bridge returns a block, the NORMAL chat model answers with it injected
into its prompt.

Sync/async: chat_routing handlers are sync but run inside FastAPI's
event loop, so the async contract function runs in a DAEMON worker
thread with its own event loop via asyncio.run, bounded by
RAG_CONTEXT_TIMEOUT_S. A daemon thread (not ThreadPoolExecutor's
context manager, whose __exit__ joins and would block the event loop
until the build finished anyway) is abandoned on timeout: the request
falls back immediately; the stray build finishes in the background and
its result is discarded. The db session it holds is used read-only by
the Lane C search (build_context_block is deterministic assembly).

Env knobs:
    RAG_AS_CONTEXT_ENABLED   default 1 — kill-switch back to legacy divert
    RAG_CONTEXT_TIMEOUT_S    default 20 — hard cap on the context build
    RAG_CONTEXT_MAX_CHARS    default 3000 — defensive clamp on the block
"""

from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    return os.getenv("RAG_AS_CONTEXT_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _timeout_s() -> float:
    try:
        return float(os.getenv("RAG_CONTEXT_TIMEOUT_S", "20"))
    except ValueError:
        return 20.0


def _max_chars() -> int:
    try:
        return int(os.getenv("RAG_CONTEXT_MAX_CHARS", "3000"))
    except ValueError:
        return 3000


def try_build_codebase_context(question: str, db) -> str:
    """Return a grounded codebase-context block for prompt injection, or "".

    Never raises. "" means: Lane C not merged yet, disabled, no results,
    timeout, or any error — the caller must fall back to legacy behaviour.
    """
    if not _enabled():
        return ""
    if not question or not question.strip():
        return ""

    try:
        from app.rag.answerer import build_codebase_context
    except (ImportError, AttributeError) as e:
        # Lane C not merged yet (or module reshaped) — graceful no-op.
        logger.debug("[rag_as_context] contract fn unavailable: %s", e)
        return ""

    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — safe to run directly.
            block = asyncio.run(build_codebase_context(question, db))
        else:
            # Inside FastAPI's loop from a sync handler: run in a daemon
            # thread with its own loop. join(timeout) enforces the cap for
            # real — on timeout the thread is abandoned, never joined.
            import threading
            result_box: dict = {}

            def _runner():
                try:
                    result_box["block"] = asyncio.run(
                        build_codebase_context(question, db)
                    )
                except Exception as e:
                    logger.warning("[rag_as_context] build failed in worker: %s", e)
                    result_box["block"] = ""

            worker = threading.Thread(
                target=_runner, daemon=True, name="rag-as-context",
            )
            worker.start()
            worker.join(timeout=_timeout_s())
            if worker.is_alive():
                logger.warning(
                    "[rag_as_context] build_codebase_context timed out "
                    "after %.0fs — falling back", _timeout_s(),
                )
                return ""
            block = result_box.get("block", "")
    except Exception as e:
        logger.warning("[rag_as_context] build_codebase_context failed: %s", e)
        return ""

    if not block or not isinstance(block, str):
        return ""
    block = block.strip()
    if not block:
        return ""
    cap = _max_chars()
    if len(block) > cap:
        block = block[:cap]
    print(f"[RAG_AS_CONTEXT] Injecting grounded codebase context ({len(block)} chars)")
    return block


__all__ = ["try_build_codebase_context"]
