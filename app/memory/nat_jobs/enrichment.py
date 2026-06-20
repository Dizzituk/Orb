# FILE: app/memory/nat_jobs/enrichment.py
# Purpose: Job 2 — pre-reply semantic enrichment. Nat SUGGESTS retrieval queries;
#          the existing DETERMINISTIC retriever fetches the data (Nat never
#          touches the data, so it can't fabricate memories). Result becomes an
#          [ENRICHMENT] block. Strictly time-boxed + circuit-breakered so it can
#          NEVER add felt latency or block a reply.
# Called-by: app.llm.routing.prompt_builders.build_full_context, app.endpoints.chat
# Depends-on: app.llm.workers.nat_worker.run_nat_job (suggest),
#             app.astra_memory.retrieval.retrieve_for_query (fetch)
# Last-renovated: 2026-06-19
"""
Enrichment (pre-reply, latency-critical).

The deterministic depth classifier is keyword/regex and stays shallow unless it
sees a trigger word, so "I'm a bit worried about my calories today" never
surfaces "you hit your target on this day last month." Nat adds the semantic
read the matcher misses — but ONLY by proposing queries; the deterministic
store does the actual fetch.

Hard rules (spec §4):
  - non-blocking-with-timeout: NAT_ENRICHMENT_TIMEOUT_MS (default 400ms). If Nat
    isn't back by then we proceed with the deterministic block alone.
  - observable: one log line per turn — `[NAT] enrichment: <ms>ms, queries=<n>,
    timed_out=<bool>` — so added latency is visible, never hidden.
  - circuit breaker: after a timeout/error we skip Nat for a short cooldown, so a
    slow/struggling Nat doesn't tax every turn.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import List

logger = logging.getLogger(__name__)

# Monotonic time of the last Nat enrichment failure (timeout/error). Module-level
# so the breaker is process-wide. [0.0] = never failed.
_last_fail: List[float] = [0.0]


def enrichment_enabled() -> bool:
    return os.getenv("NAT_ENRICHMENT_ENABLED", "1").strip() == "1"


def _timeout_s() -> float:
    try:
        return max(0.05, int(os.getenv("NAT_ENRICHMENT_TIMEOUT_MS", "400")) / 1000.0)
    except Exception:
        return 0.4


def _cooldown_s() -> float:
    try:
        return max(0.0, float(os.getenv("NAT_ENRICHMENT_COOLDOWN_S", "30")))
    except Exception:
        return 30.0


async def build_enrichment_block(db, project_id, user_message: str) -> str:
    """Return an [ENRICHMENT] block, or "" (always, on any timeout/error)."""
    if not enrichment_enabled() or not project_id or not (user_message or "").strip():
        return ""

    # Circuit breaker — skip Nat entirely during the cooldown after a failure.
    try:
        if _last_fail[0] and (time.monotonic() - _last_fail[0]) < _cooldown_s():
            return ""
    except Exception:
        pass

    t0 = time.monotonic()
    timed_out = False
    queries: List[str] = []
    try:
        queries = await asyncio.wait_for(_suggest_queries(user_message), timeout=_timeout_s())
    except asyncio.TimeoutError:
        timed_out = True
        _last_fail[0] = time.monotonic()
    except Exception as exc:  # noqa: BLE001
        _last_fail[0] = time.monotonic()
        logger.debug("[enrichment] suggest failed: %s", exc)

    block = ""
    if queries:
        try:
            block = _fetch_and_format(db, queries)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[enrichment] fetch failed: %s", exc)

    ms = int((time.monotonic() - t0) * 1000)
    # Required observability line — the user watches this to confirm zero added latency.
    logger.info("[NAT] enrichment: %dms, queries=%d, timed_out=%s", ms, len(queries), timed_out)
    return block


def build_enrichment_block_sync(project_id, message: str) -> str:
    """Sync entry for SYNC context builders (build_full_context runs in a sync
    request path). Runs the async enrichment on a worker thread with its own
    event loop + a FRESH db session, joined with the budget + a small retrieval
    margin. Returns "" on timeout/error — never raises, never over-waits.

    The request thread waits at most the budget; matches the existing behaviour
    where build_full_context already blocks on sync retrieval/embeddings.
    """
    if not enrichment_enabled() or not project_id or not (message or "").strip():
        return ""
    holder = {"block": ""}

    def _worker():
        try:
            from app.db import get_db_session
            db2 = get_db_session()
            try:
                holder["block"] = asyncio.run(
                    build_enrichment_block(db2, project_id, message)
                ) or ""
            finally:
                try:
                    db2.close()
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("[enrichment] sync worker failed: %s", exc)

    t = threading.Thread(target=_worker, name="nat-enrichment", daemon=True)
    t.start()
    join_budget = _timeout_s() + float(os.getenv("NAT_ENRICHMENT_FETCH_MARGIN_S", "0.7"))
    t.join(timeout=join_budget)
    return holder["block"]


async def _suggest_queries(user_message: str) -> List[str]:
    """Ask Nat ONLY for query strings (never data). Returns <=5 cleaned queries."""
    from app.llm.workers.nat_worker import run_nat_job, coerce_json
    system = (
        "Given this message, identify what STORED user information would enrich "
        "the reply (past logs, dates, metrics, preferences, facts). Output a JSON "
        "list of short search queries to look up, or [] if nothing stored would "
        "help. Reply ONLY with the JSON list of strings."
    )
    reply = await run_nat_job(
        system, (user_message or "")[:1500],
        enable_thinking=False,
        max_tokens=int(os.getenv("NAT_JOB_MAX_TOKENS", "150")),
        timeout_seconds=10,
    )
    qs = coerce_json(reply) if isinstance(reply, str) else None
    out: List[str] = []
    seen = set()
    if isinstance(qs, list):
        for q in qs:
            if not isinstance(q, str):
                continue
            s = q.strip()
            if not s or len(s) > 80 or s.lower() in seen:
                continue
            seen.add(s.lower())
            out.append(s)
            if len(out) >= 5:
                break
    return out


def _fetch_and_format(db, queries: List[str]) -> str:
    """Run Nat's queries through the DETERMINISTIC retriever and format results.

    Synchronous, like the existing deterministic retrieval in the front door —
    Nat never sees or generates the data, only suggested what to look up.
    """
    from app.astra_memory.retrieval import retrieve_for_query
    lines: List[str] = []
    seen = set()
    for q in queries[:5]:
        try:
            res = retrieve_for_query(db, q)
        except Exception:
            continue
        for rec in (getattr(res, "records", None) or [])[:2]:
            title = (getattr(rec, "title", "") or "").strip()
            content = (getattr(rec, "content", "") or "").strip().replace("\n", " ")
            if not content:
                continue
            line = f"- {title + ': ' if title else ''}{content[:300]}"
            key = line[:80].lower()
            if key in seen:
                continue
            seen.add(key)
            lines.append(line)
        if len(lines) >= 8:
            break
    if not lines:
        return ""
    return (
        "[ENRICHMENT]\n"
        "Relevant stored context (Nat flagged these as worth looking up):\n"
        + "\n".join(lines[:8])
    )
