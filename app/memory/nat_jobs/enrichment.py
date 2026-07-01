# FILE: app/memory/nat_jobs/enrichment.py
# Purpose: Job 2 - pre-reply semantic enrichment. Nat SUGGESTS retrieval queries;
#          the existing DETERMINISTIC retriever fetches the data (Nat never
#          touches the data, so it can't fabricate memories). Result becomes an
#          [ENRICHMENT] block. Strictly time-boxed + circuit-breakered so it can
#          NEVER add felt latency or block a reply. The turn's query embeddings
#          are batched into ONE call + cached, and the block is built best-effort:
#          whatever relevant hits are gathered by the deadline are injected
#          (never all-or-nothing).
# Called-by: app.llm.routing.prompt_builders.build_full_context
# Depends-on: app.llm.workers.nat_worker.run_nat_job (suggest),
#             app.astra_memory.retrieval.retrieve_for_query (fetch),
#             app.astra_memory.semantic_candidates.prewarm_query_embeddings (batch+cache)
# Last-renovated: 2026-06-22
"""
Enrichment (pre-reply, latency-critical).

The deterministic depth classifier is keyword/regex and stays shallow unless it
sees a trigger word, so "I'm a bit worried about my calories today" never
surfaces "you hit your target on this day last month." Nat adds the semantic
read the matcher misses - but ONLY by proposing queries; the deterministic
store does the actual fetch.

Hard rules (spec section 4):
  - non-blocking-with-timeout: NAT_ENRICHMENT_TIMEOUT_MS (default 400ms). If Nat
    isn't back by then we proceed with the deterministic block alone.
  - observable: one log line per turn - `[NAT] enrichment: <ms>ms, queries=<n>,
    cached=<n>, timed_out=<bool>` - so added latency is visible, never hidden.
  - circuit breaker: after a timeout/error we skip Nat for a short cooldown, so a
    slow/struggling Nat doesn't tax every turn.

Latency design (2026-06-22): the fetch's real cost was N serial remote embedding
round-trips (one per suggested query). Three changes keep it inside the budget
WITHOUT extending the budget:
  - BATCH: prewarm_query_embeddings() embeds the whole turn's queries in ONE call
    and caches them, so the per-query retrievals below hit the cache.
  - CAP: NAT_ENRICHMENT_MAX_QUERIES (default 3) bounds how many queries run -
    fewer, sharper lookups rather than five mushy ones.
  - BEST-EFFORT PARTIAL: lines are accumulated into a shared sink as each query
    resolves (highest-relevance query first), so a deadline-bounded caller
    injects whatever GOOD hits it has by the deadline instead of discarding the
    lot. The existing relevance/confidence filtering still applies, so partial
    means "fewer good lines," never "filler."
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


def _max_queries() -> int:
    """How many suggested queries to actually look up. With embeddings batched
    into one call, more queries no longer cost more round-trips, so this is a
    precision/cost knob, not a latency one - keep it small and sharp."""
    try:
        return max(1, int(os.getenv("NAT_ENRICHMENT_MAX_QUERIES", "3")))
    except Exception:
        return 3


async def build_enrichment_block(db, project_id, user_message: str, sink=None) -> str:
    """Return an [ENRICHMENT] block, or "" (always, on any timeout/error).

    If `sink` is provided - a {"lines": [...], "lock": threading.Lock()}
    accumulator - each relevant line is appended to it the moment it is found, so
    a caller enforcing a hard wall-clock deadline can read whatever was gathered
    so far (best-effort partial) instead of all-or-nothing.
    """
    if not enrichment_enabled() or not project_id or not (user_message or "").strip():
        return ""

    # Circuit breaker - skip Nat entirely during the cooldown after a failure.
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
    cached = 0
    if queries:
        # Batch-embed the whole turn's queries in ONE remote call (+cache) so the
        # per-query fetches below reuse the vectors instead of N serial round-trips.
        try:
            from app.astra_memory.semantic_candidates import prewarm_query_embeddings
            stats = prewarm_query_embeddings(queries)
            cached = int(stats.get("cached", 0)) if isinstance(stats, dict) else 0
        except Exception as exc:  # noqa: BLE001 - pure optimisation, never fatal
            logger.debug("[enrichment] prewarm skipped: %s", exc)
        try:
            lines = _collect_lines(db, queries, sink=sink)
            block = _format_block(lines)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[enrichment] fetch failed: %s", exc)

    ms = int((time.monotonic() - t0) * 1000)
    # Required observability line - the user watches this to confirm zero added latency.
    logger.info("[NAT] enrichment: %dms, queries=%d, cached=%d, timed_out=%s",
                ms, len(queries), cached, timed_out)
    return block


def build_enrichment_block_sync(project_id, message: str) -> str:
    """Sync entry for SYNC context builders (build_full_context runs in a sync
    request path). Runs the async enrichment on a worker thread with its own
    event loop + a FRESH db session, joined with the budget + a small retrieval
    margin. Returns "" on no-result - never raises, never over-waits.

    Best-effort partial: the worker streams relevant lines into a shared sink as
    it finds them, so at the deadline we return whatever GOOD lines were gathered
    so far rather than throwing the whole turn's work away. The request thread
    still waits at most the budget; matches the existing behaviour where
    build_full_context already blocks on sync retrieval/embeddings.
    """
    if not enrichment_enabled() or not project_id or not (message or "").strip():
        return ""
    sink = {"lines": [], "lock": threading.Lock()}

    def _worker():
        try:
            from app.db import get_db_session
            db2 = get_db_session()
            try:
                asyncio.run(build_enrichment_block(db2, project_id, message, sink=sink))
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
    # Read whatever relevant lines were gathered by the deadline (best-effort).
    try:
        with sink["lock"]:
            lines = list(sink["lines"])
    except Exception:
        lines = []
    return _format_block(lines)


async def _suggest_queries(user_message: str) -> List[str]:
    """Ask Nat ONLY for query strings (never data). Returns <= _max_queries()
    cleaned queries."""
    from app.llm.workers.nat_worker import run_nat_job, coerce_json
    system = (
        "Given this message, identify what STORED user information would enrich "
        "the reply (past logs, dates, metrics, preferences, facts). Output a JSON "
        "list of short search queries to look up, or [] if nothing stored would "
        "help. Put the single most useful query FIRST. Reply ONLY with the JSON "
        "list of strings."
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
    cap = _max_queries()
    if isinstance(qs, list):
        for q in qs:
            if not isinstance(q, str):
                continue
            s = q.strip()
            if not s or len(s) > 80 or s.lower() in seen:
                continue
            seen.add(s.lower())
            out.append(s)
            if len(out) >= cap:
                break
    return out


def _collect_lines(db, queries: List[str], sink=None) -> List[str]:
    """Run Nat's queries through the DETERMINISTIC retriever, in the order Nat
    suggested them (most useful first), appending each formatted line to `sink`
    as it is produced so a deadline-bounded caller can read partial progress.
    Returns all lines gathered.

    Synchronous, like the existing deterministic retrieval in the front door -
    Nat never sees or generates the data, only suggested what to look up. Because
    the work runs highest-relevance-query-first, a deadline cut-off drops the
    speculative tail, not the best hits.
    """
    from app.astra_memory.retrieval import retrieve_for_query
    lines: List[str] = []
    seen = set()
    for q in queries[:_max_queries()]:
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
            # Stream into the shared sink so a deadline read sees partial progress.
            if sink is not None:
                try:
                    with sink["lock"]:
                        sink["lines"].append(line)
                except Exception:
                    pass
        if len(lines) >= 8:
            break
    return lines


def _format_block(lines: List[str]) -> str:
    """Wrap gathered lines in the [ENRICHMENT] block, or "" if there are none.
    Identical shape whether the lines are complete or a best-effort partial."""
    if not lines:
        return ""
    return (
        "[ENRICHMENT]\n"
        "Relevant stored context (Nat flagged these as worth looking up):\n"
        + "\n".join(lines[:8])
    )
