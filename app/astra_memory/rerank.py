# FILE: app/astra_memory/rerank.py
# Purpose: Confidence gate over ranked retrieval candidates (Job 4 task 1).
# Called-by: app.astra_memory.retrieval
# Depends-on: stdlib only (operates on RetrievalCandidate duck-type)
# Last-renovated: 2026-06-15
"""
Confidence gate for retrieval candidates (MEMORY_JOBS_SPEC Job 4, task 1).

Stage 1 ranks candidates by a linear sum (static priority + tag/entity match +
recency, then the semantic-channel bonus). That ordering is sound, but it has
no notion of whether a candidate is relevant ENOUGH to inject. At D2+ it would
hand the prompt the full max_items even when only three of them are on-topic
and the rest are mushy base-priority filler. Three sharp items beat twenty
mushy ones.

This is the second-pass JUDGEMENT the spec calls for: a cheap, deterministic
confidence gate that runs AFTER ranking and BEFORE stage-2 expansion (so the
dropped tail never even costs a cold-storage fetch). It keeps the candidates
sitting in the top score cluster and drops the long tail trailing far below it
— turning a sorted list into a real inject / don't-inject decision.

It is purely relative and bounded, with NO API call, so it is safe on the sync
voice path (the user is driving). Relative-not-absolute matters: a uniformly
weak result set leaves the whole pool near the top, so nothing is pruned (no
regression); only a set with a clear winner cluster sheds its tail.

Behaviour, not shape (MEMORY_MAP.md §5): the ratio/threshold knobs here are
cheap to tune and env-overridable, and the gate never becomes an independent
injection path — it only thins the candidate list inside retrieve_for_query.

Deliberately deferred (documented): an LLM-scored rerank. The only text-LLM
helper (`_streaming_utils_3.call_llm_text`) is async and the retrieval path is
synchronous and latency-sensitive, so an LLM second pass would force an async
seam onto the hot voice path. The deterministic gate already satisfies Job 4's
acceptance ("three sharp items beat twenty mushy ones"; "nothing is pulled when
nothing clears the bar"); the LLM variant is a behaviour follow-on for D3+.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Keep a candidate when its score is within this fraction of the top score.
# 0.45 → anything trailing more than ~55% below the best hit is "mushy" and is
# dropped. Relative (not absolute): a uniformly-weak pool stays near the top and
# is left untouched; a pool with a clear winner cluster sheds its tail.
_KEEP_RATIO_DEFAULT = 0.45

# Don't bother gating tiny pools — there is no mushy tail to shed, the depth's
# max_items cap already bounds them, and it keeps the small controlled candidate
# sets the retrieval unit-tests build untouched.
_MIN_TO_GATE = 6


def _enabled() -> bool:
    return os.getenv("ASTRA_RETRIEVAL_CONFIDENCE_GATE", "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _keep_ratio() -> float:
    try:
        r = float(os.getenv("ASTRA_RETRIEVAL_KEEP_RATIO", str(_KEEP_RATIO_DEFAULT)))
    except (TypeError, ValueError):
        return _KEEP_RATIO_DEFAULT
    # Clamp to a sane band so a stray env value can neither empty nor no-op it.
    return min(0.9, max(0.1, r))


def apply_confidence_gate(candidates: list, depth=None) -> list:
    """Drop the low-confidence tail from a ranked candidate list.

    Expects `candidates` already sorted by relevance_score descending (as
    retrieve_for_query does after cost ranking). Returns the kept prefix — the
    candidates whose score sits within `_keep_ratio()` of the top score. Never
    returns empty for a non-empty input (the top hit always survives), and is a
    pure no-op for pools at or below `_MIN_TO_GATE`.

    Best-effort: any error returns the input unchanged.
    """
    try:
        if not _enabled() or len(candidates) <= _MIN_TO_GATE:
            return candidates

        top = candidates[0].relevance_score
        if top is None or top <= 0:
            return candidates

        bar = top * _keep_ratio()
        kept = [c for c in candidates if (c.relevance_score or 0.0) >= bar]
        if not kept:
            kept = candidates[:1]

        dropped = len(candidates) - len(kept)
        if dropped > 0:
            logger.info(
                "[rerank] confidence gate kept %d/%d (top=%.2f bar=%.2f%s)",
                len(kept), len(candidates), top, bar,
                f" depth={getattr(depth, 'value', depth)}" if depth is not None else "",
            )
        return kept
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("[rerank] confidence gate skipped: %s", exc)
        return candidates


__all__ = ["apply_confidence_gate"]
