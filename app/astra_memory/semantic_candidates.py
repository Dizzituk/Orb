# FILE: app/astra_memory/semantic_candidates.py
# Purpose: Semantic candidate augmentation for hot-index retrieval.
# Called-by: app.astra_memory.retrieval
# Depends-on: app.astra_memory.preference_models, app.astra_memory.retrieval, app.llm.clients
# Last-renovated: 2026-06-11
"""
Semantic candidate augmentation for hot-index retrieval.

Job 3 of the memory roadmap (2026-06-10): the embeddings table has been
populated for months (11k+ vectors via Gemini Embedding 2) but the
retrieval path that builds the memory context never consulted it —
matching was curated-keyword only, so anything outside the tagger vocab
(half the index sits at tags=["general"]) was unreachable by topic.

This module adds a vector channel alongside the tag/entity channel:

  - An in-process matrix of all stored embeddings (float32, lazy-loaded,
    refreshed when the table grows or every few hours). 11k x ~768-3072
    dims is well within desktop RAM and makes per-query similarity a
    single matmul.
  - augment_with_semantic(): embeds the user query (RETRIEVAL_QUERY task
    type), takes top-K cosine matches, maps them onto hot-index records,
    and merges them into the stage-1 candidate pool — boosting records
    already found by tags, and resurrecting relevant records the keyword
    channel missed entirely.

Gating: runs for D2+ depths, or at any depth when the keyword channel
produced no tags/entities (the recall-rescue case). Controlled by
ASTRA_SEMANTIC_RETRIEVAL (default on). A circuit breaker disables the
channel for 30 minutes after 3 consecutive embedding-API failures so a
Gemini outage never stalls chat.

Never raises into the caller — all failures degrade to keyword-only.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Map embeddings.source_type -> hot_index.record_type
_SOURCE_TO_RECORD_TYPE = {
    "message": "message",
    "note": "note",
    "file": "document",
    # Job 3 (2026-06-15): consolidated conversation facts. The membrane writes
    # both an embedding (source_type='memory') and a HotIndex row
    # (record_type='memory') keyed by the same PreferenceRecord id, so the
    # vector channel can resurface a durable fact the keyword channel missed.
    "memory": "memory",
}

_MIN_SIM_DEFAULT = 0.55
_TOP_K_DEFAULT = 40
_RELOAD_AFTER_SECONDS = 6 * 3600
_RELOAD_GROWTH_THRESHOLD = 200

# Circuit breaker state for the embedding API
_consecutive_failures = 0
_disabled_until = 0.0

# Query-embedding cache (2026-06-22): normalised query text -> vector. A given
# string always embeds to the same vector for a fixed model, so this NEVER goes
# stale; it's an in-process LRU (cleared on restart - self-healing) that lets
# enrichment batch a turn's queries once and reuse them, and makes repeated vague
# asks ("worried about my calories today") cost ZERO network on the live reply
# path. Keyed on Nat's suggested queries, which it tends to canonicalise, so
# varied user phrasings converge onto the same cached lookups.
_QUERY_EMBED_CACHE: "OrderedDict[str, List[float]]" = OrderedDict()
_QUERY_CACHE_MAX = int(os.getenv("ASTRA_QUERY_EMBED_CACHE_MAX", "512"))


def _normalize_query(text: str) -> str:
    return " ".join((text or "").lower().split())


def _cache_get(text: str) -> Optional[List[float]]:
    key = _normalize_query(text)
    vec = _QUERY_EMBED_CACHE.get(key)
    if vec is not None:
        _QUERY_EMBED_CACHE.move_to_end(key)  # LRU touch
    return vec


def _cache_put(text: str, vec: Optional[List[float]]) -> None:
    if not vec:
        return
    key = _normalize_query(text)
    _QUERY_EMBED_CACHE[key] = vec
    _QUERY_EMBED_CACHE.move_to_end(key)
    while len(_QUERY_EMBED_CACHE) > _QUERY_CACHE_MAX:
        _QUERY_EMBED_CACHE.popitem(last=False)


def _enabled() -> bool:
    if os.getenv("ASTRA_SEMANTIC_RETRIEVAL", "true").strip().lower() in ("0", "false", "no", "off"):
        return False
    return time.time() >= _disabled_until


def _record_failure() -> None:
    global _consecutive_failures, _disabled_until
    _consecutive_failures += 1
    if _consecutive_failures >= 3:
        _disabled_until = time.time() + 30 * 60
        logger.warning(
            "[semantic] 3 consecutive embedding failures — channel disabled for 30 min"
        )


def _record_success() -> None:
    global _consecutive_failures
    _consecutive_failures = 0


# =============================================================================
# In-process vector index
# =============================================================================

class _VectorIndex:
    """Lazy-loaded matrix of all stored embeddings."""

    def __init__(self) -> None:
        self.matrix = None              # np.ndarray (n, dim) float32, L2-normalised
        self.keys: List[Tuple[str, str]] = []  # (record_type, record_id) per row
        self.loaded_at: float = 0.0
        self.loaded_count: int = 0

    def _needs_reload(self, db) -> bool:
        if self.matrix is None:
            return True
        if time.time() - self.loaded_at > _RELOAD_AFTER_SECONDS:
            return True
        try:
            from sqlalchemy import text as _sql
            current = db.execute(_sql("SELECT COUNT(*) FROM embeddings")).scalar() or 0
            return (current - self.loaded_count) > _RELOAD_GROWTH_THRESHOLD
        except Exception:
            return False

    def ensure_loaded(self, db) -> bool:
        """Load/refresh the matrix. Returns True if usable."""
        if not self._needs_reload(db):
            return self.matrix is not None
        try:
            import numpy as np
            from sqlalchemy import text as _sql

            # Chunked load — the embeddings table has localised btree damage
            # (diagnosed 2026-06-10: ids ~1-500 unreadable, "database disk
            # image is malformed"). Reading in id windows lets us skip the
            # corrupt range(s) and still serve the ~95% of vectors that are
            # intact, instead of losing the whole channel to one bad page.
            max_id = db.execute(_sql("SELECT MAX(id) FROM embeddings")).scalar() or 0
            rows = []
            skipped_ranges = 0
            chunk = 500
            lo = 0
            while lo <= max_id:
                try:
                    part = db.execute(_sql(
                        "SELECT source_type, source_id, embedding FROM embeddings "
                        "WHERE id > :lo AND id <= :hi"
                    ), {"lo": lo, "hi": lo + chunk}).fetchall()
                    rows.extend(part)
                except Exception:
                    skipped_ranges += 1
                    try:
                        db.rollback()
                    except Exception:
                        pass
                lo += chunk
            if skipped_ranges:
                logger.warning(
                    "[semantic] skipped %d corrupt id-range(s) loading embeddings "
                    "(known table damage, 2026-06-10 diagnostic) — proceeding with %d rows",
                    skipped_ranges, len(rows),
                )

            vectors: List[List[float]] = []
            keys: List[Tuple[str, str]] = []
            dim_counts: Dict[int, int] = {}

            parsed: List[Tuple[Tuple[str, str], List[float]]] = []
            for source_type, source_id, embedding_text in rows:
                record_type = _SOURCE_TO_RECORD_TYPE.get(str(source_type))
                if not record_type or not embedding_text:
                    continue
                try:
                    vec = json.loads(embedding_text)
                except Exception:
                    continue
                if not isinstance(vec, list) or not vec:
                    continue
                dim_counts[len(vec)] = dim_counts.get(len(vec), 0) + 1
                parsed.append(((record_type, str(source_id)), vec))

            if not parsed:
                logger.info("[semantic] no usable embeddings found")
                self.matrix = None
                return False

            # Keep only the dominant dimensionality (model changes leave strays)
            dominant_dim = max(dim_counts, key=lambda d: dim_counts[d])
            for key, vec in parsed:
                if len(vec) == dominant_dim:
                    keys.append(key)
                    vectors.append(vec)

            matrix = np.asarray(vectors, dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.matrix = matrix / norms
            self.keys = keys
            self.loaded_at = time.time()
            self.loaded_count = len(rows)
            logger.info(
                "[semantic] vector index loaded: %d vectors, dim=%d (%.1f MB)",
                len(keys), dominant_dim, self.matrix.nbytes / 1e6,
            )
            return True
        except Exception as exc:
            logger.warning("[semantic] vector index load failed: %s", exc)
            self.matrix = None
            return False

    def top_matches(self, query_vec: List[float], top_k: int) -> List[Tuple[str, str, float]]:
        """Cosine top-K against the index. Returns (record_type, record_id, sim)."""
        import numpy as np
        if self.matrix is None:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        if q.shape[0] != self.matrix.shape[1]:
            logger.debug(
                "[semantic] query dim %d != index dim %d — skipping",
                q.shape[0], self.matrix.shape[1],
            )
            return []
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        sims = self.matrix @ (q / qn)
        if top_k < len(sims):
            idx = np.argpartition(sims, -top_k)[-top_k:]
        else:
            idx = np.arange(len(sims))
        idx = idx[np.argsort(sims[idx])[::-1]]

        # Collapse chunked sources to their best chunk
        best: Dict[Tuple[str, str], float] = {}
        for i in idx:
            key = self.keys[int(i)]
            sim = float(sims[int(i)])
            if key not in best or sim > best[key]:
                best[key] = sim
        return [(rt, rid, sim) for (rt, rid), sim in
                sorted(best.items(), key=lambda kv: kv[1], reverse=True)]


_index = _VectorIndex()


def _embed_query(text: str) -> Optional[List[float]]:
    """Embed the user query. Cache-first, then sync Gemini call with circuit
    breaker. A cache hit (e.g. prewarmed by enrichment's batch, or a repeated
    vague ask) makes zero network calls."""
    cached = _cache_get(text)
    if cached is not None:
        return cached
    try:
        from app.llm.clients import generate_embedding
        vec = generate_embedding(text[:4000], task_type="RETRIEVAL_QUERY")
        _record_success()
        _cache_put(text, vec)
        return vec
    except Exception as exc:
        logger.warning("[semantic] query embedding failed: %s", exc)
        _record_failure()
        return None


def prewarm_query_embeddings(texts: List[str]) -> Dict[str, int]:
    """Embed a whole turn's query strings in ONE batch call and cache them, so the
    per-query retrievals that follow hit the cache instead of N serial Gemini
    round-trips. Pure optimisation: failure-proof, returns stats, and is a no-op
    when the channel is disabled or everything's already cached. On batch failure
    the per-query lazy _embed_query path (also cached) remains as fallback.

    Returns {"queries": unique, "cached": already-warm, "embedded": newly-embedded}.
    """
    stats = {"queries": 0, "cached": 0, "embedded": 0}
    try:
        if not _enabled():
            return stats
        uniq: List[str] = []
        seen = set()
        for t in texts or []:
            if not isinstance(t, str) or not t.strip():
                continue
            key = _normalize_query(t)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(t)
        stats["queries"] = len(uniq)
        misses = [t for t in uniq if _cache_get(t) is None]
        stats["cached"] = len(uniq) - len(misses)
        if not misses:
            return stats
        from app.llm.clients import get_embeddings
        vecs = get_embeddings([m[:4000] for m in misses], task_type="RETRIEVAL_QUERY")
        if isinstance(vecs, list):
            for t, vec in zip(misses, vecs):
                if vec:
                    _cache_put(t, vec)
                    stats["embedded"] += 1
            _record_success()
    except Exception as exc:  # noqa: BLE001 - lazy per-query embed is the fallback
        logger.debug("[semantic] prewarm batch failed (lazy fallback): %s", exc)
    return stats


# =============================================================================
# Public API
# =============================================================================

def augment_with_semantic(
    db,
    candidates: list,
    user_message: str,
    depth,
    query_tags: Optional[List[str]],
    query_entities: Optional[List[str]],
    max_candidates: int = 100,
    recency_weight: float = 0.3,
) -> list:
    """
    Merge semantic matches into the stage-1 candidate list.

    - Candidates already present that also match semantically get a
      similarity bonus (the two channels agree — strong signal).
    - Strong semantic hits absent from the keyword channel are fetched
      from the hot index and appended (the rescue case).

    Returns the (re-sorted) candidate list; on any failure returns the
    input list unchanged.
    """
    try:
        from app.astra_memory.preference_models import IntentDepth, HotIndex
        from app.astra_memory.retrieval import RetrievalCandidate, RetrievalCost

        if not _enabled() or not user_message or not user_message.strip():
            return candidates

        keyword_channel_empty = not query_tags and not query_entities
        deep = depth in (IntentDepth.D2, IntentDepth.D3, IntentDepth.D4)
        if not (deep or keyword_channel_empty):
            return candidates

        if not _index.ensure_loaded(db):
            return candidates

        query_vec = _embed_query(user_message)
        if not query_vec:
            return candidates

        min_sim = float(os.getenv("ASTRA_SEMANTIC_MIN_SIM", str(_MIN_SIM_DEFAULT)))
        top_k = int(os.getenv("ASTRA_SEMANTIC_TOP_K", str(_TOP_K_DEFAULT)))
        matches = [m for m in _index.top_matches(query_vec, top_k) if m[2] >= min_sim]
        if not matches:
            return candidates

        from datetime import datetime, timezone
        existing = {(c.record_type, c.record_id): c for c in candidates}
        added = 0

        for record_type, record_id, sim in matches:
            key = (record_type, record_id)
            if key in existing:
                # Both channels agree — boost once.
                existing[key].relevance_score += sim * 0.5
                continue
            hot = db.query(HotIndex).filter(
                HotIndex.record_type == record_type,
                HotIndex.record_id == record_id,
            ).first()
            if not hot:
                continue
            score = (hot.retrieval_priority or 0.0) + sim * 0.9
            if hot.updated_at:
                age_days = (datetime.now(timezone.utc)
                            - hot.updated_at.replace(tzinfo=timezone.utc)).days
                score += max(0, 1 - (age_days / 30)) * recency_weight
            candidate = RetrievalCandidate(
                record_type=hot.record_type,
                record_id=hot.record_id,
                title=hot.title,
                one_liner=hot.one_liner,
                relevance_score=score,
                retrieval_cost=hot.retrieval_cost or RetrievalCost.TINY,
                hot_index_id=hot.id,
                tags=hot.tags or [],
            )
            existing[key] = candidate
            candidates.append(candidate)
            added += 1

        if added:
            logger.info("[semantic] merged %d semantic-only candidates", added)

        candidates.sort(key=lambda c: c.relevance_score, reverse=True)
        return candidates[:max_candidates]
    except Exception as exc:
        logger.warning("[semantic] augmentation failed, keyword-only: %s", exc)
        return candidates
