# FILE: app/embeddings/parity.py
# Purpose: Query-parity job — recall@10 overlap + latency, gemini vs local, for the cutover sign-off.
# Called-by: app.idle.tasks_reembed (embedding_parity task)
# Depends-on: app.embeddings (provider_router, models), app.memory.models, numpy
# Last-renovated: 2026-07-02 (LANE E)
"""
Cutover-gate parity job (LANE E, Task 3c).

Before EMBEDDINGS_TEXT_QUERY_CUTOVER flips to 1, this job measures how the
local embedder ranks against the gemini baseline on OUR data:

  1. Sample up to 100 recent real user queries (messages table).
  2. Baseline: embed each query with gemini, score against the full legacy
     (gemini-stamped) corpus, take top-50; the first 10 are gemini's top-10.
  3. Shadow: embed the query with the local model, re-embed the top-50
     candidates' CONTENT locally (cached across queries), rank locally;
     take the local top-10 of the same pool.
  4. recall@10 overlap = |gemini10 ∩ local10| / 10 — ranking agreement on an
     identical candidate pool (no rows are written anywhere).
  5. Embed-latency p50/p95 recorded for both providers along the way
     (acceptance 2: local p50 <= gemini baseline, warm-resident).

Output: reports/LANE-E-PARITY-REPORT.md + a dict for the ledger summary.
Runs inside the live backend (idle task) — the Gemini key only exists there
(encrypted settings store; fresh processes can't decrypt).
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

QUERY_SAMPLE = 100
POOL_SIZE = 50
TOP_K = 10
MIN_QUERY_CHARS = 12

REPORT_PATH = Path("reports") / "LANE-E-PARITY-REPORT.md"


def _sample_queries(db) -> List[str]:
    """Most recent distinct real user messages, newest first."""
    from app.memory.models import Message

    rows = (
        db.query(Message)
        .filter(Message.role == "user")
        .order_by(Message.id.desc())
        .limit(600)
        .all()
    )
    seen = set()
    out: List[str] = []
    for row in rows:
        text = (row.content or "").strip()
        if len(text) < MIN_QUERY_CHARS:
            continue
        key = " ".join(text.lower().split())[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(text[:1000])
        if len(out) >= QUERY_SAMPLE:
            break
    return out


def _load_legacy_matrix(db):
    """(ids, np.ndarray L2-normalised) of the gemini-stamped corpus. Chunked
    id-window read — the embeddings table has known localised btree damage
    (see semantic_candidates.py) and one bad page must not kill the job."""
    import numpy as np
    from sqlalchemy import text as _sql
    from app.embeddings.provider_router import LEGACY_TEXT_LABEL

    max_id = db.execute(_sql("SELECT MAX(id) FROM embeddings")).scalar() or 0
    ids: List[int] = []
    vectors: List[List[float]] = []
    lo, chunk, skipped = 0, 500, 0
    while lo <= max_id:
        try:
            part = db.execute(_sql(
                "SELECT id, embedding FROM embeddings "
                "WHERE id > :lo AND id <= :hi "
                "AND (embedding_model = :label OR embedding_model IS NULL)"
            ), {"lo": lo, "hi": lo + chunk, "label": LEGACY_TEXT_LABEL}).fetchall()
            for row_id, embedding_text in part:
                try:
                    vec = json.loads(embedding_text)
                except Exception:
                    continue
                if isinstance(vec, list) and vec:
                    ids.append(int(row_id))
                    vectors.append(vec)
        except Exception:
            skipped += 1
            try:
                db.rollback()
            except Exception:
                pass
        lo += chunk
    if skipped:
        logger.warning("[parity] skipped %d corrupt id-range(s)", skipped)
    if not vectors:
        return [], None

    dims: Dict[int, int] = {}
    for v in vectors:
        dims[len(v)] = dims.get(len(v), 0) + 1
    dominant = max(dims, key=lambda d: dims[d])
    keep = [(i, v) for i, v in zip(ids, vectors) if len(v) == dominant]
    ids = [i for i, _ in keep]
    matrix = np.asarray([v for _, v in keep], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return ids, matrix / norms


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct * (len(ordered) - 1)))))
    return ordered[idx]


def run_parity_job(session_factory, write_spec) -> Dict:
    """Run the full parity measurement. Returns the summary dict; writes the
    markdown report. Read-only on the vector stores."""
    import numpy as np
    from app.embeddings import provider_router
    from app.embeddings.models import Embedding

    gemini_spec = provider_router._GEMINI_SPEC
    db = session_factory()
    try:
        queries = _sample_queries(db)
        ids, matrix = _load_legacy_matrix(db)

        overlaps: List[float] = []
        gemini_ms: List[float] = []
        local_ms: List[float] = []
        per_query: List[Tuple[str, float]] = []
        local_vec_cache: Dict[int, Optional[List[float]]] = {}
        failures = 0

        if matrix is None or not queries:
            reason = "no legacy corpus" if matrix is None else "no sampled queries"
            logger.warning("[parity] nothing to measure: %s", reason)
        else:
            for query in queries:
                t0 = time.perf_counter()
                gvec = provider_router.embed_text(
                    query, task_type="RETRIEVAL_QUERY", spec=gemini_spec
                )
                gemini_ms.append((time.perf_counter() - t0) * 1000.0)

                t0 = time.perf_counter()
                lvec = provider_router.embed_text(
                    query, task_type="RETRIEVAL_QUERY", spec=write_spec
                )
                local_ms.append((time.perf_counter() - t0) * 1000.0)

                if not gvec or not lvec:
                    failures += 1
                    continue

                q = np.asarray(gvec, dtype=np.float32)
                qn = np.linalg.norm(q) or 1.0
                sims = matrix @ (q / qn)
                order = np.argsort(-sims)[:POOL_SIZE]
                pool_ids = [ids[i] for i in order]
                gemini_top = set(pool_ids[:TOP_K])

                # Re-embed pool content locally (cache row -> local vector)
                missing = [rid for rid in pool_ids if rid not in local_vec_cache]
                if missing:
                    rows = (
                        db.query(Embedding)
                        .filter(Embedding.id.in_(missing))
                        .all()
                    )
                    by_id = {r.id: (r.content or "") for r in rows}
                    texts = [by_id.get(rid, "") for rid in missing]
                    vecs = provider_router.embed_text_batch(texts, spec=write_spec)
                    for rid, vec in zip(missing, vecs):
                        local_vec_cache[rid] = vec

                lq = np.asarray(lvec, dtype=np.float32)
                lqn = np.linalg.norm(lq) or 1.0
                lq = lq / lqn
                scored: List[Tuple[int, float]] = []
                for rid in pool_ids:
                    vec = local_vec_cache.get(rid)
                    if not vec:
                        continue
                    dv = np.asarray(vec, dtype=np.float32)
                    dn = np.linalg.norm(dv) or 1.0
                    scored.append((rid, float(dv @ lq) / float(dn)))
                scored.sort(key=lambda x: x[1], reverse=True)
                local_top = {rid for rid, _ in scored[:TOP_K]}

                overlap = len(gemini_top & local_top) / float(TOP_K)
                overlaps.append(overlap)
                per_query.append((query[:80], overlap))
    finally:
        db.close()

    result = {
        "query_count": len(overlaps),
        "mean_overlap": (sum(overlaps) / len(overlaps)) if overlaps else 0.0,
        "min_overlap": min(overlaps) if overlaps else 0.0,
        "gemini_p50_ms": _percentile(gemini_ms, 0.50),
        "gemini_p95_ms": _percentile(gemini_ms, 0.95),
        "local_p50_ms": _percentile(local_ms, 0.50),
        "local_p95_ms": _percentile(local_ms, 0.95),
        "embed_failures": failures,
        "report_path": str(REPORT_PATH),
    }
    _write_report(result, per_query, write_spec)
    return result


def _write_report(result: Dict, per_query: List[Tuple[str, float]], write_spec) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    low = [f"- {o:.1f}  `{q}`" for q, o in sorted(per_query, key=lambda x: x[1])[:15]]
    lines = [
        "# LANE E PARITY REPORT — gemini vs local query embedding",
        f"_Generated {datetime.utcnow().isoformat()}Z by the embedding_parity idle task._",
        "",
        f"- Local model: **{write_spec.label}** (dims {write_spec.dims})",
        f"- Queries sampled: **{result['query_count']}** "
        f"(embed failures: {result['embed_failures']})",
        f"- recall@{TOP_K} overlap vs gemini baseline (same {POOL_SIZE}-candidate pool): "
        f"**mean {result['mean_overlap']:.2f}**, min {result['min_overlap']:.2f}",
        f"- Query-embed latency p50/p95: local **{result['local_p50_ms']:.0f} / "
        f"{result['local_p95_ms']:.0f} ms** vs gemini {result['gemini_p50_ms']:.0f} / "
        f"{result['gemini_p95_ms']:.0f} ms",
        "",
        "## Sign-off",
        "If the overlap and latency above look acceptable, flip "
        "`EMBEDDINGS_TEXT_QUERY_CUTOVER=1` in `.env` and restart — that enables "
        "local query-time embedding (dual-read) and unlocks the reembed_batch "
        "drain. Rollback at any point: `EMBEDDINGS_TEXT_PROVIDER=gemini`.",
        "",
        "## Lowest-overlap queries (worst 15)",
        *(low or ["- (none)"]),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("[parity] report written → %s", REPORT_PATH)
