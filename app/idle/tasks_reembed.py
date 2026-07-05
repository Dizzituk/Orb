# FILE: app/idle/tasks_reembed.py
# Purpose: Idle tasks for the local-embeddings migration — reembed_batch (gemini→local drain) + embedding_parity gate.
# Called-by: app.idle.router.ensure_builtin_tasks_registered (import side-effect)
# Depends-on: app.embeddings (provider_router, local_provider, models, parity), app.idle (ledger, router)
# Last-renovated: 2026-07-02 (LANE E)
"""
LANE E migration tasks (Task 3).

reembed_batch — walks rows stamped gemini-embedding-2-preview and re-embeds
them via the local text model, hot items first (memory/note/message), then
RAG chunks, then bulk documents, then experience patterns, then the
self-model fragment store. Checkpoints via ctx.should_yield() between
batches; progress lives in the ledger JSON (get_background_task_log shows
it). HARD GATE: no-ops until EMBEDDINGS_TEXT_PROVIDER=local AND
EMBEDDINGS_TEXT_QUERY_CUTOVER=1 — draining the gemini population before the
query path can score local rows would silently shrink recall.

embedding_parity — the sign-off artifact for that gate: 100 sampled real
queries, recall@10 overlap gemini vs local + embed-latency p50/p95, written
to reports/LANE-E-PARITY-REPORT.md (app/embeddings/parity.py). Runs once per
model pair (fingerprint-skipped after).

Rollback stays trivial throughout: rows are stamped, so flipping
EMBEDDINGS_TEXT_PROVIDER back to gemini leaves both populations searchable
in their own spaces.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import List, Optional

from app.idle.router import RecurringSpec, TaskContext, TaskOutcome, register_task_handler

logger = logging.getLogger(__name__)

REEMBED_TASK = "reembed_batch"
PARITY_TASK = "embedding_parity"

# Hot-first drain order (jobspec 3a). "rest" catches any source_type not
# listed so the legacy population really reaches zero.
_TABLE_UNITS = [
    ("hot", ["memory", "note", "message"]),
    ("rag_chunks", ["arch_code_chunk"]),
    ("bulk_docs", ["file"]),
    ("experience", ["experience_pattern"]),
    ("rest", None),
]
_LISTED_TYPES = [t for _, types in _TABLE_UNITS if types for t in types]

_MAX_TRACKED_FAILURES = 1000


def _batch_size() -> int:
    # CARRYOVER §3: drains must be preemptible — small checkpointed batches
    # (~30 items) so "finish the current batch, park, wake Chatterbox" is a
    # couple of seconds, never a long uninterruptible job.
    try:
        return max(1, int(os.getenv("ASTRA_REEMBED_BATCH_SIZE", "30")))
    except Exception:
        return 30


def _gate_reason() -> Optional[str]:
    """None when the migration may run; else why it is gated."""
    from app.embeddings import provider_router

    if provider_router.text_write_spec().provider != "local":
        return "EMBEDDINGS_TEXT_PROVIDER is not 'local'"
    if not provider_router.query_cutover_enabled():
        return ("EMBEDDINGS_TEXT_QUERY_CUTOVER=0 — parity report awaiting "
                "sign-off (reports/LANE-E-PARITY-REPORT.md)")
    return None


def _legacy_criterion():
    """Legacy space = gemini-stamped rows PLUS NULL-stamped rows (the boot
    backfill skips damaged id-windows — app/embeddings/migrations.py — and
    per-row restamping here is exactly how those heal)."""
    from sqlalchemy import or_
    from app.embeddings.models import Embedding
    from app.embeddings.provider_router import LEGACY_TEXT_LABEL

    return or_(
        Embedding.embedding_model == LEGACY_TEXT_LABEL,
        Embedding.embedding_model.is_(None),
    )


def _legacy_remaining(db) -> int:
    from app.embeddings.models import Embedding
    from sqlalchemy import func

    return int(
        db.query(func.count(Embedding.id))
        .filter(_legacy_criterion())
        .scalar()
        or 0
    )


def _reembed_table_batch(
    session_factory,
    source_types: Optional[List[str]],
    write_spec,
    failed_ids: List[int],
) -> int:
    """Re-embed one batch of legacy rows. Returns rows restamped (0 = unit
    drained). Rows whose content will not embed are tracked in failed_ids
    (excluded from later batches, surfaced in the summary)."""
    from app.embeddings import provider_router
    from app.embeddings.models import Embedding

    db = session_factory()
    try:
        q = db.query(Embedding).filter(_legacy_criterion())
        if source_types is None:
            q = q.filter(~Embedding.source_type.in_(_LISTED_TYPES))
        else:
            q = q.filter(Embedding.source_type.in_(source_types))
        if failed_ids:
            q = q.filter(~Embedding.id.in_(failed_ids))
        rows = q.order_by(Embedding.id.asc()).limit(_batch_size()).all()
        if not rows:
            return 0

        texts = [(row.content or "") for row in rows]
        vectors = provider_router.embed_text_batch(texts, spec=write_spec)

        done = 0
        arch_ids: List[int] = []
        for row, vec in zip(rows, vectors):
            if not vec:
                if len(failed_ids) < _MAX_TRACKED_FAILURES:
                    failed_ids.append(row.id)
                continue
            row.embedding = json.dumps(vec)
            row.embedding_model = write_spec.label
            row.dims = len(vec)
            if row.source_type == "arch_code_chunk":
                arch_ids.append(row.source_id)
            done += 1

        # Keep the ArchCodeChunk metadata stamp in sync with the vector rows.
        if arch_ids:
            try:
                from app.rag.models import ArchCodeChunk
                db.query(ArchCodeChunk).filter(ArchCodeChunk.id.in_(arch_ids)).update(
                    {ArchCodeChunk.embedding_model: write_spec.label},
                    synchronize_session=False,
                )
            except Exception as exc:
                logger.warning("[reembed] arch chunk stamp sync failed: %s", exc)

        db.commit()

        if done == 0:
            # Whole batch failed to embed — the server is down or every row
            # in the window is bad. Fail (retry-on-cooldown), don't spin.
            raise RuntimeError(
                f"re-embed batch produced 0/{len(rows)} vectors — local "
                f"embedding server down?"
            )
        return done
    finally:
        db.close()


def _reembed_fragments(write_spec) -> int:
    """Re-embed legacy-model fragments in the self-model store (JSON-backed,
    small). One pass; per-fragment failures are skipped and retried on the
    next run."""
    from app.embeddings import provider_router
    from app.self_model.fragments.store import get_fragment_store

    store = get_fragment_store()
    legacy = provider_router.LEGACY_TEXT_LABEL
    stale = [
        f for f in store.all()
        if f.embedding and (f.embedding_model or "") == legacy
    ]
    if not stale:
        return 0
    done = 0
    for i in range(0, len(stale), _batch_size()):
        chunk = stale[i:i + _batch_size()]
        vectors = provider_router.embed_text_batch(
            [f.text for f in chunk], spec=write_spec
        )
        for frag, vec in zip(chunk, vectors):
            if vec and store.update_embedding(frag.fragment_id, vec, write_spec.label):
                done += 1
    return done


async def reembed_handler(ctx: TaskContext) -> TaskOutcome:
    from app.embeddings import local_provider, provider_router
    from app.embeddings.service import reset_model_count_cache

    gate = _gate_reason()
    if gate:
        return TaskOutcome.completed(
            summary=f"gated: {gate}",
            coverage="no rows touched (gate)",
        )
    if not local_provider.text_available():
        raise RuntimeError(
            "local embedding server (:8004) unreachable — refusing to start "
            "a drain that would fail row by row"
        )

    write_spec = provider_router.text_write_spec()
    progress = ctx.load_progress()
    counts = dict(progress.get("counts") or {})
    failed_ids: List[int] = list(progress.get("failed_ids") or [])

    for unit, source_types in _TABLE_UNITS:
        while True:
            if ctx.should_yield():
                ctx.save_progress({"counts": counts, "failed_ids": failed_ids, "unit": unit})
                return TaskOutcome.paused(f"checkpointed in unit '{unit}'")
            n = await asyncio.to_thread(
                _reembed_table_batch, ctx.session_factory, source_types,
                write_spec, failed_ids,
            )
            if n == 0:
                break
            counts[unit] = counts.get(unit, 0) + n
            ctx.save_progress({"counts": counts, "failed_ids": failed_ids, "unit": unit})
            reset_model_count_cache()

    counts["fragments"] = counts.get("fragments", 0) + await asyncio.to_thread(
        _reembed_fragments, write_spec
    )
    ctx.save_progress({"counts": counts, "failed_ids": failed_ids, "unit": "done"})

    db = ctx.session_factory()
    try:
        remaining = _legacy_remaining(db)
    finally:
        db.close()
    reset_model_count_cache()

    total = sum(counts.values())
    summary = (
        f"re-embedded {total} row(s) → {write_spec.label} "
        f"({', '.join(f'{k}={v}' for k, v in counts.items() if v)}); "
        f"legacy remaining={remaining}"
        + (f"; unembeddable skipped={len(failed_ids)}" if failed_ids else "")
    )
    return TaskOutcome.completed(
        summary=summary,
        coverage=f"embeddings table + fragment store, batch={_batch_size()}",
    )


def reembed_fingerprint(params: dict) -> Optional[str]:
    """Skip when nothing changed: same gate state and same legacy count."""
    try:
        from app.db import SessionLocal
        from app.embeddings import provider_router

        db = SessionLocal()
        try:
            remaining = _legacy_remaining(db)
        finally:
            db.close()
        return (
            f"{provider_router.text_write_spec().provider}"
            f"|{provider_router.query_cutover_enabled()}"
            f"|remaining:{remaining}"
        )
    except Exception as exc:
        logger.warning("[reembed] fingerprint failed: %s (running anyway)", exc)
        return None


# ─── Parity gate task ────────────────────────────────────────────

async def parity_handler(ctx: TaskContext) -> TaskOutcome:
    from app.embeddings import local_provider, provider_router
    from app.embeddings.parity import run_parity_job

    write_spec = provider_router.text_write_spec()
    if write_spec.provider != "local":
        return TaskOutcome.completed(
            summary="gated: EMBEDDINGS_TEXT_PROVIDER is not 'local'",
            coverage="no parity run",
        )
    if not local_provider.text_available():
        raise RuntimeError("local embedding server (:8004) unreachable")

    report = await asyncio.to_thread(run_parity_job, ctx.session_factory, write_spec)
    return TaskOutcome.completed(
        summary=(
            f"parity report: recall@10 overlap mean {report['mean_overlap']:.2f} "
            f"over {report['query_count']} queries; local p50 "
            f"{report['local_p50_ms']:.0f}ms vs gemini p50 "
            f"{report['gemini_p50_ms']:.0f}ms → {report['report_path']}"
        ),
        coverage=f"{report['query_count']} sampled real queries",
    )


def parity_fingerprint(params: dict) -> Optional[str]:
    """One real run per model pair (+ per cutover state)."""
    try:
        from app.embeddings import provider_router

        spec = provider_router.text_write_spec()
        return f"{spec.provider}|{spec.label}|cutover:{provider_router.query_cutover_enabled()}"
    except Exception:
        return None


def _cadence_hours(env: str, default: float) -> float:
    try:
        return float(os.getenv(env, str(default)))
    except Exception:
        return default


register_task_handler(
    REEMBED_TASK,
    reembed_handler,
    fingerprint_fn=reembed_fingerprint,
    recurring=RecurringSpec(
        task_type=REEMBED_TASK,
        params={},
        cadence_hours=_cadence_hours("ASTRA_REEMBED_CADENCE_HOURS", 6.0),
    ),
)

register_task_handler(
    PARITY_TASK,
    parity_handler,
    fingerprint_fn=parity_fingerprint,
    recurring=RecurringSpec(
        task_type=PARITY_TASK,
        params={},
        cadence_hours=_cadence_hours("ASTRA_PARITY_CADENCE_HOURS", 24.0),
    ),
)
