# FILE: app/idle/tasks.py
# Purpose: Built-in idle tasks — the repo-map refresh (rescan + reindex, full-once-then-delta).
# Called-by: app.idle.router.ensure_builtin_tasks_registered (import side-effect registers)
# Depends-on: app.rag.rescan, app.rag.reindex, app.idle.router
# Last-renovated: 2026-07-01
"""
Repo-map task ("repo_map"): keeps ASTRA's code self-knowledge fresh.

- First ever run: the incremental rescan sees an empty index -> everything is
  "added" -> full map. Every later run processes only the delta the rescan
  reports (added/modified/deleted); descriptors + embeddings are generated
  only for new/changed chunks (reindex is incremental by construction).
- Fingerprint: a cheap HOST-side walk of the scan roots (path|size|mtime).
  Unchanged fingerprint since the last completed run -> the governor skips
  the whole task without touching the sandbox. Note: rescan itself reads
  file CONTENT through the sandbox controller (by design — self-knowledge
  tracks the sandbox tree); the host walk is only the change detector.
- No LLM calls at all: descriptors are deterministic string assembly,
  embeddings ride the whitelisted Gemini embedding path.

Units (checkpointed between each): rescan -> reindex -> cleanup.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import Optional

from app.idle.router import RecurringSpec, TaskContext, TaskOutcome, register_task_handler

logger = logging.getLogger(__name__)

REPO_MAP_TASK = "repo_map"


def _iter_host_files():
    """Walk the SAME roots the RAG rescan covers, on the host filesystem."""
    from app.rag.rescan import CODE_EXTENSIONS, SCAN_ROOTS, SKIP_DIRS

    for root in SCAN_ROOTS:
        if os.path.isfile(root):
            yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if os.path.splitext(fn)[1] in CODE_EXTENSIONS:
                    yield os.path.join(dirpath, fn)


def repo_map_fingerprint(params: dict) -> Optional[str]:
    h = hashlib.sha256()
    count = 0
    try:
        for path in sorted(_iter_host_files()):
            try:
                st = os.stat(path)
                h.update(f"{path}|{st.st_size}|{st.st_mtime_ns}\n".encode("utf-8"))
                count += 1
            except OSError:
                continue
    except Exception as exc:
        logger.warning("[idle.repo_map] fingerprint walk failed: %s", exc)
        return None
    h.update(f"files:{count}".encode("utf-8"))
    return h.hexdigest()


def _unit_rescan(session_factory) -> dict:
    from app.rag.rescan import rescan_codebase

    db = session_factory()
    try:
        report = rescan_codebase(db)
        return {
            "added": len(getattr(report, "added", []) or []),
            "modified": len(getattr(report, "modified", []) or []),
            "deleted": len(getattr(report, "deleted", []) or []),
            "unchanged": len(getattr(report, "unchanged", []) or []),
            "chunks_added": getattr(report, "chunks_added", 0),
            "chunks_removed": getattr(report, "chunks_removed", 0),
        }
    finally:
        db.close()


def _unit_reindex(session_factory) -> dict:
    from app.rag.reindex import reindex_unembedded

    db = session_factory()
    try:
        return reindex_unembedded(db)
    finally:
        db.close()


def _unit_cleanup(session_factory) -> dict:
    from app.rag.reindex import cleanup_orphaned_embeddings

    db = session_factory()
    try:
        return {"orphans_removed": cleanup_orphaned_embeddings(db)}
    finally:
        db.close()


_UNITS = (
    ("rescan", _unit_rescan),
    ("reindex", _unit_reindex),
    ("cleanup", _unit_cleanup),
)


async def repo_map_handler(ctx: TaskContext) -> TaskOutcome:
    progress = ctx.load_progress()
    done = list(progress.get("units_done") or [])
    stats = dict(progress.get("stats") or {})

    for name, fn in _UNITS:
        if name in done:
            continue
        if ctx.should_yield():
            ctx.save_progress({"units_done": done, "stats": stats})
            return TaskOutcome.paused(f"checkpointed before unit '{name}'")
        # Heavy sync work off the event loop; each unit opens its own session.
        stats[name] = await asyncio.to_thread(fn, ctx.session_factory)
        done.append(name)
        ctx.save_progress({"units_done": done, "stats": stats})

    r = stats.get("rescan") or {}
    ri = stats.get("reindex") or {}
    cl = stats.get("cleanup") or {}
    summary = (
        f"repo map: +{r.get('added', 0)} ~{r.get('modified', 0)} -{r.get('deleted', 0)} files "
        f"({r.get('unchanged', 0)} unchanged); {ri.get('descriptors_generated', 0)} descriptors, "
        f"{ri.get('embeddings_created', 0)} embeddings, {ri.get('errors', 0)} errors; "
        f"{cl.get('orphans_removed', 0)} orphaned vectors removed"
    )
    coverage = "RAG scan roots (app/ + main.py); content via sandbox rescan, delta-only reindex"
    return TaskOutcome.completed(summary=summary, coverage=coverage)


def _cadence_hours() -> float:
    try:
        return float(os.getenv("ASTRA_REPO_MAP_CADENCE_HOURS", "24"))
    except Exception:
        return 24.0


register_task_handler(
    REPO_MAP_TASK,
    repo_map_handler,
    fingerprint_fn=repo_map_fingerprint,
    recurring=RecurringSpec(task_type=REPO_MAP_TASK, params={}, cadence_hours=_cadence_hours()),
)
