# FILE: app/pipeline_v2/segmented/scheduler.py
# Purpose: Derek phase 5 — topological segment scheduler (deterministic waves from depends_on).
# Called-by: app.pipeline_v2.segmented.segmented_builder
# Depends-on: stdlib only
# Last-renovated: 2026-07-04
"""
The Rubik's-cube order, computed — never guessed.

Kahn's algorithm over depends_on with a deterministic tie-break (segment_id
sort) produces WAVES: every segment in wave N depends only on waves < N.
Execution today is strictly sequential (waves flattened in order —
ASTRA_SEGMENT_PARALLEL is a hard false until Derek's vLLM fleet exists),
but the wave structure is what future parallelism will consume unchanged.

Operates on plain segment dicts OR SegmentSpec objects (duck-typed via
segment_id / dependencies).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)


class SchedulingError(RuntimeError):
    """The dependency graph cannot be scheduled (cycle / unknown id)."""


def _sid(seg: Any) -> str:
    if isinstance(seg, dict):
        return str(seg.get("segment_id", ""))
    return str(getattr(seg, "segment_id", ""))


def _deps(seg: Any) -> List[str]:
    if isinstance(seg, dict):
        return list(seg.get("dependencies") or seg.get("depends_on") or [])
    return list(getattr(seg, "dependencies", None) or [])


def parallel_requested() -> bool:
    """ASTRA_SEGMENT_PARALLEL — hard default false; true logs a WARN and is
    IGNORED (true parallelism arrives with Derek's vLLM; not built now)."""
    val = os.getenv("ASTRA_SEGMENT_PARALLEL", "false").strip().lower() in ("1", "true", "yes")
    if val:
        logger.warning(
            "[scheduler] ASTRA_SEGMENT_PARALLEL=true requested but parallel "
            "execution is not built yet (arrives with Derek's vLLM fleet) — "
            "running sequentially."
        )
    return False


def schedule_waves(segments: Sequence[Any]) -> List[List[Any]]:
    """Kahn's algorithm -> waves, deterministic tie-break by segment_id.

    Raises SchedulingError on unknown dependency ids or cycles (segment
    validation upstream should have caught both — this is the last line).
    """
    by_id: Dict[str, Any] = {_sid(s): s for s in segments}
    if len(by_id) != len(segments):
        raise SchedulingError("duplicate segment_id in manifest")

    indegree: Dict[str, int] = {sid: 0 for sid in by_id}
    dependants: Dict[str, List[str]] = {sid: [] for sid in by_id}
    for seg in segments:
        sid = _sid(seg)
        for dep in _deps(seg):
            if dep not in by_id:
                raise SchedulingError(f"segment {sid} depends on unknown id {dep!r}")
            indegree[sid] += 1
            dependants[dep].append(sid)

    ready = sorted(sid for sid, d in indegree.items() if d == 0)
    waves: List[List[Any]] = []
    scheduled = 0
    while ready:
        waves.append([by_id[sid] for sid in ready])
        scheduled += len(ready)
        next_ready: List[str] = []
        for sid in ready:
            for child in dependants[sid]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    next_ready.append(child)
        ready = sorted(set(next_ready))

    if scheduled != len(by_id):
        stuck = sorted(sid for sid, d in indegree.items() if d > 0)
        raise SchedulingError(f"dependency cycle among segments: {stuck}")

    logger.info(
        "[scheduler] %d segments -> %d wave(s): %s",
        len(by_id), len(waves), [[_sid(s) for s in w] for w in waves],
    )
    return waves


def schedule_order(segments: Sequence[Any]) -> List[Any]:
    """Flattened sequential order (waves in order, id-sorted within a wave)."""
    return [seg for wave in schedule_waves(segments) for seg in wave]


__all__ = ["SchedulingError", "schedule_waves", "schedule_order", "parallel_requested"]
