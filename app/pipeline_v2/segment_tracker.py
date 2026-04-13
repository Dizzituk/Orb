# FILE: app/pipeline_v2/segment_tracker.py
"""
Reactive Per-Segment Tracking (Phase 4 Job 12)

Watches write_file tool calls and updates the active segment based on which
segment owns the path just written. This gives the llm_tools routing layer
(Phase 2 Job 8) explicit per-segment awareness without forcing the
agentic_builder into strict per-segment iteration — which would fight
GPT-5.4's cross-segment reasoning.

Design: the tracker is additive. Existing flow (single run_tool_loop with
all segments in context, path-based auto-routing) keeps working. The
tracker just adds an extra signal: it also calls set_tool_segment(seg)
explicitly when it can identify which segment a write belonged to. That
enables per-segment logging and between-segment verification hooks
without restructuring the main orchestration.

v1.0 (2026-04-12): Phase 4 Job 12, Approach B.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class SegmentProgress:
    segment_id: str
    target_id: Optional[str]
    files_in_scope: int
    files_written: Set[str] = field(default_factory=set)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def is_complete(self) -> bool:
        return self.files_in_scope > 0 and len(self.files_written) >= self.files_in_scope

    def completion_ratio(self) -> float:
        if self.files_in_scope == 0:
            return 1.0
        return len(self.files_written) / self.files_in_scope


class SegmentTracker:
    """Tracks per-segment progress across a single builder run.

    Registered as an on_tool_call observer in the agentic_builder flow.
    When a write_file tool call lands, the tracker:
      1. Identifies which segment owns the path (via manifest file_scope)
      2. Updates that segment's files_written set
      3. Calls set_tool_segment(seg) so llm_tools Phase 2 Job 8 gets
         explicit segment awareness (not just path-based auto-routing)
      4. Detects when a segment transitions from incomplete to complete
         and fires an optional on_segment_complete callback
    """

    def __init__(self, manifest: Any,
                 on_segment_start: Optional[Callable[[Any], None]] = None,
                 on_segment_complete: Optional[Callable[[Any], None]] = None):
        self.manifest = manifest
        self.on_segment_start = on_segment_start
        self.on_segment_complete = on_segment_complete

        # Build segment_id -> SegmentProgress index
        self.progress: Dict[str, SegmentProgress] = {}
        self._seg_by_id: Dict[str, Any] = {}
        self._active_segment_id: Optional[str] = None

        segments = getattr(manifest, "segments", None) or []
        for seg in segments:
            sid = getattr(seg, "segment_id", None) or (seg.get("segment_id") if isinstance(seg, dict) else None)
            if not sid:
                continue
            self._seg_by_id[sid] = seg
            scope = getattr(seg, "file_scope", None)
            if scope is None and isinstance(seg, dict):
                scope = seg.get("file_scope", [])
            self.progress[sid] = SegmentProgress(
                segment_id=sid,
                target_id=(getattr(seg, "target_id", None)
                           if not isinstance(seg, dict) else seg.get("target_id")),
                files_in_scope=len(scope or []),
            )

    def _find_owning_segment_id(self, path: str) -> Optional[str]:
        """Find which segment's file_scope owns `path`.
        Uses os.path.normpath on both sides, case-insensitive.
        """
        import os
        target = os.path.normpath(path).lower()
        for sid, seg in self._seg_by_id.items():
            scope = getattr(seg, "file_scope", None)
            if scope is None and isinstance(seg, dict):
                scope = seg.get("file_scope", [])
            for fpath in (scope or []):
                candidate = os.path.normpath(fpath).lower()
                if candidate == target:
                    return sid
                if target.endswith(os.sep + candidate) or target.endswith("/" + candidate.replace("\\", "/")):
                    return sid
        return None

    def on_tool_call(self, name: str, args: Dict[str, Any]) -> None:
        """Called for every tool invocation. Only acts on write_file."""
        if name != "write_file":
            return
        path = args.get("path", "")
        if not path:
            return
        sid = self._find_owning_segment_id(path)
        if sid is None:
            return

        # Update progress
        import time
        prog = self.progress.get(sid)
        if prog is None:
            return
        was_empty = len(prog.files_written) == 0
        was_incomplete = not prog.is_complete()
        prog.files_written.add(path)
        if was_empty:
            prog.started_at = time.monotonic()

        # Segment transition: activate this segment for the tool layer
        if self._active_segment_id != sid:
            self._active_segment_id = sid
            seg = self._seg_by_id[sid]
            try:
                from app.pipeline_v2.llm_tools import set_tool_segment
                set_tool_segment(seg)
            except Exception as e:
                logger.debug("[tracker] set_tool_segment failed: %s", e)
            logger.info(
                "[tracker] Job 12 activated segment: %s (target=%s, progress=%d/%d)",
                sid, prog.target_id, len(prog.files_written), prog.files_in_scope,
            )
            if self.on_segment_start:
                try:
                    self.on_segment_start(seg)
                except Exception as e:
                    logger.debug("[tracker] on_segment_start failed: %s", e)

        # Completion fire
        if was_incomplete and prog.is_complete():
            prog.completed_at = time.monotonic()
            logger.info(
                "[tracker] Job 12 segment COMPLETE: %s (%d files, %.1fs)",
                sid, len(prog.files_written),
                (prog.completed_at - (prog.started_at or prog.completed_at)),
            )
            if self.on_segment_complete:
                try:
                    self.on_segment_complete(self._seg_by_id[sid])
                except Exception as e:
                    logger.debug("[tracker] on_segment_complete failed: %s", e)

    def summary(self) -> str:
        completed = sum(1 for p in self.progress.values() if p.is_complete())
        return f"SegmentTracker(segments={completed}/{len(self.progress)} complete)"

    def progress_report(self) -> List[Dict[str, Any]]:
        return [
            {
                "segment_id": p.segment_id,
                "target_id": p.target_id,
                "files_written": len(p.files_written),
                "files_in_scope": p.files_in_scope,
                "complete": p.is_complete(),
                "completion_ratio": round(p.completion_ratio(), 2),
            }
            for p in self.progress.values()
        ]