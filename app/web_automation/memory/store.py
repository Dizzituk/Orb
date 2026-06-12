# FILE: app/web_automation/memory/store.py
# Purpose: Filesystem persistence for flow definitions.
# Called-by: app.web_automation.memory
# Depends-on: app.web_automation.memory.models
# Last-renovated: 2026-06-11
"""
Filesystem persistence for flow definitions.

One JSON file per (platform, task) under app/web_automation/memory/data/.
Filename pattern: '{platform}__{task}.json'. Two underscores separate the
two fields so platform / task names containing single underscores survive.

Why JSON files rather than a DB row:
  * Flow definitions are diffable in git — easy to review what the agent
    learned, easy to roll back a regression after a Meta UI redesign.
  * Per-flow file contention is zero — independent platform/task writes
    never collide.
  * No schema migrations: pydantic validates at load time and ignores
    unknown fields, so future fields don't break old files.

Stat updates (success_count, failure_count, last_run_at) are written by
the runner via update_flow_stats. They live alongside the flow definition
so a single file is the full record.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from app.web_automation.memory.models import Flow

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

# Filenames must be filesystem-safe and round-trip cleanly through the
# split. Letters, digits, dot, dash, underscore allowed.
_NAME_OK = re.compile(r"^[A-Za-z0-9._-]+$")


# =============================================================================
# PATH RESOLUTION
# =============================================================================

def _validate_name(value: str, kind: str) -> None:
    if not value or not _NAME_OK.match(value):
        raise ValueError(
            f"Invalid {kind} '{value}': only letters, digits, '.', '-', '_' allowed."
        )


def _flow_path(platform: str, task: str) -> Path:
    _validate_name(platform, "platform")
    _validate_name(task, "task")
    return DATA_DIR / f"{platform}__{task}.json"


# =============================================================================
# READ / WRITE
# =============================================================================

def load_flow(platform: str, task: str) -> Optional[Flow]:
    """Return the saved flow for (platform, task), or None if not stored."""
    try:
        path = _flow_path(platform, task)
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        return Flow.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(
            "[flow_store] Corrupt flow file %s: %s — ignoring", path, exc,
        )
        return None


def save_flow(flow: Flow) -> Path:
    """Persist a flow definition. Overwrites any existing file for that key."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _flow_path(flow.platform, flow.task)
    path.write_text(flow.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "[flow_store] Saved %s/%s v%s with %d step(s)",
        flow.platform, flow.task, flow.version, len(flow.steps),
    )
    return path


def list_flows(platform: Optional[str] = None) -> List[Tuple[str, str]]:
    """List all stored flows as (platform, task) tuples; optional filter."""
    if not DATA_DIR.exists():
        return []
    out: List[Tuple[str, str]] = []
    for f in sorted(DATA_DIR.glob("*__*.json")):
        stem = f.stem
        if "__" not in stem:
            continue
        p, t = stem.split("__", 1)
        if platform and p != platform:
            continue
        out.append((p, t))
    return out


# =============================================================================
# STATS
# =============================================================================

def update_flow_stats(
    platform: str,
    task: str,
    *,
    success: bool,
    failure_reason: Optional[str] = None,
) -> None:
    """Increment success/failure counters and stamp last_run_at."""
    flow = load_flow(platform, task)
    if flow is None:
        return
    if success:
        flow.success_count += 1
        flow.last_failure_reason = None
    else:
        flow.failure_count += 1
        if failure_reason:
            flow.last_failure_reason = failure_reason[:500]
    flow.last_run_at = datetime.now(timezone.utc).isoformat()
    save_flow(flow)
