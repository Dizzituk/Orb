# FILE: app/llm/weaver_stream_state.py
"""
Weaver Stream State (refactor target)

Contains:
- optional integration with spec_flow_state (checkpoint load/save)
- optional integration with specs service (load previous spec for UPDATE mode)

This module is defensive: if optional imports are missing, it degrades gracefully.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

# Optional: incremental helpers
try:
    from app.llm.weaver_incremental import (
        WeaverCheckpoint,
        extract_spec_core,
    )

    _INCREMENTAL_HELPERS_AVAILABLE = True
except Exception:
    WeaverCheckpoint = None  # type: ignore
    extract_spec_core = None  # type: ignore
    _INCREMENTAL_HELPERS_AVAILABLE = False

# Flow state management (optional)
try:
    from app.llm.spec_flow_state import (
        get_active_flow,
        set_flow_state,
        start_weaver_flow,
    )

    _FLOW_STATE_AVAILABLE = True
except Exception:
    get_active_flow = None  # type: ignore
    set_flow_state = None  # type: ignore
    start_weaver_flow = None  # type: ignore
    _FLOW_STATE_AVAILABLE = False

# Optional: spec retrieval (for incremental update prompts)
try:
    from app.specs.service import get_spec, get_spec_schema

    _SPECS_SERVICE_AVAILABLE = True
except Exception:
    try:
        from app.specs import get_spec, get_spec_schema  # type: ignore

        _SPECS_SERVICE_AVAILABLE = True
    except Exception:
        get_spec = None  # type: ignore
        get_spec_schema = None  # type: ignore
        _SPECS_SERVICE_AVAILABLE = False

logger = logging.getLogger(__name__)

WEAVER_ENABLE_INCREMENTAL = os.getenv("WEAVER_ENABLE_INCREMENTAL", "1").lower() in ("1", "true", "yes", "on")


def load_weaver_checkpoint(project_id: int) -> Optional["WeaverCheckpoint"]:
    """
    Load incremental checkpoint from flow state (if available).

    Stored under state.work_artifacts:
      - weaver_last_seen_message_id
      - weaver_last_weak_spots
      - weaver_last_spec_id (optional mirror of weaver_spec_id)
    """
    if not (WEAVER_ENABLE_INCREMENTAL and _FLOW_STATE_AVAILABLE and get_active_flow and _INCREMENTAL_HELPERS_AVAILABLE):
        return None

    try:
        state = get_active_flow(project_id)
        if not state:
            return None

        artifacts = getattr(state, "work_artifacts", {}) or {}
        last_seen = artifacts.get("weaver_last_seen_message_id")
        last_weak = artifacts.get("weaver_last_weak_spots") or []
        last_spec_id = artifacts.get("weaver_last_spec_id") or getattr(state, "weaver_spec_id", None)

        # Defensive typing
        if isinstance(last_seen, str) and last_seen.isdigit():
            last_seen = int(last_seen)
        if not isinstance(last_seen, int):
            last_seen = None
        if not isinstance(last_weak, list):
            last_weak = []

        return WeaverCheckpoint(  # type: ignore[call-arg]
            last_weaver_spec_id=last_spec_id,
            last_seen_message_id=last_seen,
            last_weak_spots=[str(x) for x in last_weak],
        )
    except Exception as e:
        logger.warning("[weaver] Failed to load checkpoint: %s", e)
        return None


def save_weaver_checkpoint(
    project_id: int,
    new_weaver_spec_id: str,
    last_seen_message_id: Optional[int],
    weak_spots: List[str],
) -> None:
    """
    Persist incremental checkpoint into flow state work_artifacts.

    This is strictly additive metadata.
    """
    if not (_FLOW_STATE_AVAILABLE and get_active_flow and set_flow_state):
        return

    try:
        state = get_active_flow(project_id)
        if not state:
            return

        artifacts = getattr(state, "work_artifacts", {}) or {}
        artifacts["weaver_last_spec_id"] = new_weaver_spec_id
        artifacts["weaver_last_seen_message_id"] = last_seen_message_id
        artifacts["weaver_last_weak_spots"] = list(weak_spots or [])
        state.work_artifacts = artifacts  # type: ignore[attr-defined]

        set_flow_state(state)
    except Exception as e:
        logger.warning("[weaver] Failed to save checkpoint: %s", e)


def register_weaver_flow(project_id: int, spec_id: str) -> None:
    """Register/update flow state for continuity (if available)."""
    if not (_FLOW_STATE_AVAILABLE and start_weaver_flow):
        return
    try:
        start_weaver_flow(project_id, spec_id)
    except Exception as e:
        logger.warning("[weaver] Failed to register flow state: %s", e)


def get_previous_spec_for_update(
    db: Session,
    checkpoint: "WeaverCheckpoint",
) -> Tuple[Optional[Dict], Optional[List[int]]]:
    """
    Load the previous spec core dict for UPDATE mode and return (core_dict, prior_source_message_ids).

    Returns (None, None) if unavailable.
    """
    if not (_SPECS_SERVICE_AVAILABLE and get_spec and get_spec_schema and _INCREMENTAL_HELPERS_AVAILABLE and extract_spec_core):
        return None, None
    if not checkpoint or not getattr(checkpoint, "last_weaver_spec_id", None):
        return None, None

    try:
        db_spec = get_spec(db, checkpoint.last_weaver_spec_id)  # type: ignore[misc]
        if not db_spec:
            return None, None

        spec_schema = get_spec_schema(db_spec)  # type: ignore[misc]
        core = extract_spec_core(spec_schema)

        prior_ids: Optional[List[int]] = None
        prov = getattr(spec_schema, "provenance", None)
        if prov is not None:
            prior_ids = list(getattr(prov, "source_message_ids", []) or [])

        return core, prior_ids
    except Exception as e:
        logger.warning("[weaver] Failed to load previous spec for update: %s", e)
        return None, None
