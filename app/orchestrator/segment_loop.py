# FILE: app/orchestrator/segment_loop.py
"""
Core orchestrator segment loop.

Reads a segment manifest, processes segments in dependency order through
the existing pipeline (Critical Pipeline → Critique → Overwatcher →
Implementer), threads evidence forward between segments, and tracks
state for crash recovery.

Phase 2 of Pipeline Segmentation.

Evidence collection is inlined here rather than in a separate module —
the functions are small, tightly coupled to loop state, and have no
external reuse case.

v1.0 (2026-02-08): Initial implementation
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from app.orchestrator._segment_loop_utils import SEGMENT_LOOP_BUILD_ID, _build_sibling_interfaces, _clear_stale_arch_versions, _find_latest_arch, _load_source_file_evidence, _now_iso, collect_segment_outputs, is_segment_blocked
from app.orchestrator._segment_loop_utils import _is_facade_segment, _save_execution_trace, build_evidence_bundle, build_segment_context, can_execute_segment, mark_dependents_blocked, unblock_recovered_segments, verify_contracts_fulfilled
from app.orchestrator._segment_loop_utils import update_segment_status
from app.orchestrator._segment_loop_utils import run_segment_through_pipeline
from app.orchestrator._segment_loop_utils import run_segmented_job

logger = logging.getLogger(__name__)
print(f"[SEGMENT_LOOP_LOADED] BUILD_ID={SEGMENT_LOOP_BUILD_ID}")


# --- Internal imports ---
from app.pot_spec.grounded.segment_schemas import (
    SegmentManifest,
    SegmentSpec,
    SegmentStatus,
    InterfaceContract,
)
from app.orchestrator.segment_state import (
    JobState,
    SegmentState,
    load_or_init_state,
    save_state,
    get_job_dir,
)

# --- Pipeline stage imports (optional — graceful degradation) ---
try:
    from app.llm.critical_pipeline_stream import generate_critical_pipeline_stream
    _CRITICAL_PIPELINE_AVAILABLE = True
except ImportError:
    _CRITICAL_PIPELINE_AVAILABLE = False

try:
    from app.overwatcher.overwatcher import run_overwatcher, run_pot_spec_execution
    _OVERWATCHER_AVAILABLE = True
except ImportError:
    _OVERWATCHER_AVAILABLE = False

try:
    from app.overwatcher.implementer import run_implementer
    _IMPLEMENTER_AVAILABLE = True
except ImportError:
    _IMPLEMENTER_AVAILABLE = False

try:
    from app.overwatcher.architecture_executor import run_architecture_execution
    from app.overwatcher.spec_resolution import resolve_latest_spec, ResolvedSpec
    from app.llm.overwatcher_stream import create_overwatcher_llm_fn
    _ARCH_EXECUTOR_AVAILABLE = True
except ImportError as _ae:
    _ARCH_EXECUTOR_AVAILABLE = False
    logger.warning("[SEGMENT_LOOP] Architecture executor not available: %s", _ae)
    print(f"[SEGMENT_LOOP] [WARNING] Architecture executor import failed: {_ae}")

# v5.12: Interface Reconciliation (Option A — prevent naming drift)
try:
    from app.orchestrator.interface_reconciliation import (
        read_dependency_interfaces_from_sandbox,
        inject_reconciliation_into_architecture,
    )
    _RECONCILIATION_AVAILABLE = True
except ImportError:
    _RECONCILIATION_AVAILABLE = False
    logger.debug("[SEGMENT_LOOP] Interface reconciliation not available")


# Type alias for progress callback
ProgressCallback = Optional[Callable[[str], None]]


# =============================================================================
# DEPENDENCY CHECKING
# =============================================================================


# =============================================================================
# STATE UPDATES
# =============================================================================


# =============================================================================
# EVIDENCE COLLECTION & THREADING
# =============================================================================


# =============================================================================
# SEGMENT CONTEXT BUILDER
# =============================================================================


# =============================================================================
# CORE ORCHESTRATOR LOOP
# =============================================================================
