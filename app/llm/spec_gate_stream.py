# FILE: app/llm/spec_gate_stream.py
"""
Spec Gate streaming handler for ASTRA command flow.

v2.4 (2026-02-03): POT SPEC PERSISTENCE FIX
- After persist_spec() succeeds, overwrite content_markdown with actual spot_markdown
- Fixes: spec_to_markdown() only produced ~691-char generic header, losing POT spec content
- The full POT markdown (with ## Change / ## Skip sections) is now persisted to DB
- Enables: Overwatcher POT detection, Critical Pipeline full spec context
- Uses new specs_service.update_spec_content_markdown() helper

v2.3 (2026-02-01): VISION CONTEXT FLOW FIX
- Added _get_weaver_vision_context_from_flow() to extract vision context from flow state
- Vision context is now passed in constraints_hint["vision_context"] to spec_runner
- This completes the Weaver → SpecGate vision context data flow
- Enables classifier to identify USER-VISIBLE UI elements for intelligent refactor

v2.2 (2026-01-20): Caller-side persistence (v1.5 SpecGate support)
- Persist spec to DB after validation (caller responsibility, not SpecGate)
- SpecGate remains read-only in runtime
- Actual persistence status reflected in events

v2.1 (2026-01-04): Blocking Validation Support
- Shows blocking issues prominently when validation fails
- Clear distinction between blocking vs informational questions
- Better user guidance on how to resolve blocking issues

v2.0: Original implementation with Weaver spec validation.
"""

from __future__ import annotations

import json
import logging
import asyncio
from typing import Optional, Any, AsyncGenerator, Dict

from sqlalchemy.orm import Session
from app.llm._spec_gate_stream_utils import _USE_GROUNDED_SPEC_GATE, _get_weaver_job_description_from_flow, _get_weaver_vision_context_from_flow, _load_latest_weaver_spec_json, _resolve_spec_gate_model, _safe_json_event
from app.llm._spec_gate_stream_utils import generate_spec_gate_stream

logger = logging.getLogger(__name__)

# Centralized stage model selection (ENV-driven)
try:
    from app.llm.stage_models import get_spec_gate_config
except ImportError:
    get_spec_gate_config = None

# Import Spec Gate v2
try:
    from app.pot_spec.spec_gate_v2 import run_spec_gate_v2, SpecGateResult
    _SPEC_GATE_V2_AVAILABLE = True
except Exception as e:
    _SPEC_GATE_V2_AVAILABLE = False
    run_spec_gate_v2 = None
    SpecGateResult = None
    logger.warning("[spec_gate_stream] spec_gate_v2 module not available: %s", e)

# Import Spec Gate Grounded (Contract v1)
import os

try:
    from app.pot_spec.spec_gate_grounded import run_spec_gate_grounded
    _SPEC_GATE_GROUNDED_AVAILABLE = True
except Exception as e:
    _SPEC_GATE_GROUNDED_AVAILABLE = False
    run_spec_gate_grounded = None
    logger.warning("[spec_gate_stream] spec_gate_grounded module not available: %s", e)

# Import Spec Gate Persistence (v1.5 - persist after validation)
try:
    from app.pot_spec.spec_gate_persistence import (
        persist_spec,
        build_spec_schema,
        safe_summary_from_objective,
        write_spec_artifacts,
        compute_spec_hash,
    )
    _SPEC_PERSISTENCE_AVAILABLE = True
except Exception as e:
    _SPEC_PERSISTENCE_AVAILABLE = False
    persist_spec = None
    build_spec_schema = None
    safe_summary_from_objective = None
    write_spec_artifacts = None
    compute_spec_hash = None
    logger.warning("[spec_gate_stream] spec_gate_persistence module not available: %s", e)

# Flow state management (optional)
try:
    from app.llm.spec_flow_state import (
        get_active_flow,
        advance_to_spec_gate_questions,
        advance_to_spec_validated,
        advance_to_spec_segmented,
        cancel_flow,
    )
    _FLOW_STATE_AVAILABLE = True
except Exception:
    _FLOW_STATE_AVAILABLE = False
    get_active_flow = None
    advance_to_spec_gate_questions = None
    advance_to_spec_validated = None
    advance_to_spec_segmented = None
    cancel_flow = None

# Job service (optional)
try:
    from app.jobs.service import get_active_job_for_project
except Exception:
    get_active_job_for_project = None

# Spec service for Weaver draft
try:
    from app.specs import service as specs_service
except Exception:
    specs_service = None

# Memory service (optional)
try:
    from app.memory import service as memory_service
    from app.memory import schemas as memory_schemas
except Exception:
    memory_service = None
    memory_schemas = None

# Audit logger (optional)
try:
    from app.llm.audit_logger import RoutingTrace
except Exception:
    RoutingTrace = None


__all__ = ["generate_spec_gate_stream"]