# FILE: app/llm/overwatcher_stream.py
"""
Overwatcher streaming handler for ASTRA command flow.

v3.7 (2026-01-04): LLM Wiring Fix - Use call_llm_text
- FIXED: Use call_llm_text from streaming.py instead of non-existent provider functions
- Removed broken imports: stream_openai_response, stream_anthropic_response, stream_gemini_response
- Simplified availability checking - uses get_available_streaming_provider() at runtime
- call_llm_text provides: retry logic, non-streaming fallback, unified provider routing

v3.6 (2026-01-04): LLM Wiring Fix (BROKEN - referenced non-existent functions)
v3.5 (2026-01-04): Job ID Resolution + Evidence Building Fixes
v3.4: Artifact Binding Support
v3.3: Token event 'text' field fix
v3.0: LLM function wiring
"""

import json
import logging
import asyncio
import os
import glob
from datetime import datetime
from typing import Optional, Any, AsyncGenerator, Callable, List, Dict

from sqlalchemy.orm import Session
from app.llm._overwatcher_stream_utils_4 import _build_evidence_bundle, _get_overwatcher_provider_model, _load_artifact_bindings, _resolve_job_id, _validate_artifact_bindings, sse_error, sse_event, sse_token
from app.llm._overwatcher_stream_utils_5 import ARTIFACT_ROOT, create_overwatcher_llm_fn, generate_overwatcher_stream

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports with graceful fallbacks
# ---------------------------------------------------------------------------

try:
    from app.memory import service as memory_service, schemas as memory_schemas
except ImportError:
    memory_service = None
    memory_schemas = None

try:
    from app.specs import service as specs_service
except ImportError:
    specs_service = None

try:
    from app.jobs.service import (
        get_active_job_for_project,
        get_work_artifacts,
        mark_job_complete,
        mark_job_failed,
        get_job_for_spec,  # NEW: Get job by spec_id
    )
except ImportError:
    get_active_job_for_project = None
    get_work_artifacts = None
    mark_job_complete = None
    mark_job_failed = None
    get_job_for_spec = None

try:
    from app.llm.audit_logger import RoutingTrace
except ImportError:
    RoutingTrace = None

try:
    from app.overwatcher.overwatcher_command import (
        run_overwatcher_command,
        OverwatcherCommandResult,
    )
    OVERWATCHER_AVAILABLE = True
except ImportError:
    run_overwatcher_command = None
    OverwatcherCommandResult = None
    OVERWATCHER_AVAILABLE = False

# v3.7: LLM availability is checked at runtime in create_overwatcher_llm_fn()
# No module-level streaming imports needed - call_llm_text handles all provider routing

try:
    from app.llm.stage_models import get_overwatcher_config
    STAGE_MODELS_AVAILABLE = True
except ImportError:
    get_overwatcher_config = None
    STAGE_MODELS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default artifact root


# ---------------------------------------------------------------------------
# Job ID Resolution (v3.5 NEW)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Artifact Binding Loading (v3.5 Enhanced)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Evidence Building (v3.5 NEW)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# LLM Call Function Factory (v3.7: Fixed - uses call_llm_text)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SSE Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main Stream Generator
# ---------------------------------------------------------------------------


__all__ = ["generate_overwatcher_stream", "create_overwatcher_llm_fn"]