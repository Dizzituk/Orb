# FILE: app/llm/routing/handler_registry.py
"""
Centralized handler imports and availability flags for stream routing.

v1.0 (2026-01-20): Extracted from stream_router.py for modularity.

This module provides:
- All optional stream handler imports with graceful fallbacks
- Availability flags (_HANDLER_AVAILABLE pattern)
- `log_handler_availability()` for debugging
- `get_handler_status()` for programmatic access

Design principle: Import once, check availability everywhere.
"""

from __future__ import annotations

import logging
import traceback
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)

# =============================================================================
# CORE STREAM HANDLERS (from stream_handlers.py - always available)
# =============================================================================

try:
    from app.llm.stream_handlers import (
        generate_sse_stream,
        generate_sandbox_stream,
        generate_introspection_stream,
        generate_feedback_stream,
        generate_confirmation_stream,
    )
    _CORE_HANDLERS_AVAILABLE = True
except ImportError as e:
    logger.error(f"[handler_registry] CRITICAL: Core stream handlers failed to import: {e}")
    _CORE_HANDLERS_AVAILABLE = False
    generate_sse_stream = None
    generate_sandbox_stream = None
    generate_introspection_stream = None
    generate_feedback_stream = None
    generate_confirmation_stream = None

# =============================================================================
# HIGH-STAKES STREAM
# =============================================================================

try:
    from app.llm.high_stakes_stream import generate_high_stakes_critique_stream
    _HIGH_STAKES_AVAILABLE = True
except ImportError:
    _HIGH_STAKES_AVAILABLE = False
    generate_high_stakes_critique_stream = None

# =============================================================================
# ASTRA CAPABILITY LAYER
# =============================================================================

try:
    from app.capabilities import get_capability_context
    _CAPABILITIES_AVAILABLE = True
    print("[handler_registry] ASTRA capability layer loaded successfully")
except ImportError:
    _CAPABILITIES_AVAILABLE = False
    get_capability_context = None

# =============================================================================
# STAGE TRACING
# =============================================================================

try:
    from app.llm.stage_trace import StageTrace, log_model_resolution, get_env_model_audit
    _STAGE_TRACE_AVAILABLE = True
except ImportError:
    _STAGE_TRACE_AVAILABLE = False
    StageTrace = None
    log_model_resolution = None
    get_env_model_audit = None

# =============================================================================
# WEAVER
# =============================================================================

try:
    from app.llm.weaver_stream import generate_weaver_stream
    _WEAVER_AVAILABLE = True
except ImportError as e:
    print(f"[WEAVER_IMPORT_ERROR] Failed to import weaver_stream: {e}")
    traceback.print_exc()
    _WEAVER_AVAILABLE = False
    generate_weaver_stream = None
except Exception as e:
    print(f"[WEAVER_IMPORT_ERROR] Unexpected error importing weaver_stream: {e}")
    traceback.print_exc()
    _WEAVER_AVAILABLE = False
    generate_weaver_stream = None

# =============================================================================
# SPEC GATE
# =============================================================================

try:
    from app.llm.spec_gate_stream import generate_spec_gate_stream
    _SPEC_GATE_STREAM_AVAILABLE = True
except ImportError:
    _SPEC_GATE_STREAM_AVAILABLE = False
    generate_spec_gate_stream = None

# =============================================================================
# CRITICAL PIPELINE
# =============================================================================

try:
    from app.llm.critical_pipeline_stream import generate_critical_pipeline_stream
    _CRITICAL_PIPELINE_AVAILABLE = True
except ImportError:
    _CRITICAL_PIPELINE_AVAILABLE = False
    generate_critical_pipeline_stream = None

# =============================================================================
# OVERWATCHER
# =============================================================================

try:
    from app.llm.overwatcher_stream import generate_overwatcher_stream
    _OVERWATCHER_AVAILABLE = True
except ImportError:
    _OVERWATCHER_AVAILABLE = False
    generate_overwatcher_stream = None

# =============================================================================
# FLOW STATE
# =============================================================================

try:
    from app.llm.spec_flow_state import (
        get_active_flow,
        SpecFlowStage,
        check_weaver_answer_keywords,
        capture_weaver_answers,
        get_weaver_design_state,
    )
    _FLOW_STATE_AVAILABLE = True
except ImportError:
    _FLOW_STATE_AVAILABLE = False
    get_active_flow = None
    SpecFlowStage = None
    check_weaver_answer_keywords = None
    capture_weaver_answers = None
    get_weaver_design_state = None

# =============================================================================
# SPEC SERVICE
# =============================================================================

try:
    from app.specs.service import get_latest_validated_spec
    _SPEC_SERVICE_AVAILABLE = True
except ImportError:
    _SPEC_SERVICE_AVAILABLE = False
    get_latest_validated_spec = None

# =============================================================================
# LOCAL TOOLS (Zobie, Architecture Maps, etc.)
# =============================================================================

try:
    from app.llm.local_tools.zobie_tools import (
        generate_local_architecture_map_stream,
        generate_full_architecture_map_stream,
        generate_local_zobie_map_stream,
        generate_update_architecture_stream,
        generate_sandbox_structure_scan_stream,
        generate_latest_architecture_map_stream,
        generate_latest_codebase_report_full_stream,
        generate_filesystem_query_stream,
        generate_codebase_report_stream,
    )
    _LOCAL_TOOLS_AVAILABLE = True
    print("[handler_registry] Local tools loaded successfully")
except ImportError as e:
    _LOCAL_TOOLS_AVAILABLE = False
    print(f"[LOCAL_TOOLS_IMPORT_ERROR] Failed to import local tools: {e}")
    traceback.print_exc()
    generate_local_architecture_map_stream = None
    generate_full_architecture_map_stream = None
    generate_local_zobie_map_stream = None
    generate_update_architecture_stream = None
    generate_sandbox_structure_scan_stream = None
    generate_latest_architecture_map_stream = None
    generate_latest_codebase_report_full_stream = None
    generate_filesystem_query_stream = None
    generate_codebase_report_stream = None
except Exception as e:
    _LOCAL_TOOLS_AVAILABLE = False
    print(f"[LOCAL_TOOLS_IMPORT_ERROR] Unexpected error: {e}")
    traceback.print_exc()
    generate_local_architecture_map_stream = None
    generate_full_architecture_map_stream = None
    generate_local_zobie_map_stream = None
    generate_update_architecture_stream = None
    generate_sandbox_structure_scan_stream = None
    generate_latest_architecture_map_stream = None
    generate_latest_codebase_report_full_stream = None
    generate_filesystem_query_stream = None
    generate_codebase_report_stream = None

# =============================================================================
# SANDBOX
# =============================================================================

try:
    from app.sandbox import handle_sandbox_prompt
    _SANDBOX_AVAILABLE = True
except ImportError:
    _SANDBOX_AVAILABLE = False
    handle_sandbox_prompt = None

# =============================================================================
# RAG STREAM
# =============================================================================

try:
    from app.llm.rag_stream import generate_rag_query_stream
    _RAG_STREAM_AVAILABLE = True
    print("[handler_registry] RAG stream handler loaded successfully")
except ImportError as e:
    _RAG_STREAM_AVAILABLE = False
    print(f"[handler_registry] WARNING: RAG stream import failed: {e}")
    generate_rag_query_stream = None

# =============================================================================
# EMBEDDING STREAM
# =============================================================================

try:
    from app.llm.embedding_stream import (
        generate_embedding_status_stream,
        generate_embeddings_stream,
    )
    _EMBEDDING_STREAM_AVAILABLE = True
    print("[handler_registry] Embedding stream handlers loaded successfully")
except ImportError as e:
    _EMBEDDING_STREAM_AVAILABLE = False
    print(f"[handler_registry] WARNING: Embedding stream import failed: {e}")
    generate_embedding_status_stream = None
    generate_embeddings_stream = None

# =============================================================================
# INTROSPECTION
# =============================================================================

try:
    from app.introspection.chat_integration import detect_log_intent
    _INTROSPECTION_AVAILABLE = True
except ImportError:
    _INTROSPECTION_AVAILABLE = False
    detect_log_intent = None


# =============================================================================
# PUBLIC API: Status & Debugging
# =============================================================================

def log_handler_availability() -> None:
    """Log availability of all handlers for debugging."""
    print(f"[HANDLER_STATUS] CoreHandlers: {_CORE_HANDLERS_AVAILABLE}")
    print(f"[HANDLER_STATUS] HighStakes: {_HIGH_STAKES_AVAILABLE}")
    print(f"[HANDLER_STATUS] Capabilities: {_CAPABILITIES_AVAILABLE}")
    print(f"[HANDLER_STATUS] StageTrace: {_STAGE_TRACE_AVAILABLE}")
    print(f"[HANDLER_STATUS] Weaver: {_WEAVER_AVAILABLE}")
    print(f"[HANDLER_STATUS] SpecGateStream: {_SPEC_GATE_STREAM_AVAILABLE}")
    print(f"[HANDLER_STATUS] CriticalPipeline: {_CRITICAL_PIPELINE_AVAILABLE}")
    print(f"[HANDLER_STATUS] Overwatcher: {_OVERWATCHER_AVAILABLE}")
    print(f"[HANDLER_STATUS] FlowState: {_FLOW_STATE_AVAILABLE}")
    print(f"[HANDLER_STATUS] SpecService: {_SPEC_SERVICE_AVAILABLE}")
    print(f"[HANDLER_STATUS] LocalTools: {_LOCAL_TOOLS_AVAILABLE}")
    print(f"[HANDLER_STATUS] Sandbox: {_SANDBOX_AVAILABLE}")
    print(f"[HANDLER_STATUS] RAGStream: {_RAG_STREAM_AVAILABLE}")
    print(f"[HANDLER_STATUS] EmbeddingStream: {_EMBEDDING_STREAM_AVAILABLE}")
    print(f"[HANDLER_STATUS] Introspection: {_INTROSPECTION_AVAILABLE}")


def get_handler_status() -> dict:
    """Get handler availability as a dictionary."""
    return {
        "core_handlers": _CORE_HANDLERS_AVAILABLE,
        "high_stakes": _HIGH_STAKES_AVAILABLE,
        "capabilities": _CAPABILITIES_AVAILABLE,
        "stage_trace": _STAGE_TRACE_AVAILABLE,
        "weaver": _WEAVER_AVAILABLE,
        "spec_gate_stream": _SPEC_GATE_STREAM_AVAILABLE,
        "critical_pipeline": _CRITICAL_PIPELINE_AVAILABLE,
        "overwatcher": _OVERWATCHER_AVAILABLE,
        "flow_state": _FLOW_STATE_AVAILABLE,
        "spec_service": _SPEC_SERVICE_AVAILABLE,
        "local_tools": _LOCAL_TOOLS_AVAILABLE,
        "sandbox": _SANDBOX_AVAILABLE,
        "rag_stream": _RAG_STREAM_AVAILABLE,
        "embedding_stream": _EMBEDDING_STREAM_AVAILABLE,
        "introspection": _INTROSPECTION_AVAILABLE,
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Availability flags
    "_CORE_HANDLERS_AVAILABLE",
    "_HIGH_STAKES_AVAILABLE",
    "_CAPABILITIES_AVAILABLE",
    "_STAGE_TRACE_AVAILABLE",
    "_WEAVER_AVAILABLE",
    "_SPEC_GATE_STREAM_AVAILABLE",
    "_CRITICAL_PIPELINE_AVAILABLE",
    "_OVERWATCHER_AVAILABLE",
    "_FLOW_STATE_AVAILABLE",
    "_SPEC_SERVICE_AVAILABLE",
    "_LOCAL_TOOLS_AVAILABLE",
    "_SANDBOX_AVAILABLE",
    "_RAG_STREAM_AVAILABLE",
    "_EMBEDDING_STREAM_AVAILABLE",
    "_INTROSPECTION_AVAILABLE",
    # Core handlers
    "generate_sse_stream",
    "generate_sandbox_stream",
    "generate_introspection_stream",
    "generate_feedback_stream",
    "generate_confirmation_stream",
    # High-stakes
    "generate_high_stakes_critique_stream",
    # Capabilities
    "get_capability_context",
    # Stage trace
    "StageTrace",
    "log_model_resolution",
    "get_env_model_audit",
    # Weaver
    "generate_weaver_stream",
    # Spec Gate
    "generate_spec_gate_stream",
    # Critical Pipeline
    "generate_critical_pipeline_stream",
    # Overwatcher
    "generate_overwatcher_stream",
    # Flow state
    "get_active_flow",
    "SpecFlowStage",
    "check_weaver_answer_keywords",
    "capture_weaver_answers",
    "get_weaver_design_state",
    # Spec service
    "get_latest_validated_spec",
    # Local tools
    "generate_local_architecture_map_stream",
    "generate_full_architecture_map_stream",
    "generate_local_zobie_map_stream",
    "generate_update_architecture_stream",
    "generate_sandbox_structure_scan_stream",
    "generate_latest_architecture_map_stream",
    "generate_latest_codebase_report_full_stream",
    "generate_filesystem_query_stream",
    "generate_codebase_report_stream",
    # Sandbox
    "handle_sandbox_prompt",
    # RAG
    "generate_rag_query_stream",
    # Embedding
    "generate_embedding_status_stream",
    "generate_embeddings_stream",
    # Introspection
    "detect_log_intent",
    # Status functions
    "log_handler_availability",
    "get_handler_status",
]
