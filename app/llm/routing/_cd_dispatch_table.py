# FILE: app/llm/routing/_cd_dispatch_table.py
"""
Command dispatch table — maps CanonicalIntent to stream handler configs.

Extracted from command_dispatch.py to reduce file size.
Each entry defines: intent, availability flag, handler, stage name, and
optional provider/model overrides.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
from app.llm.translation_routing import CanonicalIntent


def _get_scoping_handler():
    """Lazy import of project scoping handler."""
    try:
        from app.llm.project_scoping_stream import generate_project_scoping_stream
        return generate_project_scoping_stream
    except ImportError:
        return None


def build_dispatch_table(registry: Any) -> List[Dict[str, Any]]:
    """Build the dispatch table from the handler registry.

    Each entry is a dict with:
        intent: CanonicalIntent (or tuple of intents)
        available: bool
        handler: callable (the stream generator)
        stage: str (stage name for tracing)
        provider_override: Optional[callable] returning (provider, model)
        handler_kwargs_fn: Optional[callable] returning extra kwargs dict
        error_name: Optional[str] for unavailable error messages
        error_module: Optional[str] for unavailable error messages
    """
    import os
    from app.llm.translation_routing import _get_spec_gate_config, _get_critical_pipeline_config
    from app.llm.legacy_triggers import ARCHMAP_PROVIDER, ARCHMAP_MODEL

    table = [
        # --- Sandbox ---
        {
            "intent": CanonicalIntent.START_SANDBOX_ZOMBIE_SELF,
            "available": registry._SANDBOX_AVAILABLE,
            "handler": registry.generate_sandbox_stream,
            "stage": "sandbox",
            "provider_override": lambda: ("local", "sandbox_manager"),
        },
        # --- Update Architecture ---
        {
            "intent": CanonicalIntent.ARCHITECTURE_UPDATE_ATLAS_ONLY,
            "available": registry._LOCAL_TOOLS_AVAILABLE,
            "handler": registry.generate_update_architecture_stream,
            "stage": "update_architecture",
            "provider_override": lambda: ("local", "architecture_scanner"),
        },
        # --- Sandbox Structure Scan ---
        {
            "intent": CanonicalIntent.SCAN_SANDBOX_STRUCTURE,
            "available": registry._LOCAL_TOOLS_AVAILABLE,
            "handler": registry.generate_sandbox_structure_scan_stream,
            "stage": "sandbox_structure_scan",
            "provider_override": lambda: ("local", "sandbox_structure_scanner"),
        },
        # --- Full Architecture Map (ALL CAPS) ---
        {
            "intent": CanonicalIntent.ARCHITECTURE_MAP_WITH_FILES,
            "available": registry._LOCAL_TOOLS_AVAILABLE,
            "handler": registry.generate_full_architecture_map_stream,
            "stage": "full_architecture_map",
            "provider_override": lambda: (ARCHMAP_PROVIDER, ARCHMAP_MODEL),
        },
        # --- Architecture Map (lowercase, DB only) ---
        {
            "intent": CanonicalIntent.ARCHITECTURE_MAP_STRUCTURE_ONLY,
            "available": registry._LOCAL_TOOLS_AVAILABLE,
            "handler": registry.generate_local_architecture_map_stream,
            "stage": "architecture_map",
            "provider_override": lambda: (ARCHMAP_PROVIDER, ARCHMAP_MODEL),
        },
        # --- Weaver ---
        {
            "intent": CanonicalIntent.WEAVER_BUILD_SPEC,
            "available": registry._WEAVER_AVAILABLE,
            "handler": registry.generate_weaver_stream,
            "stage": "weaver",
            "needs_conversation_id": True,
        },
        # --- Spec Gate ---
        {
            "intent": CanonicalIntent.SEND_TO_SPEC_GATE,
            "available": registry._SPEC_GATE_STREAM_AVAILABLE,
            "handler": registry.generate_spec_gate_stream,
            "stage": "spec_gate",
            "provider_override": lambda: _get_spec_gate_config(),
            "needs_conversation_id": True,
        },
        # --- Unified Pipeline (v5.4) ---
        {
            "intent": (
                CanonicalIntent.RUN_PIPELINE,
                CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB,
                CanonicalIntent.RUN_SEGMENT_LOOP,
            ),
            "available": registry._SEGMENT_LOOP_AVAILABLE,
            "handler": registry.generate_segment_loop_stream,
            "stage": "pipeline",
            "provider_override": lambda: _get_critical_pipeline_config(),
            "handler_kwargs_fn": lambda req, tr: {"conversation_id": str(req.project_id)},
            "skip_message": True,
            "error_name": "PIPELINE HANDLER",
            "error_module": "app/orchestrator/segment_loop_stream.py",
        },
        # --- Implement Segments (v5.13) ---
        {
            "intent": CanonicalIntent.IMPLEMENT_SEGMENTS,
            "available": registry._SEGMENT_LOOP_AVAILABLE,
            "handler": registry.generate_segment_loop_stream,
            "stage": "pipeline_implement",
            "provider_override": lambda: _get_critical_pipeline_config(),
            "handler_kwargs_fn": lambda req, tr: {"conversation_id": str(req.project_id), "implement_only": True},
            "skip_message": True,
        },
        # --- Refactor Codebase (v1.9) ---
        {
            "intent": CanonicalIntent.REFACTOR_CODEBASE,
            "available": registry._REFACTOR_AVAILABLE,
            "handler": registry.generate_refactor_stream,
            "stage": "refactor_loop",
            "no_standard_args": True,
        },
        # --- Implementer (was Overwatcher — v3.2 merged into single stage) ---
        # v3.2-fix: Added provider_override so stage trace correctly shows
        # the implementer's actual model (anthropic/claude-sonnet) instead
        # of the stream router's default (google/gemini-flash).
        {
            "intent": CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES,
            "available": registry._OVERWATCHER_AVAILABLE,
            "handler": registry.generate_overwatcher_stream,
            "stage": "implementer",
            "needs_conversation_id": True,
            "provider_override": lambda: (
                os.getenv("IMPLEMENTER_PROVIDER", "anthropic"),
                os.getenv("IMPLEMENTER_MODEL", "claude-sonnet-4-6"),
            ),
        },
        # --- RAG Codebase Query ---
        {
            "intent": CanonicalIntent.RAG_CODEBASE_QUERY,
            "available": registry._RAG_STREAM_AVAILABLE,
            "handler": registry.generate_rag_query_stream,
            "stage": "rag_query",
            "provider_override": lambda: ("local", "rag_answerer"),
            "error_name": "RAG Query Handler",
            "error_module": "app/llm/rag_stream.py",
        },
        # --- Embedding Status ---
        {
            "intent": CanonicalIntent.EMBEDDING_STATUS,
            "available": registry._EMBEDDING_STREAM_AVAILABLE,
            "handler": registry.generate_embedding_status_stream,
            "stage": "embedding_status",
            "provider_override": lambda: ("local", "embedding_manager"),
            "error_name": "Embedding Status Handler",
            "error_module": "app/llm/embedding_stream.py",
        },
        # --- Generate Embeddings ---
        {
            "intent": CanonicalIntent.GENERATE_EMBEDDINGS,
            "available": registry._EMBEDDING_STREAM_AVAILABLE,
            "handler": registry.generate_embeddings_stream,
            "stage": "generate_embeddings",
            "provider_override": lambda: ("local", "embedding_manager"),
            "error_name": "Generate Embeddings Handler",
            "error_module": "app/llm/embedding_stream.py",
        },
        # --- Filesystem Query ---
        {
            "intent": CanonicalIntent.FILESYSTEM_QUERY,
            "available": registry._LOCAL_TOOLS_AVAILABLE and registry.generate_filesystem_query_stream is not None,
            "handler": registry.generate_filesystem_query_stream,
            "stage": "filesystem_query",
            "provider_override": lambda: ("local", "filesystem_scanner"),
            "error_name": "Filesystem Query Handler",
            "error_module": "app/llm/local_tools/zobie_tools.py",
        },
        # --- Codebase Report ---
        {
            "intent": CanonicalIntent.CODEBASE_REPORT,
            "available": registry._LOCAL_TOOLS_AVAILABLE and registry.generate_codebase_report_stream is not None,
            "handler": registry.generate_codebase_report_stream,
            "stage": "codebase_report",
            "provider_override": lambda: ("local", "codebase_report"),
            "error_name": "Codebase Report Handler",
            "error_module": "app/llm/local_tools/zobie_tools.py",
        },
        # --- Latest Architecture Map ---
        {
            "intent": CanonicalIntent.LATEST_ARCHITECTURE_MAP,
            "available": registry._LOCAL_TOOLS_AVAILABLE and registry.generate_latest_architecture_map_stream is not None,
            "handler": registry.generate_latest_architecture_map_stream,
            "stage": "latest_architecture_map",
            "provider_override": lambda: ("local", "latest_report_resolver"),
            "error_name": "Latest Architecture Map Handler",
            "error_module": "app/llm/local_tools/zobie_tools.py",
        },
        # --- Latest Codebase Report Full ---
        {
            "intent": CanonicalIntent.LATEST_CODEBASE_REPORT_FULL,
            "available": registry._LOCAL_TOOLS_AVAILABLE and registry.generate_latest_codebase_report_full_stream is not None,
            "handler": registry.generate_latest_codebase_report_full_stream,
            "stage": "latest_codebase_report_full",
            "provider_override": lambda: ("local", "latest_report_resolver"),
            "error_name": "Latest Codebase Report Handler",
            "error_module": "app/llm/local_tools/zobie_tools.py",
        },
        # --- Memory Ingest (v2.1) ---
        {
            "intent": CanonicalIntent.MEMORY_INGEST,
            "available": registry._WEAVER_AVAILABLE,
            "handler": registry.generate_weaver_stream,
            "stage": "memory_ingest",
            "provider_override": lambda: ("local", "ingest_pipeline"),
            "needs_conversation_id": True,
            "error_name": "Memory Ingest Handler",
            "error_module": "app/memory/ingest/pipeline.py",
        },
        # --- Memory Store (v2.1) ---
        {
            "intent": CanonicalIntent.MEMORY_STORE,
            "available": registry._WEAVER_AVAILABLE,
            "handler": registry.generate_weaver_stream,
            "stage": "memory_store",
            "provider_override": lambda: ("local", "preference_capture"),
            "needs_conversation_id": True,
            "error_name": "Memory Store Handler",
            "error_module": "app/memory/domains/preference_capture.py",
        },
        # --- Deep Research (v2.1) ---
        {
            "intent": CanonicalIntent.DEEP_RESEARCH,
            "available": registry._DEEP_RESEARCH_STREAM_AVAILABLE,
            "handler": registry.generate_deep_research_stream,
            "stage": "deep_research",
            "provider_override": lambda: ("multi", "deep_research_engine"),
            "handler_kwargs_fn": lambda req, tr: {
                "extracted_query": (tr.extracted_context.get('extracted_query') if hasattr(tr, 'extracted_context') else None),
            },
            "error_name": "Deep Research Handler",
            "error_module": "app/llm/deep_research_stream.py",
        },
        # --- Web Search (v2.1) ---
        {
            "intent": CanonicalIntent.WEB_SEARCH,
            "available": registry._WEB_SEARCH_STREAM_AVAILABLE,
            "handler": registry.generate_web_search_stream,
            "stage": "web_search",
            "provider_override": lambda: ("brave", "web_search"),
            "handler_kwargs_fn": lambda req, tr: {
                "extracted_query": (tr.extracted_context.get('extracted_query') if hasattr(tr, 'extracted_context') else None),
            },
            "error_name": "Web Search Handler",
            "error_module": "app/llm/web_search_stream.py",
        },
        # --- Multi-File Search (v1.1) ---
        {
            "intent": CanonicalIntent.MULTI_FILE_SEARCH,
            "available": registry._WEAVER_AVAILABLE,
            "handler": registry.generate_weaver_stream,
            "stage": "multi_file_search",
            "needs_conversation_id": True,
            "error_name": "Multi-File Search Handler",
            "error_module": "app/llm/weaver_stream.py",
        },
        # --- Multi-File Refactor (v1.1) ---
        {
            "intent": CanonicalIntent.MULTI_FILE_REFACTOR,
            "available": registry._WEAVER_AVAILABLE,
            "handler": registry.generate_weaver_stream,
            "stage": "multi_file_refactor",
            "needs_conversation_id": True,
            "error_name": "Multi-File Refactor Handler",
            "error_module": "app/llm/weaver_stream.py",
        },
        # --- Image Generation (v3.1) ---
        {
            "intent": CanonicalIntent.GENERATE_IMAGE,
            "available": registry._IMAGE_GEN_STREAM_AVAILABLE,
            "handler": registry.generate_image_stream,
            "stage": "image_generation",
            "provider_override": lambda: (
                os.getenv("IMAGE_GEN_PROVIDER", "openai"),
                os.getenv("IMAGE_GEN_MODEL", "gpt-image-1.5"),
            ),
            "error_name": "Image Generation Handler",
            "error_module": "app/llm/image_router.py",
        },
        # --- Project Scoping (v2.2) ---
        {
            "intent": CanonicalIntent.PROJECT_SCOPE_START,
            "available": True,  # Always available — uses chat deep model
            "handler": _get_scoping_handler(),
            "stage": "project_scoping",
            "provider_override": lambda: (
                os.getenv("CHAT_DEEP_PROVIDER", "openai"),
                os.getenv("CHAT_DEEP_MODEL", "gpt-5.4"),
            ),
            "needs_conversation_id": True,
        },
    ]

    # v11.0: Extend with project registry entries
    try:
        from app.llm.routing._cd_project_registry import get_project_registry_entries
        table.extend(get_project_registry_entries(registry))
    except Exception as _pr_err:
        import logging as _log
        _log.getLogger(__name__).warning("[dispatch_table] Project registry entries unavailable: %s", _pr_err)

    return table


__all__ = ["build_dispatch_table"]
