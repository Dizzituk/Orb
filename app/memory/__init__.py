# FILE: app/memory/__init__.py
# Purpose: ASTRA Unified Memory System.
# Called-by: app.bridge.capability_layer, app.builds.pipeline_bridge, app.builds.stage_hooks, app.db (+44 more)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
ASTRA Unified Memory System.

Public interface:
    from app.memory.router import memory_router
    from app.memory.startup import init_memory_system

The memory_router singleton is the primary entry point for all
memory queries and stores. Call init_memory_system() at boot
to register domain stores.

Domain stores:
    architecture — arch_code_chunks (read-only, populated by archmap scanner)
    knowledge    — rag_entries (knowledge, context, decision, ingested, redirect)
    preference   — astra_preferences (Job 4, wraps app/astra_memory/)
    confidence   — confidence_scores (Job 5)
"""
