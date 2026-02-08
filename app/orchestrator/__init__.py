# FILE: app/orchestrator/__init__.py
"""
Orchestrator package — segment loop execution for pipeline segmentation.

Phase 2 of Pipeline Segmentation. Consumes segment manifests produced by
SpecGate (Phase 1) and executes each segment through the pipeline in
dependency order with evidence threading and crash recovery.

Modules:
    segment_state       — State tracking models and persistence
    segment_loop        — Core orchestrator loop logic
    segment_loop_stream — SSE streaming handler for frontend integration
"""

from __future__ import annotations

ORCHESTRATOR_BUILD_ID = "2026-02-08-v1.0-phase2-orchestrator-segment-loop"
print(f"[ORCHESTRATOR_LOADED] BUILD_ID={ORCHESTRATOR_BUILD_ID}")

__all__ = [
    "ORCHESTRATOR_BUILD_ID",
]
