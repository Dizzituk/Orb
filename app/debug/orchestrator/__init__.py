# FILE: app/debug/orchestrator/__init__.py
# Purpose: ASTRA Debug Orchestrator.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.debug.orchestrator.schemas
# Last-renovated: 2026-06-11
"""
ASTRA Debug Orchestrator.

Runs a parallel investigate → plan → execute → verify loop for Debug
Projects, replacing single-model serial debugging with multi-agent
decomposition.

Public API (wired in as phases land):
    run_orchestration(request: OrchestrationRequest) -> DebugResolution

See schemas.py for all typed contracts between phases.
"""
from __future__ import annotations

from app.debug.orchestrator.schemas import (
    OrchestrationRequest,
    OrchestrationPhase,
    OrchestrationEvent,
    SubagentRole,
    SubagentBrief,
    SubagentReport,
    DecompositionResult,
    DebugPlan,
    FixStep,
    BehaviourCheck,
    VerificationResult,
    IterationRecord,
    DebugResolution,
    StepStatus,
    Finding,
)

__all__ = [
    "OrchestrationRequest",
    "OrchestrationPhase",
    "OrchestrationEvent",
    "SubagentRole",
    "SubagentBrief",
    "SubagentReport",
    "DecompositionResult",
    "DebugPlan",
    "FixStep",
    "BehaviourCheck",
    "VerificationResult",
    "IterationRecord",
    "DebugResolution",
    "StepStatus",
    "Finding",
]
