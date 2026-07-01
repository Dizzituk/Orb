# FILE: app/debug/orchestrator/schemas.py
# Purpose: Orchestrator schemas — typed contracts for every hand-off between phases.
# Called-by: app.debug.orchestrator, app.debug.orchestrator.behaviour_verifier, app.debug.orchestrator.chat_stream_bridge, app.debug.orchestrator.decomposer (+4 more)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Orchestrator schemas — typed contracts for every hand-off between phases.

The orchestrator runs a loop: decompose → investigate → plan → execute → verify.
Each phase consumes and produces typed objects defined here so that any phase
can be tested or swapped in isolation.

v1.0 (2026-04-13): Initial schemas for Debug Orchestrator v1.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Phase + role enums
# ---------------------------------------------------------------------------

class OrchestrationPhase(str, Enum):
    """Outer-loop phase the orchestrator is currently in."""
    DECOMPOSING = "decomposing"
    INVESTIGATING = "investigating"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING_CODE = "verifying_code"
    VERIFYING_BEHAVIOUR = "verifying_behaviour"
    RESOLVED = "resolved"
    FAILED = "failed"
    MAX_ITERATIONS = "max_iterations"
    CANCELLED = "cancelled"


class SubagentRole(str, Enum):
    """Role determines which tools a subagent can call."""
    INVESTIGATOR = "investigator"       # read-only: find root cause
    PATTERN_MATCHER = "pattern_matcher" # read-only: find similar code/fixes
    EXECUTOR = "executor"               # read + write: apply the fix
    CODE_VERIFIER = "code_verifier"     # read + run_command: tests/compile
    BEHAVIOUR_VERIFIER = "behaviour_verifier"  # adb + desktop control


class StepStatus(str, Enum):
    """Status of an individual step within a phase."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Input to the orchestrator
# ---------------------------------------------------------------------------

class OrchestrationRequest(BaseModel):
    """Top-level request to run the debug orchestrator."""
    project_id: str = Field(..., description="Debug Project ID from project_service")
    bug_list: str = Field(..., description="Free-text bug list (usually project.description)")
    target_project: Optional[str] = Field(
        None,
        description="Project ID from pipeline_v2.target_registry. If None, decomposer infers."
    )
    max_iterations: int = Field(3, ge=1, le=10)
    max_subagents_parallel: int = Field(5, ge=1, le=20)
    enable_behaviour_verify: bool = Field(True, description="Run adb-driven scenario checks after code verify")


# ---------------------------------------------------------------------------
# Decomposition: bug list → investigation briefs
# ---------------------------------------------------------------------------

class SubagentBrief(BaseModel):
    """A single scoped task handed to a subagent."""
    brief_id: str = Field(..., description="Stable ID (e.g. 'inv_01')")
    role: SubagentRole
    task: str = Field(..., description="Plain-English scoped question/task")
    target_project: str = Field(..., description="BuildTargetProfile project_id")
    context_files: List[str] = Field(default_factory=list, description="Hint files to seed investigation")
    related_bugs: List[str] = Field(default_factory=list, description="Which bugs from the list this covers")
    depends_on: List[str] = Field(default_factory=list, description="brief_ids that must finish first")
    max_tool_calls: int = Field(15, ge=1, le=1000)


class DecompositionResult(BaseModel):
    """Output of the decomposer."""
    briefs: List[SubagentBrief]
    rationale: str = Field(..., description="Why the decomposer broke the problem up this way")
    unaddressed_bugs: List[str] = Field(
        default_factory=list,
        description="Bugs that could not be mapped to any brief"
    )


# ---------------------------------------------------------------------------
# Investigation: brief → report
# ---------------------------------------------------------------------------

class Finding(BaseModel):
    """A single structured finding from a subagent."""
    summary: str = Field(..., description="One-line statement of what was found")
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="File refs, line numbers, log snippets, etc."
    )
    tags: List[str] = Field(default_factory=list, description="e.g. 'race-condition', 'missing-null-check'")


class SubagentReport(BaseModel):
    """Output of a single subagent run."""
    brief_id: str
    role: SubagentRole
    status: StepStatus
    findings: List[Finding] = Field(default_factory=list)
    dead_ends: List[str] = Field(default_factory=list, description="Hypotheses that did not pan out")
    files_read: List[str] = Field(default_factory=list)
    files_modified: List[str] = Field(default_factory=list, description="Only non-empty for executors")
    tool_call_count: int = 0
    tokens_used: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    elapsed_ms: int = 0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Planning: reports → ordered fix plan
# ---------------------------------------------------------------------------

class FixStep(BaseModel):
    """A single step in the fix plan."""
    step_id: str = Field(..., description="Stable ID (e.g. 'fix_01')")
    description: str = Field(..., description="What this step does, in plain English")
    target_project: str
    target_files: List[str] = Field(
        default_factory=list,
        description="Files this step is expected to modify. Scopes executor writes."
    )
    depends_on: List[str] = Field(default_factory=list, description="step_ids that must finish first")
    parallelisable_with: List[str] = Field(
        default_factory=list,
        description="step_ids that can run concurrently (file scopes must not overlap)"
    )
    rationale: str = Field(..., description="Why this step is needed, drawn from investigation findings")


class DebugPlan(BaseModel):
    """Output of the planner."""
    root_cause: str = Field(..., description="Synthesised root cause across all reports")
    steps: List[FixStep]
    contradictions: List[str] = Field(
        default_factory=list,
        description="Conflicts between subagent findings that need human judgement"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    unresolved_bugs: List[str] = Field(
        default_factory=list,
        description="Bugs the plan does not address"
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

class BehaviourCheck(BaseModel):
    """A scripted behaviour-level test."""
    check_id: str
    description: str = Field(..., description="What the check proves, user-facing")
    scenario_type: Literal["android", "desktop", "http", "pipeline"]
    steps: List[Dict[str, Any]] = Field(
        ...,
        description="Ordered list of tool calls with args (e.g. tap, type, wait, assert)"
    )
    expected: str = Field(..., description="Natural-language expected outcome")
    timeout_seconds: int = Field(30, ge=1, le=300)


class VerificationResult(BaseModel):
    """Result of a single verification (code or behaviour)."""
    check_id: str
    check_type: Literal["code", "behaviour"]
    status: StepStatus
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    failure_signature: Optional[str] = Field(
        None,
        description="Stable hash of the failure for regression detection"
    )
    message: str = ""
    elapsed_ms: int = 0


# ---------------------------------------------------------------------------
# Iteration + final resolution
# ---------------------------------------------------------------------------

class IterationRecord(BaseModel):
    """One full pass through investigate→plan→execute→verify."""
    iteration: int = Field(..., ge=1)
    phase_reached: OrchestrationPhase
    decomposition: Optional[DecompositionResult] = None
    reports: List[SubagentReport] = Field(default_factory=list)
    plan: Optional[DebugPlan] = None
    execution_reports: List[SubagentReport] = Field(default_factory=list)
    code_verifications: List[VerificationResult] = Field(default_factory=list)
    behaviour_verifications: List[VerificationResult] = Field(default_factory=list)
    passed: bool = False
    elapsed_ms: int = 0


class DebugResolution(BaseModel):
    """Final result returned to the Debug tab."""
    project_id: str
    final_phase: OrchestrationPhase
    resolved: bool
    iterations: List[IterationRecord]
    total_elapsed_ms: int
    total_tokens: int
    total_cost_gbp: float = 0.0
    summary: str = Field(..., description="Human-readable summary of what was done")
    surfaced_contradictions: List[str] = Field(default_factory=list)
    unresolved_bugs: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# SSE stream events (what the frontend consumes)
# ---------------------------------------------------------------------------

class OrchestrationEvent(BaseModel):
    """Event streamed to the frontend during orchestration."""
    event_type: Literal[
        "phase_change",
        "decomposition_complete",
        "subagent_spawn",
        "subagent_start",
        "subagent_progress",
        "subagent_complete",
        "subagent_spawn_complete",
        "plan_complete",
        "execution_start",
        "execution_complete",
        "verification_start",
        "verification_complete",
        "iteration_complete",
        "resolution",
        "cancelled",
        "error",
    ]
    iteration: int = 0
    phase: OrchestrationPhase
    message: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)
