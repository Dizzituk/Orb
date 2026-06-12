# FILE: app/optimize/models.py
# Purpose: Optimize Tab data models.
# Called-by: app.optimize.decomposer, app.optimize.executor, app.optimize.orchestrator, app.optimize.profiler (+3 more)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Optimize Tab data models.

Structured types for the four-phase cycle:
  Phase A: Decompose → ChunkManifest, CodeChunk, DependencyEdge
  Phase B: Profile → ProfileResult, ChunkMetrics, Bottleneck
  Phase C: Propose → Proposal, ProposalCategory, RiskLevel
  Phase D: Execute → ExecutionResult, BeforeAfterMetrics

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-OPT-001.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════

class ProposalCategory(str, Enum):
    """The eight optimisation categories."""
    DEAD_CODE = "dead_code"
    SEQUENTIAL_TO_PARALLEL = "seq_to_parallel"
    REDUNDANT_CALL = "redundant_call"
    LOGIC_SIMPLIFICATION = "logic_simplification"
    CACHE_INTRODUCTION = "cache_introduction"
    MODEL_CALL_REDUCTION = "model_call_reduction"
    FILE_SPLIT_MERGE = "file_split_merge"
    DEPENDENCY_CLEANUP = "dependency_cleanup"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ProposalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    PASSED = "passed"
    FAILED = "failed"
    PROMOTED = "promoted"


class OptimizeTarget(str, Enum):
    """What the optimize tab is pointed at."""
    ASTRA_CORE = "astra-backend"
    ASTRA_FRONTEND = "astra-frontend"
    DRIVER_COPILOT = "driver-copilot"
    CONTENT_PIPELINE = "content-pipeline"
    FINANCE_PIPELINE = "finance-pipeline"


# ═══════════════════════════════════════════════════════════════════
# Phase A: Decompose
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DependencyEdge:
    """A directed dependency between two code chunks."""
    source: str             # File/module path
    target: str             # What it imports/depends on
    edge_type: str = "import"   # import, call, data_flow
    weight: float = 1.0


@dataclass
class CodeChunk:
    """A discrete functional unit in the codebase."""
    path: str               # File path
    name: str = ""          # Module/function/class name
    responsibility: str = ""
    lines: int = 0
    size_bytes: int = 0
    complexity_estimate: float = 0.0  # 0-1 rough complexity
    imports: List[str] = field(default_factory=list)
    dependents: int = 0     # How many other chunks import this
    dependencies: int = 0   # How many chunks this imports
    is_dead_code: bool = False
    is_oversized: bool = False
    tags: List[str] = field(default_factory=list)

    @property
    def coupling_score(self) -> float:
        """Higher = more coupled. dependents * dependencies normalised."""
        return (self.dependents + self.dependencies) / max(1, self.lines / 100)


@dataclass
class DeadCodeCandidate:
    """A file, function, or import that appears unreferenced."""
    path: str
    item_type: str = "file"  # file, function, import, config
    name: str = ""
    reason: str = ""
    confidence: float = 0.8


@dataclass
class ChunkManifest:
    """Complete decomposition output."""
    target: str = ""
    chunks: List[CodeChunk] = field(default_factory=list)
    dependency_edges: List[DependencyEdge] = field(default_factory=list)
    dead_code: List[DeadCodeCandidate] = field(default_factory=list)
    total_files: int = 0
    total_lines: int = 0
    total_size_bytes: int = 0
    oversized_files: int = 0
    generated_at: str = ""

    def summary(self) -> str:
        return (
            f"Decompose: {self.total_files} files, {self.total_lines:,} lines, "
            f"{self.total_size_bytes // 1024}KB | "
            f"Dead code: {len(self.dead_code)} candidates | "
            f"Oversized: {self.oversized_files} files"
        )


# ═══════════════════════════════════════════════════════════════════
# Phase B: Profile
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ChunkMetrics:
    """Profiling metrics for a single chunk."""
    path: str
    import_time_ms: float = 0.0
    line_count: int = 0
    size_bytes: int = 0
    cyclomatic_complexity: int = 0
    coupling_in: int = 0    # Incoming dependencies
    coupling_out: int = 0   # Outgoing dependencies
    duplicate_ratio: float = 0.0  # 0-1
    test_coverage: float = 0.0    # 0-1 if available
    last_modified_days: int = 0
    anomaly_flags: List[str] = field(default_factory=list)


@dataclass
class Bottleneck:
    """A ranked bottleneck from profiling."""
    path: str
    metric: str             # Which metric makes it a bottleneck
    value: float = 0.0
    impact_score: float = 0.0  # 0-1 impact on overall system
    description: str = ""


@dataclass
class ProfileResult:
    """Complete profiling output."""
    target: str = ""
    chunk_metrics: List[ChunkMetrics] = field(default_factory=list)
    bottlenecks: List[Bottleneck] = field(default_factory=list)
    total_import_time_ms: float = 0.0
    total_complexity: int = 0
    generated_at: str = ""


# ═══════════════════════════════════════════════════════════════════
# Phase C: Propose
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Proposal:
    """A single optimisation proposal."""
    proposal_id: str = ""
    category: ProposalCategory = ProposalCategory.DEAD_CODE
    title: str = ""
    description: str = ""
    target_chunks: List[str] = field(default_factory=list)
    predicted_improvement: str = ""
    risk: RiskLevel = RiskLevel.LOW
    estimated_token_cost: float = 0.0
    confidence: float = 0.7
    status: ProposalStatus = ProposalStatus.PENDING
    impact_score: float = 0.0  # For ranking
    created_at: str = ""
    executed_at: str = ""
    actual_improvement: str = ""

    @property
    def impact_risk_ratio(self) -> float:
        """Higher = better. Used for ranking."""
        risk_penalty = {"low": 1.0, "medium": 0.6, "high": 0.3}
        return self.impact_score * risk_penalty.get(self.risk.value, 0.5)


# ═══════════════════════════════════════════════════════════════════
# Phase D: Execute
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BeforeAfterMetrics:
    """Metrics captured before and after an optimisation."""
    metric_name: str
    before: float = 0.0
    after: float = 0.0

    @property
    def delta(self) -> float:
        return self.after - self.before

    @property
    def delta_pct(self) -> float:
        if self.before == 0:
            return 0.0
        return (self.delta / self.before) * 100


@dataclass
class ExecutionResult:
    """Result of executing an optimisation proposal."""
    proposal_id: str = ""
    success: bool = False
    bvl_passed: bool = False
    metrics: List[BeforeAfterMetrics] = field(default_factory=list)
    files_changed: List[str] = field(default_factory=list)
    lines_removed: int = 0
    lines_added: int = 0
    token_cost: float = 0.0
    duration_seconds: float = 0.0
    error: str = ""
    ready_for_promotion: bool = False


# ═══════════════════════════════════════════════════════════════════
# Full optimize report
# ═══════════════════════════════════════════════════════════════════

@dataclass
class OptimizeReport:
    """Complete report from an optimisation pass."""
    target: str = ""
    manifest: Optional[ChunkManifest] = None
    profile: Optional[ProfileResult] = None
    proposals: List[Proposal] = field(default_factory=list)
    execution_results: List[ExecutionResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    total_token_cost: float = 0.0

    @property
    def approved_count(self) -> int:
        return sum(1 for p in self.proposals if p.status != ProposalStatus.PENDING)

    @property
    def executed_count(self) -> int:
        return len(self.execution_results)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.execution_results if r.success)
