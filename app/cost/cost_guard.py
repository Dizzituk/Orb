# FILE: app/overwatcher/cost_guard.py
"""
Cost Guard for ASTRA Pipeline (Job 3)

Spec §10: Cost Controls
- Token caps per stage/role
- Break-glass mode (explicit, user-approved, rare)
- Cost tracking and budget enforcement

Roles and their token budgets:
- Overwatcher: 2,000 output tokens (strict)
- Spec Gate: 4,000 output tokens
- Implementer: 16,000 output tokens (can be higher for full files)
- Architect: 32,000 output tokens (architecture maps)
- Critique: 8,000 output tokens
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class ModelRole(str, Enum):
    """Model roles with different token budgets."""
    OVERWATCHER = "overwatcher"
    SPEC_GATE = "spec_gate"
    IMPLEMENTER = "implementer"
    ARCHITECT = "architect"
    CRITIQUE = "critique"
    MEDIATOR = "mediator"


class BudgetStatus(str, Enum):
    """Budget check result."""
    WITHIN_BUDGET = "within_budget"
    APPROACHING_LIMIT = "approaching_limit"  # >80% of budget
    EXCEEDED = "exceeded"
    BREAK_GLASS_REQUIRED = "break_glass_required"


@dataclass
class TokenBudget:
    """Token budget configuration for a role."""
    role: ModelRole
    max_output_tokens: int
    max_input_tokens: int
    warning_threshold: float = 0.8  # Warn at 80% usage
    break_glass_multiplier: float = 2.0  # Break-glass allows 2x normal budget
    
    def get_effective_limit(self, break_glass: bool = False) -> int:
        """Get effective output token limit."""
        if break_glass:
            return int(self.max_output_tokens * self.break_glass_multiplier)
        return self.max_output_tokens


# Default budgets per role
DEFAULT_BUDGETS: Dict[ModelRole, TokenBudget] = {
    ModelRole.OVERWATCHER: TokenBudget(
        role=ModelRole.OVERWATCHER,
        max_output_tokens=2_000,
        max_input_tokens=120_000,
    ),
    ModelRole.SPEC_GATE: TokenBudget(
        role=ModelRole.SPEC_GATE,
        max_output_tokens=4_000,
        max_input_tokens=100_000,
    ),
    ModelRole.IMPLEMENTER: TokenBudget(
        role=ModelRole.IMPLEMENTER,
        max_output_tokens=16_000,
        max_input_tokens=100_000,
    ),
    ModelRole.ARCHITECT: TokenBudget(
        role=ModelRole.ARCHITECT,
        max_output_tokens=32_000,
        max_input_tokens=100_000,
    ),
    ModelRole.CRITIQUE: TokenBudget(
        role=ModelRole.CRITIQUE,
        max_output_tokens=8_000,
        max_input_tokens=100_000,
    ),
    ModelRole.MEDIATOR: TokenBudget(
        role=ModelRole.MEDIATOR,
        max_output_tokens=4_000,
        max_input_tokens=50_000,
    ),
}


@dataclass
class UsageRecord:
    """Record of token usage for a single call."""
    job_id: str
    role: ModelRole
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_estimate: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    break_glass_used: bool = False
    stage: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "role": self.role.value,
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_estimate": self.cost_estimate,
            "timestamp": self.timestamp,
            "break_glass": self.break_glass_used,
            "stage": self.stage,
        }


@dataclass
class JobBudgetSummary:
    """Cumulative budget summary for a job."""
    job_id: str
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    calls_by_role: Dict[str, int] = field(default_factory=dict)
    tokens_by_role: Dict[str, int] = field(default_factory=dict)
    break_glass_count: int = 0
    budget_warnings: List[str] = field(default_factory=list)
    
    def add_usage(self, record: UsageRecord) -> None:
        """Add a usage record to the summary."""
        self.total_prompt_tokens += record.prompt_tokens
        self.total_completion_tokens += record.completion_tokens
        self.total_tokens += record.total_tokens
        self.total_cost += record.cost_estimate
        
        role_key = record.role.value
        self.calls_by_role[role_key] = self.calls_by_role.get(role_key, 0) + 1
        self.tokens_by_role[role_key] = self.tokens_by_role.get(role_key, 0) + record.total_tokens
        
        if record.break_glass_used:
            self.break_glass_count += 1
    
    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "calls_by_role": self.calls_by_role,
            "tokens_by_role": self.tokens_by_role,
            "break_glass_count": self.break_glass_count,
            "warnings": self.budget_warnings,
        }


@dataclass
class BudgetCheckResult:
    """Result of a budget check."""
    status: BudgetStatus
    allowed_tokens: int
    requested_tokens: int
    message: str
    break_glass_available: bool = True
    
    def is_allowed(self) -> bool:
        return self.status in (BudgetStatus.WITHIN_BUDGET, BudgetStatus.APPROACHING_LIMIT)


@dataclass
class BreakGlassRequest:
    """Request to use break-glass budget override."""
    job_id: str
    role: ModelRole
    reason: str
    requested_tokens: int
    normal_limit: int
    break_glass_limit: int
    approved: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    
    def approve(self, approver: str = "system") -> None:
        """Approve the break-glass request."""
        self.approved = True
        self.approved_by = approver
        self.approved_at = datetime.now(timezone.utc).isoformat()


class CostGuard:
    """
    Cost guard for ASTRA pipeline.
    
    Enforces token budgets and tracks usage across jobs.
    """
    
    def __init__(self, budgets: Optional[Dict[ModelRole, TokenBudget]] = None):
        self._budgets = budgets or DEFAULT_BUDGETS.copy()
        self._job_summaries: Dict[str, JobBudgetSummary] = {}
        self._usage_records: List[UsageRecord] = []
        self._break_glass_requests: List[BreakGlassRequest] = []
        
        # Load overrides from environment
        self._load_env_overrides()
    
    def _load_env_overrides(self) -> None:
        """Load budget overrides from environment variables."""
        env_map = {
            "ORB_OVERWATCHER_MAX_OUTPUT_TOKENS": ModelRole.OVERWATCHER,
            "ORB_SPEC_GATE_MAX_OUTPUT_TOKENS": ModelRole.SPEC_GATE,
            "ORB_IMPLEMENTER_MAX_OUTPUT_TOKENS": ModelRole.IMPLEMENTER,
            "ORB_ARCHITECT_MAX_OUTPUT_TOKENS": ModelRole.ARCHITECT,
            "ORB_CRITIQUE_MAX_OUTPUT_TOKENS": ModelRole.CRITIQUE,
        }
        
        for env_var, role in env_map.items():
            value = os.getenv(env_var)
            if value:
                try:
                    self._budgets[role].max_output_tokens = int(value)
                    logger.info(f"[cost_guard] Override {role.value} max_output_tokens = {value}")
                except ValueError:
                    pass
    
    def get_budget(self, role: ModelRole) -> TokenBudget:
        """Get budget for a role."""
        return self._budgets.get(role, DEFAULT_BUDGETS[ModelRole.IMPLEMENTER])
    
    def check_budget(
        self,
        role: ModelRole,
        requested_output_tokens: int,
        break_glass: bool = False,
    ) -> BudgetCheckResult:
        """
        Check if a request is within budget.
        
        Args:
            role: The model role making the request
            requested_output_tokens: Tokens being requested
            break_glass: Whether break-glass mode is active
        
        Returns:
            BudgetCheckResult with status and allowed tokens
        """
        budget = self.get_budget(role)
        normal_limit = budget.max_output_tokens
        break_glass_limit = budget.get_effective_limit(break_glass=True)
        
        if requested_output_tokens <= normal_limit:
            # Within normal budget
            warning_threshold = int(normal_limit * budget.warning_threshold)
            if requested_output_tokens >= warning_threshold:
                return BudgetCheckResult(
                    status=BudgetStatus.APPROACHING_LIMIT,
                    allowed_tokens=normal_limit,
                    requested_tokens=requested_output_tokens,
                    message=f"Approaching {role.value} token limit ({requested_output_tokens}/{normal_limit})",
                )
            return BudgetCheckResult(
                status=BudgetStatus.WITHIN_BUDGET,
                allowed_tokens=normal_limit,
                requested_tokens=requested_output_tokens,
                message="Within budget",
            )
        
        # Exceeds normal budget
        if break_glass and requested_output_tokens <= break_glass_limit:
            return BudgetCheckResult(
                status=BudgetStatus.WITHIN_BUDGET,
                allowed_tokens=break_glass_limit,
                requested_tokens=requested_output_tokens,
                message=f"Within break-glass budget ({requested_output_tokens}/{break_glass_limit})",
            )
        
        if not break_glass and requested_output_tokens <= break_glass_limit:
            return BudgetCheckResult(
                status=BudgetStatus.BREAK_GLASS_REQUIRED,
                allowed_tokens=normal_limit,
                requested_tokens=requested_output_tokens,
                message=f"Exceeds {role.value} budget ({requested_output_tokens} > {normal_limit}). Break-glass available up to {break_glass_limit}.",
                break_glass_available=True,
            )
        
        return BudgetCheckResult(
            status=BudgetStatus.EXCEEDED,
            allowed_tokens=break_glass_limit if break_glass else normal_limit,
            requested_tokens=requested_output_tokens,
            message=f"Exceeds maximum allowed tokens even with break-glass ({requested_output_tokens} > {break_glass_limit})",
            break_glass_available=False,
        )
    
    def get_max_tokens_for_role(self, role: ModelRole, break_glass: bool = False) -> int:
        """Get maximum output tokens allowed for a role."""
        budget = self.get_budget(role)
        return budget.get_effective_limit(break_glass)
    
    def record_usage(
        self,
        job_id: str,
        role: ModelRole,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_estimate: float = 0.0,
        break_glass_used: bool = False,
        stage: Optional[str] = None,
    ) -> UsageRecord:
        """
        Record token usage for a call.
        
        Args:
            job_id: Job identifier
            role: Model role
            provider: LLM provider
            model: Model identifier
            prompt_tokens: Input tokens used
            completion_tokens: Output tokens used
            cost_estimate: Estimated cost in USD
            break_glass_used: Whether break-glass was used
            stage: Pipeline stage name
        
        Returns:
            UsageRecord for the call
        """
        record = UsageRecord(
            job_id=job_id,
            role=role,
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_estimate=cost_estimate,
            break_glass_used=break_glass_used,
            stage=stage,
        )
        
        self._usage_records.append(record)
        
        # Update job summary
        if job_id not in self._job_summaries:
            self._job_summaries[job_id] = JobBudgetSummary(job_id=job_id)
        
        summary = self._job_summaries[job_id]
        summary.add_usage(record)
        
        # Check for budget warnings
        budget = self.get_budget(role)
        if completion_tokens > budget.max_output_tokens:
            warning = f"{role.value} exceeded output budget: {completion_tokens} > {budget.max_output_tokens}"
            summary.budget_warnings.append(warning)
            logger.warning(f"[cost_guard] {warning}")
        
        return record
    
    def get_job_summary(self, job_id: str) -> Optional[JobBudgetSummary]:
        """Get budget summary for a job."""
        return self._job_summaries.get(job_id)
    
    def create_break_glass_request(
        self,
        job_id: str,
        role: ModelRole,
        reason: str,
        requested_tokens: int,
    ) -> BreakGlassRequest:
        """
        Create a break-glass request.
        
        Break-glass must be explicit, logged, and should be rare.
        """
        budget = self.get_budget(role)
        
        request = BreakGlassRequest(
            job_id=job_id,
            role=role,
            reason=reason,
            requested_tokens=requested_tokens,
            normal_limit=budget.max_output_tokens,
            break_glass_limit=budget.get_effective_limit(break_glass=True),
        )
        
        self._break_glass_requests.append(request)
        logger.warning(
            f"[cost_guard] Break-glass requested: job={job_id}, role={role.value}, "
            f"tokens={requested_tokens}, reason={reason}"
        )
        
        return request
    
    def approve_break_glass(
        self,
        request: BreakGlassRequest,
        approver: str = "auto",
    ) -> bool:
        """
        Approve a break-glass request.
        
        In production, this should require explicit user approval.
        For automated pipelines, can use "auto" approver with logging.
        """
        if request.requested_tokens > request.break_glass_limit:
            logger.error(
                f"[cost_guard] Break-glass denied: {request.requested_tokens} > {request.break_glass_limit}"
            )
            return False
        
        request.approve(approver)
        logger.warning(
            f"[cost_guard] Break-glass approved: job={request.job_id}, role={request.role.value}, "
            f"tokens={request.requested_tokens}, approver={approver}"
        )
        
        return True
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get aggregate usage statistics."""
        total_tokens = sum(r.total_tokens for r in self._usage_records)
        total_cost = sum(r.cost_estimate for r in self._usage_records)
        total_calls = len(self._usage_records)
        break_glass_count = sum(1 for r in self._usage_records if r.break_glass_used)
        
        by_role: Dict[str, Dict[str, int]] = {}
        for r in self._usage_records:
            role_key = r.role.value
            if role_key not in by_role:
                by_role[role_key] = {"calls": 0, "tokens": 0}
            by_role[role_key]["calls"] += 1
            by_role[role_key]["tokens"] += r.total_tokens
        
        return {
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "break_glass_count": break_glass_count,
            "by_role": by_role,
            "jobs_tracked": len(self._job_summaries),
        }


# =============================================================================
# Global Instance
# =============================================================================

_cost_guard: Optional[CostGuard] = None


def get_cost_guard() -> CostGuard:
    """Get the global CostGuard instance."""
    global _cost_guard
    if _cost_guard is None:
        _cost_guard = CostGuard()
    return _cost_guard


def check_budget(role: ModelRole, requested_tokens: int, break_glass: bool = False) -> BudgetCheckResult:
    """Convenience function to check budget."""
    return get_cost_guard().check_budget(role, requested_tokens, break_glass)


def get_max_tokens(role: ModelRole, break_glass: bool = False) -> int:
    """Convenience function to get max tokens for a role."""
    return get_cost_guard().get_max_tokens_for_role(role, break_glass)


def record_usage(
    job_id: str,
    role: ModelRole,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_estimate: float = 0.0,
    break_glass_used: bool = False,
    stage: Optional[str] = None,
) -> UsageRecord:
    """Convenience function to record usage."""
    return get_cost_guard().record_usage(
        job_id, role, provider, model,
        prompt_tokens, completion_tokens, cost_estimate,
        break_glass_used, stage,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Types
    "ModelRole",
    "BudgetStatus",
    "TokenBudget",
    "UsageRecord",
    "JobBudgetSummary",
    "BudgetCheckResult",
    "BreakGlassRequest",
    # Class
    "CostGuard",
    # Functions
    "get_cost_guard",
    "check_budget",
    "get_max_tokens",
    "record_usage",
    # Constants
    "DEFAULT_BUDGETS",
]
