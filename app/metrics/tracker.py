# FILE: app/metrics/tracker.py
"""
Pipeline Metrics Tracker

Tracks cost and accuracy metrics per job for analysis.
Outputs JSON logs that can be aggregated for cost-accuracy optimization.

Usage:
    tracker = MetricsTracker(job_id)
    tracker.log_stage("spec_gate", model="gpt-5.2-pro", input_tokens=500, output_tokens=200)
    tracker.log_critique(iteration=1, passed=False, blocking_issues=2)
    tracker.finalize(success=True)
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Cost per 1M tokens (Standard tier) - update as pricing changes
MODEL_COSTS = {
    # OpenAI
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5.2-chat-latest": {"input": 1.75, "output": 14.00},
    "gpt-5.2-pro": {"input": 21.00, "output": 168.00},
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "gpt-5": {"input": 1.25, "output": 10.00},
    # Anthropic
    "claude-opus-4-5-20251101": {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    # Google
    "gemini-3-pro-preview": {"input": 2.00, "output": 12.00},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
}


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a model call."""
    costs = MODEL_COSTS.get(model, {"input": 0, "output": 0})
    input_cost = (input_tokens / 1_000_000) * costs["input"]
    output_cost = (output_tokens / 1_000_000) * costs["output"]
    return round(input_cost + output_cost, 6)


@dataclass
class StageMetric:
    """Metrics for a single pipeline stage."""
    stage: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    success: bool = True
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class CritiqueMetric:
    """Metrics for a critique iteration."""
    iteration: int
    passed: bool
    blocking_issues: int = 0
    non_blocking_issues: int = 0
    issue_ids: List[str] = field(default_factory=list)
    model: str = ""
    tokens_used: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class JobMetrics:
    """Complete metrics for a job."""
    job_id: str
    project_id: int = 0
    job_type: str = ""
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    success: bool = False
    
    # Stage metrics
    stages: List[Dict[str, Any]] = field(default_factory=list)
    
    # Critique loop metrics
    critiques: List[Dict[str, Any]] = field(default_factory=list)
    total_critique_iterations: int = 0
    critique_passed: bool = False
    
    # Cost summary
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    
    # Accuracy indicators
    spec_gate_questions: int = 0  # 0 = spec was complete
    stage3_verified: bool = False
    
    # Error tracking
    error_message: Optional[str] = None


class MetricsTracker:
    """Track metrics for a single job."""
    
    def __init__(self, job_id: str, project_id: int = 0, job_type: str = ""):
        self.metrics = JobMetrics(
            job_id=job_id,
            project_id=project_id,
            job_type=job_type,
        )
        self._enabled = os.getenv("ORB_TRACKING_ENABLED", "1") == "1"
        self._metrics_dir = Path(os.getenv("ORB_METRICS_DIR", "metrics"))
    
    def log_stage(
        self,
        stage: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_ms: int = 0,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Log metrics for a pipeline stage."""
        if not self._enabled:
            return
        
        cost = _calculate_cost(model, input_tokens, output_tokens)
        
        metric = StageMetric(
            stage=stage,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            duration_ms=duration_ms,
            success=success,
            error=error,
        )
        
        self.metrics.stages.append(asdict(metric))
        self.metrics.total_input_tokens += input_tokens
        self.metrics.total_output_tokens += output_tokens
        self.metrics.total_cost_usd += cost
        
        if os.getenv("ORB_LOG_TOKEN_USAGE", "0") == "1":
            logger.info(
                f"[metrics] {stage}: model={model}, "
                f"tokens={input_tokens}/{output_tokens}, "
                f"cost=${cost:.4f}"
            )
    
    def log_critique(
        self,
        iteration: int,
        passed: bool,
        blocking_issues: int = 0,
        non_blocking_issues: int = 0,
        issue_ids: Optional[List[str]] = None,
        model: str = "",
        tokens_used: int = 0,
    ) -> None:
        """Log metrics for a critique iteration."""
        if not self._enabled:
            return
        
        metric = CritiqueMetric(
            iteration=iteration,
            passed=passed,
            blocking_issues=blocking_issues,
            non_blocking_issues=non_blocking_issues,
            issue_ids=issue_ids or [],
            model=model,
            tokens_used=tokens_used,
        )
        
        self.metrics.critiques.append(asdict(metric))
        self.metrics.total_critique_iterations = iteration
        
        if passed:
            self.metrics.critique_passed = True
        
        if os.getenv("ORB_LOG_CRITIQUE_DETAILS", "0") == "1":
            status = "PASS" if passed else "FAIL"
            logger.info(
                f"[metrics] critique iteration {iteration}: {status}, "
                f"blocking={blocking_issues}, non_blocking={non_blocking_issues}"
            )
    
    def log_spec_gate(self, open_questions: int) -> None:
        """Log spec gate result."""
        self.metrics.spec_gate_questions = open_questions
        if os.getenv("ORB_LOG_ROUTING_DECISIONS", "0") == "1":
            logger.info(f"[metrics] spec_gate: open_questions={open_questions}")
    
    def log_stage3_verification(self, verified: bool) -> None:
        """Log Stage 3 verification result."""
        self.metrics.stage3_verified = verified
        if os.getenv("ORB_LOG_ROUTING_DECISIONS", "0") == "1":
            logger.info(f"[metrics] stage3_verification: verified={verified}")
    
    def finalize(self, success: bool = True, error_message: Optional[str] = None) -> Dict[str, Any]:
        """Finalize and save metrics."""
        self.metrics.completed_at = datetime.utcnow().isoformat()
        self.metrics.success = success
        self.metrics.error_message = error_message
        
        result = asdict(self.metrics)
        
        if self._enabled and os.getenv("ORB_TRACK_JOB_COSTS", "0") == "1":
            self._save_metrics(result)
        
        # Log summary
        logger.info(
            f"[metrics] Job {self.metrics.job_id} complete: "
            f"success={success}, "
            f"iterations={self.metrics.total_critique_iterations}, "
            f"cost=${self.metrics.total_cost_usd:.4f}"
        )
        
        return result
    
    def _save_metrics(self, data: Dict[str, Any]) -> None:
        """Save metrics to JSON file."""
        try:
            self._metrics_dir.mkdir(parents=True, exist_ok=True)
            
            # Daily file for aggregation
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            filepath = self._metrics_dir / f"metrics_{date_str}.jsonl"
            
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")
            
            logger.debug(f"[metrics] Saved to {filepath}")
        except Exception as e:
            logger.warning(f"[metrics] Failed to save: {e}")


# Convenience function for quick cost estimation
def estimate_job_cost(
    spec_gate_tokens: int = 2000,
    draft_tokens: int = 8000,
    critique_iterations: int = 2,
    critique_tokens_per_iter: int = 3000,
    revision_tokens_per_iter: int = 6000,
) -> Dict[str, float]:
    """Estimate cost for a typical job."""
    costs = {
        "spec_gate": _calculate_cost("gpt-5.2-pro", spec_gate_tokens, 500),
        "draft": _calculate_cost("claude-opus-4-5-20251101", 5000, draft_tokens),
        "critique": critique_iterations * _calculate_cost(
            "gemini-3-pro-preview", critique_tokens_per_iter, 1000
        ),
        "revision": (critique_iterations - 1) * _calculate_cost(
            "claude-opus-4-5-20251101", 8000, revision_tokens_per_iter
        ),
    }
    costs["total"] = sum(costs.values())
    return costs


__all__ = [
    "MetricsTracker",
    "estimate_job_cost",
    "MODEL_COSTS",
]