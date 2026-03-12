# FILE: app/pipeline_v2/bvl/bvl_models.py
"""
BVL Data Models — structured types for the Behavioral Verification Layer.

Every tier produces TierResult objects. Every test flow is a TestFlow.
The retry loop consumes RetryContext. The orchestrator reads BVLReport.

All models are plain dataclasses — no ORM, no Pydantic dependency.

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-BVL-001.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════

class TierName(str, Enum):
    """The three verification tiers."""
    SANITY = "tier1_sanity"
    BEHAVIORAL = "tier2_behavioral"
    ADVERSARIAL = "tier3_adversarial"


class TierStatus(str, Enum):
    """Outcome of a single tier run."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


# ═══════════════════════════════════════════════════════════════════
# Tier Results
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SanityCheck:
    """Result of a single Tier 1 sanity check."""
    name: str                   # e.g. "gradle_build", "apk_install", "app_launch"
    passed: bool = False
    message: str = ""
    duration_ms: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TierResult:
    """Output of one tier run against one component."""
    tier: TierName
    status: TierStatus = TierStatus.FAILED
    checks: List[SanityCheck] = field(default_factory=list)
    error_context: str = ""     # Plain-language error for retry loop
    screenshot_b64: Optional[str] = None
    logcat_excerpt: str = ""
    duration_ms: int = 0

    @property
    def passed(self) -> bool:
        return self.status == TierStatus.PASSED

    @property
    def failed_checks(self) -> List[SanityCheck]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        icon = {"passed": "✅", "failed": "❌", "skipped": "⏭️", "error": "⚠️"}
        parts = [f"{icon.get(self.status.value, '?')} {self.tier.value}"]
        for c in self.checks:
            mark = "✓" if c.passed else "✗"
            parts.append(f"  {mark} {c.name}: {c.message}")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════
# Behavioral Test Models (Tier 2)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TestAction:
    """A single UI interaction in a test flow."""
    action_type: str            # "tap", "type", "swipe", "scroll", "wait", "back"
    target: str = ""            # Element description or coordinates
    value: str = ""             # Text to type, swipe direction, wait duration
    timeout_ms: int = 5000


@dataclass
class TestAssertion:
    """Expected outcome after one or more actions."""
    assertion_type: str         # "element_visible", "text_matches", "screen_changed", "data_persisted"
    target: str = ""            # Element or field to check
    expected: str = ""          # Expected value
    passed: bool = False
    actual: str = ""            # What was actually found
    message: str = ""


@dataclass
class TestFlow:
    """A complete behavioral test sequence for one user story."""
    flow_id: str                # e.g. "login_flow", "add_stop_flow"
    description: str = ""       # Human-readable: "User logs in with valid credentials"
    user_story_ref: str = ""    # Which spec requirement this tests
    setup_actions: List[TestAction] = field(default_factory=list)
    actions: List[TestAction] = field(default_factory=list)
    assertions: List[TestAssertion] = field(default_factory=list)
    teardown_actions: List[TestAction] = field(default_factory=list)
    passed: bool = False
    error: str = ""
    duration_ms: int = 0


# ═══════════════════════════════════════════════════════════════════
# Flow Coverage Matrix (Tier 2)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FlowCoverage:
    """Maps user stories to test flows and their results."""
    user_story: str
    flow_id: str
    tested: bool = False
    passed: bool = False
    failure_reason: str = ""


# ═══════════════════════════════════════════════════════════════════
# Retry Context
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RetryContext:
    """Minimal context package sent to the Builder on failure.

    Designed to be small (~1,200–3,800 tokens) for cost control.
    Includes only the failing file, error signal, and spec slice.
    """
    component_path: str         # The file that failed
    tier: TierName              # Which tier caught the failure
    attempt: int = 1            # Which retry attempt (1-3)
    error_signal: str = ""      # Concise error: crash log, assertion failure, etc.
    spec_slice: str = ""        # Relevant portion of the spec
    previous_errors: List[str] = field(default_factory=list)  # Cumulative
    logcat_excerpt: str = ""    # Relevant logcat lines (filtered)

    def to_builder_prompt(self) -> str:
        """Format as a prompt for the Agentic Builder."""
        lines = [
            f"BVL {self.tier.value} FAILED for: {self.component_path}",
            f"Attempt: {self.attempt}/3",
            "",
            "ERROR:",
            self.error_signal,
        ]
        if self.logcat_excerpt:
            lines.extend(["", "LOGCAT:", self.logcat_excerpt[:1500]])
        if self.previous_errors:
            lines.extend(["", "PREVIOUS ATTEMPTS:"])
            for i, err in enumerate(self.previous_errors, 1):
                lines.append(f"  Attempt {i}: {err[:300]}")
        if self.spec_slice:
            lines.extend(["", "RELEVANT SPEC:", self.spec_slice[:2000]])
        lines.extend([
            "",
            "Read the failing file, diagnose the issue, and fix it.",
            "Make SURGICAL changes — do not rewrite from scratch.",
        ])
        return "\n".join(lines)

    @property
    def estimated_tokens(self) -> int:
        """Rough token estimate for cost tracking."""
        text = self.to_builder_prompt()
        return len(text) // 4  # ~4 chars per token approximation


# ═══════════════════════════════════════════════════════════════════
# Component-Level Result
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ComponentResult:
    """Full BVL result for a single component (file or module).

    Tracks all tier results and retry attempts. A component is
    SIGNED_OFF only if all three tiers pass. It is BLOCKED if
    it exhausts its retry budget at any tier.
    """
    component_path: str
    tier_results: List[TierResult] = field(default_factory=list)
    retry_attempts: int = 0
    signed_off: bool = False
    blocked: bool = False
    block_reason: str = ""
    total_duration_ms: int = 0

    @property
    def highest_tier_passed(self) -> Optional[TierName]:
        """The highest tier this component has cleared."""
        passed_tiers = [
            tr.tier for tr in self.tier_results if tr.passed
        ]
        if TierName.ADVERSARIAL in passed_tiers:
            return TierName.ADVERSARIAL
        if TierName.BEHAVIORAL in passed_tiers:
            return TierName.BEHAVIORAL
        if TierName.SANITY in passed_tiers:
            return TierName.SANITY
        return None


# ═══════════════════════════════════════════════════════════════════
# Full BVL Report
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BVLReport:
    """Complete Behavioral Verification Layer report for a build job.

    This is what the orchestrator and debrief system consume.
    """
    job_id: str = ""
    component_results: List[ComponentResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    total_retries: int = 0
    total_llm_calls: int = 0
    estimated_cost_usd: float = 0.0

    @property
    def all_signed_off(self) -> bool:
        return all(cr.signed_off for cr in self.component_results)

    @property
    def blocked_components(self) -> List[ComponentResult]:
        return [cr for cr in self.component_results if cr.blocked]

    @property
    def signed_off_components(self) -> List[ComponentResult]:
        return [cr for cr in self.component_results if cr.signed_off]

    def summary(self) -> str:
        total = len(self.component_results)
        signed = len(self.signed_off_components)
        blocked = len(self.blocked_components)
        lines = [
            f"BVL Report: {signed}/{total} signed off, {blocked} blocked",
            f"Duration: {self.total_duration_seconds:.1f}s | "
            f"Retries: {self.total_retries} | "
            f"LLM calls: {self.total_llm_calls} | "
            f"Cost: ${self.estimated_cost_usd:.2f}",
        ]
        if self.blocked_components:
            lines.append("BLOCKED:")
            for cr in self.blocked_components:
                lines.append(f"  ✗ {cr.component_path}: {cr.block_reason}")
        return "\n".join(lines)

    def to_debrief(self) -> Dict[str, Any]:
        """Format for the debug/debrief system."""
        return {
            "job_id": self.job_id,
            "overall_passed": self.all_signed_off,
            "components_total": len(self.component_results),
            "components_signed_off": len(self.signed_off_components),
            "components_blocked": len(self.blocked_components),
            "blocked_details": [
                {
                    "path": cr.component_path,
                    "reason": cr.block_reason,
                    "retries": cr.retry_attempts,
                    "highest_tier": (
                        cr.highest_tier_passed.value
                        if cr.highest_tier_passed else "none"
                    ),
                }
                for cr in self.blocked_components
            ],
            "total_duration_seconds": self.total_duration_seconds,
            "total_retries": self.total_retries,
            "total_llm_calls": self.total_llm_calls,
            "estimated_cost_usd": self.estimated_cost_usd,
        }
