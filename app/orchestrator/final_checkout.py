# FILE: app/orchestrator/final_checkout.py
"""
Final Project Checkout — Stage 10.

Project-level verification that runs after ALL phases complete.
Catches issues that per-phase checkout (Stage 9) cannot:

1. Cross-phase import integrity — do files from Phase 2 correctly
   import from Phase 1 files?
2. Full project boot test — does the entire assembled project start?
   v2.0: Now uses the fix-loop boot test with surgical import repair.
3. Spec coverage — are all files from the original spec accounted for?
4. Final AI Review — quality, security, and performance assessment
   against the original POT spec. Advisory only (doesn't block pass/fail).
5. Deliverable summary — final report of what was built.

Token cost management:
  - Does NOT read the entire codebase. Scans only file metadata + targeted reads.
  - Boot test reads only failing files (parsed from traceback).
  - AI review reads only a SAMPLE of files (highest-risk: entry points, auth, API).
  - Surgical fixes write only the specific lines that need changing.

For single-phase jobs, this is largely redundant with Stage 9 but
still runs the spec coverage check, AI review, and produces the summary.

v2.0 (2026-02-15): Fix-loop boot test, surgical import repair, AI review.
v1.0 (2026-02-14): Initial implementation — Stage 10.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .construction_planner_models import ConstructionPlan
from .construction_skeleton import verify_phase_deliverables
from .phase_checkout_checks import (
    run_boot_test_with_fix_loop,
    _read_file_via_sandbox,
)
from app.pot_spec.grounded.size_models import MAX_FILE_LINES

logger = logging.getLogger(__name__)

FINAL_CHECKOUT_BUILD_ID = "2026-02-15-v2.0-surgical-fix-and-ai-review"
print(f"[FINAL_CHECKOUT_LOADED] BUILD_ID={FINAL_CHECKOUT_BUILD_ID}")


# =============================================================================
# RESULT MODELS
# =============================================================================

@dataclass
class SpecCoverageResult:
    """Did the build produce all files from the original spec?"""
    status: str  # "pass", "fail"
    expected_files: int = 0
    found_files: int = 0
    missing_files: List[str] = field(default_factory=list)
    extra_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "expected_files": self.expected_files,
            "found_files": self.found_files,
            "missing_files": self.missing_files,
            "extra_files": self.extra_files,
        }


@dataclass
class CrossPhaseResult:
    """Did all phase contracts get honoured?"""
    status: str  # "pass", "fail", "skipped"
    phases_checked: int = 0
    violations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "phases_checked": self.phases_checked,
            "violations": self.violations,
        }


@dataclass
class AIReviewResult:
    """Final AI quality/security/performance review."""
    status: str  # "pass", "fail", "error", "skipped"
    overall_score: float = 0.0  # 0-10
    quality_score: float = 0.0
    security_score: float = 0.0
    performance_score: float = 0.0
    critical_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    files_reviewed: int = 0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "overall_score": self.overall_score,
            "quality_score": self.quality_score,
            "security_score": self.security_score,
            "performance_score": self.performance_score,
            "critical_issues": self.critical_issues,
            "recommendations": self.recommendations,
            "files_reviewed": self.files_reviewed,
            "notes": self.notes,
        }


@dataclass
class FinalCheckoutResult:
    """Complete result of Final Checkout (Stage 10)."""
    job_id: str
    status: str = "pending"  # "pass", "fail", "error"
    boot_test_status: str = ""
    spec_coverage: Optional[SpecCoverageResult] = None
    cross_phase: Optional[CrossPhaseResult] = None
    ai_review: Optional[AIReviewResult] = None
    total_files_built: int = 0
    total_phases: int = 0
    duration_ms: int = 0
    timestamp: str = ""
    checks_run: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "boot_test_status": self.boot_test_status,
            "spec_coverage": self.spec_coverage.to_dict() if self.spec_coverage else None,
            "cross_phase": self.cross_phase.to_dict() if self.cross_phase else None,
            "ai_review": self.ai_review.to_dict() if self.ai_review else None,
            "total_files_built": self.total_files_built,
            "total_phases": self.total_phases,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "checks_run": self.checks_run,
        }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def run_final_checkout(
    job_id: str,
    plan: ConstructionPlan,
    original_file_scope: List[str],
    job_dir: str,
    sandbox_base: str = r"D:\Orb",
    original_spec: Optional[str] = None,
    state: Optional[Any] = None,
    emit: Optional[Callable] = None,
) -> FinalCheckoutResult:
    """
    Run final project-level verification after all phases complete.

    Args:
        job_id: Job identifier
        plan: The completed construction plan
        original_file_scope: Full file scope from the original spec
        job_dir: Job directory for saving result
        sandbox_base: Root path for file checks
        original_spec: The original POT spec markdown (for AI review)
        state: JobState with segment info (for boot fix mapping)
        emit: Progress callback

    Returns:
        FinalCheckoutResult with aggregated pass/fail
    """
    start = time.time()
    _emit = emit or (lambda msg: None)
    result = FinalCheckoutResult(job_id=job_id, total_phases=plan.total_phases)

    _emit(f"\n{'='*60}")
    _emit(">>> FINAL PROJECT CHECKOUT - Stage 10")
    _emit(f"   {plan.total_phases} phase(s), {len(original_file_scope)} expected files")

    # --- Check 1: Spec coverage ---
    _emit("\n[CHECK 1] Spec file coverage...")
    result.spec_coverage = _check_spec_coverage(original_file_scope, sandbox_base)
    result.checks_run.append("spec_coverage")
    result.total_files_built = result.spec_coverage.found_files

    if result.spec_coverage.status == "pass":
        _emit(f"  [OK] All {result.spec_coverage.expected_files} spec files found on disk")
    else:
        _emit(f"  [FAIL] {len(result.spec_coverage.missing_files)} file(s) missing:")
        for mf in result.spec_coverage.missing_files[:10]:
            _emit(f"    - {mf}")

    # --- Check 2: Cross-phase contract verification ---
    if plan.is_multi_phase:
        _emit("\n[CHECK 2] Cross-phase contract verification...")
        result.cross_phase = _check_cross_phase_contracts(plan, sandbox_base)
        result.checks_run.append("cross_phase")

        if result.cross_phase.status == "pass":
            _emit(f"  [OK] All {result.cross_phase.phases_checked} phase contracts honoured")
        else:
            _emit(f"  [FAIL] {len(result.cross_phase.violations)} contract violation(s)")
            for v in result.cross_phase.violations[:5]:
                _emit(f"    - Phase {v.get('phase_id', '?')}: {v.get('detail', '?')}")
    else:
        _emit("\n[CHECK 2] Cross-phase contracts -- SKIPPED (single phase)")

    # --- Check 3: Full project boot test with surgical fix loop ---
    _emit("\n[CHECK 3] Full project boot test (with surgical fix loop)...")
    # Build state for boot fix mapping if not provided
    _boot_state = state
    if _boot_state is None:
        _boot_state = _build_minimal_state_from_plan(plan, sandbox_base)

    boot = await run_boot_test_with_fix_loop(
        sandbox_base=sandbox_base,
        state=_boot_state,
        emit=_emit,
    )
    result.boot_test_status = boot.status
    result.checks_run.append("boot_test")

    if boot.status == "pass":
        _emit("  [OK] Project boots cleanly")
    elif boot.status == "fail":
        _emit(f"  [FAIL] Boot failed: {boot.error_summary}")
        if boot.traceback_file:
            _emit(f"    Failing file: {boot.traceback_file}")
    else:
        _emit(f"  [WARNING] Boot error: {boot.error_summary}")

    # --- Check 4: Final AI Review (quality, security, performance) ---
    _emit("\n[CHECK 4] Final AI Review (quality/security/performance)...")
    result.ai_review = await _run_final_ai_review(
        original_spec=original_spec,
        file_scope=original_file_scope,
        sandbox_base=sandbox_base,
        emit=_emit,
    )
    result.checks_run.append("ai_review")

    if result.ai_review and result.ai_review.status == "pass":
        _emit(f"  [OK] AI Review passed (score: {result.ai_review.overall_score}/10)")
    elif result.ai_review and result.ai_review.status == "fail":
        _emit(f"  [WARNING] AI Review flagged issues (score: {result.ai_review.overall_score}/10)")
        for issue in result.ai_review.critical_issues[:5]:
            _emit(f"    - {issue}")
    else:
        _emit("  [SKIPPED] AI Review could not run")

    # --- Aggregate ---
    # Boot test + spec coverage are pass/fail gates.
    # Cross-phase and AI review are advisory.
    all_ok = (
        result.spec_coverage.status == "pass"
        and result.boot_test_status == "pass"
    )
    result.status = "pass" if all_ok else "fail"
    result.duration_ms = int((time.time() - start) * 1000)

    if all_ok:
        _emit(f"\n>>> FINAL CHECKOUT PASSED - project fully verified")
    else:
        _emit(f"\n>>> FINAL CHECKOUT FAILED")
        if result.boot_test_status != "pass":
            _emit("   Reason: Boot test failed")
        if result.spec_coverage.status != "pass":
            _emit(f"   Reason: {len(result.spec_coverage.missing_files)} spec files missing")

    _emit(f"   Files built: {result.total_files_built}/{len(original_file_scope)}")
    _emit(f"   Duration: {result.duration_ms}ms")

    _save_result(result, job_dir)
    return result


# =============================================================================
# CHECK 1: SPEC COVERAGE
# =============================================================================

def _check_spec_coverage(
    file_scope: List[str],
    sandbox_base: str,
) -> SpecCoverageResult:
    """Verify all files from the original spec exist on disk."""
    missing = []
    found = 0

    for rel_path in file_scope:
        normalised = rel_path.replace("/", os.sep).replace("\\", os.sep)
        abs_path = os.path.join(sandbox_base, normalised)
        if os.path.isfile(abs_path):
            found += 1
        else:
            missing.append(rel_path)

    return SpecCoverageResult(
        status="fail" if missing else "pass",
        expected_files=len(file_scope),
        found_files=found,
        missing_files=missing,
    )


# =============================================================================
# CHECK 2: CROSS-PHASE CONTRACTS
# =============================================================================

def _check_cross_phase_contracts(
    plan: ConstructionPlan,
    sandbox_base: str,
) -> CrossPhaseResult:
    """Verify all phase contracts were honoured."""
    violations = []

    for phase in plan.phases:
        result = verify_phase_deliverables(plan, phase, sandbox_base)
        if result["status"] == "fail":
            for mf in result.get("missing", []):
                violations.append({
                    "phase_id": phase.phase_id,
                    "violation_type": "missing_deliverable",
                    "detail": f"Phase {phase.phase_number} promised '{mf}' but it's missing",
                })

    return CrossPhaseResult(
        status="fail" if violations else "pass",
        phases_checked=len(plan.phases),
        violations=violations,
    )


# =============================================================================
# CHECK 4: FINAL AI REVIEW
# =============================================================================

# Files to prioritise for review (highest risk, minimal token cost)
_REVIEW_PRIORITY_PATTERNS = [
    r"main\.py$",                    # Entry point
    r"__init__\.py$",                # Package init (import chains)
    r"auth|security|password",       # Security-sensitive
    r"api|router|endpoint|views",    # External-facing
    r"database|models|migration",    # Data integrity
    r"config|settings|env",          # Configuration
]

_MAX_REVIEW_FILES = 12       # Cap total files reviewed
_MAX_REVIEW_CHARS = 8000     # Cap per-file content in prompt
_REVIEW_MODEL = "claude-sonnet-4-5-20250929"  # Sonnet for cost efficiency


async def _run_final_ai_review(
    original_spec: Optional[str],
    file_scope: List[str],
    sandbox_base: str,
    emit: Optional[Callable] = None,
) -> AIReviewResult:
    """
    Final AI quality/security/performance review.

    Token cost management:
    - Only reviews a SAMPLE of high-risk files (entry points, auth, API)
    - Each file capped at 8K chars in the prompt
    - Uses Sonnet (not Opus) for cost efficiency
    - Total prompt stays under ~40K tokens

    This is ADVISORY — flagged issues don't block the checkout.
    """
    _emit = emit or (lambda msg: None)

    if not original_spec:
        return AIReviewResult(
            status="skipped",
            notes="No original spec provided for review comparison",
        )

    # --- Select files for review (prioritise high-risk) ---
    priority_files: List[str] = []
    other_files: List[str] = []

    for rel_path in file_scope:
        if any(re.search(pat, rel_path, re.IGNORECASE) for pat in _REVIEW_PRIORITY_PATTERNS):
            priority_files.append(rel_path)
        else:
            other_files.append(rel_path)

    # Take all priority files + fill remaining slots with others
    review_files = priority_files[:_MAX_REVIEW_FILES]
    remaining_slots = _MAX_REVIEW_FILES - len(review_files)
    if remaining_slots > 0:
        review_files.extend(other_files[:remaining_slots])

    if not review_files:
        return AIReviewResult(
            status="skipped",
            notes="No files available for review",
        )

    # --- Read file contents ---
    file_contents: Dict[str, str] = {}
    for rel_path in review_files:
        abs_path = os.path.join(sandbox_base, rel_path.replace("/", os.sep))
        if os.path.isfile(abs_path):
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read(_MAX_REVIEW_CHARS)
                file_contents[rel_path] = content
            except Exception:
                pass

    if not file_contents:
        return AIReviewResult(
            status="skipped",
            notes="Could not read any files for review",
        )

    _emit(f"  Reviewing {len(file_contents)} file(s) (of {len(file_scope)} total)")

    # --- Build review prompt ---
    prompt = _build_ai_review_prompt(original_spec, file_contents)

    # --- Call LLM ---
    try:
        from app.providers.registry import llm_call

        provider_id = os.getenv("ASTRA_REVIEW_PROVIDER", "anthropic")
        model_id = os.getenv("ASTRA_REVIEW_MODEL", _REVIEW_MODEL)

        llm_result = await llm_call(
            provider_id=provider_id,
            model_id=model_id,
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_REVIEW_SYSTEM_PROMPT,
            max_tokens=4096,
            timeout_seconds=120,
        )

        response = llm_result.content if llm_result else None
        if not response:
            return AIReviewResult(status="error", notes="Empty LLM response")

        return _parse_review_response(response, len(file_contents))

    except Exception as exc:
        logger.warning("[final_checkout] AI review failed: %s", exc)
        return AIReviewResult(
            status="error",
            notes=f"AI review failed: {exc}",
        )


_REVIEW_SYSTEM_PROMPT = """\
You are a senior software reviewer performing a final quality gate assessment.
Review the code files against the original specification.

Focus on:
1. QUALITY: Code structure, error handling, naming consistency, dead code
2. SECURITY: Input validation, injection risks, hardcoded secrets, auth gaps
3. PERFORMANCE: Obvious bottlenecks, N+1 queries, unbounded loops, missing caching

Be concise and actionable. Only flag real issues, not style preferences.
Respond with valid JSON only.\
"""


def _build_ai_review_prompt(
    spec: str,
    file_contents: Dict[str, str],
) -> str:
    """Build the AI review prompt with spec + sampled files."""
    parts = [
        "# Final AI Review\n",
        "## Original Specification (summary)\n",
    ]

    # Cap spec at 6K chars
    spec_display = spec[:6000]
    if len(spec) > 6000:
        spec_display += "\n... [truncated]"
    parts.append(spec_display)
    parts.append("\n\n## Code Files for Review\n")

    for path, content in file_contents.items():
        parts.append(f"### `{path}`\n```python\n{content}\n```\n")

    parts.append("""
## Response Format

```json
{
  "overall_score": 7.5,
  "quality_score": 8.0,
  "security_score": 7.0,
  "performance_score": 7.5,
  "critical_issues": [
    "Brief description of critical issue 1",
    "Brief description of critical issue 2"
  ],
  "recommendations": [
    "Brief actionable recommendation 1",
    "Brief actionable recommendation 2"
  ],
  "notes": "Optional overall assessment"
}
```

Scores are 0-10. critical_issues are things that should be fixed before production.
recommendations are improvements. Keep both lists under 8 items each.
""")

    return "\n".join(parts)


def _parse_review_response(response: str, files_reviewed: int) -> AIReviewResult:
    """Parse the LLM review response."""
    cleaned = response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    # Clean trailing commas
    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("[final_checkout] Failed to parse AI review: %s", e)
        return AIReviewResult(
            status="error",
            notes=f"Failed to parse review response: {e}",
            files_reviewed=files_reviewed,
        )

    overall = float(data.get("overall_score", 0))
    return AIReviewResult(
        status="pass" if overall >= 6.0 else "fail",
        overall_score=overall,
        quality_score=float(data.get("quality_score", 0)),
        security_score=float(data.get("security_score", 0)),
        performance_score=float(data.get("performance_score", 0)),
        critical_issues=data.get("critical_issues", [])[:8],
        recommendations=data.get("recommendations", [])[:8],
        files_reviewed=files_reviewed,
        notes=data.get("notes", ""),
    )


# =============================================================================
# HELPERS
# =============================================================================

def _build_minimal_state_from_plan(
    plan: ConstructionPlan,
    sandbox_base: str,
) -> Any:
    """
    Build a minimal state-like object from a ConstructionPlan for boot fix mapping.

    The boot fix loop needs state.segments to map failing files to segments.
    When called from final checkout (cross-phase), we may not have a single
    JobState — build a compatible shim from the plan.
    """
    class _MinimalSegState:
        def __init__(self, output_files):
            self.output_files = output_files
            self.status = "complete"

    class _MinimalState:
        def __init__(self):
            self.segments = {}

    state = _MinimalState()

    # Collect files from all phases
    for phase in plan.phases:
        for file_path in phase.file_scope:
            # Use phase_id as pseudo-segment
            state.segments[phase.phase_id] = _MinimalSegState(phase.file_scope)

    return state


# =============================================================================
# PERSISTENCE
# =============================================================================

def _save_result(result: FinalCheckoutResult, job_dir: str) -> None:
    """Save final checkout result to job directory."""
    path = os.path.join(job_dir, "final_checkout_result.json")
    try:
        os.makedirs(job_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("[final_checkout] Result saved to %s", path)
    except Exception as exc:
        logger.warning("[final_checkout] Failed to save: %s", exc)
