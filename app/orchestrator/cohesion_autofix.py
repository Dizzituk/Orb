# FILE: app/orchestrator/cohesion_autofix.py
"""
Three-Tier Cohesion Auto-Fix System

When cohesion check finds issues, this module attempts to fix them
in-place on the architecture markdown before falling back to expensive
full regeneration.

Tier 1 — Deterministic (zero API cost):
    Regex/string replacements for known patterns:
    - Import depth fixes (from .X → from ..X)
    - Missing stdlib imports (import logging + logger setup)
    - Naming mismatches (function name corrections)

Tier 2 — Micro LLM patch (tiny API cost):
    Small, focused LLM call with ONLY the affected section and the fix
    instruction. Used for issues that need context-aware editing but
    don't require a full redraft. ~500-1000 tokens.

Tier 3 — Full regeneration (existing pipeline):
    Falls through to the existing targeted regen path for structural
    issues that can't be patched.

v1.0 (2026-02-13): Initial implementation — full three-tier system.
"""

from __future__ import annotations

import logging
import os
import re
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from app.orchestrator._cohesion_autofix_utils import COHESION_AUTOFIX_BUILD_ID, _build_micro_patch_prompt, _build_sibling_export_context, _fix_import_depth, _fix_missing_import, _fix_naming_mismatch, _is_import_depth_issue, _save_patched_architecture

logger = logging.getLogger(__name__)
print(f"[COHESION_AUTOFIX_LOADED] BUILD_ID={COHESION_AUTOFIX_BUILD_ID}")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class AutofixAttempt:
    """Record of a single fix attempt on one issue."""
    issue_id: str
    tier: int  # 1, 2, or 3
    success: bool
    description: str = ""
    changes_made: str = ""

@dataclass
class AutofixResult:
    """Result of the full autofix pass."""
    attempts: List[AutofixAttempt] = field(default_factory=list)
    architectures_modified: Dict[str, str] = field(default_factory=dict)  # seg_id → patched text
    issues_fixed: List[str] = field(default_factory=list)  # issue_ids
    issues_remaining: List[str] = field(default_factory=list)  # issue_ids needing regen
    tier2_tokens_used: int = 0

    @property
    def any_fixed(self) -> bool:
        return len(self.issues_fixed) > 0

    @property
    def all_fixed(self) -> bool:
        return len(self.issues_remaining) == 0


# =============================================================================
# TIER CLASSIFICATION
# =============================================================================

def classify_issue(issue) -> int:
    """
    Classify a CohesionIssue into a fix tier.

    Returns:
        1 = deterministic regex fix
        2 = micro LLM patch
        3 = full regeneration required
    """
    cat = issue.category.lower()
    desc = issue.description.lower()
    fix = (issue.suggested_fix or "").lower()

    # ----- TIER 1: Deterministic patterns -----

    # Import depth fix: "from .X" → "from ..X"
    if cat == "import_mismatch" and _is_import_depth_issue(issue):
        return 1

    # Missing stdlib import (logging, os, json, etc.)
    if cat == "missing_import" and ("import logging" in desc or "import logging" in fix):
        return 1

    # Naming mismatch with known expected/actual
    if cat == "naming_mismatch" and issue.expected and issue.actual:
        return 1

    # ----- TIER 2: Micro LLM patch -----

    # Missing exports in __init__.py
    if cat in ("missing_export", "contract_violation") and ("re-export" in desc or "export" in fix):
        return 2

    # Import mismatch that isn't a simple depth fix
    if cat == "import_mismatch" and not _is_import_depth_issue(issue):
        return 2

    # Any issue with a clear suggested_fix that we can't pattern-match
    if issue.suggested_fix and len(issue.suggested_fix) > 10:
        # Has a substantive fix suggestion — try micro LLM
        return 2

    # ----- TIER 3: Full regen -----
    return 3


# =============================================================================
# TIER 1: DETERMINISTIC FIXES
# =============================================================================

def apply_tier1_fix(issue, arch_text: str) -> Tuple[str, bool, str]:
    """
    Apply a deterministic fix to architecture text.

    Returns:
        (fixed_text, success, description_of_change)
    """
    cat = issue.category.lower()

    if cat == "import_mismatch":
        return _fix_import_depth(issue, arch_text)
    elif cat == "missing_import":
        return _fix_missing_import(issue, arch_text)
    elif cat == "naming_mismatch":
        return _fix_naming_mismatch(issue, arch_text)

    return arch_text, False, "No Tier 1 handler for this category"


# =============================================================================
# TIER 2: MICRO LLM PATCH
# =============================================================================

async def apply_tier2_fix(
    issue,
    arch_text: str,
    segment_id: str,
    job_dir: str = "",
) -> Tuple[str, bool, str, int]:
    """
    Apply a micro LLM patch to fix a specific issue.

    Uses a small, cheap model (gpt-4.1-mini) with a focused prompt
    containing only the relevant section and the fix instruction.

    Returns:
        (fixed_text, success, description, tokens_used)
    """
    try:
        from app.llm.stage_models import get_stage_config
        # Use a cheap model for micro patches
        try:
            cfg = get_stage_config("COHESION_MICRO_PATCH")
            provider = cfg.provider
            model = cfg.model
        except Exception:
            # Fallback to cheapest available
            provider = "openai"
            model = "gpt-4.1-mini"
    except ImportError:
        provider = "openai"
        model = "gpt-4.1-mini"

    # Build focused prompt (v3.8: now includes sibling export evidence)
    prompt = _build_micro_patch_prompt(issue, arch_text, segment_id, job_dir=job_dir)

    try:
        from app.llm.streaming import call_llm_text
        response = await call_llm_text(
            provider=provider,
            model=model,
            system_prompt=(
                "You are an architecture document editor. You receive an architecture "
                "markdown document and a specific issue to fix. Apply ONLY the requested "
                "fix — do not change anything else. Return the COMPLETE fixed document "
                "with no preamble, no explanation, just the document."
            ),
            user_prompt=prompt,
            max_tokens=len(arch_text) // 2 + 2000,  # Enough for the full doc
            timeout_seconds=300,  # v3.3: Large arch docs need more time
        )

        if not response or not response.strip():
            return arch_text, False, "Empty LLM response", 0

        # Validate the response looks like architecture markdown
        fixed_text = response.strip()
        if len(fixed_text) < len(arch_text) * 0.5:
            logger.warning(
                "[cohesion_autofix] Tier 2 response too short (%d vs %d) — rejecting",
                len(fixed_text), len(arch_text),
            )
            return arch_text, False, "Response too short — likely truncated", 0

        # Rough token estimate
        tokens_used = (len(prompt) + len(fixed_text)) // 4

        return fixed_text, True, f"Micro LLM patch applied via {provider}/{model}", tokens_used

    except ImportError:
        logger.warning("[cohesion_autofix] LLM module not available for Tier 2")
        return arch_text, False, "LLM module not available", 0
    except Exception as e:
        # v3.3: Retry once on timeout errors
        is_timeout = "timeout" in str(e).lower() or "ReadTimeout" in type(e).__name__
        if is_timeout:
            logger.warning("[cohesion_autofix] Tier 2 timeout, retrying once: %s", e)
            try:
                response = await call_llm_text(
                    provider=provider,
                    model=model,
                    system_prompt=(
                        "You are an architecture document editor. Apply ONLY the requested "
                        "fix. Return the COMPLETE fixed document with no preamble."
                    ),
                    user_prompt=prompt,
                    max_tokens=len(arch_text) // 2 + 2000,
                    timeout_seconds=600,  # Double timeout on retry
                )
                if response and response.strip():
                    fixed_text = response.strip()
                    if len(fixed_text) >= len(arch_text) * 0.5:
                        tokens_used = (len(prompt) + len(fixed_text)) // 4
                        return fixed_text, True, f"Micro LLM patch applied (retry) via {provider}/{model}", tokens_used
            except Exception as retry_err:
                logger.warning("[cohesion_autofix] Tier 2 retry also failed: %s", retry_err)
        else:
            logger.warning("[cohesion_autofix] Tier 2 LLM call failed: %s", e)
        return arch_text, False, f"LLM call failed: {e}", 0


# =============================================================================
# ORCHESTRATOR — Main entry point
# =============================================================================

async def run_autofix(
    cohesion_result,
    architectures: Dict[str, str],
    job_dir: str,
    on_progress=None,
) -> AutofixResult:
    """
    Run the three-tier autofix system on cohesion issues.

    Args:
        cohesion_result: CohesionResult from the cohesion check
        architectures: {segment_id: architecture_text} dict
        job_dir: Job directory path (for saving patched architectures)
        on_progress: Optional callback for UI updates

    Returns:
        AutofixResult with details of what was fixed and what remains
    """
    result = AutofixResult()

    def _emit(msg):
        if on_progress:
            on_progress(msg)
        logger.info("[cohesion_autofix] %s", msg)

    # Collect ALL issues (blocking + warnings that are fixable)
    all_issues = cohesion_result.blocking_issues + [
        w for w in cohesion_result.warning_issues
        if w.category in ("missing_import", "naming_mismatch", "import_mismatch")
    ]

    if not all_issues:
        _emit("No fixable issues found")
        return result

    # Classify all issues
    classified = []
    for issue in all_issues:
        tier = classify_issue(issue)
        classified.append((issue, tier))
        _emit(f"  📋 {issue.issue_id} [{issue.category}] → Tier {tier}")

    # Track which architectures have been modified
    patched_archs = dict(architectures)  # Working copy

    # ----- TIER 1: Deterministic fixes -----
    tier1_issues = [(i, t) for i, t in classified if t == 1]
    if tier1_issues:
        _emit(f"\n🔧 Tier 1: Applying {len(tier1_issues)} deterministic fix(es)...")

    for issue, _ in tier1_issues:
        seg_id = issue.source_segment
        if seg_id not in patched_archs:
            result.attempts.append(AutofixAttempt(
                issue_id=issue.issue_id, tier=1, success=False,
                description=f"Segment {seg_id} not in architectures dict",
            ))
            result.issues_remaining.append(issue.issue_id)
            continue

        arch_text = patched_archs[seg_id]
        fixed_text, success, change_desc = apply_tier1_fix(issue, arch_text)

        attempt = AutofixAttempt(
            issue_id=issue.issue_id, tier=1, success=success,
            description=change_desc,
            changes_made=change_desc if success else "",
        )
        result.attempts.append(attempt)

        if success:
            patched_archs[seg_id] = fixed_text
            result.architectures_modified[seg_id] = fixed_text
            result.issues_fixed.append(issue.issue_id)
            _emit(f"  ✅ {issue.issue_id}: {change_desc}")
        else:
            # Tier 1 failed — escalate to Tier 2
            _emit(f"  ⚠️ {issue.issue_id}: Tier 1 failed ({change_desc}) — escalating to Tier 2")
            classified = [
                (i, 2 if i.issue_id == issue.issue_id else t)
                for i, t in classified
            ]

    # ----- TIER 2: Micro LLM patches -----
    tier2_issues = [(i, t) for i, t in classified if t == 2 and i.issue_id not in result.issues_fixed]
    if tier2_issues:
        _emit(f"\n🤖 Tier 2: Applying {len(tier2_issues)} micro LLM patch(es)...")

    for issue, _ in tier2_issues:
        seg_id = issue.source_segment
        if seg_id not in patched_archs:
            result.attempts.append(AutofixAttempt(
                issue_id=issue.issue_id, tier=2, success=False,
                description=f"Segment {seg_id} not in architectures dict",
            ))
            result.issues_remaining.append(issue.issue_id)
            continue

        arch_text = patched_archs[seg_id]
        fixed_text, success, change_desc, tokens = await apply_tier2_fix(
            issue, arch_text, seg_id, job_dir=job_dir,
        )

        attempt = AutofixAttempt(
            issue_id=issue.issue_id, tier=2, success=success,
            description=change_desc,
            changes_made=change_desc if success else "",
        )
        result.attempts.append(attempt)
        result.tier2_tokens_used += tokens

        if success:
            patched_archs[seg_id] = fixed_text
            result.architectures_modified[seg_id] = fixed_text
            result.issues_fixed.append(issue.issue_id)
            _emit(f"  ✅ {issue.issue_id}: {change_desc} (~{tokens} tokens)")
        else:
            # Tier 2 failed — escalate to Tier 3
            _emit(f"  ❌ {issue.issue_id}: Tier 2 failed ({change_desc}) — needs full regen")
            result.issues_remaining.append(issue.issue_id)

    # ----- TIER 3: Remaining issues -----
    tier3_issues = [(i, t) for i, t in classified if t == 3 and i.issue_id not in result.issues_fixed]
    for issue, _ in tier3_issues:
        result.attempts.append(AutofixAttempt(
            issue_id=issue.issue_id, tier=3, success=False,
            description="Structural issue — requires full regeneration",
        ))
        result.issues_remaining.append(issue.issue_id)
        _emit(f"  🔄 {issue.issue_id}: Tier 3 — queued for full regen")

    # ----- Save patched architectures to disk -----
    if result.architectures_modified:
        _emit(f"\n💾 Saving {len(result.architectures_modified)} patched architecture(s)...")
        for seg_id, patched_text in result.architectures_modified.items():
            _save_patched_architecture(seg_id, patched_text, job_dir)
            _emit(f"  💾 Saved patched architecture for {seg_id}")

    # ----- Summary -----
    _emit(f"\n📊 Autofix complete: {len(result.issues_fixed)} fixed, "
           f"{len(result.issues_remaining)} remaining")
    if result.tier2_tokens_used > 0:
        _emit(f"  💰 Tier 2 token usage: ~{result.tier2_tokens_used}")

    return result


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "classify_issue",
    "apply_tier1_fix",
    "apply_tier2_fix",
    "run_autofix",
    "AutofixResult",
    "AutofixAttempt",
    "COHESION_AUTOFIX_BUILD_ID",
]
