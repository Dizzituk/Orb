from __future__ import annotations
import logging
from app.orchestrator._cohesion_check_utils_10 import CohesionResult, load_segment_architectures
from app.orchestrator._cohesion_check_utils_8 import _classify_fix_tier, _save_patched_architecture
from app.orchestrator._cohesion_check_utils_9 import _apply_tier1_fix, _apply_tier2_fix
from typing import Dict, List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


async def attempt_auto_fixes(
    result: CohesionResult,
    job_dir: str,
    architectures: Dict[str, str],
    skeleton_json: Optional[str] = None,
    manifest_dict: Optional[dict] = None,
    tier2_provider: str = "anthropic",
    tier2_model: str = "claude-sonnet-4-5-20250929",
    max_tier2_fixes: int = 3,
) -> CohesionResult:
    """
    Attempt to auto-fix cohesion issues using tiered approach.

    1. Classify each issue into tier (1, 2, or 3)
    2. Apply Tier 1 (deterministic) fixes — zero API cost
    3. Apply Tier 2 (micro-LLM) fixes — tiny API cost
    4. Save patched architectures to disk
    5. Re-validate with skeleton compliance
    6. Return updated result

    Tier 3 issues are left untouched for the existing regen flow.

    Args:
        result: The CohesionResult from initial check
        job_dir: Path to job directory
        architectures: {segment_id: architecture_content}
        skeleton_json: Skeleton contracts JSON
        manifest_dict: Manifest dict for re-validation
        tier2_provider: LLM provider for Tier 2 fixes
        tier2_model: LLM model for Tier 2 fixes
        max_tier2_fixes: Maximum number of Tier 2 LLM calls to make

    Returns:
        Updated CohesionResult with fixed issues marked
    """
    from .cohesion_check import run_skeleton_compliance
    if not result.issues:
        return result

    # =========================================================================
    # Step 1: Classify all issues
    # =========================================================================
    for issue in result.issues:
        issue.auto_fix_tier = _classify_fix_tier(issue)
        logger.debug(
            "[cohesion_auto_fix] %s (%s/%s) → Tier %d",
            issue.issue_id, issue.category, issue.severity, issue.auto_fix_tier,
        )

    tier1_issues = [i for i in result.issues if i.auto_fix_tier == 1]
    tier2_issues = [i for i in result.issues if i.auto_fix_tier == 2]
    tier3_issues = [i for i in result.issues if i.auto_fix_tier == 3]

    logger.info(
        "[cohesion_auto_fix] Classification: %d Tier-1, %d Tier-2, %d Tier-3",
        len(tier1_issues), len(tier2_issues), len(tier3_issues),
    )

    if not tier1_issues and not tier2_issues:
        logger.info("[cohesion_auto_fix] No auto-fixable issues found")
        return result

    # =========================================================================
    # Step 2: Apply Tier 1 fixes (deterministic, per-segment)
    # =========================================================================
    patched_archs: Dict[str, str] = {}  # seg_id → patched text
    fix_log: Dict[str, List[str]] = {}  # seg_id → list of fix notes
    tier1_fixed = 0

    for issue in tier1_issues:
        seg_id = issue.source_segment
        if not seg_id or seg_id not in architectures:
            continue

        current_text = patched_archs.get(seg_id, architectures[seg_id])
        patched = _apply_tier1_fix(issue, current_text)

        if patched and patched != current_text:
            patched_archs[seg_id] = patched
            fix_log.setdefault(seg_id, []).append(issue.auto_fix_note)
            issue.auto_fixed = True
            tier1_fixed += 1
            logger.info(
                "[cohesion_auto_fix] Tier 1 FIX: %s in %s — %s",
                issue.issue_id, seg_id, issue.auto_fix_note,
            )
        else:
            logger.warning(
                "[cohesion_auto_fix] Tier 1 SKIP: %s in %s — pattern not found in arch text",
                issue.issue_id, seg_id,
            )
            # v5.20: When Tier 1 fails for missing_import, cascade to Tier 2
            # instead of immediately escalating to blocking (which triggers
            # expensive Opus regen). Tier 2 can often patch it for ~500 tokens.
            if issue.category == "missing_import" and issue.severity == "warning":
                if issue not in tier2_issues:
                    issue.auto_fix_tier = 2
                    tier2_issues.append(issue)
                    logger.info(
                        "[cohesion_auto_fix] Tier 1→2 CASCADE: %s — will attempt micro-LLM fix",
                        issue.issue_id,
                    )

    print(f"[cohesion_auto_fix] Tier 1: {tier1_fixed}/{len(tier1_issues)} fixes applied")

    # =========================================================================
    # Step 3: Apply Tier 2 fixes (micro-LLM, per-segment)
    # =========================================================================
    tier2_fixed = 0
    tier2_calls = 0

    for issue in tier2_issues:
        if tier2_calls >= max_tier2_fixes:
            logger.info(
                "[cohesion_auto_fix] Tier 2: reached max calls (%d), skipping rest",
                max_tier2_fixes,
            )
            break

        seg_id = issue.source_segment
        if not seg_id or seg_id not in architectures:
            continue

        current_text = patched_archs.get(seg_id, architectures[seg_id])
        tier2_calls += 1

        patched = await _apply_tier2_fix(
            issue=issue,
            arch_text=current_text,
            seg_id=seg_id,
            provider=tier2_provider,
            model=tier2_model,
        )

        if patched and patched != current_text:
            patched_archs[seg_id] = patched
            fix_log.setdefault(seg_id, []).append(issue.auto_fix_note)
            issue.auto_fixed = True
            tier2_fixed += 1
            logger.info(
                "[cohesion_auto_fix] Tier 2 FIX: %s in %s — %s",
                issue.issue_id, seg_id, issue.auto_fix_note,
            )
        else:
            logger.warning(
                "[cohesion_auto_fix] Tier 2 SKIP: %s in %s — LLM fix failed",
                issue.issue_id, seg_id,
            )

    print(
        f"[cohesion_auto_fix] Tier 2: {tier2_fixed}/{len(tier2_issues)} fixes applied "
        f"({tier2_calls} LLM call(s))"
    )

    # v5.20: Escalate any missing_import issues that BOTH Tier 1 and Tier 2 failed to fix.
    # These are guaranteed NameErrors at runtime and must trigger regen.
    for issue in tier2_issues:
        if not issue.auto_fixed and issue.category == "missing_import" and issue.severity == "warning":
            issue.severity = "blocking"
            issue.auto_fix_note = (
                "Tier 1+2 fix FAILED \u2014 escalated to blocking. "
                "Missing import will cause NameError at runtime."
            )
            logger.warning(
                "[cohesion_auto_fix] \u26a0\ufe0f ESCALATED %s to blocking \u2014 Tier 1+2 both failed",
                issue.issue_id,
            )

    # =========================================================================
    # Step 4: Save patched architectures to disk
    # =========================================================================
    if not patched_archs:
        logger.info("[cohesion_auto_fix] No patches applied — skipping save")
        return result

    for seg_id, patched_text in patched_archs.items():
        notes = fix_log.get(seg_id, [])
        saved_path = _save_patched_architecture(job_dir, seg_id, patched_text, notes)
        print(f"[cohesion_auto_fix] 💾 Saved: {saved_path}")

    # =========================================================================
    # Step 5: Re-validate with skeleton compliance
    # =========================================================================
    print("[cohesion_auto_fix] 🔍 Re-validating after auto-fixes...")

    # Reload architectures (now includes patched versions)
    segment_ids = list(architectures.keys())
    reloaded_archs = load_segment_architectures(job_dir, segment_ids)

    recheck_issues = run_skeleton_compliance(
        architectures=reloaded_archs,
        skeleton_json=skeleton_json,
        manifest_dict=manifest_dict,
    )

    # =========================================================================
    # Step 6: Build updated result
    # =========================================================================
    # Keep unfixed issues + any NEW issues from re-validation
    # Remove issues that were fixed and no longer appear in re-check
    recheck_ids = {(i.category, i.source_segment, i.description[:80]) for i in recheck_issues}

    updated_issues = []

    # Add fixed issues as resolved (downgraded to info)
    for issue in result.issues:
        if issue.auto_fixed:
            # Check if it still appears in re-validation
            key = (issue.category, issue.source_segment, issue.description[:80])
            if key in recheck_ids:
                # Fix didn't work — keep as original severity
                issue.auto_fixed = False
                issue.auto_fix_note += " (FIX FAILED — issue persists)"
                updated_issues.append(issue)
                logger.warning(
                    "[cohesion_auto_fix] Fix FAILED for %s — issue persists after patch",
                    issue.issue_id,
                )
            else:
                # Fix worked — downgrade to resolved
                issue.severity = "resolved"
                updated_issues.append(issue)
                logger.info(
                    "[cohesion_auto_fix] ✅ Fix CONFIRMED for %s",
                    issue.issue_id,
                )
        else:
            # Unfixed issue — check if resolved as side-effect of another fix
            key = (issue.category, issue.source_segment, issue.description[:80])
            if key not in recheck_ids:
                # Side-effect fix! Issue no longer appears in re-validation
                issue.severity = "resolved"
                issue.auto_fixed = True
                issue.auto_fix_note = "Resolved as side-effect of related fix"
                updated_issues.append(issue)
                logger.info(
                    "[cohesion_auto_fix] ✅ Side-effect fix for %s — resolved by related patch",
                    issue.issue_id,
                )

                # v3.2: Annotate affected arch text so implementer knows about the change
                _affected_seg = issue.related_segment or issue.source_segment
                if _affected_seg and _affected_seg in patched_archs:
                    _annotation = (
                        f"\n\n<!-- COHESION ANNOTATION ({issue.issue_id}): "
                        f"{issue.description[:200]} "
                        f"Fix: {issue.suggested_fix[:200]} "
                        f"(resolved as side-effect of related fix) -->\n"
                    )
                    patched_archs[_affected_seg] += _annotation
                    fix_log.setdefault(_affected_seg, []).append(
                        f"Side-effect annotation for {issue.issue_id}"
                    )
                    logger.info(
                        "[cohesion_auto_fix] 📝 Annotated %s arch for side-effect fix %s",
                        _affected_seg, issue.issue_id,
                    )
                elif _affected_seg and _affected_seg in architectures:
                    # Segment wasn't patched yet — start a new patch
                    _annotation = (
                        f"\n\n<!-- COHESION ANNOTATION ({issue.issue_id}): "
                        f"{issue.description[:200]} "
                        f"Fix: {issue.suggested_fix[:200]} "
                        f"(resolved as side-effect of related fix) -->\n"
                    )
                    patched_archs[_affected_seg] = architectures[_affected_seg] + _annotation
                    fix_log.setdefault(_affected_seg, []).append(
                        f"Side-effect annotation for {issue.issue_id}"
                    )
                    logger.info(
                        "[cohesion_auto_fix] 📝 Annotated %s arch for side-effect fix %s (new patch)",
                        _affected_seg, issue.issue_id,
                    )
            else:
                updated_issues.append(issue)

    # Add any NEW issues from re-validation that weren't in original
    original_ids = {(i.category, i.source_segment, i.description[:80]) for i in result.issues}
    for new_issue in recheck_issues:
        key = (new_issue.category, new_issue.source_segment, new_issue.description[:80])
        if key not in original_ids:
            new_issue.auto_fix_note = "NEW: appeared after auto-fix patching"
            updated_issues.append(new_issue)
            logger.warning(
                "[cohesion_auto_fix] NEW issue after patching: %s",
                new_issue.issue_id,
            )

    result.issues = updated_issues

    # Recalculate status
    remaining_blocking = [
        i for i in result.issues
        if i.severity == "blocking"
    ]
    result.status = "fail" if remaining_blocking else "pass"

    total_fixed = tier1_fixed + tier2_fixed
    remaining = len(remaining_blocking)
    result.notes = (
        f"Auto-fix: {total_fixed} fixed "
        f"(T1:{tier1_fixed}, T2:{tier2_fixed}), "
        f"{remaining} blocking remain, "
        f"{len(tier3_issues)} deferred to regen"
    )

    print(
        f"[cohesion_auto_fix] ═══ RESULT: {total_fixed} fixed, "
        f"{remaining} blocking remain ═══"
    )

    return result
