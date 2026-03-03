from __future__ import annotations
import logging
import os
import re
import time
from .construction_planner_models import ConstructionPlan
from .construction_skeleton import verify_phase_deliverables
from .strike_tracker import FixOutcome, StrikeTracker, StrikeVerdict
from app.orchestrator._final_checkout_utils_2 import _MAX_REVIEW_CHARS, _MAX_REVIEW_FILES, _REVIEW_PRIORITY_PATTERNS, _REVIEW_SYSTEM_PROMPT
from app.orchestrator._final_checkout_utils_3 import _build_ai_review_prompt, _detect_and_fix_integration_issues
from app.orchestrator._final_checkout_utils_4 import _apply_ai_review_fix, _generate_missing_file, _parse_review_response
from typing import Any, Callable, Dict, List, Optional
logger = logging.getLogger(__name__)

# v3.2-fix: Sandbox-aware filesystem checks for codebase paths.
from app.sandbox_fs import (
    sandbox_isfile as _sbx_isfile,
    sandbox_isdir as _sbx_isdir,
    sandbox_exists as _sbx_exists,
    sandbox_read_text as _sbx_read_text,
)
logger = logging.getLogger(__name__)


async def _check_spec_coverage_with_fixes(
    file_scope: List[str],
    sandbox_base: str,
    sandbox_client: Any,
    original_spec: Optional[str],
    tracker: StrikeTracker,
    emit: Optional[Callable] = None,
) -> SpecCoverageResult:
    """
    Verify all spec files exist. If missing, attempt to generate them.

    Uses StrikeTracker: if generation fails for the same file repeatedly,
    escalate after 3 strikes.
    """
    from .final_checkout import SpecCoverageResult, _file_exists_on_sandbox
    _emit = emit or (lambda msg: None)

    # First pass: check what exists
    missing = []
    found = 0
    for rel_path in file_scope:
        exists = _file_exists_on_sandbox(sandbox_client, rel_path, sandbox_base)
        if exists:
            found += 1
        else:
            missing.append(rel_path)

    if not missing:
        tracker.record_resolution()
        return SpecCoverageResult(
            status="pass",
            expected_files=len(file_scope),
            found_files=found,
        )

    _emit(f"  {len(missing)} file(s) missing from spec — attempting generation...")
    generated = []

    for rel_path in missing:
        fix_start = time.time()
        error_text = f"missing_file:{rel_path}"
        verdict = tracker.report_error(error_text)

        if verdict == StrikeVerdict.HARD_STOP:
            _emit(f"    [STRIKE 3] Hard stop on {rel_path} — cannot generate")
            tracker.record_hard_stop(error_text)
            break

        strategy = "generate_from_spec" if verdict == StrikeVerdict.PROCEED else "generate_stub"
        _emit(f"    [{verdict.value}] Generating {rel_path} (strategy: {strategy})")

        success = await _generate_missing_file(
            rel_path=rel_path,
            sandbox_base=sandbox_base,
            sandbox_client=sandbox_client,
            original_spec=original_spec,
            strategy=strategy,
            emit=_emit,
        )

        duration = int((time.time() - fix_start) * 1000)

        if success:
            generated.append(rel_path)
            found += 1
            tracker.record_attempt(
                error_detail=error_text,
                fix_strategy=strategy,
                fix_description=f"Generated {rel_path}",
                outcome=FixOutcome.RESOLVED,
                duration_ms=duration,
            )
        else:
            tracker.record_attempt(
                error_detail=error_text,
                fix_strategy=strategy,
                fix_description=f"Failed to generate {rel_path}",
                outcome=FixOutcome.SAME_ERROR,
                duration_ms=duration,
            )

    # Recheck
    still_missing = [f for f in missing if f not in generated]

    if not still_missing:
        tracker.record_resolution()

    return SpecCoverageResult(
        status="pass" if not still_missing else "fail",
        expected_files=len(file_scope),
        found_files=found,
        missing_files=still_missing,
        files_generated=generated,
    )

async def _check_cross_phase_with_fixes(
    plan: ConstructionPlan,
    sandbox_base: str,
    sandbox_client: Any,
    tracker: StrikeTracker,
    emit: Optional[Callable] = None,
    state: Any = None,
    manifest: Any = None,
) -> CrossPhaseResult:
    """
    Verify cross-phase contracts and fix violations.

    Three layers of checking:
    1. Deliverable existence: does each promised file exist?
    2. Import resolution: do cross-segment imports resolve correctly?
    3. Interface contracts: do exposes/consumes match actual definitions?

    Fix strategy:
    - missing_deliverable: generate the file from spec
    - import_mismatch: read the provider file to see what it exports,
      fix the consumer's import to use the correct name. Provider is
      ground truth (written first), consumer is what gets fixed.
    - missing_export: add the missing name to the provider's __all__ or
      re-export it from __init__.py
    """
    from .final_checkout import CrossPhaseResult, _file_exists_on_sandbox
    _emit = emit or (lambda msg: None)
    violations = []
    fixes_applied = []

    # --- Layer 1: Deliverable existence ---
    for phase in plan.phases:
        result = verify_phase_deliverables(plan, phase, sandbox_base)
        if result["status"] == "fail":
            for mf in result.get("missing", []):
                violation = {
                    "phase_id": phase.phase_id,
                    "violation_type": "missing_deliverable",
                    "detail": f"Phase {phase.phase_number} promised '{mf}' but it's missing",
                }

                error_text = f"cross_phase_missing:{mf}"
                verdict = tracker.report_error(error_text)

                if verdict == StrikeVerdict.HARD_STOP:
                    violations.append(violation)
                    tracker.record_hard_stop(error_text)
                    continue

                fix_start = time.time()
                exists = _file_exists_on_sandbox(sandbox_client, mf, sandbox_base)

                if exists:
                    tracker.record_attempt(
                        error_detail=error_text,
                        fix_strategy="path_verification",
                        fix_description=f"File exists on sandbox: {mf}",
                        outcome=FixOutcome.RESOLVED,
                        duration_ms=int((time.time() - fix_start) * 1000),
                    )
                    fixes_applied.append({
                        "violation": violation,
                        "fix": "File found on sandbox (host path mismatch)",
                    })
                else:
                    tracker.record_attempt(
                        error_detail=error_text,
                        fix_strategy="verify_existence",
                        fix_description=f"File not found on sandbox either: {mf}",
                        outcome=FixOutcome.SAME_ERROR,
                        duration_ms=int((time.time() - fix_start) * 1000),
                    )
                    violations.append(violation)

    # --- Layer 2: Import resolution + interface contracts ---
    # Use the integration check's detection, then fix what it finds
    if state and manifest:
        import_issues = await _detect_and_fix_integration_issues(
            state=state,
            manifest=manifest,
            sandbox_base=sandbox_base,
            sandbox_client=sandbox_client,
            tracker=tracker,
            emit=_emit,
        )
        for issue_dict in import_issues.get("unresolved", []):
            violations.append(issue_dict)
        for fix_dict in import_issues.get("fixed", []):
            fixes_applied.append(fix_dict)

    if not violations:
        tracker.record_resolution()

    return CrossPhaseResult(
        status="pass" if not violations else "fail",
        phases_checked=len(plan.phases),
        violations=violations,
        fixes_applied=fixes_applied,
    )

async def _run_ai_review_with_fixes(
    original_spec: Optional[str],
    file_scope: List[str],
    sandbox_base: str,
    sandbox_client: Any,
    tracker: StrikeTracker,
    emit: Optional[Callable] = None,
) -> AIReviewResult:
    """
    Run AI review. If critical issues are found, attempt targeted fixes.

    v3.0: After initial review, if score < 6/10 and critical issues exist,
    send each critical issue back to the LLM with the file content for
    a targeted fix. Then re-score. Uses strike rule for repeated failures.
    """
    from .final_checkout import AIReviewResult
    _emit = emit or (lambda msg: None)

    if not original_spec:
        tracker.record_resolution()
        return AIReviewResult(
            status="skipped",
            notes="No original spec provided for review comparison",
        )

    # --- Initial review ---
    review = await _run_ai_review_pass(
        original_spec=original_spec,
        file_scope=file_scope,
        sandbox_base=sandbox_base,
        emit=_emit,
    )

    if review.status in ("skipped", "error"):
        tracker.record_resolution()
        return review

    if review.overall_score >= 6.0 or not review.critical_issues:
        tracker.record_resolution()
        return review

    # --- Score below threshold — attempt fixes ---
    _emit(f"  Score {review.overall_score}/10 with {len(review.critical_issues)} critical issues — attempting fixes...")

    for issue in review.critical_issues[:3]:  # Cap at 3 fixes per round
        error_text = f"ai_review_issue:{issue[:100]}"
        verdict = tracker.report_error(error_text)

        if verdict == StrikeVerdict.HARD_STOP:
            _emit(f"    [STRIKE 3] Hard stop on AI review issue")
            tracker.record_hard_stop(error_text)
            break

        strategy = "targeted_fix" if verdict == StrikeVerdict.PROCEED else "conservative_fix"
        _emit(f"    [{verdict.value}] Fixing: {issue[:80]}...")

        fix_start = time.time()
        fixed = await _apply_ai_review_fix(
            issue=issue,
            file_scope=file_scope,
            sandbox_base=sandbox_base,
            sandbox_client=sandbox_client,
            original_spec=original_spec,
            strategy=strategy,
            emit=_emit,
        )
        duration = int((time.time() - fix_start) * 1000)

        tracker.record_attempt(
            error_detail=error_text,
            fix_strategy=strategy,
            fix_description=f"AI review fix for: {issue[:100]}",
            outcome=FixOutcome.RESOLVED if fixed else FixOutcome.SAME_ERROR,
            duration_ms=duration,
        )

    # --- Re-score after fixes ---
    _emit("  Re-scoring after fixes...")
    review = await _run_ai_review_pass(
        original_spec=original_spec,
        file_scope=file_scope,
        sandbox_base=sandbox_base,
        emit=_emit,
    )

    if review.overall_score >= 6.0:
        tracker.record_resolution()
    else:
        tracker.record_hard_stop(f"AI review score still {review.overall_score}/10")

    return review

async def _run_ai_review_pass(
    original_spec: Optional[str],
    file_scope: List[str],
    sandbox_base: str,
    emit: Optional[Callable] = None,
) -> AIReviewResult:
    """Single pass of AI review (no fix attempts)."""
    from .final_checkout import AIReviewResult
    _emit = emit or (lambda msg: None)

    # Select files for review
    priority_files: List[str] = []
    other_files: List[str] = []

    for rel_path in file_scope:
        if any(re.search(pat, rel_path, re.IGNORECASE) for pat in _REVIEW_PRIORITY_PATTERNS):
            priority_files.append(rel_path)
        else:
            other_files.append(rel_path)

    review_files = priority_files[:_MAX_REVIEW_FILES]
    remaining_slots = _MAX_REVIEW_FILES - len(review_files)
    if remaining_slots > 0:
        review_files.extend(other_files[:remaining_slots])

    if not review_files:
        return AIReviewResult(status="skipped", notes="No files available for review")

    # Read file contents
    file_contents: Dict[str, str] = {}
    for rel_path in review_files:
        abs_path = os.path.join(sandbox_base, rel_path.replace("/", os.sep))
        if _sbx_isfile(abs_path):
            try:
                # v3.4-fix: Read from sandbox, not host
                _fc_content = _sbx_read_text(abs_path)
                if _fc_content is not None:
                    file_contents[rel_path] = _fc_content[:_MAX_REVIEW_CHARS]
                else:
                    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                        file_contents[rel_path] = f.read(_MAX_REVIEW_CHARS)
            except Exception:
                pass

    if not file_contents:
        return AIReviewResult(status="skipped", notes="Could not read any files for review")

    _emit(f"  Reviewing {len(file_contents)} file(s)")

    # Build prompt and call LLM
    prompt = _build_ai_review_prompt(original_spec, file_contents)

    try:
        from app.providers.registry import llm_call

        provider_id = os.getenv("FINAL_CHECKOUT_PROVIDER", "anthropic")
        model_id = os.getenv("FINAL_CHECKOUT_MODEL", "claude-opus-4-6")

        llm_result = await llm_call(
            provider_id=provider_id,
            model_id=model_id,
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_REVIEW_SYSTEM_PROMPT,
            max_tokens=int(os.getenv("FINAL_CHECKOUT_MAX_OUTPUT_TOKENS", "8000")),
            timeout_seconds=int(os.getenv("FINAL_CHECKOUT_TIMEOUT_SECONDS", "240")),
        )

        response = llm_result.content if llm_result else None
        if not response:
            return AIReviewResult(status="error", notes="Empty LLM response")

        return _parse_review_response(response, len(file_contents))

    except Exception as exc:
        logger.warning("[final_checkout] AI review failed: %s", exc)
        return AIReviewResult(status="error", notes=f"AI review failed: {exc}")

# v3.3-fix: Frontend path prefixes that must resolve to D:\orb-desktop
_FC_FRONTEND_PREFIX = "orb-desktop/"
_FC_FRONTEND_ROOT = r"D:\orb-desktop"
_FC_FRONTEND_BARE_PREFIXES = ("src/", "src\\", "public/", "public\\")

def _write_file_to_sandbox(
    client: Any,
    rel_path: str,
    content: str,
    sandbox_base: str,
) -> bool:
    """Write a file to the sandbox.

    v3.3-fix: Now resolves frontend paths (orb-desktop/ prefix or bare
    src/, public/) to D:\\orb-desktop instead of D:\\Orb\\src.
    """
    if not client:
        return False

    try:
        import base64
        normed = rel_path.replace("/", "\\")
        normalized_fwd = rel_path.replace("\\", "/")

        if normed.startswith("C:") or normed.startswith("D:"):
            # Already absolute
            abs_path = normed
        elif normalized_fwd.startswith(_FC_FRONTEND_PREFIX):
            # Strip orb-desktop/ prefix, resolve against frontend root
            frontend_rel = normalized_fwd[len(_FC_FRONTEND_PREFIX):]
            abs_path = _FC_FRONTEND_ROOT + "\\" + frontend_rel.replace("/", "\\")
        elif any(normalized_fwd.startswith(bp.replace("\\", "/")) for bp in _FC_FRONTEND_BARE_PREFIXES):
            # Bare src/ or public/ — these only exist under orb-desktop
            abs_path = _FC_FRONTEND_ROOT + "\\" + normed
        else:
            abs_path = f"{sandbox_base}\\{normed}"

        # v3.4-fix: Strip surviving scaffold markers before write
        from app.overwatcher._implementer_utils_6 import _strip_scaffold_markers
        content = _strip_scaffold_markers(content, rel_path)

        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        temp_path = abs_path + ".tmp_fc"

        # Ensure parent directory exists
        parent = "\\".join(abs_path.replace("/", "\\").split("\\")[:-1])
        client.shell_run(
            f'New-Item -ItemType Directory -Force -Path "{parent}" | Out-Null',
            timeout_seconds=10,
        )

        # Write via base64
        client.shell_run(
            f'Set-Content -Path "{temp_path}" -Value "{b64}" -NoNewline -Encoding ASCII',
            timeout_seconds=10,
        )
        client.shell_run(
            f'$b = [System.IO.File]::ReadAllText("{temp_path}"); '
            f'$bytes = [System.Convert]::FromBase64String($b); '
            f'[System.IO.File]::WriteAllBytes("{abs_path}", $bytes); '
            f'Remove-Item -Path "{temp_path}" -Force -ErrorAction SilentlyContinue; '
            f'"WRITE_OK"',
            timeout_seconds=15,
        )
        return True
    except Exception as exc:
        logger.warning("[final_checkout] write_file_to_sandbox failed: %s", exc)
        return False
