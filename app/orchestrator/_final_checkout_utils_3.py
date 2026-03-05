from __future__ import annotations
import json
import logging
import os
import time
from .construction_planner_models import ConstructionPlan
from .phase_checkout_checks import _read_file_via_sandbox, run_boot_test_with_fix_loop
from .strike_tracker import FixOutcome, StrikeTracker, StrikeVerdict
from app.orchestrator._final_checkout_utils_2 import _fix_interface_contract_issue
from typing import Any, Callable, Dict, List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


async def _detect_and_fix_integration_issues(
    state: Any,
    manifest: Any,
    sandbox_base: str,
    sandbox_client: Any,
    tracker: StrikeTracker,
    emit: Optional[Callable] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run the integration check's import resolution and interface contract
    checks, then fix each issue found.

    Fix logic (provider is ground truth, consumer gets fixed):
    - import_resolution error (name not found in target):
      1. Read the provider file, get its actual exports
      2. Find the closest match to the missing name
      3. Rewrite the consumer's import line with the correct name
    - interface_contract error (expose promise broken):
      1. Check if the name exists under a different spelling
      2. Add re-export to __init__.py if needed
    """
    from .final_checkout import _fix_import_resolution_issue
    _emit = emit or (lambda msg: None)
    result = {"fixed": [], "unresolved": []}

    try:
        from .integration_check import (
            run_integration_check,
            IntegrationIssue,
        )
        from .ast_helpers import get_all_defined_names
    except ImportError as exc:
        _emit(f"  [SKIP] Cannot import integration_check: {exc}")
        return result

    # Run the read-only integration check to detect issues
    try:
        job_dir = os.path.join(sandbox_base, "jobs", "jobs")
        check_result = run_integration_check(
            manifest=manifest,
            state=state,
            job_dir=job_dir,
            on_progress=lambda msg: _emit(f"    {msg}"),
        )
    except Exception as exc:
        _emit(f"  [SKIP] Integration check failed: {exc}")
        return result

    errors = [i for i in check_result.tier1_issues if i.severity == "error"]
    if not errors:
        _emit("  [OK] No cross-segment integration errors")
        return result

    _emit(f"  {len(errors)} integration error(s) found — attempting fixes...")

    for issue in errors:
        error_text = f"{issue.check_type}:{issue.expected[:80]}"
        verdict = tracker.report_error(error_text)

        if verdict == StrikeVerdict.HARD_STOP:
            result["unresolved"].append({
                "check_type": issue.check_type,
                "detail": issue.message,
                "file_a": issue.file_a,
                "file_b": issue.file_b,
            })
            tracker.record_hard_stop(error_text)
            continue

        fix_start = time.time()
        strategies_tried = tracker.get_strategies_tried()

        if issue.check_type == "import_resolution":
            fixed = await _fix_import_resolution_issue(
                issue=issue,
                sandbox_base=sandbox_base,
                sandbox_client=sandbox_client,
                verdict=verdict,
                strategies_tried=strategies_tried,
                emit=_emit,
            )
        elif issue.check_type == "interface_contract":
            fixed = await _fix_interface_contract_issue(
                issue=issue,
                sandbox_base=sandbox_base,
                sandbox_client=sandbox_client,
                verdict=verdict,
                emit=_emit,
            )
        else:
            fixed = False
            _emit(f"    [SKIP] No auto-fix for {issue.check_type}")

        duration = int((time.time() - fix_start) * 1000)
        strategy = "import_rewrite" if issue.check_type == "import_resolution" else "contract_fix"

        if fixed:
            tracker.record_attempt(
                error_detail=error_text,
                fix_strategy=strategy,
                fix_description=f"Fixed {issue.check_type} in {issue.file_b}",
                outcome=FixOutcome.RESOLVED,
                duration_ms=duration,
            )
            result["fixed"].append({
                "check_type": issue.check_type,
                "detail": issue.message,
                "fix": f"Rewrote import in {issue.file_b}",
            })
        else:
            tracker.record_attempt(
                error_detail=error_text,
                fix_strategy=strategy,
                fix_description=f"Could not fix {issue.check_type}",
                outcome=FixOutcome.SAME_ERROR,
                duration_ms=duration,
            )
            result["unresolved"].append({
                "check_type": issue.check_type,
                "detail": issue.message,
                "file_a": issue.file_a,
                "file_b": issue.file_b,
            })

    return result

async def _llm_fix_import(
    consumer_file: str,
    consumer_content: str,
    provider_file: str,
    missing_name: str,
    sandbox_base: str,
    sandbox_client: Any,
) -> bool:
    """Ask LLM to fix an import mismatch between two files."""
    from .final_checkout import _write_file_to_sandbox
    try:
        from app.providers.registry import llm_call

        # v3.5: Read provider file from sandbox (only source of truth)
        provider_content = ""
        if provider_file and sandbox_client:
            provider_content = _read_file_via_sandbox(
                sandbox_client, provider_file, sandbox_base
            ) or ""
            provider_content = provider_content[:6000]
        elif provider_file:
            try:
                from app.sandbox_fs import sandbox_read_text
                _pc = sandbox_read_text(provider_file)
                if _pc:
                    provider_content = _pc[:6000]
            except Exception:
                pass

        prompt = (
            f"Fix the import error in the CONSUMER file.\n\n"
            f"ERROR: '{missing_name}' is not defined in the provider file.\n\n"
            f"PROVIDER FILE ({os.path.basename(provider_file or 'unknown')}): "
            f"Shows what names are actually available:\n"
            f"```python\n{provider_content[:4000]}\n```\n\n"
            f"CONSUMER FILE ({os.path.basename(consumer_file)}): "
            f"Needs the import fixed:\n"
            f"```python\n{consumer_content[:4000]}\n```\n\n"
            f"Output ONLY the complete fixed consumer file. No explanations."
        )

        result = await llm_call(
            provider_id=os.getenv("FINAL_CHECKOUT_PROVIDER", "anthropic"),
            model_id=os.getenv("FINAL_CHECKOUT_MODEL", "claude-opus-4-6"),
            messages=[{"role": "user", "content": prompt}],
            system_prompt="Fix the import. Output only the complete file content.",
            max_tokens=int(os.getenv("FINAL_CHECKOUT_MAX_OUTPUT_TOKENS", "8000")),
            timeout_seconds=int(os.getenv("FINAL_CHECKOUT_TIMEOUT_SECONDS", "240")),
        )

        fixed_content = result.content if result else None
        if fixed_content and len(fixed_content) > 50:
            return _write_file_to_sandbox(
                sandbox_client, consumer_file, fixed_content, sandbox_base
            )

        return False
    except Exception:
        return False

async def _run_boot_with_strike_tracking(
    sandbox_base: str,
    state: Any,
    tracker: StrikeTracker,
    emit: Optional[Callable] = None,
) -> Any:
    """
    Run boot test with StrikeTracker integration.

    The boot fix loop in phase_checkout_checks already implements retry
    logic. Here we wrap it with strike tracking for RAG recording.
    """
    _emit = emit or (lambda msg: None)

    boot = await run_boot_test_with_fix_loop(
        sandbox_base=sandbox_base,
        state=state,
        emit=_emit,
    )

    if boot.status == "pass":
        tracker.record_resolution()
    else:
        error_text = boot.error_summary or "Unknown boot failure"
        tracker.report_error(error_text)
        tracker.record_attempt(
            error_detail=error_text,
            fix_strategy="boot_fix_loop",
            fix_description=f"Boot fix loop exhausted (file: {boot.traceback_file or 'unknown'})",
            outcome=FixOutcome.SAME_ERROR,
        )
        tracker.record_hard_stop(error_text)

    return boot

def _extract_files_from_issue(
    issue: str,
    file_scope: List[str],
) -> List[str]:
    """
    Parse an AI review issue description to identify affected files.

    Looks for filenames, paths, and module references in the issue text,
    then matches them against the known file scope.
    """
    import re as _re

    affected = []
    issue_lower = issue.lower()

    # Direct filename mentions (e.g. "main.py", "executor.py")
    file_mentions = _re.findall(r'[\w/\\.-]+\.(?:py|ts|tsx|js|jsx)', issue)

    for mention in file_mentions:
        mention_base = os.path.basename(mention).lower()
        for scope_path in file_scope:
            if os.path.basename(scope_path).lower() == mention_base:
                if scope_path not in affected:
                    affected.append(scope_path)

    # Module references (e.g. "app.overwatcher.architecture_executor")
    module_refs = _re.findall(r'app\.[\w.]+', issue)
    for mod_ref in module_refs:
        expected_path = mod_ref.replace(".", "/") + ".py"
        for scope_path in file_scope:
            if scope_path.replace("\\", "/").endswith(expected_path):
                if scope_path not in affected:
                    affected.append(scope_path)

    # If no specific files found, try keyword matching
    if not affected:
        keywords = _re.findall(r'\b(?:auth|security|config|main|init|route|model|database)\b', issue_lower)
        for kw in keywords:
            for scope_path in file_scope:
                if kw in os.path.basename(scope_path).lower():
                    if scope_path not in affected:
                        affected.append(scope_path)
                    break  # One file per keyword

    return affected[:3]  # Cap at 3 files per issue

def _build_ai_review_prompt(
    spec: str,
    file_contents: Dict[str, str],
) -> str:
    """Build the AI review prompt with spec + sampled files."""
    parts = [
        "# Final AI Review\n",
        "## Original Specification (summary)\n",
    ]

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

def _write_file_to_sandbox_stub(
    client: Any,
    rel_path: str,
    sandbox_base: str,
) -> bool:
    """Write a minimal stub file to the sandbox."""
    from .final_checkout import _write_file_to_sandbox
    if rel_path.endswith(".py"):
        stub = f'# STUB: Auto-generated by Final Checkout\n"""Stub for {rel_path}"""\n'
    elif rel_path.endswith("__init__.py"):
        stub = '# Auto-generated __init__.py\n'
    else:
        stub = f'// STUB: Auto-generated\n'
    return _write_file_to_sandbox(client, rel_path, stub, sandbox_base)

def _build_plan_from_manifest(
    job_id: str,
    manifest: Any,
) -> ConstructionPlan:
    """
    v3.2: Build a minimal ConstructionPlan from a SegmentManifest.

    When Final Checkout is called from the segment loop (which doesn't
    have a formal multi-phase plan), we synthesise one from the manifest
    so all existing plan-based checks work unchanged.
    """
    from .construction_planner_models import PhaseDefinition, PhaseContract

    all_files = []
    if manifest:
        for seg in manifest.segments:
            all_files.extend(seg.file_scope)

    phase = PhaseDefinition(
        phase_id="phase-1",
        phase_number=1,
        title="Segmented execution (single phase)",
        file_scope=list(set(all_files)),
        status="complete",
        contract=PhaseContract(
            phase_id="phase-1",
            exports=list(set(all_files)),
        ),
    )

    return ConstructionPlan(
        job_id=job_id,
        total_phases=1,
        phases=[phase],
        is_multi_phase=False,
        total_files=len(set(all_files)),
        estimated_total_segments=manifest.total_segments if manifest else 0,
    )

def _save_result(result: FinalCheckoutResult, job_dir: str) -> None:
    """Save final checkout result to job directory."""
    from .final_checkout import FinalCheckoutResult
    path = os.path.join(job_dir, "final_checkout_result.json")
    try:
        os.makedirs(job_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("[final_checkout] Result saved to %s", path)
    except Exception as exc:
        logger.warning("[final_checkout] Failed to save: %s", exc)
