# FILE: app/pot_spec/grounded/spec_runner.py
"""
SpecGate v4.0 - Direct Spec Builder

NO GATES. NO CLASSIFICATION. NO RISK ASSESSMENT.

Flow:
1. Get Weaver spec (what to do)
2. Run scan (evidence of where)
3. Build POT spec (output for Implementer)
4. Segmentation check
5. Return result

v5.0 (2026-02): Extracted segmentation to _spec_runner_segmentation.py
                 Extracted result assembly to _spec_runner_result.py
                 Extracted deterministic refactor to _spec_runner_deterministic_refactor.py
v4.0 (2026-02-01): Stripped all gates - simple but powerful
"""
from __future__ import annotations

import hashlib
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session
from app.pot_spec.grounded._sbx_fs import _sbx_isfile, _sbx_isdir
from app.pot_spec.grounded._spec_runner_utils_10 import (
    SPEC_RUNNER_BUILD_ID,
    _build_simple_spec,
)

# v3.2-fix: Sandbox-aware filesystem checks for codebase paths.
# v4.3: Spec gate uses HOST filesystem, not sandbox.
from app.pot_spec.grounded._spec_runner_utils_12 import _extract_project_paths
from app.pot_spec.grounded._spec_runner_utils_13 import _extract_file_scope_from_spec
from app.pot_spec.grounded._spec_runner_segmentation import run_segmentation_check
from app.pot_spec.grounded._spec_runner_result import build_spec_result

logger = logging.getLogger(__name__)
print(f"[SPEC_RUNNER_LOADED] BUILD_ID={SPEC_RUNNER_BUILD_ID}")

from .spec_models import GroundedFact, FileTarget, GroundedPOTSpec
from .domain_detection import detect_domains
from .sandbox_discovery import extract_sandbox_hints
from .evidence_gathering import gather_filesystem_evidence, sandbox_read_file
from .multi_file_detection import _detect_multi_file_intent, _build_multi_file_operation
from .weaver_parser import parse_weaver_intent, _is_placeholder_goal

try:
    from .simple_refactor import build_direct_spec, SIMPLE_REFACTOR_BUILD_ID
    _DIRECT_BUILDER_AVAILABLE = True
except ImportError:
    _DIRECT_BUILDER_AVAILABLE = False
    build_direct_spec = None

try:
    from .simple_create import build_grounded_create_spec, SIMPLE_CREATE_BUILD_ID
    _CREATE_BUILDER_AVAILABLE = True
except ImportError:
    _CREATE_BUILDER_AVAILABLE = False
    build_grounded_create_spec = None

try:
    from ..evidence_collector import EvidenceBundle, load_evidence
    _EVIDENCE_AVAILABLE = True
except ImportError:
    _EVIDENCE_AVAILABLE = False
    EvidenceBundle = None
    load_evidence = None

try:
    from ..spec_gate_types import SpecGateResult
except ImportError:
    from dataclasses import dataclass, field
    @dataclass
    class SpecGateResult:
        ready_for_pipeline: bool = False
        open_questions: List[str] = field(default_factory=list)
        spot_markdown: Optional[str] = None
        db_persisted: bool = False
        spec_id: Optional[str] = None
        spec_hash: Optional[str] = None
        spec_version: Optional[int] = None
        hard_stopped: bool = False
        hard_stop_reason: Optional[str] = None
        notes: Optional[str] = None
        blocking_issues: List[str] = field(default_factory=list)
        validation_status: str = "pending"
        grounding_data: Optional[Dict] = None


__all__ = ["run_spec_gate_grounded"]


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def run_spec_gate_grounded(
    db: Session,
    job_id: str,
    user_intent: str,
    provider_id: str,
    model_id: str,
    project_id: int,
    constraints_hint: Optional[Dict] = None,
    spec_version: int = 1,
    user_answers: Optional[Dict[str, str]] = None,
) -> SpecGateResult:
    """
    v4.0: Direct spec builder - NO GATES.
    """
    try:
        round_n = max(1, min(3, int(spec_version or 1)))
        logger.info("[spec_runner] v4.0 Starting: job=%s, round=%d", job_id, round_n)

        # =============================================================
        # STEP 1: Get Weaver spec
        # =============================================================
        weaver_job_text = (constraints_hint or {}).get('weaver_job_description_text', '')
        combined_text = f"{user_intent or ''} {weaver_job_text}"
        intent = parse_weaver_intent(constraints_hint or {})

        goal = _extract_goal(intent, weaver_job_text, user_intent)
        logger.info("[spec_runner] v4.0 Weaver goal: %s", goal[:100])

        # =============================================================
        # STEP 2: Detect if this needs a scan
        # =============================================================

        # v2.4 (2026-04-08): Check for greenfield external project.
        # v2.3 assumed external = greenfield. Wrong — AstraBridge has 46 .kt
        # files. Now we check whether source files actually exist at the root.
        _build_profile = (constraints_hint or {}).get("build_target_profile")
        _is_greenfield = False
        if _build_profile:
            _profile_root = _build_profile.get("project_root", "")
            _known_self_roots = {"D:/Orb", "D:\\Orb", "D:/orb-desktop", "D:\\orb-desktop"}
            _norm_root = _profile_root.replace("\\", "/").rstrip("/")
            if _norm_root and _norm_root not in _known_self_roots:
                _src_root = _build_profile.get("source_root", "")
                _check_dir = os.path.join(_profile_root, _src_root) if _src_root else _profile_root
                _has_source = False
                try:
                    for _root, _dirs, _files in os.walk(_check_dir):
                        if any(f.endswith(('.kt', '.java', '.py', '.ts', '.tsx', '.js', '.jsx')) for f in _files):
                            _has_source = True
                            break
                except Exception:
                    _has_source = False
                if _has_source:
                    logger.info("[spec_runner] v2.4 EXISTING external project: %s at %s", _build_profile.get("project_id"), _profile_root)
                    print(f"[spec_runner] v2.4 EXISTING PROJECT (not greenfield): {_build_profile.get('project_id')} at {_profile_root}")
                else:
                    _is_greenfield = True
                    logger.info("[spec_runner] v2.4 GREENFIELD: %s (%s) at %s", _build_profile.get("project_id"), _build_profile.get("language"), _profile_root)
                    print(f"[spec_runner] v2.4 GREENFIELD MODE: {_build_profile.get('project_id')} ({_build_profile.get('language')}) at {_profile_root}")

        if _is_greenfield:
            # Greenfield: skip discovery, use profile root directly
            project_paths = [_profile_root]
        else:
            project_paths = _extract_project_paths(combined_text)

        # v11.0: Early sandbox check — if ALL paths are ASTRA repos and sandbox
        # is offline, hard-stop with clear user message instead of silently
        # falling through to a useless simple spec.
        from app.pot_spec.grounded._sbx_fs import _is_astra_repo_path, sandbox_available
        _all_astra = all(_is_astra_repo_path(p) for p in project_paths) if project_paths else False
        if _all_astra and project_paths and not sandbox_available():
            logger.warning("[spec_runner] v11.0 Sandbox offline — attempting auto-start...")
            print("[spec_runner] v11.0 Sandbox offline — attempting auto-start...")

            from app.pot_spec.grounded._sbx_fs import launch_sandbox
            sandbox_started = launch_sandbox(wait_seconds=60)

            if not sandbox_started:
                logger.error("[spec_runner] v11.0 Sandbox auto-start failed — hard stop")
                return SpecGateResult(
                    ready_for_pipeline=False,
                    hard_stopped=True,
                    hard_stop_reason=(
                        "🔴 **Sandbox Unavailable — Auto-Start Failed**\n\n"
                        "This job requires access to ASTRA's codebase clone in the Zobie-Orb "
                        "Hyper-V VM. ASTRA attempted to start the VM automatically but the "
                        "sandbox controller did not come online within 60 seconds.\n\n"
                        "**To fix manually:**\n"
                        "1. Open Hyper-V Manager and start the **Zobie-Orb** VM\n"
                        "2. Wait for the VM to boot and the sandbox controller to start at `192.168.250.2:8765`\n"
                        "3. Retry this command\n"
                    ),
                    spec_version=round_n,
                    validation_status="blocked",
                )
            else:
                logger.info("[spec_runner] v11.0 Sandbox auto-started successfully")
                print("[spec_runner] v11.0 Sandbox is now online — continuing with spec generation")
        # v8.0: Multi-file refactor detection DISABLED.
        # It repeatedly misclassifies CREATE/architecture jobs as refactors
        # by inferring false search/replace terms from natural language
        # (e.g. 'placeholder'->'real', 'Orb'->'project', 'frontend'->'backend').
        # The CREATE spec path handles all job types correctly.
        # TODO: Re-enable with stricter guards when actual refactor jobs are needed.
        multi_file_meta = None

        # =============================================================
        # STEP 3: Build spec (scan path or create path)
        # =============================================================
        multi_file_op = None
        spot_markdown = None
        valid_paths = []

        if multi_file_meta and multi_file_meta.get("is_multi_file"):
            result = await _handle_multi_file_scan(
                multi_file_meta, project_paths, weaver_job_text,
                combined_text, provider_id, model_id, goal, round_n, constraints_hint,
            )
            if isinstance(result, SpecGateResult):
                return result  # Early return (no matches / scan error)
            multi_file_op, spot_markdown = result
        else:
            result = await _handle_create_path(
                goal, weaver_job_text, user_intent, project_paths,
                provider_id, model_id, round_n,
                greenfield=_is_greenfield,
                build_profile=_build_profile,
            )
            if isinstance(result, SpecGateResult):
                return result  # Early return (no goal)
            spot_markdown, valid_paths = result

        # =============================================================
        # STEP 4: Segmentation check
        # =============================================================
        _is_create_job = not (multi_file_meta and multi_file_meta.get("is_multi_file"))
        seg_manifest, needle_est, early_return = await run_segmentation_check(
            spot_markdown, combined_text, multi_file_op, job_id, goal, round_n,
            is_create_job=_is_create_job,
        )

        if early_return and seg_manifest:
            # Segmented job — return early for segment loop routing
            return _build_segmented_result(
                spot_markdown, seg_manifest, goal, job_id, round_n,
            )

        # =============================================================
        # STEP 5: Assemble result
        # =============================================================
        return build_spec_result(
            spot_markdown=spot_markdown,
            multi_file_op=multi_file_op,
            valid_paths=valid_paths,
            goal=goal,
            job_id=job_id,
            round_n=round_n,
            segmentation_manifest=seg_manifest,
            needle_estimate=needle_est,
            create_builder_available=_CREATE_BUILDER_AVAILABLE,
            SpecGateResult=SpecGateResult,
        )

    except Exception as e:
        logger.exception("[spec_runner] v4.0 HARD STOP: %s", e)
        return SpecGateResult(
            ready_for_pipeline=False,
            hard_stopped=True,
            hard_stop_reason=str(e),
            spec_version=int(spec_version) if isinstance(spec_version, int) else None,
            validation_status="error",
        )


# =============================================================================
# STEP HELPERS
# =============================================================================

def _extract_goal(intent: Dict, weaver_job_text: str, user_intent: str) -> str:
    """Extract goal with placeholder filtering. Priority: intent > weaver > user."""
    intent_goal = intent.get("goal", "")
    if intent_goal and not _is_placeholder_goal(intent_goal):
        return intent_goal

    if weaver_job_text:
        for line in weaver_job_text.split('\n'):
            line = line.strip()
            if line and not _is_placeholder_goal(line):
                return line[:200]

    if user_intent:
        return user_intent[:200]
    return ""


async def _handle_multi_file_scan(
    multi_file_meta, project_paths, weaver_job_text,
    combined_text, provider_id, model_id, goal, round_n, constraints_hint,
):
    """Handle multi-file scan path. Returns (multi_file_op, spot_markdown) or SpecGateResult."""
    logger.info("[spec_runner] v4.0 Multi-file detected: %s", multi_file_meta.get("operation_type"))

    multi_file_op = await _build_multi_file_operation(
        operation_type=multi_file_meta.get("operation_type", "search"),
        search_pattern=multi_file_meta.get("search_pattern", ""),
        replacement_pattern=multi_file_meta.get("replacement_pattern", ""),
        file_filter=multi_file_meta.get("file_filter"),
        sandbox_client=None,
        job_description=weaver_job_text or combined_text,
        provider_id=provider_id,
        model_id=model_id,
        explicit_roots=project_paths if project_paths else None,
        vision_context=constraints_hint.get("vision_context", "") if constraints_hint else "",
    )

    if multi_file_op.total_occurrences == 0 and not multi_file_op.error_message:
        return SpecGateResult(
            ready_for_pipeline=False,
            open_questions=[
                f"No matches found for '{multi_file_meta.get('search_pattern')}' in {project_paths}."
            ],
            spec_version=round_n,
            validation_status="needs_clarification",
        )

    if multi_file_op.error_message:
        return SpecGateResult(
            ready_for_pipeline=False,
            blocking_issues=[f"Scan error: {multi_file_op.error_message}"],
            spec_version=round_n,
            validation_status="blocked",
        )

    if _DIRECT_BUILDER_AVAILABLE and multi_file_op.raw_matches:
        spot_markdown = build_direct_spec(
            search_term=multi_file_op.search_pattern,
            replace_term=multi_file_op.replacement_pattern,
            raw_matches=multi_file_op.raw_matches,
            goal=goal,
            total_files=multi_file_op.total_files,
        )
    else:
        spot_markdown = multi_file_op.classification_markdown
        if not spot_markdown:
            spot_markdown = (
                f"# SPoT Spec — {multi_file_op.search_pattern} → {multi_file_op.replacement_pattern}\n\n"
                f"## Goal\n{goal}\n\n"
                f"## Evidence\nFound **{multi_file_op.total_occurrences}** in **{multi_file_op.total_files} files**\n\n"
                f"## Acceptance\n- [ ] App boots\n- [ ] Changes applied\n"
            )

    return multi_file_op, spot_markdown


async def _handle_create_path(
    goal, weaver_job_text, user_intent, project_paths, provider_id, model_id, round_n,
    greenfield: bool = False,
    build_profile: Optional[Dict] = None,
):
    """Handle CREATE/MODIFY path. Returns (spot_markdown, valid_paths) or SpecGateResult.

    v2.3: greenfield mode — skip sandbox scanning, build spec directly.
    """
    if not goal and not weaver_job_text and not user_intent:
        return SpecGateResult(
            ready_for_pipeline=False,
            open_questions=["What would you like me to do?"],
            spec_version=round_n,
            validation_status="needs_clarification",
        )

    if greenfield and build_profile:
        # Greenfield external project — no existing codebase to scan.
        # Use the dedicated greenfield spec builder to generate a full
        # project architecture with file creation targets.
        logger.info("[spec_runner] v2.3 GREENFIELD CREATE: skipping sandbox scan")
        print(f"[spec_runner] v2.3 GREENFIELD CREATE: {build_profile.get('project_id')} at {build_profile.get('project_root')}")

        try:
            from app.pot_spec.grounded._greenfield_spec_builder import build_greenfield_spec
            spot_markdown = build_greenfield_spec(
                goal=goal,
                what_to_do=weaver_job_text or user_intent,
                build_profile=build_profile,
            )
        except Exception as e:
            logger.warning("[spec_runner] Greenfield spec builder failed, using simple: %s", e)
            spot_markdown = _build_simple_spec(
                goal=goal,
                what_to_do=weaver_job_text or user_intent,
            )
        return spot_markdown, project_paths

    # v11.0: Sandbox availability check for ASTRA repo paths
    from app.pot_spec.grounded._sbx_fs import _is_astra_repo_path, sandbox_available
    _astra_paths = [p for p in project_paths if _is_astra_repo_path(p)]
    _app_paths = [p for p in project_paths if not _is_astra_repo_path(p)]

    if _astra_paths and not sandbox_available():
        logger.warning("[spec_runner] v11.0 Sandbox offline — ASTRA repo paths unavailable: %s", _astra_paths)
        if not _app_paths:
            # All paths need sandbox — cannot proceed
            from app.pot_spec.grounded._spec_runner_utils_10 import _build_simple_spec as _fallback_spec
            logger.error("[spec_runner] v11.0 HARD: All paths are ASTRA repos and sandbox is offline")
            print("[spec_runner] v11.0 SANDBOX OFFLINE — all project paths are ASTRA repos, cannot inspect codebase")
            spot_markdown = _fallback_spec(goal=goal, what_to_do=weaver_job_text or user_intent)
            # Return with a note so the stream can surface the sandbox warning
            return spot_markdown, []
        else:
            print(f"[spec_runner] v11.0 Sandbox offline. ASTRA paths {_astra_paths} unavailable. Using app paths: {_app_paths}")

    valid_paths = [p for p in project_paths if _sbx_isdir(p)]

    if _CREATE_BUILDER_AVAILABLE and valid_paths:
        try:
            spot_markdown, _ = await build_grounded_create_spec(
                goal=goal,
                what_to_do=weaver_job_text or user_intent,
                project_paths=valid_paths,
                sandbox_client=None,
                provider_id=provider_id,
                model_id=model_id,
            )
        except Exception as err:
            logger.warning("[spec_runner] v4.3 Grounded CREATE failed: %s", err)
            spot_markdown = _build_simple_spec(
                goal=goal, what_to_do=weaver_job_text or user_intent,
            )
    else:
        spot_markdown = _build_simple_spec(
            goal=goal, what_to_do=weaver_job_text or user_intent,
        )

    return spot_markdown, valid_paths


def _build_segmented_result(
    spot_markdown, seg_manifest, goal, job_id, round_n,
):
    """Build SpecGateResult for segmented jobs."""
    from app.pot_spec.grounded._spec_runner_utils_11 import _get_job_dir_for_segmentation

    _seg_spec_id = f"sg-{uuid.uuid4().hex[:12]}"
    _seg_spec_hash = hashlib.sha256(spot_markdown.encode()).hexdigest() if spot_markdown else ""

    return SpecGateResult(
        ready_for_pipeline=True,
        open_questions=[],
        spot_markdown=spot_markdown,
        db_persisted=False,
        spec_id=_seg_spec_id,
        spec_hash=_seg_spec_hash,
        spec_version=round_n,
        notes="v4.9: Job segmented — use segment loop for execution",
        blocking_issues=[],
        validation_status="segmented",
        grounding_data={
            "job_kind": "architecture",
            "job_kind_confidence": 0.9,
            "job_kind_reason": "Segmented job — Phase 2 segment loop",
            "goal": goal,
            "segmentation": {
                "segmented": True,
                "total_segments": seg_manifest.total_segments,
                "segment_ids": [s.segment_id for s in seg_manifest.segments],
                "manifest_path": os.path.join(
                    _get_job_dir_for_segmentation(job_id),
                    'segments', 'manifest.json',
                ),
            },
        },
    )