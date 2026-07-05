# FILE: app/pot_spec/grounded/spec_runner.py
# Purpose: SpecGate v4.0 - Direct Spec Builder
# Called-by: app.pot_spec.grounded.spec_generation
# Depends-on: app.builds.models, app.pipeline_v2.target_registry, app.pot_spec.evidence_collector, app.pot_spec.grounded._greenfield_spec_builder (+16 more)
# Last-renovated: 2026-06-11
"""
SpecGate v4.0 - Direct Spec Builder

NO GATES. NO CLASSIFICATION. NO RISK ASSESSMENT.

Flow:
1. Get Weaver spec (what to do)
2. Run scan (evidence of where)
3. Build POT spec (output for Implementer)
4. Segmentation check
5. Return result

v12.0 (2026-07-04): No-build-target greenfield gate (_greenfield_gate.py).
                 New-app jobs with NO build profile hard-stop with a scoping
                 prompt instead of falling into the ASTRA/sandbox lane; the
                 v2.4 greenfield check moved to the same leaf.
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
from app.pot_spec.grounded._greenfield_gate import (
    NO_BUILD_TARGET_STOP_REASON,
    is_external_root,
    is_unrouted_greenfield_job,
    profile_is_greenfield,
)

logger = logging.getLogger(__name__)
print(f"[SPEC_RUNNER_LOADED] BUILD_ID={SPEC_RUNNER_BUILD_ID}")

from .spec_models import GroundedFact, FileTarget, GroundedPOTSpec
from .domain_detection import detect_domains
from .sandbox_discovery import extract_sandbox_hints
from .evidence_gathering import gather_filesystem_evidence, sandbox_read_file
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

        # v1.1 (2026-04-12): Phase 1 Job 14 - wire the ambient job target hint.
        # Read target_ids from the BuildProject row (populated by pipeline_bridge
        # at Weaver-save time, Phase 0 Job 3) so the segmenter can restrict
        # resolve_target_for_files() to only the targets this job actually touches.
        # Lookup order: chat_project_id (most reliable) -> job_id -> spec_id.
        try:
            from app.pipeline_v2.target_registry import set_job_target_hint
            from app.builds.models import BuildProject, BuildStatus
            _bp = None
            if project_id is not None:
                _bp = (
                    db.query(BuildProject)
                    .filter(
                        BuildProject.chat_project_id == project_id,
                        BuildProject.status == BuildStatus.active,
                    )
                    .order_by(BuildProject.updated_at.desc())
                    .first()
                )
            if _bp is None:
                _bp = db.query(BuildProject).filter(BuildProject.job_id == job_id).first()
            if _bp is None:
                _bp = db.query(BuildProject).filter(BuildProject.spec_id == job_id).first()
            if _bp is not None and getattr(_bp, "target_ids", None):
                set_job_target_hint(_bp.target_ids)
            else:
                set_job_target_hint(None)
                logger.debug(
                    "[spec_runner] No target_ids on BuildProject for "
                    "job=%s project_id=%s (bp_found=%s)",
                    job_id, project_id, _bp is not None,
                )
        except Exception as _hint_err:
            logger.debug("[spec_runner] set_job_target_hint failed (non-fatal): %s", _hint_err)

        # =============================================================
        # STEP 1: Get Weaver spec
        # =============================================================
        weaver_job_text = (constraints_hint or {}).get('weaver_job_description_text', '')
        combined_text = f"{user_intent or ''} {weaver_job_text}"
        intent = parse_weaver_intent(constraints_hint or {})

        goal = _extract_goal(intent, weaver_job_text, user_intent)
        logger.info("[spec_runner] v4.0 Weaver goal: %s", goal[:100])

        # 2026-07-04 vision upgrade: re-analyse the user's ORIGINAL uploaded
        # images (paths carried via flow state -> constraints_hint) with the
        # CHECKOUT_EYES vision tier, and finally CONSUME the chat-time
        # vision_context (which previously dead-ended in constraints_hint).
        # The evidence block rides weaver_job_text so EVERY spec-building
        # path (greenfield, grounded CREATE, simple fallback) sees it.
        try:
            from app.llm.image_refs import build_image_evidence
            _img_evidence = await build_image_evidence(
                image_refs=list((constraints_hint or {}).get("image_refs") or []),
                goal=goal,
                job_id=job_id,
                chat_vision_context=str((constraints_hint or {}).get("vision_context") or ""),
            )
            if _img_evidence:
                # NOTE: deliberately NOT folded into combined_text — that feeds
                # the deterministic file-scope scraper, and screenshots often
                # contain path-like text that must not become file_scope.
                weaver_job_text = (weaver_job_text or "") + _img_evidence
                print(f"[spec_runner] IMAGE EVIDENCE attached ({len(_img_evidence)} chars)")
        except Exception as _img_err:
            logger.warning("[spec_runner] image evidence skipped (non-fatal): %s", _img_err)

        # =============================================================
        # STEP 2: Detect if this needs a scan
        # =============================================================

        # v2.4 (2026-04-08): Check for greenfield external project.
        # v2.3 assumed external = greenfield. Wrong — AstraBridge has 46 .kt
        # files. Now we check whether source files actually exist at the root
        # (the check itself lives in _greenfield_gate.py since v12.0).
        _build_profile = (constraints_hint or {}).get("build_target_profile")

        # v12.0 (2026-07-04): NO-build-target greenfield gate. A new-app job
        # with no profile must never degrade into "modify ASTRA" — the old
        # fallback path-scraped the chat, matched ASTRA repos and died on the
        # v11.0 sandbox stop. Detect it (Weaver job_class flag OR
        # greenfield/game domains) BEFORE any path extraction and stop with a
        # scoping prompt instead. Genuine ASTRA jobs fall through untouched.
        if not _build_profile:
            _gf_stop, _gf_why = is_unrouted_greenfield_job(
                constraints_hint, weaver_job_text, user_intent,
            )
            if _gf_stop:
                logger.warning(
                    "[spec_runner] v12.0 NO BUILD TARGET — greenfield stop (%s)", _gf_why,
                )
                print(f"[spec_runner] v12.0 NO BUILD TARGET — greenfield job detected ({_gf_why}); prompting for project scoping")
                return SpecGateResult(
                    ready_for_pipeline=False,
                    hard_stopped=True,
                    hard_stop_reason=NO_BUILD_TARGET_STOP_REASON,
                    spec_version=round_n,
                    validation_status="blocked",
                    notes="v12.0: greenfield job with no build target — scoping required",
                )

        _profile_root = (_build_profile or {}).get("project_root", "")
        _is_greenfield = profile_is_greenfield(_build_profile)
        if _is_greenfield:
            logger.info("[spec_runner] v2.4 GREENFIELD: %s (%s) at %s", _build_profile.get("project_id"), _build_profile.get("language"), _profile_root)
            print(f"[spec_runner] v2.4 GREENFIELD MODE: {_build_profile.get('project_id')} ({_build_profile.get('language')}) at {_profile_root}")
        elif _build_profile and is_external_root(_profile_root):
            logger.info("[spec_runner] v2.4 EXISTING external project: %s at %s", _build_profile.get("project_id"), _profile_root)
            print(f"[spec_runner] v2.4 EXISTING PROJECT (not greenfield): {_build_profile.get('project_id')} at {_profile_root}")

        if _is_greenfield:
            # Greenfield: skip discovery, use profile root directly
            project_paths = [_profile_root]
        else:
            project_paths = _extract_project_paths(combined_text)
            # Ground the injected build-target root, not only the paths
            # scraped from the chat text. Without this, a job whose
            # conversation mentions the backend (Orb) grounds Orb instead of
            # the actual target (e.g. Astra-Bridge on the host), producing a
            # spec that is blind to the real target code.
            if _build_profile:
                _tgt_root = _build_profile.get("project_root", "")
                if _tgt_root:
                    _existing_norm = [p.replace("\\", "/").rstrip("/") for p in project_paths]
                    if _tgt_root.replace("\\", "/").rstrip("/") not in _existing_norm:
                        project_paths.insert(0, _tgt_root)

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
        # =============================================================
        # STEP 3: Build spec (create path)
        # v8.0: multi-file refactor SCAN path retired 2026-06-20 (Taz decision).
        # multi_file_op stays None so STEP 4/5 (segmentation + result assembly) are unchanged.
        # =============================================================
        multi_file_op = None
        spot_markdown = None
        valid_paths = []

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
        _is_create_job = True  # multi-file scan path retired 2026-06-20; always the create path
        seg_manifest, needle_est, early_return = await run_segmentation_check(
            spot_markdown, combined_text, multi_file_op, job_id, goal, round_n,
            is_create_job=_is_create_job,
            # live10: greenfield files resolve against the NEW project's root
            # only — never the ASTRA repos (sandbox-routed, 15s/probe wedge).
            greenfield_root=_profile_root if _is_greenfield else None,
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

        # live9 (2026-07-04): the generic greenfield template embeds NO file
        # paths, so segmentation shipped a 0-file manifest and the scaffold
        # had nothing to build (first live Tetris run). If the spec names no
        # files, SPECGATE_MAIN plans the modular layout NOW — the file plan
        # belongs to the spec. Fail-soft: no plan = spec unchanged, and the
        # pipeline's 0-file hard-stop still protects the run.
        try:
            from app.pot_spec.grounded._spec_runner_utils_13 import (
                _extract_file_scope_from_spec as _live9_extract,
            )
            if not _live9_extract(spot_markdown):
                from app.pot_spec.grounded._greenfield_file_planner import plan_greenfield_files
                _plan_section = await plan_greenfield_files(
                    goal=goal,
                    what_to_do=weaver_job_text or user_intent,
                    build_profile=build_profile,
                )
                if _plan_section:
                    spot_markdown = spot_markdown.rstrip() + "\n\n" + _plan_section
                else:
                    logger.warning(
                        "[spec_runner] live9 Greenfield file planner produced no "
                        "usable plan — manifest will be empty and the pipeline "
                        "will hard-stop at scaffold",
                    )
        except Exception as _plan_err:
            logger.warning("[spec_runner] live9 Greenfield file planner failed: %s", _plan_err)

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
                job_id=job_id,  # Derek p4: agent-dispatch ledger evidence
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