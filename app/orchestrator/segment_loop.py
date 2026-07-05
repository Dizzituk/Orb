# FILE: app/orchestrator/segment_loop.py
# Purpose: Core orchestrator — routes to ASTRA v2.2 Pipeline.
# Called-by: app.orchestrator.segment_loop_stream
# Depends-on: app.builds.models, app.builds.pipeline_bridge, app.db, app.orchestrator.segment_state (+9 more)
# Last-renovated: 2026-06-11
"""
Core orchestrator — routes to ASTRA v2.2 Pipeline.

v9.0 (2026-03-07): V2.1 is the ONLY pipeline. The old per-segment
architecture→critique→overwatcher→implementer loop and the v8.0
agentic pipeline have been removed.

v9.2 (2026-03-10): Multi-project targeting.
v9.3 (2026-04-06): Gradle daemon pre-warm for Android builds.. Reads build_target_id from
the build project and passes the correct BuildTargetProfile to the
v2.2 pipeline orchestrator.

Flow: segment_loop_stream.py → run_segmented_job() → run_v2_pipeline()
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Optional

from app.orchestrator.segment_state import (
    JobState,
    get_job_dir,
)

logger = logging.getLogger(__name__)

SEGMENT_LOOP_BUILD_ID = "2026-03-10-v9.2-multi-project-target"
print(f"[SEGMENT_LOOP_LOADED] BUILD_ID={SEGMENT_LOOP_BUILD_ID}")

# Type alias for progress callback
ProgressCallback = Optional[Callable[[str], None]]


async def _android_install_and_launch(
    profile: "BuildTargetProfile",
    emit: Callable,
) -> None:
    """Install and launch the debug APK on the connected emulator.

    Called automatically after a successful Android build. Finds the
    debug APK in the build output directory, installs it via ADB,
    and launches the main activity.
    """
    import glob
    import asyncio

    project_root = profile.project_root.replace("/", os.sep)
    apk_pattern = os.path.join(project_root, "app", "build", "outputs", "apk", "debug", "*.apk")
    apk_files = glob.glob(apk_pattern)

    if not apk_files:
        emit("   \u26a0\ufe0f No debug APK found — skipping install")
        return

    apk_path = apk_files[0]
    adb_path = r"C:\Users\dizzi\AppData\Local\Android\Sdk\platform-tools\adb.exe"

    if not os.path.isfile(adb_path):
        emit("   \u26a0\ufe0f ADB not found — skipping install")
        return

    emit(f"   \U0001f4f1 Installing APK ({os.path.getsize(apk_path) // 1024}KB)...")

    # Install
    proc = await asyncio.create_subprocess_exec(
        adb_path, "install", "-r", apk_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
    output = (stdout or b"").decode(errors="replace")

    if "Success" not in output:
        emit(f"   \u274c APK install failed: {output[:200]}")
        return

    emit("   \u2705 APK installed")

    # Read package name from manifest
    package_name = profile.package_name if hasattr(profile, "package_name") and profile.package_name else None
    if not package_name:
        # Try to extract from AndroidManifest.xml
        manifest_path = os.path.join(project_root, "app", "src", "main", "AndroidManifest.xml")
        if os.path.isfile(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                content = f.read()
            import re
            match = re.search(r'package="([^"]+)"', content)
            if match:
                package_name = match.group(1)

    if not package_name:
        emit("   \u26a0\ufe0f Could not determine package name — skipping launch")
        return

    # Launch
    emit(f"   \U0001f680 Launching {package_name}...")
    proc = await asyncio.create_subprocess_exec(
        adb_path, "shell", "am", "start",
        "-n", f"{package_name}/.MainActivity",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
    output = (stdout or b"").decode(errors="replace")

    if "Error" in output:
        emit(f"   \u26a0\ufe0f Launch warning: {output[:200]}")
    else:
        emit("   \u2705 App launched on emulator")


def _load_build_target_profile(project_id: int, emit: Callable):
    """Load the BuildTargetProfile from the active build project.

    Looks up the build project linked to this chat project, reads
    its build_target_id, and loads the corresponding profile.

    Returns the profile, or None (which means the orchestrator
    will use the default — astra-backend).
    """
    try:
        from app.db import SessionLocal
        from app.builds.models import BuildProject, BuildStatus
        from app.builds.pipeline_bridge import get_build_target_profile

        db = SessionLocal()
        try:
            build_project = (
                db.query(BuildProject)
                .filter(
                    BuildProject.chat_project_id == project_id,
                    BuildProject.status == BuildStatus.active,
                )
                .order_by(BuildProject.updated_at.desc())
                .first()
            )
            if build_project and build_project.build_target_id:
                profile = get_build_target_profile(build_project)
                if profile:
                    emit(f"   🎯 Build target: {profile.project_name} ({profile.language}/{profile.framework})")
                    emit(f"   📁 Project root: {profile.project_root}")
                    logger.info(
                        "[SEGMENT_LOOP] Loaded build target profile: %s (%s)",
                        profile.project_id, profile.project_name,
                    )
                    return profile
                else:
                    emit(f"   ⚠️ Build target '{build_project.build_target_id}' not found in registry")
            else:
                emit("   ℹ️ No build target set — using default (ASTRA backend)")
        finally:
            db.close()
    except Exception as e:
        logger.debug("[SEGMENT_LOOP] Could not load build target profile: %s", e)
    return None


async def run_segmented_job(
    job_id: str,
    manifest_path: str,
    parent_spec: dict,
    db: Any = None,
    project_id: int = 0,
    on_progress: ProgressCallback = None,
    implement_only: bool = False,
) -> JobState:
    """
    Main entry point for pipeline execution.

    v9.2: Routes to ASTRA v2.2 Pipeline with multi-project targeting.
    Reads build_target_id from the build project and passes the
    correct BuildTargetProfile to the orchestrator.
    """
    emit = on_progress or (lambda msg: None)

    logger.info("[SEGMENT_LOOP] v9.2 ASTRA V2.2 Pipeline — loading job %s", job_id)

    try:
        from app.pipeline_v2.orchestrator import run_v2_pipeline
    except ImportError as _imp_err:
        logger.error("[SEGMENT_LOOP] v9.2 pipeline_v2 import failed: %s", _imp_err)
        emit(f"❌ Pipeline v2.2 module not available: {_imp_err}")
        return JobState(job_id=job_id, overall_status="failed", total_segments=0)

    _v2_job_dir = os.path.join("D:\\Orb", "jobs", "jobs", job_id)

    # v9.2: Load the build target profile from the build project
    profile = _load_build_target_profile(project_id, emit)

    # PROVING RUN (2026-06-10): clone freshness gate for self-builds. ASTRA
    # only ever builds ITSELF in the sandbox clone; building on a stale or
    # dirty clone wastes the run, and promoting from one would roll live
    # work backwards. Gate runs BEFORE any pre-warm or builder spend, and
    # FAILS CLOSED for self-builds: a broken gate must never wave a
    # self-build through unchecked.
    if profile is not None:
        _gate_passed = True
        try:
            from app.pipeline_v2.clone_freshness import is_self_build, check_clone_freshness
            if is_self_build(profile):
                _fresh = await check_clone_freshness(profile)
                for _fl in _fresh.render_lines():
                    emit(_fl)
                _gate_passed = _fresh.ok
                if not _gate_passed:
                    emit("❌ Self-build refused: sandbox clone is not coherent with the host (see gate above)")
        except Exception as _cf_err:
            logger.error("[SEGMENT_LOOP] clone freshness gate crashed: %s", _cf_err)
            try:
                from app.pipeline_v2.clone_freshness import is_self_build as _isb
                if _isb(profile):
                    emit(f"❌ Self-build refused: freshness gate failed to run ({_cf_err}) — failing closed")
                    _gate_passed = False
            except Exception:
                emit(f"❌ Self-build refused: freshness gate unavailable — failing closed")
                _gate_passed = False
        if not _gate_passed:
            return JobState(job_id=job_id, overall_status="failed", total_segments=0)

    # v9.3: Pre-warm Gradle daemon for Android builds
    # v9.4 (2026-04-12): Phase 2 Job 9 — multi-target parallel pre-warming.
    # When the build project has target_ids set (Phase 0 Job 3 column), walk
    # ALL registered targets in parallel using asyncio.gather so both Gradle
    # daemon (Android) and Python import-check (backend) warm up concurrently.
    # Single-target jobs keep the single-profile path.
    if profile:
        try:
            from app.pipeline_v2.gradle_daemon import needs_gradle, pre_warm_daemon
            from app.pipeline_v2.target_registry import get_profile as _get_profile
            # Discover all targets to warm. Prefer target_ids from the build
            # project (set by Phase 0 Job 4's multi-target detection). Fall back
            # to [profile] for single-target legacy jobs.
            _targets_to_warm = [profile]
            try:
                from app.db import SessionLocal
                from app.builds.models import BuildProject, BuildStatus
                _db = SessionLocal()
                try:
                    _bp = (_db.query(BuildProject)
                             .filter(BuildProject.chat_project_id == project_id,
                                     BuildProject.status == BuildStatus.active)
                             .order_by(BuildProject.updated_at.desc())
                             .first())
                    if _bp and _bp.target_ids:
                        _discovered = []
                        for _tid in _bp.target_ids:
                            _p = _get_profile(_tid)
                            if _p is not None:
                                _discovered.append(_p)
                        if len(_discovered) > 1:
                            _targets_to_warm = _discovered
                            emit(f"   🎯 Multi-target build: {len(_discovered)} targets")
                finally:
                    _db.close()
            except Exception as _lookup_err:
                logger.debug("[SEGMENT_LOOP] target_ids lookup failed (non-fatal): %s", _lookup_err)

            # Kick off pre-warm tasks for every target in parallel
            _warm_tasks = []
            for _prof in _targets_to_warm:
                if needs_gradle(_prof):
                    _warm_tasks.append(("gradle", _prof, pre_warm_daemon(_prof)))
                else:
                    # Python/backend target — import-check is cheap and warms
                    # the venv's import cache. Use a tiny async wrapper.
                    async def _backend_warm(_p):
                        try:
                            # Touching the profile-resolved project_root is
                            # enough to confirm the venv is accessible; no
                            # subprocess needed.
                            import os as _os
                            _exists = _os.path.isdir(_p.project_root)
                            return _exists
                        except Exception:
                            return False
                    _warm_tasks.append(("backend", _prof, _backend_warm(_prof)))

            if _warm_tasks:
                import asyncio as _asyncio
                _results = await _asyncio.gather(
                    *[t[2] for t in _warm_tasks], return_exceptions=True
                )
                for (_kind, _prof, _), _res in zip(_warm_tasks, _results):
                    if isinstance(_res, Exception):
                        logger.warning(
                            "[SEGMENT_LOOP] Job 9 pre-warm (%s/%s) raised: %s",
                            _kind, _prof.project_id, _res,
                        )
                    elif _res:
                        emit(f"   🔥 Pre-warm {_kind}: {_prof.project_name}")
                        logger.info(
                            "[SEGMENT_LOOP] Job 9 pre-warm ok: %s (%s)",
                            _prof.project_id, _kind,
                        )
        except Exception as _warm_err:
            logger.warning("[SEGMENT_LOOP] Pre-warm failed (non-fatal): %s", _warm_err)


    # Load manifest
    _v2_manifest = {}
    _v2_spec = {}
    _v2_intent = ""
    try:
        with open(manifest_path, "r", encoding="utf-8") as _mf:
            _v2_manifest = json.load(_mf)

        # Try loading spec from segments dir
        _spec_path = os.path.join(os.path.dirname(manifest_path), "..", "spec.json")
        if os.path.isfile(_spec_path):
            with open(_spec_path, "r", encoding="utf-8") as _sf:
                _v2_spec = json.load(_sf)

        # Load intent from weaver if available
        _intent_path = os.path.join(_v2_job_dir, "intent.txt")
        if os.path.isfile(_intent_path):
            with open(_intent_path, "r", encoding="utf-8") as _if:
                _v2_intent = _if.read()
        elif parent_spec:
            _v2_intent = parent_spec.get("summary", str(parent_spec)[:2000])
    except Exception as _load_err:
        logger.error("[SEGMENT_LOOP] v9.2 Failed to load v2 inputs: %s", _load_err)

    if not _v2_manifest:
        emit("❌ Could not load manifest — cannot run pipeline")
        return JobState(job_id=job_id, overall_status="failed", total_segments=0)

    try:
        v2_result = await run_v2_pipeline(
            job_id=job_id,
            manifest=_v2_manifest,
            spec=_v2_spec or parent_spec,
            intent_text=_v2_intent or str(parent_spec)[:2000],
            job_dir=_v2_job_dir,
            on_progress=on_progress,
            profile=profile,
        )

        # live9 (2026-07-04): nothing built = nothing to verify. An empty
        # manifest used to sail through as a vacuous 0/0 "PASS" + an
        # "OVERALL: VERIFIED" merged verdict, then the eyes launched a
        # nonexistent app and the judge blocked on the wreckage. Skip every
        # advisory verification honestly instead.
        _nothing_built = not any(
            seg.get("file_scope") for seg in (_v2_manifest or {}).get("segments", [])
        )
        if _nothing_built:
            emit("   ⛔ Manifest has 0 files — nothing was built, skipping verification and checkout")

        # v9.4 (2026-04-12): Phase 3 Job 10 — multi-target verification.
        # After the builder finishes writing segments, verify every target's
        # toolchain can build its code and run the contract verifier across
        # the whole manifest. For single-target jobs this is equivalent to
        # the legacy build_check + boot_check path, just run through a
        # unified report. For multi-target jobs it catches the case where
        # one target compiles but the other doesn't.
        _mt_report = None
        try:
            from app.pipeline_v2.multi_target_verifier import verify_manifest_build
            from app.pot_spec.grounded.segment_schemas import SegmentManifest as _SM
            _manifest_obj = _v2_manifest
            if isinstance(_v2_manifest, dict):
                try:
                    _manifest_obj = _SM.from_dict(_v2_manifest)
                except Exception:
                    _manifest_obj = None
            if _manifest_obj is not None and not _nothing_built:
                _mt_report = await verify_manifest_build(_manifest_obj)
                emit(f"   🎯 Job 10 verification: {_mt_report.summary()}")
                if not _mt_report.is_passing():
                    for _fr in _mt_report.failed_targets():
                        emit(f"      ❌ {_fr.target_id}: {_fr.status.value} — {_fr.detail}")
        except Exception as _mtv_err:
            logger.warning("[SEGMENT_LOOP] Job 10 verifier raised (non-fatal): %s", _mtv_err)

        # Overall success = v2 pipeline + all targets passed multi-target verify
        _all_pass = bool(v2_result.success) and (_mt_report is None or _mt_report.is_passing())

        # JOB 3 (2026-06-10): one merged verdict (BVL + contract + spec review)
        # so the human signs off from a single table instead of refereeing
        # three contradicting reports.
        _verdict = None
        if not _nothing_built:
            try:
                from app.pipeline_v2.final_verdict import build_final_verdict, push_verdict_narrative
                _verdict = build_final_verdict(
                    pipeline_result=v2_result,
                    contract_report=_mt_report,
                    spec=_v2_spec or parent_spec,
                )
                for _vline in _verdict.render_lines():
                    emit(_vline)
                push_verdict_narrative(_verdict)
            except Exception as _fv_err:
                logger.warning("[SEGMENT_LOOP] Final verdict assembly failed (non-fatal): %s", _fv_err)

        # JOB 11 (2026-06-10): agentic verifier final checkout — Claude with
        # hands, driving the real app per acceptance criterion. Gated behind
        # ASTRA_AGENTIC_VERIFIER=1 (real token spend) and Android targets
        # only for now (perception tools are ADB-based). Advisory: it does
        # not change _all_pass; the human sign-off remains the gate.
        try:
            from app.pipeline_v2.config import VERIFIER_AGENT_ENABLED
            from app.pipeline_v2.clone_freshness import is_self_build as _vg_isb
            if (not _nothing_built) and VERIFIER_AGENT_ENABLED and profile and (profile.language == "kotlin" or _vg_isb(profile)):
                from app.pipeline_v2.verifier_agent.agent import run_verifier_agent
                _ev_text = "\n".join(_verdict.render_lines()) if _verdict else ""
                await run_verifier_agent(
                    spec=_v2_spec or parent_spec or {},
                    profile=profile,
                    intent_text="",
                    evidence_summary=_ev_text,
                    job_id=job_id,
                    emit=emit,
                )
        except Exception as _va_err:
            logger.warning("[SEGMENT_LOOP] Agentic verifier failed (non-fatal): %s", _va_err)

        # DEREK p6 (2026-07-04): eyes+judge behavioural checkout for the
        # GREENFIELD/host lane (which the Claude verifier above never covers
        # — its gate is kotlin-or-self-build). Cheap SMALL-tier eyes + HEAVY
        # judge, so it gets its own always-affordable gate:
        # ASTRA_AGENTIC_VERIFIER_GREENFIELD (default ON — the Fable-cost
        # reason the Claude verifier is off does not apply here). Advisory;
        # verdict + failures land in the Decision Ledger, and a FAIL yields
        # a Phase-5-shaped re-dispatch signal for the segmented builder.
        try:
            _gf_eyes_on = os.getenv("ASTRA_AGENTIC_VERIFIER_GREENFIELD", "1").strip().lower() in ("1", "true", "yes")
            from app.pipeline_v2.android_sandbox import is_host_build as _hj_ihb
            from app.pipeline_v2.clone_freshness import is_self_build as _hj_isb
            if (
                _gf_eyes_on and profile
                and not _nothing_built
                and profile.language != "kotlin"
                and not _hj_isb(profile)
                and _hj_ihb(profile)
            ):
                # live13: FAIL verdicts now trigger the verifier's own repair
                # fleet (surgical fix worker -> re-run eyes+judge, bounded by
                # ASTRA_CHECKOUT_REPAIR_ROUNDS), and the final verdict GATES
                # the greenfield run's overall status — no 🎉 on a game that
                # doesn't boot.
                from app.pipeline_v2.verifier_agent.eyes_judge import run_checkout_with_repair
                _spec_txt = ""
                try:
                    _spec_txt = json.dumps(_v2_spec or parent_spec or {}, default=str)[:8000]
                except Exception:
                    _spec_txt = str(_v2_spec or parent_spec or "")[:8000]
                _judge = await run_checkout_with_repair(
                    profile=profile,
                    spec_text=_spec_txt,
                    job_id=job_id,
                    job_dir=_v2_job_dir,
                    spec=_v2_spec or parent_spec,
                    manifest=_v2_manifest,
                    scaffold=getattr(v2_result, "scaffold_result", None),
                    emit=emit,
                )
                if _judge.verdict != "PASS":
                    _all_pass = False
                    emit(f"   ⛔ Behavioural checkout {_judge.verdict} — run marked FAILED")
                if _judge.redispatch_signal:
                    emit("   ♻️ Checkout FAIL evidence recorded — re-dispatch signal in the ledger")
        except Exception as _ej_err:
            logger.warning("[SEGMENT_LOOP] Eyes+judge checkout failed (non-fatal): %s", _ej_err)

        # v2.3: Auto-install and launch APK for Android builds
        if v2_result.success and profile and profile.language == "kotlin":
            try:
                await _android_install_and_launch(profile, emit)
            except Exception as _apk_err:
                logger.warning("[SEGMENT_LOOP] APK install/launch failed: %s", _apk_err)
                emit(f"\u26a0\ufe0f APK install failed: {_apk_err}")

        return JobState(
            job_id=job_id,
            overall_status="complete" if _all_pass else "failed",
            total_segments=len(_v2_manifest.get("segments", [])),
        )
    except Exception as _v2_err:
        logger.error(
            "[SEGMENT_LOOP] v9.2 V2 pipeline CRASHED: %s", _v2_err, exc_info=True,
        )
        emit(f"❌ Pipeline crashed: {_v2_err}")
        return JobState(job_id=job_id, overall_status="failed", total_segments=0)
