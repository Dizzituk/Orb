# FILE: app/orchestrator/segment_loop.py
"""
Core orchestrator — routes to ASTRA v2.2 Pipeline.

v9.0 (2026-03-07): V2.1 is the ONLY pipeline. The old per-segment
architecture→critique→overwatcher→implementer loop and the v8.0
agentic pipeline have been removed.

v9.2 (2026-03-10): Multi-project targeting. Reads build_target_id from
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

        # v2.3: Auto-install and launch APK for Android builds
        if v2_result.success and profile and profile.language == "kotlin":
            try:
                await _android_install_and_launch(profile, emit)
            except Exception as _apk_err:
                logger.warning("[SEGMENT_LOOP] APK install/launch failed: %s", _apk_err)
                emit(f"\u26a0\ufe0f APK install failed: {_apk_err}")

        return JobState(
            job_id=job_id,
            overall_status="complete" if v2_result.success else "failed",
            total_segments=len(_v2_manifest.get("segments", [])),
        )
    except Exception as _v2_err:
        logger.error(
            "[SEGMENT_LOOP] v9.2 V2 pipeline CRASHED: %s", _v2_err, exc_info=True,
        )
        emit(f"❌ Pipeline crashed: {_v2_err}")
        return JobState(job_id=job_id, overall_status="failed", total_segments=0)
