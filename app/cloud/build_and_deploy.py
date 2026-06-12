# FILE: app/cloud/build_and_deploy.py
# Purpose: Build Android APKs and deploy to Proton Drive.
# Called-by: app.cloud.router, app.llm.stream_router
# Depends-on: app.cloud.rclone_service, app.pipeline_v2.target_registry
# Last-renovated: 2026-06-11
"""
Build Android APKs and deploy to Proton Drive.

Full flow: Gradle build → APK → Proton Drive cloud.

Used by chat routing when user says things like:
  "Build the Bridge APK and put it in the cloud"
  "Create the APK for Driver CoPilot and upload it"

v1.0 (2026-03-14): Initial — build + cloud deploy for Android targets.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_JAVA_HOME = r"C:\Program Files\Android\Android Studio\jbr"
_ANDROID_HOME = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Android", "Sdk")


def _get_target_config(target_id: str) -> Optional[Dict[str, str]]:
    """Look up build config from the target registry."""
    try:
        from app.pipeline_v2.target_registry import get_profile
        profile = get_profile(target_id)
        if not profile:
            return None
        return {
            "name": profile.project_name,
            "root": profile.project_root,
            "target_id": profile.project_id,
        }
    except Exception:
        return None


def _resolve_target_from_text(text: str) -> Optional[Dict[str, str]]:
    """Resolve which Android project to build from natural language."""
    try:
        from app.pipeline_v2.target_registry import resolve_project_from_message
        profile = resolve_project_from_message(text)
        if profile:
            return {
                "name": profile.project_name,
                "root": profile.project_root,
                "target_id": profile.project_id,
            }
    except Exception:
        pass
    return None


async def build_apk(project_root: str) -> Dict[str, Any]:
    """Run Gradle assembleDebug and return the APK path."""
    gradlew = os.path.join(project_root, "gradlew.bat")
    if not os.path.exists(gradlew):
        return {"success": False, "error": f"No gradlew.bat found at {project_root}"}

    env = os.environ.copy()
    env["JAVA_HOME"] = _JAVA_HOME
    env["ANDROID_HOME"] = _ANDROID_HOME

    try:
        proc = await asyncio.create_subprocess_exec(
            gradlew, "assembleDebug", "--no-daemon",
            cwd=project_root,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)

        if proc.returncode != 0:
            err_text = stderr.decode("utf-8", errors="replace")[-1000:]
            return {"success": False, "error": f"Gradle failed (exit {proc.returncode}): {err_text}"}

        # Find the APK
        apk_dir = os.path.join(project_root, "app", "build", "outputs", "apk", "debug")
        apk_candidates = list(Path(apk_dir).glob("*.apk")) if os.path.isdir(apk_dir) else []

        if not apk_candidates:
            return {"success": False, "error": f"Build succeeded but no APK found in {apk_dir}"}

        apk_path = str(apk_candidates[0])
        apk_size = os.path.getsize(apk_path)
        logger.info("[build_deploy] APK built: %s (%d bytes)", apk_path, apk_size)

        return {
            "success": True,
            "apk_path": apk_path,
            "apk_size": apk_size,
            "apk_name": os.path.basename(apk_path),
        }

    except asyncio.TimeoutError:
        return {"success": False, "error": "Gradle build timed out (300s)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def deploy_to_cloud(
    local_path: str,
    cloud_filename: str,
    cloud_folder: str = "APKs",
) -> Dict[str, Any]:
    """Upload a file to Proton Drive via rclone."""
    from app.cloud.rclone_service import upload_file
    cloud_dest = f"{cloud_folder}/{cloud_filename}"
    result = await upload_file(local_path, cloud_dest)
    return result


async def build_and_deploy(
    text: str,
    target_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Full pipeline: resolve target → build APK → upload to cloud.

    Args:
        text: Natural language describing what to build.
        target_id: Optional explicit target (e.g. "astra-bridge").

    Returns dict with build result, cloud result, and instructions.
    """
    # Step 1: Resolve the Android project
    config = None
    if target_id:
        config = _get_target_config(target_id)
    if not config:
        config = _resolve_target_from_text(text)
    if not config:
        return {
            "success": False,
            "error": "Couldn't determine which Android project to build. "
                     "Try: 'Build the Astra Bridge APK' or 'Build the Driver CoPilot APK'.",
        }

    result: Dict[str, Any] = {
        "project": config["name"],
        "target_id": config["target_id"],
        "project_root": config["root"],
    }

    # Step 2: Build the APK
    build_result = await build_apk(config["root"])
    result["build"] = build_result

    if not build_result["success"]:
        result["success"] = False
        result["error"] = build_result["error"]
        return result

    # Step 3: Upload to cloud — try Proton first, fall back to Google Drive
    apk_name = config["name"].replace(" ", "-") + ".apk"
    cloud_result = await deploy_to_cloud(
        build_result["apk_path"],
        cloud_filename=apk_name,
    )
    result["cloud"] = cloud_result

    if cloud_result.get("success"):
        result["success"] = True
        result["message"] = (
            f"Built {config['name']} APK and uploaded to Proton Drive: APKs/{apk_name}. "
            f"Open Proton Drive on your phone to install."
        )
    else:
        # Proton failed — try Google Drive via rclone (reliable writes)
        logger.info("[build_deploy] Proton upload failed, trying Google Drive via rclone...")
        gdrive_result = None
        try:
            from app.cloud.rclone_service import _run_rclone
            _gd_r = await _run_rclone(
                ["copyto", build_result["apk_path"], f"gdrive:ASTRA APKs/{apk_name}"],
                timeout=120,
            )
            if _gd_r["exit_code"] == 0:
                gdrive_result = {
                    "success": True,
                    "provider": "google_drive",
                    "cloud_path": f"ASTRA APKs/{apk_name}",
                }
                logger.info("[build_deploy] Uploaded to Google Drive: ASTRA APKs/%s", apk_name)
            else:
                gdrive_result = {"success": False, "error": _gd_r["stderr"][:200]}
        except Exception as e:
            logger.debug("[build_deploy] Google Drive fallback failed: %s", e)

        if gdrive_result and gdrive_result.get("success"):
            result["success"] = True
            result["cloud"] = gdrive_result
            _link = gdrive_result.get('web_link', 'Google Drive')
            result["message"] = (
                f"Built {config['name']} APK and uploaded to Google Drive: ASTRA APKs/{apk_name}. "
                f"Download from your Google Drive app or: {_link}"
            )
        else:
            # Both failed
            result["success"] = True
            result["upload_failed"] = True
            _gdrive_err = gdrive_result.get('error', 'not configured') if gdrive_result else 'not available'
            result["message"] = (
                f"Built {config['name']} APK successfully ({build_result['apk_size'] // 1024}KB). "
                f"Cloud upload failed (Proton: rclone beta limitation, Google Drive: {_gdrive_err}). "
                f"APK ready at: {build_result['apk_path']}"
            )

    return result
