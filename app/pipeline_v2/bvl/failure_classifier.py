# FILE: app/pipeline_v2/bvl/failure_classifier.py
"""
JOB 7 (2026-06-10) - Environment-vs-code classification for tier failures.

Before the retry loop spends an expensive diagnose+fix cycle, decide whether
the failure is even code-caused. Login screens, missing emulators and
unreachable backends are ENVIRONMENT problems: route them to preflight
remediation, never to the builder. Deterministic checks only - zero LLM cost.

Classification contract:
  kind = "environment" -> retry loop runs preflight once, re-tests, and if
         still environmental returns BLOCKED with the reason, having consumed
         ZERO builder fix attempts.
  kind = "code"        -> normal cross-model diagnosis + builder fix.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile
    from app.pipeline_v2.bvl.bvl_models import TierResult

logger = logging.getLogger(__name__)

ENVIRONMENT = "environment"
CODE = "code"

_ENV_ERROR_PATTERNS = (
    "no emulator device found",
    "emulator not available",
    "start the avd",
    "device offline",
    "no devices/emulators found",
    "device unauthorized",
)


@dataclass
class Classification:
    kind: str
    reason: str
    remediation: str = ""


async def classify_failure(
    tier_result: "TierResult",
    profile: "BuildTargetProfile",
) -> Classification:
    """Classify a failed tier result. Never raises; defaults to CODE."""
    try:
        return await _classify(tier_result, profile)
    except Exception as exc:
        logger.warning("[failure_classifier] classification crashed: %s", exc)
        return Classification(CODE, f"classifier error ({exc}) - defaulting to code")


async def _classify(tier_result, profile) -> Classification:
    err = (getattr(tier_result, "error_context", "") or "").lower()

    # 1) Emulator-level signals straight from the error text
    for pat in _ENV_ERROR_PATTERNS:
        if pat in err:
            return Classification(
                ENVIRONMENT,
                "emulator not running / not reachable",
                "start the emulator (preflight can auto-start if avd_name is configured)",
            )

    failed_checks = [
        c for c in (getattr(tier_result, "checks", []) or [])
        if not getattr(c, "passed", True)
    ]
    failed_names = {getattr(c, "name", "") for c in failed_checks}

    # 2) Compile failures are code by definition
    if "gradle_build" in failed_names:
        return Classification(CODE, "gradle build failed - compile error")

    # 3) Emulator check failed
    if "emulator_ready" in failed_names:
        return Classification(
            ENVIRONMENT,
            "no emulator device connected",
            "start the AVD (set avd_name in config/test_env.json for auto-start)",
        )

    # 4) Render failures: look at the CURRENT screen to tell login/loading
    #    (environment) apart from a crashed or empty app (code).
    if "first_render" in failed_names:
        from app.pipeline_v2.bvl.emulator_bridge import (
            dump_ui_tree, is_app_running, is_emulator_running,
        )
        if not await is_emulator_running():
            return Classification(ENVIRONMENT, "emulator went away during verification",
                                  "restart the emulator")
        package = str(getattr(profile, "package_name", "") or "")
        xml = (await dump_ui_tree()) or ""
        blob = xml.lower()

        login_markers = ["password", "log in", "sign in"]
        try:
            from app.pipeline_v2.test_env import get_test_env_for_profile
            env = get_test_env_for_profile(profile)
            if env:
                login_markers += [m.lower() for m in (
                    env.password_field_marker, env.login_button_text, env.url_field_marker,
                ) if m]
        except Exception:
            pass

        if any(m in blob for m in login_markers):
            return Classification(
                ENVIRONMENT,
                "app is sitting on its login screen - not logged in",
                "set test credentials in config/test_env.json so preflight can log in",
            )
        if any(m in blob for m in ("disconnected", "connection refused", "unable to connect", "waiting for signal")):
            return Classification(
                ENVIRONMENT,
                "app reports backend unreachable from the device",
                "check backend is running and backend_url_from_device (10.0.2.2 mapping)",
            )
        if package and not await is_app_running(package):
            return Classification(CODE, "app process is not running after launch - likely crash")
        if not xml.strip():
            return Classification(ENVIRONMENT, "UI dump returned nothing - device/uiautomator state",
                                  "reboot the emulator")
        return Classification(CODE, "screen rendered no interactive elements with app alive - UI-level bug")

    # 5) Install failures are usually device-state problems
    if "apk_install" in failed_names:
        return Classification(
            ENVIRONMENT,
            "APK install failed - device storage/signature/state",
            "wipe or restart the emulator, or uninstall the existing package",
        )

    # 6) Crash on launch is code
    if "app_launch" in failed_names:
        return Classification(CODE, "app crashed on launch")

    return Classification(CODE, "no environmental signature found - treating as code")
