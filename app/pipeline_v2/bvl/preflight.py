# FILE: app/pipeline_v2/bvl/preflight.py
"""
JOB 6 (2026-06-10) - Pre-flight stage: get the app into a TESTABLE state
before any verification runs, unattended.

Why: the tier-1 first-render check requires a rendered interactive screen,
but a fresh emulator boots the Bridge to a login screen - environmental, not
code. Previously this blocked the whole cascade and burnt builder fix cycles.

Steps (all best-effort, never raises):
  1. Load the target's TestEnv (Job 5)
  2. Emulator running? If not and avd_name is configured, start it headless
     and wait for boot
  3. Launch the currently-installed app (install -r preserves app data, so a
     login session established here survives tier-1's reinstall)
  4. If the logged-in marker is on screen -> ready
  5. If the login screen is showing and credentials are configured, drive the
     login: optionally fill the backend URL + Save, fill password, tap Login,
     wait for the logged-in marker

Returns a PreflightResult; the BVL orchestrator emits it and continues either
way (a not-ready preflight becomes an ENVIRONMENT classification in Job 7
rather than a hard stop here).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING

from app.pipeline_v2.bvl.emulator_bridge import (
    dump_ui_tree,
    find_element,
    is_emulator_running,
    launch_app,
    press_enter,
    start_emulator,
    tap_element,
    type_text,
    wait_for_boot,
)
from app.pipeline_v2.test_env import TestEnv, get_test_env_for_profile

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile

logger = logging.getLogger(__name__)


@dataclass
class PreflightResult:
    ready: bool = False
    reason: str = ""
    steps: List[Tuple[str, bool, str]] = field(default_factory=list)

    def add(self, name: str, ok: bool, msg: str = "") -> None:
        self.steps.append((name, ok, msg))


async def run_preflight(
    profile: "BuildTargetProfile",
    emit: Optional[Callable[[str], None]] = None,
) -> PreflightResult:
    """Bring the target app to a logged-in, testable state. Never raises."""
    emit = emit or (lambda msg: None)
    res = PreflightResult()
    try:
        return await _run(profile, emit, res)
    except Exception as exc:
        logger.warning("[preflight] crashed (non-fatal): %s", exc)
        res.ready = False
        res.reason = f"preflight crashed: {exc}"
        return res


async def _run(profile, emit, res: PreflightResult) -> PreflightResult:
    package = str(getattr(profile, "package_name", "") or "")
    env = get_test_env_for_profile(profile)
    if env is None:
        res.add("test_env", False, "no test_env entry for this target (config/test_env.json)")
        emit("   [preflight] no test environment configured for this target")
    else:
        res.add("test_env", True, "loaded" + ("" if env.has_credentials() else " (credentials NOT set)"))

    # -- Emulator ------------------------------------------------------
    if await is_emulator_running():
        res.add("emulator", True, "already running")
        emit("   [preflight] emulator running")
    else:
        avd = env.avd_name if env else ""
        if avd:
            emit(f"   [preflight] no emulator - starting AVD '{avd}' headless...")
            await start_emulator(avd)
            boot = await wait_for_boot(timeout_sec=120)
            if boot.ok:
                res.add("emulator", True, f"started AVD {avd}")
                emit("   [preflight] emulator booted")
            else:
                res.add("emulator", False, f"AVD {avd} did not boot in time")
                res.reason = "environment: emulator did not boot"
                emit("   [preflight] emulator did not boot - cannot continue")
                return res
        else:
            res.add("emulator", False, "no emulator running and no avd_name configured")
            res.reason = "environment: no emulator running (set avd_name in config/test_env.json to auto-start)"
            emit("   [preflight] no emulator and no AVD configured - cannot continue")
            return res

    # -- App state -----------------------------------------------------
    if not package:
        res.add("app", False, "profile has no package_name")
        res.reason = "profile missing package_name"
        return res

    activity = (env.main_activity if env else None) or ".MainActivity"
    launch = await launch_app(package, activity)
    if not launch.ok:
        res.add("launch", False, f"am start failed: {launch.stderr[:150]} (app may not be installed yet - tier 1 installs)")
        res.reason = "app not installed yet - login state will be established after first install"
        return res
    res.add("launch", True, "")
    await asyncio.sleep(3)

    marker = env.logged_in_marker if env else ""
    if marker and await _marker_present(marker):
        res.add("logged_in", True, "session already present")
        res.ready = True
        emit("   [preflight] app is logged in - ready")
        return res

    # -- Login drive ---------------------------------------------------
    if not env or not env.has_credentials():
        res.add("login", False, "test credentials not set (config/test_env.json or ASTRA_TESTENV_*_PASSWORD)")
        res.reason = "environment: app not logged in and no test credentials configured"
        emit("   [preflight] not logged in and no test credentials - cannot log in")
        return res

    login_visible = await _login_screen_visible(env)
    if not login_visible:
        res.add("login", False, "neither logged-in marker nor login screen detected")
        res.reason = "environment: app state unrecognised (no login screen, no logged-in marker)"
        return res

    emit("   [preflight] login screen detected - driving login with test credentials")

    # Optional: backend URL + Save & Check first
    if env.backend_url_from_device and env.url_field_marker:
        url_field = await find_element(text=env.url_field_marker, content_desc=env.url_field_marker)
        if url_field is not None:
            await tap_element(url_field)
            await asyncio.sleep(0.4)
            await type_text(env.backend_url_from_device)
            await asyncio.sleep(0.3)
            if env.save_button_text:
                save_btn = await find_element(text=env.save_button_text)
                if save_btn is not None:
                    await tap_element(save_btn)
                    await asyncio.sleep(2.0)
            res.add("backend_url", True, env.backend_url_from_device)

    pw_field = await find_element(text=env.password_field_marker, content_desc=env.password_field_marker)
    if pw_field is None:
        res.add("login", False, f"password field '{env.password_field_marker}' not found")
        res.reason = "environment: login screen present but password field not found"
        return res
    await tap_element(pw_field)
    await asyncio.sleep(0.4)
    await type_text(env.password)
    await asyncio.sleep(0.3)

    login_btn = await find_element(text=env.login_button_text or "Login")
    if login_btn is not None:
        await tap_element(login_btn)
    else:
        await press_enter()

    # Wait for the logged-in marker (up to ~12s)
    for _ in range(6):
        await asyncio.sleep(2)
        if marker and await _marker_present(marker):
            res.add("login", True, "logged in")
            res.ready = True
            emit("   [preflight] login successful - app ready")
            return res

    res.add("login", False, "logged-in marker did not appear after login attempt")
    res.reason = "environment: login attempt did not reach logged-in state (check test credentials / backend reachability from emulator)"
    emit("   [preflight] login attempt did not complete")
    return res


async def _marker_present(marker: str) -> bool:
    node = await find_element(text=marker, content_desc=marker)
    return node is not None


async def _login_screen_visible(env: TestEnv) -> bool:
    xml = await dump_ui_tree()
    if not xml:
        return False
    blob = xml.lower()
    candidates = [env.password_field_marker, env.login_button_text, env.url_field_marker]
    return any(c and c.lower() in blob for c in candidates)
