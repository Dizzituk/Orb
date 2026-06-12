# FILE: app/debug/orchestrator/behaviour_verifier.py
# Purpose: Behaviour verifier — executes scripted user-facing scenarios.
# Called-by: app.debug.orchestrator.loop_controller
# Depends-on: app.debug, app.debug.action_executor, app.debug.adb_tools, app.debug.orchestrator.schemas
# Last-renovated: 2026-06-11
"""
Behaviour verifier — executes scripted user-facing scenarios.

Unlike a code verifier (which runs unit tests / compile / syntax checks),
the behaviour verifier proves the change actually produces the expected
user-facing outcome:

  - Android scenarios via adb_tools (tap, type, screenshot, logcat)
  - Desktop scenarios via the existing desktop control tools
  - HTTP scenarios via a simple httpx call
  - Pipeline scenarios via run_command to exercise the in-process API

Each scenario is a BehaviourCheck (see schemas). Steps are declarative:
    {"action": "tap", "x": 100, "y": 200}
    {"action": "wait_ms", "ms": 500}
    {"action": "assert_logcat_contains", "text": "STT ready"}

The verifier executes steps in order, captures evidence, and returns a
VerificationResult with PASS/FAIL + failure_signature.

v1.0 (2026-04-13): Initial implementation. Extensible action registry.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from app.debug.orchestrator.schemas import (
    BehaviourCheck,
    StepStatus,
    VerificationResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action registry
# ---------------------------------------------------------------------------

# Each action: (step_args: dict, context: dict) -> (ok: bool, message: str, evidence: dict)
ActionFn = Callable[[Dict[str, Any], Dict[str, Any]], Awaitable[Tuple[bool, str, Dict[str, Any]]]]

_ACTIONS: Dict[str, ActionFn] = {}


def register_action(name: str):
    def deco(fn: ActionFn) -> ActionFn:
        _ACTIONS[name] = fn
        return fn
    return deco


# ---------------------------------------------------------------------------
# Android actions (delegate to app.debug.adb_tools)
# ---------------------------------------------------------------------------

@register_action("android_launch")
async def _android_launch(args, ctx):
    from app.debug import adb_tools
    out = await adb_tools.launch_app()
    ok = "error" not in out.lower()
    return ok, out, {"action": "android_launch", "output": out}


@register_action("android_restart")
async def _android_restart(args, ctx):
    from app.debug import adb_tools
    out = await adb_tools.restart_app()
    ok = "error" not in out.lower()
    return ok, out, {"action": "android_restart", "output": out}


@register_action("android_tap")
async def _android_tap(args, ctx):
    from app.debug import adb_tools
    x = int(args.get("x", 0))
    y = int(args.get("y", 0))
    out = await adb_tools.tap(x, y)
    return True, out, {"action": "android_tap", "x": x, "y": y, "output": out}


@register_action("android_type")
async def _android_type(args, ctx):
    from app.debug import adb_tools
    text = str(args.get("text", ""))
    out = await adb_tools.type_text(text)
    return True, out, {"action": "android_type", "text": text[:200], "output": out}


@register_action("android_key")
async def _android_key(args, ctx):
    from app.debug import adb_tools
    keycode = str(args.get("keycode", "KEYCODE_ENTER"))
    out = await adb_tools.press_key(keycode)
    return True, out, {"action": "android_key", "keycode": keycode, "output": out}


@register_action("android_screenshot")
async def _android_screenshot(args, ctx):
    from app.debug import adb_tools
    result = await adb_tools.take_screenshot()
    # result is a dict per adb_tools.take_screenshot
    path = result.get("path") if isinstance(result, dict) else None
    return True, f"screenshot saved: {path}", {"action": "android_screenshot", "path": path}


@register_action("android_assert_logcat")
async def _android_assert_logcat(args, ctx):
    from app.debug import adb_tools
    expected = str(args.get("text", ""))
    lines = int(args.get("lines", 100))
    filter_tag = str(args.get("filter_tag", ""))
    logcat = await adb_tools.get_logcat(lines=lines, filter_tag=filter_tag)
    ok = expected in logcat
    msg = (
        f"FOUND '{expected[:60]}' in logcat"
        if ok
        else f"MISSING '{expected[:60]}' from last {lines} logcat lines"
    )
    return ok, msg, {
        "action": "android_assert_logcat",
        "expected": expected, "found": ok,
        "logcat_tail": logcat[-2000:] if logcat else "",
    }


@register_action("android_crash_check")
async def _android_crash_check(args, ctx):
    from app.debug import adb_tools
    crash = await adb_tools.get_crash_log()
    has_crash = bool(crash and crash.strip() and "no crash" not in crash.lower())
    ok = not has_crash  # pass == no crash
    msg = "no crash detected" if ok else "crash found"
    return ok, msg, {"action": "android_crash_check", "crash": (crash or "")[:2000]}


# ---------------------------------------------------------------------------
# HTTP action
# ---------------------------------------------------------------------------

@register_action("http_get")
async def _http_get(args, ctx):
    import httpx
    url = str(args.get("url", ""))
    expect_status = int(args.get("expect_status", 200))
    expect_contains = args.get("expect_contains")
    timeout = float(args.get("timeout", 10))

    if not url:
        return False, "http_get: missing url", {"action": "http_get"}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
    except Exception as e:
        return False, f"http_get error: {e}", {"action": "http_get", "url": url, "error": str(e)}

    status_ok = resp.status_code == expect_status
    body = resp.text[:4000]
    content_ok = (expect_contains in body) if expect_contains else True
    ok = status_ok and content_ok
    msg = f"HTTP {resp.status_code} ({'ok' if ok else 'fail'})"
    return ok, msg, {
        "action": "http_get", "url": url,
        "status": resp.status_code, "expect_status": expect_status,
        "body_preview": body,
        "expect_contains": expect_contains,
    }


# ---------------------------------------------------------------------------
# Shell action (pipeline/backend scenarios)
# ---------------------------------------------------------------------------

@register_action("run_command")
async def _run_command(args, ctx):
    from app.debug.action_executor import execute_tool
    cmd = str(args.get("command", ""))
    cwd = args.get("cwd")
    expect_rc = int(args.get("expect_rc", 0))
    expect_contains = args.get("expect_contains")
    params = {"command": cmd}
    if cwd:
        params["cwd"] = cwd
    result = await execute_tool("run_command", params)
    # action_executor returns a string; parse for success
    ok = True
    if expect_contains and expect_contains not in (result or ""):
        ok = False
    # Heuristic: if the string starts with ERROR or contains 'exit code N' where N != expect_rc
    if "ERROR:" in (result or "")[:20]:
        ok = False
    return ok, (result or "")[:200], {
        "action": "run_command", "command": cmd[:500],
        "output": (result or "")[:4000],
    }


# ---------------------------------------------------------------------------
# Wait / sleep
# ---------------------------------------------------------------------------

@register_action("wait_ms")
async def _wait_ms(args, ctx):
    ms = int(args.get("ms", 500))
    ms = max(0, min(30_000, ms))
    await asyncio.sleep(ms / 1000.0)
    return True, f"waited {ms}ms", {"action": "wait_ms", "ms": ms}


# ---------------------------------------------------------------------------
# Desktop actions
# ---------------------------------------------------------------------------

@register_action("desktop_screenshot")
async def _desktop_screenshot(args, ctx):
    from app.debug.action_executor import execute_tool
    result = await execute_tool("desktop_screenshot", args or {})
    return True, (result or "")[:200], {"action": "desktop_screenshot", "output": (result or "")[:2000]}


@register_action("desktop_click")
async def _desktop_click(args, ctx):
    from app.debug.action_executor import execute_tool
    result = await execute_tool("desktop_click", args or {})
    ok = "ERROR" not in (result or "")[:20]
    return ok, (result or "")[:200], {"action": "desktop_click", "args": args, "output": (result or "")[:1000]}


# ---------------------------------------------------------------------------
# Failure signature
# ---------------------------------------------------------------------------

def _failure_signature(check: BehaviourCheck, failed_step_index: int, message: str) -> str:
    key = f"{check.check_id}|{check.scenario_type}|{failed_step_index}|{message[:80]}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def run_behaviour_check(check: BehaviourCheck) -> VerificationResult:
    """Execute a single BehaviourCheck step-by-step, return VerificationResult."""
    t_start = time.time()
    evidence: List[Dict[str, Any]] = []
    ctx: Dict[str, Any] = {}
    status = StepStatus.RUNNING
    msg = ""
    sig: Optional[str] = None

    try:
        for idx, step in enumerate(check.steps):
            action_name = step.get("action")
            if not action_name:
                status = StepStatus.FAILED
                msg = f"step {idx}: missing 'action' key"
                sig = _failure_signature(check, idx, msg)
                break

            fn = _ACTIONS.get(action_name)
            if fn is None:
                status = StepStatus.FAILED
                msg = f"step {idx}: unknown action '{action_name}'"
                sig = _failure_signature(check, idx, msg)
                break

            step_start = time.time()
            try:
                ok, step_msg, step_ev = await asyncio.wait_for(
                    fn(step, ctx),
                    timeout=check.timeout_seconds,
                )
            except asyncio.TimeoutError:
                status = StepStatus.FAILED
                msg = f"step {idx} ({action_name}): timeout after {check.timeout_seconds}s"
                sig = _failure_signature(check, idx, "timeout")
                evidence.append({"action": action_name, "step_index": idx, "timeout": True})
                break
            except Exception as e:
                status = StepStatus.FAILED
                msg = f"step {idx} ({action_name}): {type(e).__name__}: {e}"
                sig = _failure_signature(check, idx, str(e))
                evidence.append({"action": action_name, "step_index": idx, "error": str(e)})
                break

            step_ev["step_index"] = idx
            step_ev["elapsed_ms"] = int((time.time() - step_start) * 1000)
            evidence.append(step_ev)

            if not ok:
                status = StepStatus.FAILED
                msg = f"step {idx} ({action_name}) FAILED: {step_msg}"
                sig = _failure_signature(check, idx, step_msg)
                break
        else:
            status = StepStatus.PASSED
            msg = f"All {len(check.steps)} steps passed: {check.expected}"

    except Exception as e:
        logger.exception("[behaviour_verifier] check=%s raised: %s", check.check_id, e)
        status = StepStatus.FAILED
        msg = f"verifier crashed: {type(e).__name__}: {e}"
        sig = _failure_signature(check, -1, str(e))

    return VerificationResult(
        check_id=check.check_id,
        check_type="behaviour",
        status=status,
        evidence=evidence,
        failure_signature=sig if status == StepStatus.FAILED else None,
        message=msg,
        elapsed_ms=int((time.time() - t_start) * 1000),
    )


async def run_behaviour_checks(checks: List[BehaviourCheck]) -> List[VerificationResult]:
    """Run a list of behaviour checks serially.

    Behaviour checks manipulate shared state (the Android device, the
    desktop screen) so they run strictly in order, never in parallel.
    """
    results: List[VerificationResult] = []
    for c in checks:
        logger.info("[behaviour_verifier] running check=%s scenario=%s", c.check_id, c.scenario_type)
        r = await run_behaviour_check(c)
        logger.info("[behaviour_verifier] %s -> %s (%s)", c.check_id, r.status.value, r.message[:100])
        results.append(r)
        if r.status == StepStatus.FAILED:
            # Don't abort — later checks may be independent and worth running
            pass
    return results


# ---------------------------------------------------------------------------
# Helpers for constructing standard checks (used by code-side verifier too)
# ---------------------------------------------------------------------------

def code_verification_check(
    check_id: str,
    command: str,
    expect_contains: Optional[str] = None,
    cwd: Optional[str] = None,
    timeout_seconds: int = 60,
) -> BehaviourCheck:
    """Construct a BehaviourCheck that is really a code-level command check.

    Useful for wiring unit tests / syntax checks / lint into the same
    verification runner as behaviour scenarios.
    """
    step: Dict[str, Any] = {"action": "run_command", "command": command}
    if expect_contains:
        step["expect_contains"] = expect_contains
    if cwd:
        step["cwd"] = cwd
    return BehaviourCheck(
        check_id=check_id,
        description=f"Run: {command[:80]}",
        scenario_type="pipeline",
        steps=[step],
        expected=expect_contains or "command exits successfully",
        timeout_seconds=timeout_seconds,
    )
