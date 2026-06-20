# FILE: app/debug/sandbox_selfheal.py
# Purpose: The autonomous boot self-heal LOOP for the sandbox clone (Job 03). Boots the
#          clone, inspects how it came up, classifies any failure, applies the known
#          remediation IN THE CLONE (visibly), reboots, and re-inspects -- looping until
#          the boot is clean or it has a precise un-auto-fixable error to report. The
#          hands-free version of the manual boot -> see error -> fix -> recheck loop.
# Called-by: app.llm.chat_tool_loop (execute_chat_tool routing + write-tier schema),
#            app.debug.orchestrator.phase_narration (PHASE_TOOLS -- narrated move)
# Depends-on: app.sandbox.manager (start/stop -- PROTECTED, call only),
#             app.debug.sandbox_boot_tool.inspect_sandbox_boot (the eyes -- Job 02),
#             app.debug.sandbox_console.visible_shell_run (Job 02 visible console),
#             app.debug.boot_failure_classifier (the failure->fix table)
# Last-renovated: 2026-06-17 (created -- Job 03 sandbox-selfheal)
"""Sandbox boot self-heal loop.

    boot -> inspect -> classify -> (known + auto-fixable?) -> fix in clone -> reboot -> re-inspect
                                 -> clean? -> report success
                                 -> unknown/unfixable? -> report verbatim error + what was tried
                                 -> per-class cycle cap hit? -> report

Every clone-side command (the npm/pip fixes) runs through the Job 02 visible console
(``visible_shell_run``) so Taz watches the fix happen. Clone paths are derived from the
controller's reported REPO_ROOT -- never a literal drive. The protected sandbox manager
(start/stop) and the boot inspector are CALLED, never modified.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# npm/pip installs are slow; the sandbox client hard-caps a single shell call at 300s.
_INSTALL_TIMEOUT = 300

SELFHEAL_SANDBOX_BOOT_TOOL: Dict[str, Any] = {
    "name": "selfheal_sandbox_boot",
    "description": (
        "AUTONOMOUS boot self-heal for the SANDBOX CLONE. Boots the clone, inspects how it "
        "came up (backend health + clone astra.log + a screenshot/vision read of the Electron "
        "window), and if the boot is UNHEALTHY it CLASSIFIES the failure and applies the known "
        "fix IN THE CLONE, visibly -- e.g. `npm install` for a missing frontend module (the "
        "@univerjs case), `pip install` for a missing backend module, or a clean reboot for a "
        "stale port -- then reboots and re-inspects, looping until the boot is clean or it has a "
        "precise, un-auto-fixable error to report. This is the hands-free version of "
        "start_sandbox_clone + inspect_sandbox_boot + fix + reboot + verify: use it when the user "
        "says things like 'boot the sandbox and fix whatever's wrong', 'get the clone booting "
        "clean', or 'self-heal the sandbox boot'. The clone boot is SLOW, so it WAITS ~90s for the "
        "startup to settle before judging it. Pass auto_fix=false to JUST boot, wait, inspect and "
        "REPORT how it came up (backend health + clone log + a screenshot/OCR of the Electron "
        "window) WITHOUT changing anything -- the right call for 'boot yourself and tell me how "
        "the boot went'. auto_fix=true (default) also applies the fixes and reboots. Acts on the "
        "CLONE only, never the host. Everything shows in the visible ASTRA sandbox console."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "wait_seconds": {
                "type": "integer",
                "description": "Seconds to wait for the slow clone backend on each inspection. Floored at 60 and defaults to 90 -- the boot genuinely takes ~60-90s, so do not shorten it.",
            },
            "auto_fix": {
                "type": "boolean",
                "description": "true (default) = apply known fixes and reboot, looping until clean. false = boot, wait, inspect and REPORT how it came up WITHOUT changing anything (use for 'boot and tell me how it went').",
            },
            "max_cycles_per_class": {
                "type": "integer",
                "description": "Cap on remediation attempts for any single failure class before giving up on it. Default 2.",
            },
            "screenshot_delay_seconds": {
                "type": "integer",
                "description": "Settle time before each screenshot so the Electron window can paint. Default 3.",
            },
        },
        "required": [],
    },
}


def _clamp(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(int(value), hi))
    except (TypeError, ValueError):
        return default


def _sq(s: str) -> str:
    """Escape a string for safe embedding inside a single-quoted PowerShell literal
    (defense-in-depth: clone paths come from the trusted controller, but doubling any
    embedded single quote keeps a stray quote from breaking out of -LiteralPath)."""
    return (s or "").replace("'", "''")


def _derive_clone_paths(mgr) -> Tuple[str, str, str]:
    """(repo_root, frontend_dir, error). Both paths derive from the controller's
    REPO_ROOT -- no hardcoded drive. error non-empty => controller unreachable / no root."""
    try:
        health = mgr.client.health()
        repo_root = (getattr(health, "repo_root", "") or "").rstrip("\\/")
    except Exception as e:  # pragma: no cover - needs a live controller
        return "", "", (
            f"Sandbox controller unreachable ({e}). Start the Windows Sandbox first -- "
            "cannot boot or self-heal the clone."
        )
    if not repo_root:
        return "", "", (
            "The sandbox controller did not report a REPO_ROOT, so the clone's layout "
            "cannot be located. Aborting (no hardcoded drive fallback)."
        )
    # Frontend is the sibling 'orb-desktop' repo, exactly as manager.start_clone resolves it.
    frontend_dir = os.path.join(os.path.dirname(repo_root), "orb-desktop")
    return repo_root, frontend_dir, ""


def _summarize_cmd(label: str, where: str, res) -> Tuple[bool, str]:
    exit_code = getattr(res, "exit_code", None)
    out = (getattr(res, "stdout", "") or "").strip()
    tail = "\n".join(out.splitlines()[-6:])[:600]
    ok = exit_code == 0
    detail = f"{label} in {where} -> exit {exit_code}"
    if tail:
        detail += "\n    " + tail.replace("\n", "\n    ")
    return ok, detail


def _pip_command(repo_root: str, package: str) -> str:
    """Reinstall backend deps in the clone: prefer requirements.txt (covers the common
    'a dep was added but not installed' drift); fall back to the specific missing module."""
    req = os.path.join(repo_root, "requirements.txt")
    safe_pkg = re.sub(r"[^A-Za-z0-9_.\-]", "", package or "")
    root_q, req_q = _sq(repo_root), _sq(req)
    if safe_pkg:
        return (
            f"Set-Location -LiteralPath '{root_q}'; "
            f"if (Test-Path -LiteralPath '{req_q}') {{ python -m pip install -r '{req_q}' }} "
            f"else {{ python -m pip install '{safe_pkg}' }}"
        )
    return f"Set-Location -LiteralPath '{root_q}'; python -m pip install -r '{req_q}'"


# Controller-safe stop: kill ONLY the clone (backend :8000, Vite :5173, Electron/node),
# by PORT + process name, so the sandbox CONTROLLER (python on :8765) is NEVER touched.
# The protected manager.stop_clone kills "all python except *sandbox_controller*", but
# Get-Process usually returns a NULL CommandLine, so that filter misfires and kills the
# controller too (live 2026-06-18: WinError 10054 then 10060 -> "controller unreachable
# mid-loop"). The self-heal loop reboots through THIS instead, so it never severs its own
# lifeline. manager.stop_clone is PROTECTED and unchanged -- we simply don't use it here.
_SAFE_STOP_PS = r"""$ErrorActionPreference='SilentlyContinue'
foreach ($port in 8000,5173) {
  try {
    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
      Select-Object -ExpandProperty OwningProcess -Unique |
      ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
  } catch {}
}
Get-Process electron, node -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Output 'SAFE_STOP_DONE'
"""


def _safe_stop_clone(mgr) -> Tuple[bool, str]:
    """Stop ONLY the clone (backend :8000 / Vite :5173 / electron+node) -- never the
    controller. Runs via the Job 02 visible console. Never raises."""
    try:
        from app.debug.sandbox_console import visible_shell_run
        res = visible_shell_run(_SAFE_STOP_PS, cwd_target="REPO", timeout_seconds=30, client=mgr.client)
        out = getattr(res, "stdout", "") or ""
        if "SAFE_STOP_DONE" in out:
            return True, "clone stopped (backend :8000 / Vite :5173 / electron+node; controller spared)"
        return False, f"stop ran but no confirmation (out: {out.strip()[:120]})"
    except Exception as e:
        return False, f"safe stop failed: {e}"


def _apply_remediation(rem, repo_root: str, frontend_dir: str, mgr) -> Tuple[bool, str]:
    """Run the remediation IN THE CLONE via the Job 02 visible console. Never raises."""
    from app.debug import boot_failure_classifier as clf
    try:
        from app.debug.sandbox_console import visible_shell_run
    except Exception as e:  # pragma: no cover
        return False, f"visible console unavailable: {e}"

    try:
        if rem.action == clf.ACTION_NPM_INSTALL:
            cmd = f"Set-Location -LiteralPath '{_sq(frontend_dir)}'; npm install"
            res = visible_shell_run(cmd, cwd_target="REPO", timeout_seconds=_INSTALL_TIMEOUT, client=mgr.client)
            return _summarize_cmd("npm install", frontend_dir, res)

        if rem.action == clf.ACTION_PIP_INSTALL:
            cmd = _pip_command(repo_root, rem.package)
            res = visible_shell_run(cmd, cwd_target="REPO", timeout_seconds=_INSTALL_TIMEOUT, client=mgr.client)
            return _summarize_cmd("pip install", repo_root, res)

        if rem.action == clf.ACTION_KILL_STALE:
            # stop_clone (run as the first half of the reboot below) already kills stale
            # node/electron/python-except-controller, which frees the port. No extra
            # command needed -- the reboot IS the remediation.
            return True, "stale procs will be cleared by the stop step of the upcoming reboot"

        return False, f"no remediation runner for action '{rem.action}'"
    except Exception as e:  # pragma: no cover - needs a live sandbox
        return False, f"remediation '{rem.action}' raised: {e}"


def _render(log, status: str, report: str, *, classification=None, note: str = "") -> str:
    lines = [f"=== SANDBOX BOOT SELF-HEAL: {status} ===", ""]
    if status == "CLEAN":
        lines.append("The clone booted clean (backend healthy, log clean, no frontend error).")
    elif status == "UNFIXABLE":
        lines.append("The boot is unhealthy and the failure is NOT a known auto-fix. "
                     "Reporting the error verbatim and what was tried.")
    elif status == "CYCLE_CAP":
        lines.append("Hit the remediation cycle cap before the boot came up clean. "
                     "Reporting the latest state and what was tried.")
    elif status == "REPORT":
        lines.append("Boot inspected (report-only -- nothing was changed). Here is how the clone "
                     "came up" + (", and the known fix that would resolve it." if classification is not None else "."))
    elif status == "ABORTED":
        lines.append("Self-heal aborted before it could run.")
    lines.append("")
    lines.append("Loop transcript:")
    lines.extend("  " + l for l in log)
    if classification is not None and getattr(classification, "kind", "unknown") != "unknown":
        lines.append("")
        lines.append(f"Failure class: {classification.kind} (matched pattern: {classification.matched_pattern})")
        if classification.remediation.package:
            lines.append(f"Offending package: {classification.remediation.package}")
        if classification.evidence:
            lines.append(f"Evidence: {classification.evidence}")
    if note:
        lines.append("")
        lines.append(f"Note: {note}")
    if report:
        lines.append("")
        lines.append("--- latest boot inspection (verbatim) ---")
        lines.append(report)
    return "\n".join(lines)[:14000]


def selfheal_sandbox_boot(params: Optional[Dict[str, Any]] = None) -> str:
    """Run the detect->classify->remediate->reboot->re-verify loop on the clone.

    Sync (the tool loop runs it via ``asyncio.to_thread``). Never raises -- every outcome
    is returned as a readable report string for the brain to relay.
    """
    params = params or {}
    # The clone boot is SLOW: wait generously before judging it (floor 60s, default 90s).
    wait_seconds = _clamp(params.get("wait_seconds"), 90, 60, 180)
    max_cycles_per_class = _clamp(params.get("max_cycles_per_class"), 2, 1, 4)
    screenshot_delay = _clamp(params.get("screenshot_delay_seconds"), 3, 0, 30)
    _af = params.get("auto_fix", True)
    auto_fix = True if _af is None else bool(_af)
    inspect_params = {"wait_seconds": wait_seconds, "screenshot_delay_seconds": screenshot_delay}

    try:
        # NB: stop_zombie_orb (manager.stop_clone) is deliberately NOT imported -- it
        # kills the controller (see _safe_stop_clone). The loop reboots via _safe_stop_clone.
        from app.sandbox.manager import get_sandbox_manager, start_zombie_orb
        from app.debug.sandbox_boot_tool import inspect_sandbox_boot
        from app.debug import boot_failure_classifier as clf
    except Exception as e:
        return _render([f"[setup] import failed: {e}"], status="ABORTED", report="")

    mgr = get_sandbox_manager()
    log = []

    # Resolve clone layout from the controller (also our controller-reachability gate).
    repo_root, frontend_dir, perr = _derive_clone_paths(mgr)
    if perr:
        log.append(f"[setup] {perr}")
        return _render(log, status="ABORTED", report="", note=perr)
    log.append(f"[setup] clone repo_root={repo_root} frontend_dir={frontend_dir}")

    # Is the clone already running? You can't boot a second instance over a running one
    # (the ports just conflict) -- so if it's up we either stop+reboot it (fix mode, to
    # get a genuinely fresh boot that reflects any fix) or inspect it as-is (report mode).
    try:
        already_up, _ = mgr.check_backend()
    except Exception:
        already_up = False

    if already_up and not auto_fix:
        # Report-only: inspect the running instance as-is; don't disrupt it.
        log.append("[boot] clone already running -- inspecting the current instance as-is "
                   "(report-only; not rebooting).")
    else:
        # Fix mode, or not running: ensure a FRESH boot. If it's already up, STOP it
        # first -- starting over a running clone just conflicts on :8000 / :5173.
        if already_up:
            log.append("[boot] clone already running -- stopping it first so we boot fresh "
                       "(can't boot a second instance over a running one).")
            _sok, smsg = _safe_stop_clone(mgr)
            log.append(f"[boot] stop -> {smsg}")
        try:
            _ok, msg = start_zombie_orb(full=True)
        except Exception as e:
            log.append(f"[boot] start_sandbox_clone raised: {e}")
            return _render(log, status="ABORTED", report="", note=str(e))
        log.append(f"[boot] start_sandbox_clone -> {msg}")
        if any(w in (msg or "").lower() for w in ("error", "timed out", "timeout")):
            log.append("[boot] (a start timeout/error usually just means the slow launch was "
                       "triggered anyway -- proceeding to wait + inspect, not reporting failure)")

    attempts: Dict[str, int] = {}
    # Overall safety bound: caps every class can be retried, plus a small buffer for
    # legitimate class-to-class progress (fixing one error reveals the next).
    max_total = max_cycles_per_class * len(clf.REMEDIATION_RULES) + 2
    report = ""

    for cycle in range(1, max_total + 1):
        report = inspect_sandbox_boot(inspect_params)
        assessment = clf.assess_boot(report)
        log.append(f"[inspect #{cycle}] {assessment.summary}")

        if assessment.controller_down:
            return _render(log, status="ABORTED", report=report,
                           note="sandbox controller went unreachable mid-loop")

        if assessment.healthy:
            return _render(log, status="CLEAN", report=report)

        cls = assessment.classification

        # Report-only mode ("boot and tell me how it went"): the boot already had the full
        # wait above; surface the state (and the fix that WOULD apply) and stop -- no changes.
        if not auto_fix:
            note = (
                "a known fix is available -- ask me to fix it and I'll apply it + reboot"
                if cls.auto_fixable else
                "no known auto-fix for this -- reporting the state verbatim"
            )
            return _render(log, status="REPORT", report=report,
                           classification=cls if cls.kind != "unknown" else None, note=note)

        if cls.kind == "unknown" or not cls.auto_fixable:
            return _render(log, status="UNFIXABLE", report=report, classification=cls)

        attempts[cls.kind] = attempts.get(cls.kind, 0) + 1
        if attempts[cls.kind] > max_cycles_per_class:
            return _render(log, status="CYCLE_CAP", report=report, classification=cls,
                           note=f"{cls.kind} still failing after {max_cycles_per_class} fix attempt(s)")

        # Apply the known fix in the clone (visible), then reboot and re-inspect.
        ran, detail = _apply_remediation(cls.remediation, repo_root, frontend_dir, mgr)
        log.append(f"[fix {attempts[cls.kind]}/{max_cycles_per_class}] {cls.kind}: {detail}")

        try:
            _sok, smsg = _safe_stop_clone(mgr)
            log.append(f"[reboot] stop -> {smsg}")
            _bok, bmsg = start_zombie_orb(full=True)
            log.append(f"[reboot] start -> {bmsg}")
        except Exception as e:
            log.append(f"[reboot] raised: {e}")
            return _render(log, status="ABORTED", report=report, note=f"reboot failed: {e}")

    return _render(log, status="CYCLE_CAP", report=report,
                   note=f"overall cycle bound ({max_total}) reached without a clean boot")


__all__ = ["SELFHEAL_SANDBOX_BOOT_TOOL", "selfheal_sandbox_boot"]
