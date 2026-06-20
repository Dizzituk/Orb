# FILE: app/debug/boot_failure_classifier.py
# Purpose: Config-driven classifier for the sandbox boot self-heal loop (Job 03).
#          Reads the combined boot report (backend health + clone astra.log scan +
#          frontend vision/OCR) produced by inspect_sandbox_boot and decides: is the
#          boot HEALTHY, or -- if not -- which KNOWN failure is it and what remediation
#          fixes it. The trigger->remediation mapping lives in a declarative table
#          (REMEDIATION_RULES), never hardcoded inline in the loop.
# Called-by: app.debug.sandbox_selfheal (the loop reads assess_boot() each cycle)
# Depends-on: (none -- pure text classification; no LLM, no sandbox, no model IDs)
# Last-renovated: 2026-06-17 (created -- Job 03 sandbox-selfheal)
"""Boot failure classifier.

The self-heal loop is only as good as its reading of the boot. This module turns the
free-text boot report into a structured verdict the loop can branch on:

  * ``classify(report)``    -- map the report to a KNOWN failure + its remediation
                               (or kind="unknown" when nothing matches).
  * ``assess_boot(report)`` -- combine health signals (backend up? log clean? screen
                               clean?) with classify() into one BootAssessment the
                               loop consumes: healthy / auto-fixable / unfixable.

Why config-driven: the failure->fix knowledge is DATA (REMEDIATION_RULES), so a new
failure class is one table row, not a new branch in the loop. The loop stays a dumb
detect->classify->remediate->reboot->re-verify engine.

Contract with app.debug.sandbox_boot_tool (the report producer): read_sandbox_boot
emits the stable markers below (``backend_ok=True``, ``boot looks clean``,
``BOOT ERRORS:``). They are the health signal; the regex rules are the failure signal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# --- remediation action names (what the loop knows how to run in the clone) ---
ACTION_NPM_INSTALL = "npm_install"   # in the frontend dir (clone)
ACTION_PIP_INSTALL = "pip_install"   # in the backend dir (clone)
ACTION_KILL_STALE = "kill_stale"     # stop_clone already kills stale procs; reboot fixes it
ACTION_NONE = "none"                 # no auto-fix -- report verbatim

# --- health markers emitted by app.debug.sandbox_boot_tool.read_sandbox_boot ---
# These are the stable contract between the report producer and this classifier.
BACKEND_OK_MARKER = "backend_ok=True"
BACKEND_CLEAN_MARKER = "boot looks clean"   # "...no ERROR/CRITICAL/Traceback... -- boot looks clean."
BOOT_ERRORS_MARKER = "BOOT ERRORS:"
CONTROLLER_DOWN_MARKERS = (
    "controller is UNREACHABLE",
    "controller check failed",
    "Sandbox unavailable",
    "Sandbox controller is UNREACHABLE",
)

# Negative frontend signals: the vision/OCR readout describing a broken screen that
# the rules below did NOT pin to a specific remediation. An AFFIRMATIVE occurrence
# (not negated -- see _has_visual_negative) flips an otherwise-"clean backend" verdict
# to unhealthy-unclassified, so the loop reports a real on-screen problem rather than
# declaring a false success. Negation matters because the vision PROMPT lists these very
# terms, so a clean answer often echoes them ("no error overlay, booted fine").
_VISUAL_NEGATIVE_MARKERS = (
    "error overlay",
    "red error screen",
    "stack trace",
    "blank white",
    "blank black",
    "crash dialog",
    "white screen",
    "uncaught",
)

# ---------------------------------------------------------------------------
# The config-driven failure -> remediation table.
# Order matters: the first rule whose ANY pattern matches wins, so put the most
# specific rules first. Each rule is pure data; the loop never branches on text.
# ---------------------------------------------------------------------------
REMEDIATION_RULES: List[dict] = [
    {
        # Backend python import error -> reinstall backend deps in the clone.
        # Checked before the broad frontend "Cannot find module" so a python
        # ModuleNotFoundError is never mistaken for a node module miss.
        "kind": "backend_missing_module",
        "patterns": [r"ModuleNotFoundError:\s*No module named"],
        "action": ACTION_PIP_INSTALL,
        "target": "backend",
        "auto_fixable": True,
        "module_regex": r"No module named ['\"]([^'\"]+)['\"]",
    },
    {
        # Frontend Vite/bundler resolve failure (the live @univerjs case) -> npm
        # install in the clone's frontend dir.
        "kind": "frontend_missing_module",
        "patterns": [
            r"Failed to resolve import",
            r"Cannot find module",
            r"Cannot resolve dependency",
            r"Module not found",
        ],
        "action": ACTION_NPM_INSTALL,
        "target": "frontend",
        "auto_fixable": True,
        "module_regex": r"(?:Failed to resolve import|Cannot find module|Module not found:?)\s+[\"']?([@\w./-]+)[\"']?",
    },
    {
        # Stale process still holding the port -> a clean stop (kills node/electron/
        # python-except-controller) + reboot frees it.
        "kind": "port_in_use",
        "patterns": [
            r"address already in use",
            r"EADDRINUSE",
            r"only one usage of each socket address",
            r"port\s+\d+\s+is (?:already )?in use",
            r"\[Errno 98\]",
            r"\[WinError 10048\]",
        ],
        "action": ACTION_KILL_STALE,
        "target": "",
        "auto_fixable": True,
        "module_regex": None,
    },
]


@dataclass
class Remediation:
    """A concrete fix the loop can run in the clone."""
    action: str = ACTION_NONE   # one of the ACTION_* constants
    target: str = ""            # "frontend" | "backend" | "" (for kill_stale)
    package: str = ""           # extracted module name, for reporting only


@dataclass
class Classification:
    """The failure-signal verdict: which known failure (if any) + its remediation."""
    kind: str = "unknown"
    auto_fixable: bool = False
    remediation: Remediation = field(default_factory=Remediation)
    matched_pattern: str = ""
    evidence: str = ""          # the report line that matched (for the report/log)


@dataclass
class BootAssessment:
    """The full verdict the loop branches on each cycle."""
    healthy: bool = False
    backend_ok: bool = False
    controller_down: bool = False
    classification: Classification = field(default_factory=Classification)
    summary: str = ""           # one-line human summary of the verdict


def _first_matching_line(report: str, pattern: str) -> str:
    """Return the first report line containing a match for `pattern` (for evidence)."""
    rx = re.compile(pattern, re.IGNORECASE)
    for line in report.splitlines():
        if rx.search(line):
            return line.strip()[:300]
    return ""


def classify(report: str) -> Classification:
    """Map a boot report to a KNOWN failure + remediation, or kind='unknown'.

    Pure text matching against REMEDIATION_RULES (config). The first rule with any
    matching pattern wins; module_regex (when present) extracts the offending package
    name for reporting and targeted installs.
    """
    text = report or ""
    for rule in REMEDIATION_RULES:
        for pat in rule["patterns"]:
            if re.search(pat, text, re.IGNORECASE):
                package = ""
                mod_re = rule.get("module_regex")
                if mod_re:
                    mm = re.search(mod_re, text)
                    if mm:
                        package = (mm.group(1) or "").strip()
                return Classification(
                    kind=rule["kind"],
                    auto_fixable=bool(rule["auto_fixable"]),
                    remediation=Remediation(
                        action=rule["action"],
                        target=rule.get("target", ""),
                        package=package,
                    ),
                    matched_pattern=pat,
                    evidence=_first_matching_line(text, pat),
                )
    return Classification()  # kind="unknown", not auto-fixable


# Negation tokens that, when they immediately precede a marker, mean the screen is
# DESCRIBED AS FINE ("no error overlay", "without a crash dialog", "free of stack traces").
_NEGATORS = (
    "no ", "not ", "n't", "without", "free of", "none", "no sign of",
    "does not", "did not", "isn't", "aren't", "never",
)


def _has_visual_negative(report: str) -> bool:
    """True only if a negative screen marker appears AFFIRMATIVELY (not negated in the
    short window before it). This keeps a genuinely clean vision answer -- which often
    echoes the prompt's example terms in a negative sentence -- from reading as broken."""
    low = (report or "").lower()
    for m in _VISUAL_NEGATIVE_MARKERS:
        start = 0
        while True:
            idx = low.find(m, start)
            if idx == -1:
                break
            window = low[max(0, idx - 24):idx]
            if not any(neg in window for neg in _NEGATORS):
                return True  # an un-negated mention -> the screen really looks broken
            start = idx + len(m)
    return False


def assess_boot(report: str) -> BootAssessment:
    """Combine health signals + classify() into the loop's per-cycle verdict.

    healthy  := backend up AND log clean AND no known failure AND no negative screen.
    Otherwise the classification tells the loop whether there is an auto-fix
    (auto_fixable) or it must report the error verbatim (kind="unknown").
    """
    text = report or ""
    controller_down = any(m in text for m in CONTROLLER_DOWN_MARKERS)
    backend_ok = BACKEND_OK_MARKER in text
    log_clean = BACKEND_CLEAN_MARKER in text
    has_boot_errors = BOOT_ERRORS_MARKER in text
    cls = classify(text)

    if controller_down:
        return BootAssessment(
            healthy=False, backend_ok=False, controller_down=True,
            classification=cls,
            summary="Sandbox controller unreachable -- cannot boot or self-heal; start the Windows Sandbox.",
        )

    visual_negative = _has_visual_negative(text)
    healthy = (
        backend_ok
        and log_clean
        and not has_boot_errors
        and cls.kind == "unknown"
        and not visual_negative
    )

    if healthy:
        summary = "Boot looks clean: backend healthy, log clean, no frontend errors."
    elif cls.kind != "unknown":
        fix = "auto-fixable" if cls.auto_fixable else "no auto-fix"
        pkg = f" ({cls.remediation.package})" if cls.remediation.package else ""
        summary = f"Known failure: {cls.kind}{pkg} [{fix}]."
    else:
        bits = []
        if not backend_ok:
            bits.append("backend not responding")
        if has_boot_errors:
            bits.append("boot errors in log")
        if visual_negative:
            bits.append("error on screen")
        why = ", ".join(bits) or "unhealthy"
        summary = f"Unhealthy but UNCLASSIFIED ({why}) -- no known auto-fix; report verbatim."

    return BootAssessment(
        healthy=healthy,
        backend_ok=backend_ok,
        controller_down=False,
        classification=cls,
        summary=summary,
    )


__all__ = [
    "ACTION_NPM_INSTALL", "ACTION_PIP_INSTALL", "ACTION_KILL_STALE", "ACTION_NONE",
    "REMEDIATION_RULES",
    "Remediation", "Classification", "BootAssessment",
    "classify", "assess_boot",
]
