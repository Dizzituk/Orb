# FILE: app/pipeline_v2/verifier_agent/eyes_judge.py
# Purpose: Derek phase 6 — checkout split into cheap vision EYES + HEAVY JUDGE with ledger verdicts and re-dispatch signals.
# Called-by: app.orchestrator.segment_loop (greenfield host lane)
# Depends-on: app.pipeline_v2.verifier_agent.host_perception, app.llm.stage_roles, app.pipeline_v2.ledger
# Last-renovated: 2026-07-04
"""
Eyes and judge, separated.

EYES (CHECKOUT_EYES, SMALL vision tier — cheap enough to run always):
drive the built app deterministically — launch, settle, screenshot xN,
window inventory, exit/traceback capture — then describe each screenshot
with the vision model. Perception only; no opinions about the spec.

JUDGE (CHECKOUT_JUDGE, HEAVY tier): evaluates the eyes' evidence against
the spec — "does this work from a customer's point of view" — and returns
a structured verdict. FAIL verdicts land in the Decision Ledger as flags
and produce a re-dispatch evidence block in exactly the shape the Phase 5
integrator feeds to segment workers. Repeated judge failures escalate once
to the APEX model (env ASTRA_JUDGE_ESCALATE_AFTER, default 2).

Deterministic checks (BVL tier-1 etc.) stay deterministic — this replaces
nothing; it adds behavioural sight to the greenfield lane, which had none.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _eyes_shots() -> int:
    try:
        return max(1, int(os.getenv("ASTRA_EYES_SHOTS", "3")))
    except ValueError:
        return 3


def _judge_escalate_after() -> int:
    try:
        return max(1, int(os.getenv("ASTRA_JUDGE_ESCALATE_AFTER", "2")))
    except ValueError:
        return 2


@dataclass
class EyesEvidence:
    """Everything the eyes saw. Perception, no judgement."""
    launched: bool = False
    still_running: bool = False
    returncode: Optional[int] = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    screenshots: List[str] = field(default_factory=list)
    descriptions: List[str] = field(default_factory=list)
    window_titles: List[str] = field(default_factory=list)
    launch_error: str = ""
    evidence_dir: str = ""
    # live11: the eyes now PLAY the app — record exactly what was driven so
    # the judge can correlate inputs with the screenshot sequence.
    input_driven: bool = False
    input_log: List[str] = field(default_factory=list)
    # live13: real boot signal — window visible, not an assumed instant boot.
    boot_ready: bool = False
    boot_wait_s: float = 0.0
    boot_window: str = ""

    def render(self) -> str:
        parts = [
            f"launched={self.launched} still_running={self.still_running} returncode={self.returncode}",
            (
                f"BOOT SIGNAL: window '{self.boot_window}' visible after {self.boot_wait_s}s"
                if self.boot_ready else
                f"BOOT SIGNAL: NO window appeared within {self.boot_wait_s}s"
                + (" (process exited)" if not self.still_running else " (process alive but windowless)")
            ),
            f"windows: {self.window_titles[:12]}",
        ]
        if self.input_driven and self.input_log:
            parts.append(
                "INPUT DRIVEN BY EYES (in order, screenshots interleaved):\n"
                + "\n".join(self.input_log)
            )
        if self.launch_error:
            parts.append(f"LAUNCH ERROR: {self.launch_error}")
        if self.stderr_tail:
            parts.append(f"STDERR TAIL:\n{self.stderr_tail}")
        if self.stdout_tail:
            parts.append(f"STDOUT TAIL:\n{self.stdout_tail}")
        for i, (path, desc) in enumerate(zip(self.screenshots, self.descriptions)):
            parts.append(f"SCREENSHOT {i + 1} ({path}):\n{desc}")
        return "\n\n".join(parts)


@dataclass
class JudgeReport:
    verdict: str = "BLOCKED"           # PASS | FAIL | BLOCKED
    reasoning: str = ""
    failures: List[Dict[str, str]] = field(default_factory=list)
    redispatch_signal: str = ""        # Phase-5-shaped evidence block ("" when PASS)
    escalated: bool = False
    model_used: str = ""


async def _describe_screenshot(path: str, spec_hint: str) -> str:
    """One SMALL-tier vision description. Never raises; degrades to a note."""
    try:
        from app.llm.stage_roles import resolve_stage_role
        role = resolve_stage_role("CHECKOUT_EYES")
        prompt = (
            "Describe exactly what is visible in this screenshot of a desktop. "
            "Name any application window that appears related to: "
            f"{spec_hint[:200]}. List visible interactive elements, any error "
            "dialogs, blank/frozen windows, or stack traces. Facts only."
        )
        if role.provider in ("google", "gemini"):
            from app.llm.gemini_vision import ask_about_image
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: ask_about_image(path, prompt),
            )
            text = (result or {}).get("answer") or (result or {}).get("text") or str(result)[:800]
            try:
                from app.cost.cost_recorder import record_llm_cost
                usage = (result or {}).get("usage") or {}
                record_llm_cost(
                    provider="google", model=role.model,
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    stage="checkout_eyes",
                )
            except Exception:
                pass
            return str(text)[:1500]
        return f"(no vision path for provider {role.provider} — screenshot saved at {path})"
    except Exception as exc:
        return f"(vision description unavailable: {exc})"


async def run_eyes(
    profile: Any,
    spec_hint: str,
    job_dir: str,
    emit: Optional[Callable[[str], None]] = None,
) -> EyesEvidence:
    """Deterministic drive + SMALL-tier sight. Cheap enough to run always."""
    from app.pipeline_v2.verifier_agent import host_perception as hp

    emit = emit or (lambda m: None)
    ev = EyesEvidence()
    ev.evidence_dir = os.path.join(job_dir, "eyes_evidence")
    os.makedirs(ev.evidence_dir, exist_ok=True)

    emit("   👀 EYES: launching the built app on the host…")
    launch = await hp.launch_app(profile.project_root, settle_seconds=2.0)
    ev.launched = bool(launch.get("launched"))
    ev.launch_error = str(launch.get("error", ""))
    ev.still_running = bool(launch.get("running"))
    ev.returncode = launch.get("returncode")
    ev.stdout_tail = str(launch.get("stdout", ""))
    ev.stderr_tail = str(launch.get("stderr", ""))

    # live13: WAIT for the boot signal — process alive AND window visible —
    # instead of assuming the app is instantly there. A death while waiting
    # hands the judge the traceback.
    _game_rect = None  # live21: the game window's rect for targeted capture
    if ev.launched and ev.still_running:
        emit("   ⏳ EYES: waiting for boot signal (window visible)…")
        ready = await hp.wait_ready(launch)
        ev.boot_ready = bool(ready.get("ready"))
        ev.boot_wait_s = float(ready.get("waited_s") or 0.0)
        ev.boot_window = str(ready.get("window_title") or "")
        if ready.get("exited"):
            ev.still_running = False
            ev.returncode = ready.get("returncode")
            ev.stdout_tail = ev.stdout_tail or str(ready.get("stdout", ""))
            ev.stderr_tail = ev.stderr_tail or str(ready.get("stderr", ""))
            emit(f"   💥 EYES: app exited during boot wait (rc={ev.returncode})")
        elif ev.boot_ready:
            emit(f"   ✅ EYES: boot signal — window '{ev.boot_window or '(borderless)'}' after {ev.boot_wait_s}s")
            # live21: locate + FOREGROUND the game window so screenshots are
            # the game, not whatever the user has on top of their desktop.
            try:
                _w = await hp.find_pid_window(int(launch["pid"]), foreground=True)
                if _w.get("found"):
                    _game_rect = _w.get("rect")
                    emit(f"   🎯 EYES: game window foregrounded {_game_rect}")
            except Exception as _fw_err:
                logger.debug("[eyes] window foreground failed: %s", _fw_err)
        else:
            emit(f"   ⚠️ EYES: no window within {ev.boot_wait_s}s — proceeding on screenshots only")

    # live11 (2026-07-05, Taz's explicit trust call): when the app is alive,
    # the eyes PLAY it — arrow bursts with a screenshot after each, so the
    # judge sees the piece actually moving. Kill switch: ASTRA_EYES_INPUT=0.
    _input_on = os.getenv("ASTRA_EYES_INPUT", "1").strip().lower() in ("1", "true", "yes")
    _play_script = [
        ("baseline", []),
        ("move left x3", ["{LEFT}", "{LEFT}", "{LEFT}"]),
        ("move right x3", ["{RIGHT}", "{RIGHT}", "{RIGHT}"]),
        ("rotate x2", ["{UP}", "{UP}"]),
        ("soft drop x6", ["{DOWN}"] * 6),
    ]

    try:
        # live13: only drive an app whose window actually appeared — keys
        # into a windowless process prove nothing.
        if _input_on and ev.boot_ready and ev.still_running and launch.get("pid"):
            emit("   🕹️ EYES: driving the app — arrows in, screenshots after each burst")
            for label, keys in _play_script:
                if keys:
                    sent = await hp.send_keys(int(launch["pid"]), keys)
                    ev.input_log.append(
                        f"{label}: {' '.join(keys)} ({'sent' if sent else 'INJECTION FAILED'})"
                    )
                    ev.input_driven = ev.input_driven or sent
                    await asyncio.sleep(0.8)
                shot = await hp.capture_screen(ev.evidence_dir, label="eyes", rect=_game_rect)
                if shot:
                    ev.screenshots.append(shot)
                    emit(f"   👀 EYES: screenshot after '{label}' captured")
            # exit-state check after play: did the app survive being driven?
            _proc = launch.get("_proc")
            if _proc is not None and _proc.poll() is not None:
                ev.still_running = False
                ev.returncode = _proc.poll()
                ev.input_log.append(f"app EXITED during play, returncode={ev.returncode}")
        else:
            for i in range(_eyes_shots()):
                shot = await hp.capture_screen(ev.evidence_dir, label="eyes", rect=_game_rect)
                if shot:
                    ev.screenshots.append(shot)
                    emit(f"   👀 EYES: screenshot {i + 1} captured")
                if i < _eyes_shots() - 1:
                    await asyncio.sleep(2)
        ev.window_titles = await hp.window_titles()
    finally:
        hp.stop_app(launch)

    # live17: a crashed launch needs no vision — the traceback IS the
    # evidence, and desktop screenshots of a nonexistent window are noise.
    if ev.launched and not ev.still_running and ev.returncode not in (0, None):
        ev.descriptions = [
            "(vision skipped — app crashed before any window; the stderr traceback is the evidence)"
        ] * len(ev.screenshots)
    else:
        for shot in ev.screenshots:
            ev.descriptions.append(await _describe_screenshot(shot, spec_hint))

    emit(
        f"   👀 EYES: done — launched={ev.launched}, running={ev.still_running}, "
        f"rc={ev.returncode}, {len(ev.screenshots)} screenshot(s)"
    )
    return ev


_JUDGE_SYSTEM = (
    "You are the final checkout judge for a software build. You receive the "
    "EYES' evidence (launch state, exit codes, tracebacks, screenshot "
    "descriptions — and, when the eyes DROVE the app, the exact key inputs "
    "sent between screenshots; correlate them: inputs with no visible effect "
    "across frames are evidence of broken controls) and the build spec. "
    "Decide whether this works from a CUSTOMER'S point of view. Output strictly:\n"
    "VERDICT: PASS or FAIL\n"
    "REASONING: <2-4 sentences>\n"
    "FAILURES_JSON: [{\"description\": \"...\", \"evidence\": \"...\", \"suspected_files\": [\"...\"]}]\n"
    "FAILURES_JSON must be [] when the verdict is PASS."
)


def _parse_judge(text: str) -> Optional[JudgeReport]:
    m = re.search(r"VERDICT:\s*(PASS|FAIL)", text, re.IGNORECASE)
    if not m:
        return None
    report = JudgeReport(verdict=m.group(1).upper())
    rm = re.search(r"REASONING:\s*(.+?)(?:FAILURES_JSON:|$)", text, re.DOTALL | re.IGNORECASE)
    if rm:
        report.reasoning = rm.group(1).strip()[:1500]
    jm = re.search(r"FAILURES_JSON:\s*(\[.*\])", text, re.DOTALL | re.IGNORECASE)
    if jm:
        try:
            failures = json.loads(jm.group(1))
            report.failures = [f for f in failures if isinstance(f, dict)]
        except json.JSONDecodeError:
            logger.warning("[judge] FAILURES_JSON did not parse")
    return report


def _redispatch_block(failures: List[Dict[str, str]]) -> str:
    """Phase-5-shaped evidence block — same shape the integrator hands workers."""
    lines = []
    for f in failures:
        lines.append(
            f"- [checkout_failure] {f.get('description', '?')}\n"
            f"  evidence: {f.get('evidence', '')}\n"
            f"  suspected files: {', '.join(f.get('suspected_files') or []) or 'unknown'}\n"
            f"  remediation: targeted repair in the owning segment"
        )
    return "\n".join(lines)


def _record_verdict_to_ledger(report: JudgeReport, ledger: Any, job_dir: str) -> None:
    if ledger is None:
        return
    try:
        from app.pipeline_v2.ledger import ledger_append, save_ledger
        ledger_append(
            ledger, entry_type="verification", stage="checkout_judge",
            relevant_to=["checkout"],
            summary=f"Checkout verdict: {report.verdict} — {report.reasoning[:200]}",
            category="checkout_verdict",
        )
        for f in report.failures[:10]:
            ledger_append(
                ledger, entry_type="flag", stage="checkout_judge",
                relevant_to=["checkout"],
                summary=f"[checkout_failure] {str(f.get('description', ''))[:250]}",
                description=str(f.get("evidence", ""))[:1000],
                category="checkout_failure",
            )
        save_ledger(ledger, job_dir)
    except Exception as exc:
        logger.warning("[judge] ledger write skipped: %s", exc)


async def run_judge(
    evidence: EyesEvidence,
    spec_text: str,
    job_id: str,
    job_dir: str,
    ledger: Any = None,
    emit: Optional[Callable[[str], None]] = None,
) -> JudgeReport:
    """HEAVY-tier judgement over the eyes' evidence. Escalates to APEX after
    repeated diagnostic failures (unparseable/errored judge rounds)."""
    from app.llm.stage_roles import resolve_stage_role
    from app.providers.registry import llm_call

    emit = emit or (lambda m: None)
    role = resolve_stage_role("CHECKOUT_JUDGE")
    prompt = (
        f"BUILD SPEC (what the customer asked for):\n{spec_text[:6000]}\n\n"
        f"EYES' EVIDENCE:\n{evidence.render()[:14000]}\n\n"
        "Judge it now."
    )

    attempts = 0
    escalate_after = _judge_escalate_after()
    provider, model, escalated = role.provider, role.model, False
    report: Optional[JudgeReport] = None

    while attempts < escalate_after + 1:
        attempts += 1
        if attempts == escalate_after + 1:
            fp = os.getenv("FRONTIER_PROVIDER", "").strip()
            fm = os.getenv("FRONTIER_MODEL", "").strip()
            if not (fp and fm):
                break
            provider, model, escalated = fp, fm, True
            emit(f"   ⚖️ JUDGE: escalating to APEX ({fp}/{fm}) after {attempts - 1} failed round(s)")
        try:
            result = await llm_call(
                provider_id=provider, model_id=model,
                messages=[{"role": "user", "content": prompt}],
                system_prompt=_JUDGE_SYSTEM,
                max_tokens=2000, timeout_seconds=120,
                stage="checkout_judge",
            )
            if result.is_success() and result.content:
                report = _parse_judge(result.content)
                if report:
                    break
            logger.warning("[judge] round %d unusable: %s", attempts,
                           getattr(result, "error_message", "unparseable"))
        except Exception as exc:
            logger.warning("[judge] round %d error: %s", attempts, exc)

    if report is None:
        report = JudgeReport(verdict="BLOCKED",
                             reasoning="judge produced no parseable verdict")
    report.escalated = escalated
    report.model_used = f"{provider}/{model}"
    if report.verdict == "FAIL" and report.failures:
        report.redispatch_signal = _redispatch_block(report.failures)

    _record_verdict_to_ledger(report, ledger, job_dir)
    emit(f"   ⚖️ JUDGE: {report.verdict} ({report.model_used})"
         + (f" — {len(report.failures)} failure(s), re-dispatch signal ready" if report.failures else ""))
    return report


async def run_eyes_judge_checkout(
    profile: Any,
    spec_text: str,
    job_id: str,
    job_dir: str,
    emit: Optional[Callable[[str], None]] = None,
) -> JudgeReport:
    """The greenfield lane's behavioural checkout: eyes then judge."""
    emit = emit or (lambda m: None)
    ledger = None
    try:
        from app.pipeline_v2.config import LEDGER_ENABLED
        if LEDGER_ENABLED:
            from app.pipeline_v2.ledger import load_or_create_ledger
            ledger = load_or_create_ledger(job_id, job_dir)
    except Exception as exc:
        logger.debug("[eyes_judge] ledger unavailable: %s", exc)

    evidence = await run_eyes(profile, spec_text, job_dir, emit)
    report = await run_judge(evidence, spec_text, job_id, job_dir, ledger, emit)

    # live11: close the Builds-tab final_checkout stage — the orchestrator
    # defers it to this checkout for external host apps, so a verdict here
    # must not leave the stage dangling at "running".
    try:
        from app.pipeline_v2.orchestrator_notify import _update_stage_status
        _update_stage_status(
            "final_checkout",
            "passed" if report.verdict == "PASS" else "failed",
            f"Eyes+judge: {report.verdict} — {report.reasoning[:150]}",
        )
    except Exception as exc:
        logger.debug("[eyes_judge] stage status update skipped: %s", exc)

    return report


async def run_checkout_with_repair(
    profile: Any,
    spec_text: str,
    job_id: str,
    job_dir: str,
    spec: Any = None,
    manifest: Any = None,
    scaffold: Any = None,
    emit: Optional[Callable[[str], None]] = None,
    fix_fn: Optional[Callable] = None,
) -> JudgeReport:
    """live13 — the verifier's own repair fleet (v1: one surgical fixer).

    Taz's directive: a FAIL verdict must not just sit in the ledger — the
    verifier diagnoses, hands the evidence to a fix worker for SURGICAL
    changes, and re-tests, up to ASTRA_CHECKOUT_REPAIR_ROUNDS times. PASS
    stops the loop; BLOCKED (no usable evidence) never triggers repair.
    fix_fn is injectable for tests; default is the agentic builder with a
    checkout_repair cost stage.
    """
    emit = emit or (lambda m: None)
    try:
        rounds = max(0, int(os.getenv("ASTRA_CHECKOUT_REPAIR_ROUNDS", "2")))
    except ValueError:
        rounds = 2

    ledger = None
    try:
        from app.pipeline_v2.config import LEDGER_ENABLED
        if LEDGER_ENABLED:
            from app.pipeline_v2.ledger import load_or_create_ledger
            ledger = load_or_create_ledger(job_id, job_dir)
    except Exception as exc:
        logger.debug("[eyes_judge] ledger unavailable: %s", exc)

    async def _default_fix(handover: str):
        from app.pipeline_v2.agentic_builder import run_agentic_builder
        return await run_agentic_builder(
            spec=spec or {}, manifest=manifest or {}, scaffold=scaffold,
            job_dir=job_dir, handover_context=handover,
            on_progress=emit, profile=profile,
            stage="checkout_repair", job_id=job_id,
        )

    _fix = fix_fn or _default_fix

    def _deterministic_crash_report(ev: EyesEvidence) -> Optional[JudgeReport]:
        """live17: a launch crash is a fact, not a judgement — FAIL without
        spending a single token on vision or the judge."""
        if not (ev.launched and not ev.still_running and ev.returncode not in (0, None)):
            return None
        tb_tail = (ev.stderr_tail or ev.stdout_tail or "no output captured")[-1200:]
        rep = JudgeReport(
            verdict="FAIL",
            reasoning=f"Deterministic: app crashed at launch (rc={ev.returncode}) before any window appeared.",
            failures=[{
                "description": f"entry point crashes at launch (rc={ev.returncode})",
                "evidence": tb_tail,
                "suspected_files": [],
            }],
            model_used="deterministic/boot-probe",
        )
        rep.redispatch_signal = _redispatch_block(rep.failures)
        _record_verdict_to_ledger(rep, ledger, job_dir)
        emit(f"   ⚖️ JUDGE: FAIL (deterministic — launch crash rc={ev.returncode}, no tokens spent)")
        return rep

    evidence = await run_eyes(profile, spec_text, job_dir, emit)
    report = _deterministic_crash_report(evidence) or await run_judge(
        evidence, spec_text, job_id, job_dir, ledger, emit,
    )

    attempted = 0
    while report.verdict == "FAIL" and attempted < rounds:
        attempted += 1
        emit(f"\n   🛠️ CHECKOUT REPAIR {attempted}/{rounds} — surgical fix from the judge's evidence")
        handover = (
            "FINAL CHECKOUT FAILED — you are the repair worker. Fix ONLY what "
            "the evidence below indicts; surgical, targeted changes, never "
            "rewrites. Then STOP — the checkout re-runs automatically.\n"
            "NEVER launch or kill any process (no Start-Process, Stop-Process, "
            "taskkill): the checkout eyes own the app's lifecycle, and such "
            "commands are blocked. Verify with py_compile / pytest / python -c "
            "imports only.\n\n"
            f"JUDGE REASONING:\n{report.reasoning[:1200]}\n\n"
            f"FAILURES:\n{report.redispatch_signal[:2500]}\n\n"
            f"RUNTIME EVIDENCE:\nstderr: {evidence.stderr_tail[:2000]}\n"
            f"stdout: {evidence.stdout_tail[:800]}\n"
            f"boot: ready={evidence.boot_ready} rc={evidence.returncode}"
        )
        try:
            fix_result = await _fix(handover)
            _written = getattr(fix_result, "all_files_written", None) or []
            emit(f"   🛠️ Repair worker touched {len(_written)} file(s)")
            if not _written:
                # live18: a 0-file repair round is no progress — re-testing
                # an unchanged build reproduces the same verdict and burns a
                # full eyes+judge cycle for nothing (round 2 of the first
                # window-up run: 275s, ~$2, zero changes).
                emit("   ⛔ Repair made NO changes — stopping; the last verdict stands")
                break
        except Exception as exc:
            logger.warning("[eyes_judge] repair round %d failed: %s", attempted, exc)
            emit(f"   ⚠️ Repair round {attempted} errored: {exc}")
            break
        emit("   🔁 Re-running eyes+judge after repair…")
        evidence = await run_eyes(profile, spec_text, job_dir, emit)
        report = _deterministic_crash_report(evidence) or await run_judge(
            evidence, spec_text, job_id, job_dir, ledger, emit,
        )

    if report.verdict == "PASS" and attempted:
        emit(f"   ✅ Checkout GREEN after {attempted} repair round(s)")
    elif report.verdict == "FAIL":
        emit(f"   ❌ Checkout still FAILING after {attempted} repair round(s) — evidence in the ledger")

    try:
        from app.pipeline_v2.orchestrator_notify import _update_stage_status
        _update_stage_status(
            "final_checkout",
            "passed" if report.verdict == "PASS" else "failed",
            f"Eyes+judge{f' (+{attempted} repair)' if attempted else ''}: "
            f"{report.verdict} — {report.reasoning[:140]}",
        )
    except Exception as exc:
        logger.debug("[eyes_judge] stage status update skipped: %s", exc)

    return report


__all__ = [
    "EyesEvidence", "JudgeReport",
    "run_eyes", "run_judge", "run_eyes_judge_checkout",
    "run_checkout_with_repair",
]
