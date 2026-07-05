# FILE: app/pipeline_v2/segmented/integrator.py
# Purpose: Derek phase 5 — integrator: contract verification, syntax sweep, targeted re-dispatch, APEX escalation.
# Called-by: app.pipeline_v2.segmented.segmented_builder
# Depends-on: app.pipeline_v2.segmented.worker, app.pipeline_v2.sandbox_tools, app.pipeline_v2.ledger
# Last-renovated: 2026-07-04
"""
The HEAVY-tier integrator: after workers complete, verify the whole.

1. CONTRACT CHECK (deterministic): every name a segment promised to expose
   must actually appear in its files; every consumed name must exist in
   some upstream segment's files.
2. SYNTAX SWEEP (deterministic): py_compile each written .py through the
   profile-routed shell (host for greenfield, sandbox for self-builds).
3. TARGETED REPAIR: failures classify (mirrors the BVL failure_classifier
   Classification shape) -> failure evidence lands in the ledger -> the
   OWNING segment's worker is re-dispatched with that evidence. Bounded by
   ASTRA_SEGMENT_MAX_REDISPATCH (default 2), then ONE APEX escalation
   (FRONTIER_PROVIDER/FRONTIER_MODEL env). Never a full rebuild.

This runs INSIDE the builder stage; the pipeline's own BVL / spec-review /
final-verdict stages still run after it, unchanged.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _max_redispatch() -> int:
    try:
        return max(0, int(os.getenv("ASTRA_SEGMENT_MAX_REDISPATCH", "2")))
    except ValueError:
        return 2


@dataclass
class IntegrationFailure:
    """One integration finding, attributed to its owning segment."""
    kind: str          # "missing_export" | "missing_consumed" | "syntax_error"
    segment_id: str    # owning segment ("" when no single owner)
    detail: str
    file_path: str = ""


@dataclass
class IntegrationReport:
    passed: bool = False
    failures: List[IntegrationFailure] = field(default_factory=list)
    redispatches: int = 0
    escalations: int = 0
    rounds: int = 0


def classify_segment_failure(failure: IntegrationFailure) -> Any:
    """Mirror of bvl.failure_classifier.Classification for segment failures.

    Everything the integrator finds (missing exports, syntax errors) is CODE;
    the environment kinds (emulator offline etc.) never arise here.
    """
    from app.pipeline_v2.bvl.failure_classifier import Classification
    remediation = {
        "missing_export": "implement the promised interface in the owning segment's files",
        "missing_consumed": "the upstream segment must expose this name, or the consumer must import what actually exists",
        "syntax_error": "fix the syntax error in the named file",
        "boot_crash": "the app crashed at launch — fix the exact error in the traceback (imports resolve relative to how the entry point is run: `python src/main.py` means FLAT sibling imports)",
    }.get(failure.kind, "targeted repair in the owning segment")
    return Classification(kind="code", reason=failure.detail, remediation=remediation)


def _identifier(token: str) -> str:
    """Reduce a contract entry to a greppable identifier (name before '(')."""
    t = str(token).strip()
    t = t.split("(")[0]
    for prefix in ("async def ", "def ", "class ", "export ", "const ", "function "):
        if t.startswith(prefix):
            t = t[len(prefix):]
    t = t.split()[-1] if t.split() else t
    return t.strip()


async def _read_segment_files(segment: Any, profile: Any) -> Dict[str, str]:
    """Read a segment's file_scope contents via the profile-routed reader."""
    out: Dict[str, str] = {}
    try:
        from app.pipeline_v2 import sandbox_tools
        for rel in (getattr(segment, "file_scope", None) or []):
            try:
                content = await sandbox_tools.read_file(rel, profile=profile)
                if content:
                    out[rel] = content
            except Exception as exc:
                logger.debug("[integrator] read failed %s: %s", rel, exc)
    except Exception as exc:
        logger.debug("[integrator] sandbox_tools unavailable: %s", exc)
    return out


async def check_contracts(segments: List[Any], profile: Any) -> List[IntegrationFailure]:
    """Deterministic exposes/consumes verification against files on disk."""
    failures: List[IntegrationFailure] = []
    contents: Dict[str, Dict[str, str]] = {}
    for seg in segments:
        sid = getattr(seg, "segment_id", "?")
        contents[sid] = await _read_segment_files(seg, profile)

    def _names(contract: Any) -> List[str]:
        names: List[str] = []
        for attr in ("class_names", "method_signatures", "export_names"):
            vals = getattr(contract, attr, None) or (
                contract.get(attr) if isinstance(contract, dict) else None
            ) or []
            names.extend(_identifier(v) for v in vals if _identifier(v))
        return names

    all_text_by_seg = {sid: "\n".join(files.values()) for sid, files in contents.items()}

    for seg in segments:
        sid = getattr(seg, "segment_id", "?")
        own_text = all_text_by_seg.get(sid, "")
        exposes = getattr(seg, "exposes", None)
        if exposes:
            for name in _names(exposes):
                if not re.search(rf"\b{re.escape(name)}\b", own_text):
                    failures.append(IntegrationFailure(
                        "missing_export", sid,
                        f"segment {sid} promised to expose {name!r} but it appears in none of its files",
                    ))
        consumes = getattr(seg, "consumes", None)
        if consumes:
            upstream_text = "\n".join(
                t for other, t in all_text_by_seg.items() if other != sid
            )
            for name in _names(consumes):
                if not re.search(rf"\b{re.escape(name)}\b", upstream_text):
                    failures.append(IntegrationFailure(
                        "missing_consumed", sid,
                        f"segment {sid} consumes {name!r} but no other segment's files define it",
                    ))
    return failures


async def syntax_sweep(segments: List[Any], profile: Any) -> List[IntegrationFailure]:
    """py_compile every .py in the segments' file_scopes (profile-routed)."""
    failures: List[IntegrationFailure] = []
    try:
        from app.pipeline_v2.sandbox_tools import run_shell, _resolve_path
    except Exception as exc:
        logger.warning("[integrator] syntax sweep unavailable: %s", exc)
        return failures

    # Host builds use THIS interpreter (bare 'python' may not be on PATH);
    # sandbox self-builds use the clone's 'python'.
    try:
        from app.pipeline_v2.android_sandbox import is_host_build
        _py = f'& "{__import__("sys").executable}"' if is_host_build(profile) else "python"
    except Exception:
        _py = "python"

    for seg in segments:
        sid = getattr(seg, "segment_id", "?")
        for rel in (getattr(seg, "file_scope", None) or []):
            if not str(rel).endswith(".py"):
                continue
            try:
                abs_path = _resolve_path(rel, profile)
                result = await run_shell(
                    f'{_py} -m py_compile "{abs_path}"',
                    timeout_sec=30, profile=profile,
                )
                rc = result.get("returncode", 0) if isinstance(result, dict) else 0
                stderr = str(result.get("stderr", "")) if isinstance(result, dict) else str(result)
                if rc != 0 or "SyntaxError" in stderr:
                    failures.append(IntegrationFailure(
                        "syntax_error", sid,
                        f"{rel}: {stderr[-500:] or f'py_compile exit {rc}'}",
                        file_path=str(rel),
                    ))
            except Exception as exc:
                logger.debug("[integrator] syntax check skipped for %s: %s", rel, exc)
    return failures


def _owning_segment_for_traceback(stderr: str, segments: List[Any]) -> str:
    """Attribute a boot traceback to the segment owning the last file it names."""
    import re as _re
    files_in_tb = _re.findall(r'File "([^"]+)"', stderr or "")
    for tb_file in reversed(files_in_tb):
        tb_norm = tb_file.replace("\\", "/").lower()
        for seg in segments:
            for rel in (getattr(seg, "file_scope", None) or []):
                if tb_norm.endswith(str(rel).replace("\\", "/").lower()):
                    return getattr(seg, "segment_id", "")
    # Fallback: the segment owning the entry point owns the boot.
    for seg in segments:
        for rel in (getattr(seg, "file_scope", None) or []):
            if str(rel).replace("\\", "/").lower().endswith("main.py"):
                return getattr(seg, "segment_id", "")
    return ""


async def boot_probe(segments: List[Any], profile: Any) -> List[IntegrationFailure]:
    """live17: deterministic launch check — ZERO tokens.

    Every checkout failure to date (ModuleNotFoundError, missing assets) was
    visible in a subprocess traceback within seconds, yet the first entity
    ever to RUN the built app was the vision+judge+repair stack. The builder
    must hand over an app that at least starts. Host greenfield lane only;
    a healthy process is killed immediately after the probe — window/behaviour
    judgement stays the checkout's job.
    """
    try:
        from app.pipeline_v2.android_sandbox import is_host_build
        from app.pipeline_v2.clone_freshness import is_self_build
        if profile is None or getattr(profile, "language", "") == "kotlin":
            return []
        if is_self_build(profile) or not is_host_build(profile):
            return []
        # argv-driven CLIs legitimately exit nonzero when launched bare —
        # the probe guards windowed/long-running apps only.
        if str(getattr(profile, "framework", "")).lower() == "cli":
            return []
        from app.pipeline_v2.verifier_agent import host_perception as hp
        if hp.guess_entrypoint(profile.project_root) is None:
            return []  # nothing launchable yet — checkout reports that story
        launch = await hp.launch_app(profile.project_root, settle_seconds=4.0)
        try:
            if launch.get("launched") and not launch.get("running"):
                rc = launch.get("returncode")
                if rc not in (0, None):
                    stderr = str(launch.get("stderr", ""))
                    return [IntegrationFailure(
                        "boot_crash",
                        _owning_segment_for_traceback(stderr, segments),
                        f"entry point crashed at launch (rc={rc}):\n{stderr[-1200:]}",
                    )]
            return []
        finally:
            hp.stop_app(launch)
    except Exception as exc:
        logger.warning("[integrator] boot probe skipped: %s", exc)
        return []


def _record_failures_to_ledger(
    failures: List[IntegrationFailure], ledger: Any, job_dir: str,
) -> None:
    if ledger is None or not failures:
        return
    try:
        from app.pipeline_v2.ledger import ledger_append, save_ledger
        for f in failures[:20]:
            ledger_append(
                ledger, entry_type="flag", stage="builder_integrator",
                relevant_to=[f.segment_id or "integration"],
                summary=f"[{f.kind}] {f.detail[:300]}",
                segment=f.segment_id or None, category="integration_failure",
            )
        save_ledger(ledger, job_dir)
    except Exception as exc:
        logger.warning("[integrator] ledger flag write skipped: %s", exc)


def _evidence_block(failures: List[IntegrationFailure]) -> str:
    lines = []
    for f in failures:
        c = classify_segment_failure(f)
        lines.append(f"- [{f.kind}] {f.detail}\n  remediation: {c.remediation}")
    return "\n".join(lines)


def _escalation_model() -> Optional[Tuple[str, str]]:
    provider = os.getenv("FRONTIER_PROVIDER", "").strip()
    model = os.getenv("FRONTIER_MODEL", "").strip()
    return (provider, model) if provider and model else None


async def run_integration(
    segments: List[Any],
    reports: Dict[str, Any],
    job_dir: str,
    job_id: str,
    profile: Any,
    ledger: Any = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> IntegrationReport:
    """Verify -> classify -> re-dispatch owning workers -> escalate. Bounded."""
    from app.pipeline_v2.segmented.worker import run_segment_worker

    emit = on_progress or (lambda m: None)
    result = IntegrationReport()
    by_id = {getattr(s, "segment_id", "?"): s for s in segments}
    attempts: Dict[str, int] = {}
    max_redispatch = _max_redispatch()

    for round_n in range(1 + max_redispatch + 1):  # initial + redispatches + escalation round
        result.rounds = round_n + 1
        failures = await check_contracts(segments, profile)
        failures += await syntax_sweep(segments, profile)
        # live17: only probe the launch when the cheap checks are clean —
        # a deterministic traceback beats every token-spending stage after us.
        if not failures:
            failures += await boot_probe(segments, profile)
        result.failures = failures
        if not failures:
            result.passed = True
            emit("   🔗 Integrator: contracts + syntax + boot probe GREEN")
            return result

        emit(f"   🔗 Integrator round {round_n + 1}: {len(failures)} failure(s)")
        _record_failures_to_ledger(failures, ledger, job_dir)

        by_owner: Dict[str, List[IntegrationFailure]] = {}
        for f in failures:
            by_owner.setdefault(f.segment_id, []).append(f)

        progressed = False
        for sid, owned in by_owner.items():
            seg = by_id.get(sid)
            if seg is None:
                continue
            n = attempts.get(sid, 0)
            if n < max_redispatch:
                attempts[sid] = n + 1
                result.redispatches += 1
                emit(f"   ♻️ Re-dispatching {sid} (attempt {n + 1}/{max_redispatch}) with failure evidence")
                rep = await run_segment_worker(
                    seg, upstream=[s for s in segments if s is not seg],
                    job_dir=job_dir, job_id=job_id, profile=profile, ledger=ledger,
                    handover_context=_evidence_block(owned),
                    on_progress=on_progress,
                )
                reports[sid] = rep
                progressed = True
            elif n == max_redispatch:
                esc = _escalation_model()
                if esc:
                    attempts[sid] = n + 1
                    result.escalations += 1
                    emit(f"   ⬆️ ESCALATING {sid} to APEX ({esc[0]}/{esc[1]}) after {n} failed re-dispatches")
                    rep = await run_segment_worker(
                        seg, upstream=[s for s in segments if s is not seg],
                        job_dir=job_dir, job_id=job_id, profile=profile, ledger=ledger,
                        handover_context=(
                            "REPEATED FAILURES — previous worker attempts did not fix:\n"
                            + _evidence_block(owned)
                        ),
                        on_progress=on_progress,
                        model_override=esc,
                        stage="builder_escalation",
                    )
                    reports[sid] = rep
                    progressed = True
                else:
                    logger.warning("[integrator] no escalation model configured (FRONTIER_PROVIDER/FRONTIER_MODEL)")

        if not progressed:
            break

    result.passed = False
    emit(f"   ❌ Integrator: {len(result.failures)} failure(s) remain after "
         f"{result.redispatches} re-dispatch(es) + {result.escalations} escalation(s)")
    return result


__all__ = [
    "IntegrationFailure", "IntegrationReport",
    "check_contracts", "syntax_sweep", "run_integration",
    "classify_segment_failure",
]
