# FILE: app/pipeline_v2/segmented/worker.py
# Purpose: Derek phase 5 — segment worker: fresh small context, scoped writes, ledger-guided, HANDS tier.
# Called-by: app.pipeline_v2.segmented.segmented_builder, app.pipeline_v2.segmented.integrator
# Depends-on: app.llm.stage_roles, app.pipeline_v2.llm_tools, app.pipeline_v2.ledger, app.pipeline_v2.ledger_format
# Last-renovated: 2026-07-04
"""
One segment, one fresh context, one HANDS-tier worker.

The worker sees ONLY: its segment spec, the ledger slice (ground truth),
the interface contracts of its depends_on segments, and its file slice.
Writes outside file_scope are REJECTED and flagged. It has its own
tool-call and token budgets, records its decisions/interfaces to the
ledger, and ends with a structured completion report.

Env:
    ASTRA_WORKER_MAX_TOOL_CALLS  default 40
    ASTRA_WORKER_MAX_TOKENS      default 200000 (in+out per segment)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

COMPLETE_MARKER = "SEGMENT_COMPLETE"


def _worker_max_calls() -> int:
    try:
        return max(4, int(os.getenv("ASTRA_WORKER_MAX_TOOL_CALLS", "40")))
    except ValueError:
        return 40


def _worker_max_tokens() -> int:
    try:
        return max(10_000, int(os.getenv("ASTRA_WORKER_MAX_TOKENS", "200000")))
    except ValueError:
        return 200_000


@dataclass
class WorkerReport:
    """Structured completion report for one segment worker run."""
    segment_id: str
    completed: bool = False
    files_written: List[str] = field(default_factory=list)
    scope_violations: List[str] = field(default_factory=list)
    exposes: Dict[str, List[str]] = field(default_factory=dict)
    decisions: List[Dict[str, str]] = field(default_factory=list)
    notes: str = ""
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""


def _norm(path: str) -> str:
    """Normalise a path for scope comparison. normpath collapses a/./b and
    a/x/../b so traversal survives as a leading '..' (NOT stripped — the
    old lstrip('./') ate it, which is exactly how the escape got in)."""
    import posixpath
    p = str(path).replace("\\", "/").strip()
    p = posixpath.normpath(p)
    if p.startswith("./"):
        p = p[2:]
    return p.lower()


def path_in_scope(path: str, scope: List[str]) -> bool:
    """True when a write path belongs to the segment's file_scope.

    Comparison is suffix-tolerant both ways: workers may write repo-relative
    or root-relative forms of the same file; a scope entry matches when one
    normalised path ends with the other on a path boundary. Paths that still
    contain parent-traversal after normalisation are NEVER in scope (Derek
    p7 review fix: 'core.py/../../evil.py' must not suffix-match its way in).
    """
    p = _norm(path)
    if not p or p == ".":
        return False
    if p == ".." or p.startswith("../") or "/../" in p:
        return False
    for entry in scope:
        s = _norm(entry)
        if p == s or p.endswith("/" + s) or s.endswith("/" + p):
            return True
    return False


def make_scoped_executor(
    segment_id: str,
    file_scope: List[str],
    violations: List[str],
    base_executor: Optional[Callable] = None,
) -> Callable:
    """Wrap the standard tool executor with a file_scope write guard."""
    async def _scoped(name: str, args: Dict) -> Any:
        if name in ("write_file", "edit_file"):
            path = str(args.get("path", ""))
            if not path_in_scope(path, file_scope):
                violations.append(path)
                logger.warning(
                    "[worker:%s] SCOPE VIOLATION rejected: %s(%s) — scope is %s",
                    segment_id, name, path, file_scope,
                )
                return (
                    f"SCOPE VIOLATION: {path} is OUTSIDE this segment's file_scope. "
                    f"You may only write: {', '.join(file_scope)}. If another "
                    f"segment owns that file, record the need in your completion "
                    f"notes instead of writing it."
                )
        nonlocal base_executor
        if base_executor is None:
            from app.pipeline_v2.llm_tools import _execute_tool
            base_executor = _execute_tool
        return await base_executor(name, args)

    return _scoped


def _contract_lines(contract: Any) -> List[str]:
    out = []
    for attr in ("class_names", "method_signatures", "endpoint_paths", "export_names"):
        vals = getattr(contract, attr, None) or (contract.get(attr) if isinstance(contract, dict) else None)
        if vals:
            out.append(f"  {attr}: {', '.join(str(v) for v in vals)}")
    return out


def build_worker_prompt(
    segment: Any,
    upstream: List[Any],
    ledger_block: str,
    handover_context: Optional[str] = None,
    profile: Any = None,
) -> str:
    """The worker's whole world: segment spec + contracts + ledger slice."""
    sid = getattr(segment, "segment_id", "?")
    _root = str(getattr(profile, "project_root", "") or "")
    parts: List[str] = [
        f"You are building ONE segment of a larger job: {sid} — {getattr(segment, 'title', '')}.",
        "",
    ]
    if _root:
        # live16 (2026-07-05): workers kept exploring the ASTRA backend's own
        # folder (their shell cwd) hunting for the project — and "ran" test
        # suites that never touched the built code.
        parts += [
            f"PROJECT ROOT: {_root}",
            "  Every run_shell command MUST operate on THIS root (cd there first).",
            "  The ASTRA backend folder your shell starts in is NOT your project.",
            f"  Tests live under the project root and run from it, e.g.: "
            f'cd "{_root}"; python -m pytest tests -q',
        ]
        if str(getattr(profile, "language", "")).lower() == "python":
            # live17: the #1 recurring launch killer — package-style imports
            # under a script-path entry point.
            parts += [
                "  IMPORT CONVENTION: the app is run as `python src/main.py`, which",
                "  puts src/ itself on sys.path. Modules import siblings FLAT",
                "  (`from board import Board`) — NEVER `from src.board import ...`",
                "  and never relative dots. Tests add src/ to sys.path the same way.",
            ]
        parts += [""]
    parts += [
        "YOUR FILE SCOPE (the ONLY files you may create or modify):",
        *[f"  - {f}" for f in (getattr(segment, "file_scope", None) or [])],
        "",
    ]
    reqs = getattr(segment, "requirements", None) or []
    if reqs:
        parts += ["REQUIREMENTS:", *[f"  - {r}" for r in reqs], ""]
    acc = getattr(segment, "acceptance_criteria", None) or []
    if acc:
        parts += ["ACCEPTANCE CRITERIA:", *[f"  - {a}" for a in acc], ""]

    exposes = getattr(segment, "exposes", None)
    if exposes:
        parts += ["THIS SEGMENT MUST EXPOSE (interface contract — binding):",
                  *_contract_lines(exposes), ""]
    consumes = getattr(segment, "consumes", None)
    if consumes:
        parts += ["THIS SEGMENT CONSUMES (do not re-implement — import/use these):",
                  *_contract_lines(consumes), ""]

    if upstream:
        parts.append("UPSTREAM SEGMENTS ALREADY BUILT (their exposed interfaces are real — read their files if unsure):")
        for up in upstream:
            up_id = getattr(up, "segment_id", "?")
            up_exp = getattr(up, "exposes", None)
            parts.append(f"  {up_id} (files: {', '.join(getattr(up, 'file_scope', []) or [])})")
            if up_exp:
                parts += ["  exposes:"] + ["  " + l for l in _contract_lines(up_exp)]
        parts.append("")

    if ledger_block:
        parts += [ledger_block, ""]

    if handover_context:
        parts += [
            "FAILURE EVIDENCE FROM THE INTEGRATOR (fix exactly this — targeted repair, not a rebuild):",
            handover_context, "",
        ]

    parts += [
        "RULES:",
        "  - Read any file you need for context; WRITE only within your file scope.",
        "  - Honour the ledger decisions above; contradicting one silently is a defect.",
        "  - After writing each file, syntax-check it (run_shell with the toolchain available).",
        "  - CONTRACTS ARE LAW: implement your declared exposes EXACTLY as signed —",
        "    names, signatures, and any stated return meanings. Consume upstream",
        "    ONLY through their declared exposes, using their EXACT names and string",
        "    values (an action string like 'move_left' is a contract — never invent",
        "    your own variant of it).",
        "  - If something you need is NOT in an upstream contract, do not guess or",
        "    re-implement it: note the gap in the completion JSON 'notes' field.",
        "",
        f"When the segment is complete, output {COMPLETE_MARKER} followed by a JSON block:",
        '  {"files_written": [...], "exposes": {"class_names": [], "method_signatures": [],',
        '   "endpoint_paths": [], "export_names": []}, "decisions": [{"key": "", "value": "", "why": ""}],',
        '   "notes": ""}',
        "Report interfaces you ACTUALLY created, decisions you ACTUALLY made.",
    ]
    return "\n".join(parts)


def parse_completion_report(text: str) -> Tuple[bool, Dict[str, Any]]:
    """Extract the SEGMENT_COMPLETE JSON block. Lenient: completion can be
    true even when the JSON fails to parse (files tracked via tools)."""
    if COMPLETE_MARKER not in (text or ""):
        return False, {}
    tail = text.split(COMPLETE_MARKER, 1)[1]
    match = re.search(r"\{.*\}", tail, re.DOTALL)
    if not match:
        return True, {}
    try:
        return True, json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("[worker] completion JSON did not parse — keeping tool-observed facts")
        return True, {}


def record_report_to_ledger(report: WorkerReport, ledger: Any, job_dir: str) -> None:
    """Land the worker's decisions + exposed interfaces in the ledger."""
    if ledger is None:
        return
    try:
        from app.pipeline_v2.ledger import ledger_append, save_ledger, LedgerConflictError
        for d in report.decisions[:12]:
            key = str(d.get("key", "")).strip()
            if not key:
                continue
            try:
                ledger_append(
                    ledger, entry_type="decision", stage="builder_worker",
                    relevant_to=[report.segment_id],
                    summary=f"[{report.segment_id}] {key} = {d.get('value', '')}",
                    key=f"{report.segment_id}.{key}", value=str(d.get("value", "")),
                    rationale=str(d.get("why", "")), segment=report.segment_id,
                )
            except LedgerConflictError as conflict:
                logger.warning("[worker:%s] ledger conflict surfaced: %s", report.segment_id, conflict)
        if any(report.exposes.values() if report.exposes else []):
            ledger_append(
                ledger, entry_type="codebase_fact", stage="builder_worker",
                relevant_to=[report.segment_id],
                summary=f"[{report.segment_id}] exposes: " + json.dumps(report.exposes)[:400],
                segment=report.segment_id, category="interface",
            )
        save_ledger(ledger, job_dir)
    except Exception as exc:
        logger.warning("[worker:%s] ledger record skipped: %s", report.segment_id, exc)


async def run_segment_worker(
    segment: Any,
    upstream: List[Any],
    job_dir: str,
    job_id: str,
    profile: Any,
    ledger: Any = None,
    handover_context: Optional[str] = None,
    on_progress: Optional[Callable[[str], None]] = None,
    model_override: Optional[Tuple[str, str]] = None,
    stage: str = "builder_worker",
) -> WorkerReport:
    """Run one segment in a fresh, small context on the BUILDER_WORKER role.

    model_override=(provider, model) is the APEX escalation path.
    """
    emit = on_progress or (lambda m: None)
    sid = getattr(segment, "segment_id", "?")
    scope = list(getattr(segment, "file_scope", None) or [])
    report = WorkerReport(segment_id=sid)

    if model_override:
        provider, model = model_override
    else:
        from app.llm.stage_roles import resolve_stage_role
        role = resolve_stage_role("BUILDER_WORKER")
        provider, model = role.provider, role.model

    # Ledger slice — ground truth for this segment only
    ledger_block = ""
    if ledger is not None:
        try:
            from app.pipeline_v2.ledger_format import format_ledger_for_prompt
            ledger_block = format_ledger_for_prompt(
                ledger, segment_id=sid, job_dir=job_dir,
                include_file_content=False, max_content_chars=4000,
            )
        except Exception as exc:
            logger.debug("[worker:%s] ledger slice skipped: %s", sid, exc)

    if profile is not None:
        from app.pipeline_v2.builder_prompts import build_system_prompt
        system_prompt = build_system_prompt(profile)
    else:
        system_prompt = "You are a precise software builder with file tools."
    system_prompt += (
        "\n\nSEGMENT-WORKER MODE: you build exactly one segment. Writes outside "
        "your file scope are rejected by the harness."
    )

    prompt = build_worker_prompt(segment, upstream, ledger_block, handover_context, profile=profile)

    executor = make_scoped_executor(sid, scope, report.scope_violations)

    texts: List[str] = []
    tool_counter = {"n": 0}

    def on_tool(name, args):
        tool_counter["n"] += 1
        p = args.get("path", "") or args.get("cmd", "")[:50]
        emit(f"      🔧 [{sid}] {name}({p})")
        if name == "write_file" and args.get("path") and path_in_scope(args["path"], scope):
            if args["path"] not in report.files_written:
                report.files_written.append(args["path"])

    def on_text(text):
        texts.append(text)
        emit(f"      💬 [{sid}] {text[:120]}")

    from app.pipeline_v2.llm_tools import (
        run_tool_loop, set_tool_profile, set_tool_segment, set_tool_manifest,
    )
    # Fresh tool-routing state per worker: stale segment/manifest globals
    # from a previous build would hijack write routing (refuse or misroute).
    set_tool_profile(profile)
    set_tool_segment(None)
    set_tool_manifest(None)

    try:
        _, in_tok, out_tok = await run_tool_loop(
            system_prompt=system_prompt,
            initial_user_message=prompt,
            provider=provider,
            model=model,
            max_iterations=_worker_max_calls(),
            max_tokens=16384,
            on_tool_call=on_tool,
            on_text=on_text,
            stage=stage,
            job_id=job_id,
            tool_executor=executor,
            max_total_tokens=_worker_max_tokens(),
        )
        report.input_tokens = in_tok
        report.output_tokens = out_tok
    except Exception as exc:
        report.error = f"worker loop failed: {exc}"
        logger.exception("[worker:%s] loop failed", sid)
        return report

    report.tool_calls = tool_counter["n"]
    full_text = "\n".join(texts)
    completed, parsed = parse_completion_report(full_text)
    report.completed = completed and bool(report.files_written)
    if parsed:
        report.exposes = parsed.get("exposes") or {}
        report.decisions = [d for d in (parsed.get("decisions") or []) if isinstance(d, dict)]
        report.notes = str(parsed.get("notes", ""))[:2000]

    record_report_to_ledger(report, ledger, job_dir)
    emit(
        f"      {'✓' if report.completed else '✗'} [{sid}] "
        f"{len(report.files_written)} file(s), {report.tool_calls} tool calls, "
        f"{report.input_tokens + report.output_tokens:,} tokens"
        + (f", {len(report.scope_violations)} scope violation(s) rejected" if report.scope_violations else "")
    )
    return report


__all__ = [
    "WorkerReport", "run_segment_worker", "make_scoped_executor",
    "path_in_scope", "build_worker_prompt", "parse_completion_report",
    "COMPLETE_MARKER",
]
