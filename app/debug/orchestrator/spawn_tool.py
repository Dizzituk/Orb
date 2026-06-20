# FILE: app/debug/orchestrator/spawn_tool.py
# Purpose: The `spawn_agents` tool - lets the Debug-tab orchestrator brain fan work
#          out to parallel role-scoped sub-agents (Claude-Code shape) and stream
#          their live activity back. Wraps the single-brief run_subagent primitive
#          with per-brief model resolution + per-agent progress events.
# Called-by: app.llm.chat_tool_loop (tool registration + streaming splice + blocking fallback)
# Depends-on: app.debug.orchestrator.subagent_runner, app.debug.orchestrator.schemas,
#             app.debug.debug_model_config
# Last-renovated: 2026-06-17
"""spawn_agents - dynamic sub-agent orchestration for the Debug tab.

The orchestrator model calls this mid-tool-loop to delegate. One call spawns N
briefs in parallel, each a scoped task run by run_subagent with role-scoped tools
(investigators/pattern-matchers read-only; executors read+write+run_command;
verifiers run tests/builds).

Per-brief model resolution (debug_model_config): a brief's optional `model` wins,
else the per-role ENV default -- investigators/pattern-matchers on
DEBUG_SUBAGENT_MODEL (gpt-5.5), executors on DEBUG_EXECUTOR_MODEL (gpt-5.4-mini),
verifiers on DEBUG_CODE_VERIFIER_MODEL.

Streaming (spec section 2.2): spawn_agents_streaming() yields per-agent progress
events (subagent_spawn / subagent_start / subagent_progress / subagent_complete /
subagent_spawn_complete) as the fan-out runs, then a terminal
{"type": "result", "content": <serialised reports>} that feeds back to the model.
run_spawn_agents_blocking() is the non-streaming variant for other call paths.

This drives the single-brief run_subagent primitive directly (not
run_subagents_parallel) because we need per-brief models AND per-agent event
attribution -- run_subagents_parallel takes one model for all briefs and a single
un-attributed on_tool_call callback.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from app.debug.debug_model_config import get_model_for_role
from app.debug.orchestrator.schemas import (
    StepStatus,
    SubagentBrief,
    SubagentReport,
    SubagentRole,
)
from app.debug.orchestrator.subagent_runner import run_subagent

logger = logging.getLogger(__name__)

_VALID_ROLES = {r.value for r in SubagentRole}
_RESULT_SERIALISE_CAP = 24000  # hard ceiling on the serialised reports fed to the model


# ---------------------------------------------------------------------------
# Caps (WI-7 formalises; sensible ENV-backed defaults here)
# ---------------------------------------------------------------------------

def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int((os.getenv(name) or "").strip() or default))
    except Exception:
        return default


def max_spawn_per_call() -> int:
    """Hard cap on agents per single spawn_agents call. DEBUG_MAX_SPAWN_PER_CALL."""
    return _int_env("DEBUG_MAX_SPAWN_PER_CALL", 8)


def default_max_parallel() -> int:
    """Concurrency cap when the call doesn't specify. DEBUG_MAX_SPAWN_PARALLEL."""
    return _int_env("DEBUG_MAX_SPAWN_PARALLEL", 5)


# ---------------------------------------------------------------------------
# Tool schema (flat; chat_tool_loop converts to Anthropic input_schema form)
# ---------------------------------------------------------------------------

SPAWN_AGENTS_TOOL: Dict[str, Any] = {
    "name": "spawn_agents",
    "description": (
        "Delegate investigation or fix-application to parallel sub-agents, the way "
        "Claude Code does. Each brief is a scoped task run by its own sub-agent with "
        "ROLE-SCOPED tools: investigators & pattern_matchers are READ-ONLY (find root "
        "cause / find prior art); executors can WRITE files + run commands (apply ONE "
        "scoped fix, stay inside the named files); code_verifier / behaviour_verifier "
        "run tests, builds, or scripted device/desktop checks. "
        "Spawn parallel investigators for multi-file or multi-repo problems and gather "
        "their findings BEFORE writing anything. Scope executors to specific files. "
        "Do NOT spawn for a trivial single-file read -- just call read_file. Returns "
        "each sub-agent's structured findings (summary, confidence, file:line evidence, "
        "files modified)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "briefs": {
                "type": "array",
                "description": "One or more scoped sub-agent tasks to run in parallel.",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "type": "string",
                            "enum": sorted(_VALID_ROLES),
                            "description": (
                                "investigator/pattern_matcher = read-only; executor = "
                                "read+write+run_command; code_verifier/behaviour_verifier = verify."
                            ),
                        },
                        "task": {
                            "type": "string",
                            "description": "Plain-English scoped task or question for this sub-agent.",
                        },
                        "target_project": {
                            "type": "string",
                            "description": "Which repo/project this agent works in (e.g. backend, desktop, bridge).",
                        },
                        "context_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional hint files to seed the investigation.",
                        },
                        "model": {
                            "type": "string",
                            "description": "Optional model override; omit to use the per-role ENV default.",
                        },
                        "max_tool_calls": {
                            "type": "integer",
                            "description": "Optional per-agent tool-call budget (default 15).",
                        },
                    },
                    "required": ["role", "task", "target_project"],
                },
            },
            "max_parallel": {
                "type": "integer",
                "description": "Optional cap on how many sub-agents run concurrently (default from ENV).",
            },
        },
        "required": ["briefs"],
    },
}


# ---------------------------------------------------------------------------
# Brief parsing + per-role model resolution
# ---------------------------------------------------------------------------

def _parse_briefs(input_dict: Dict[str, Any]) -> Tuple[List[Tuple[SubagentBrief, str]], List[str]]:
    """Build [(brief, resolved_model)] from the tool input. Collects warnings for
    skipped/capped briefs. Raises ValueError only if NO valid brief survives."""
    raw = input_dict.get("briefs") if isinstance(input_dict, dict) else None
    if not isinstance(raw, list) or not raw:
        raise ValueError("spawn_agents requires a non-empty 'briefs' array")

    out: List[Tuple[SubagentBrief, str]] = []
    warnings: List[str] = []
    cap = max_spawn_per_call()

    for i, b in enumerate(raw):
        if len(out) >= cap:
            warnings.append(f"capped at {cap} agents; {len(raw) - cap} further brief(s) dropped")
            break
        if not isinstance(b, dict):
            warnings.append(f"brief #{i} ignored (not an object)")
            continue
        role = str(b.get("role", "")).strip().lower()
        if role not in _VALID_ROLES:
            warnings.append(f"brief #{i} ignored (invalid role {role!r})")
            continue
        task = str(b.get("task", "")).strip()
        if not task:
            warnings.append(f"brief #{i} ignored (empty task)")
            continue
        target_project = str(b.get("target_project", "")).strip() or "unknown"
        context_files = [str(p) for p in (b.get("context_files") or []) if str(p).strip()]
        try:
            max_tool_calls = int(b.get("max_tool_calls") or 15)
        except Exception:
            max_tool_calls = 15
        max_tool_calls = max(1, min(max_tool_calls, 1000))

        brief = SubagentBrief(
            brief_id=f"spawn_{i + 1:02d}",
            role=SubagentRole(role),
            task=task,
            target_project=target_project,
            context_files=context_files,
            related_bugs=[],
            depends_on=[],
            max_tool_calls=max_tool_calls,
        )
        model = str(b.get("model", "")).strip() or get_model_for_role(role)
        out.append((brief, model))

    if not out:
        raise ValueError("spawn_agents: no valid briefs after validation: " + "; ".join(warnings))
    return out, warnings


# ---------------------------------------------------------------------------
# Report serialisation (compact, evidence-truncated)
# ---------------------------------------------------------------------------

def _serialise_reports(reports: List[SubagentReport], warnings: List[str]) -> str:
    out: List[Dict[str, Any]] = []
    for r in reports:
        findings: List[Dict[str, Any]] = []
        for f in r.findings:
            refs: List[Dict[str, Any]] = []
            for e in (f.evidence or [])[:5]:
                if isinstance(e, dict):
                    fp = str(e.get("file", "")).strip()
                    ln = e.get("line", "")
                    loc = fp + (f":{ln}" if ln not in ("", None) else "")
                    note = str(e.get("note", ""))[:160]
                    ref: Dict[str, Any] = {"loc": loc or note[:80]}
                    if note:
                        ref["note"] = note
                    refs.append(ref)
                else:
                    refs.append({"loc": str(e)[:160]})
            findings.append({
                "summary": str(f.summary)[:300],
                "confidence": round(float(f.confidence), 2),
                "evidence": refs,
                "tags": list(f.tags or [])[:6],
            })
        out.append({
            "brief_id": r.brief_id,
            "role": r.role.value,
            "status": r.status.value,
            "findings": findings,
            "dead_ends": list(r.dead_ends or [])[:5],
            "files_modified": list(r.files_modified or []),
            "tool_call_count": r.tool_call_count,
            "error": r.error,
        })

    payload: Dict[str, Any] = {"reports": out}
    if warnings:
        payload["warnings"] = warnings
    s = json.dumps(payload, ensure_ascii=False)
    if len(s) > _RESULT_SERIALISE_CAP:
        s = s[:_RESULT_SERIALISE_CAP] + f"\n...[truncated; {len(out)} report(s) total]"
    return s


# ---------------------------------------------------------------------------
# Streaming fan-out (the spawn primitive the tool loop splices in)
# ---------------------------------------------------------------------------

async def spawn_agents_streaming(
    input_dict: Dict[str, Any],
    *,
    extra_context: Optional[str] = None,
    max_parallel: Optional[int] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Fan briefs out over run_subagent, yielding per-agent progress events as they
    happen, then a terminal {"type": "result", "content": <serialised reports>}.

    Never raises: a bad input or a crashed agent is reported in-band so the model's
    tool loop continues. (extra_context is reserved for WI-5 arch-memory injection.)
    """
    try:
        briefs, warnings = _parse_briefs(input_dict)
    except ValueError as e:
        yield {"type": "result", "content": json.dumps({"error": str(e), "reports": []})}
        return

    if max_parallel is None:
        try:
            max_parallel = int(input_dict.get("max_parallel") or 0) or default_max_parallel()
        except Exception:
            max_parallel = default_max_parallel()
    max_parallel = max(1, min(int(max_parallel), len(briefs)))

    # Batch start -> frontend draws the fan-out box with one empty cell per brief.
    yield {
        "type": "subagent_spawn",
        "data": {
            "count": len(briefs),
            "max_parallel": max_parallel,
            "warnings": warnings,
            "briefs": [
                {
                    "brief_id": b.brief_id,
                    "role": b.role.value,
                    "target_project": b.target_project,
                    "task": b.task[:160],
                    "model": m,
                }
                for b, m in briefs
            ],
        },
    }

    queue: "asyncio.Queue[Any]" = asyncio.Queue()
    sem = asyncio.Semaphore(max_parallel)
    reports: Dict[str, SubagentReport] = {}

    async def _run_one(brief: SubagentBrief, model: str) -> None:
        async with sem:
            queue.put_nowait({
                "type": "subagent_start",
                "data": {
                    "brief_id": brief.brief_id,
                    "role": brief.role.value,
                    "target_project": brief.target_project,
                    "task": brief.task[:160],
                    "model": model,
                },
            })

            def _on_tool(name: str, args: Dict) -> None:
                try:
                    path = ""
                    if isinstance(args, dict):
                        path = str(args.get("path", ""))[:200]
                    queue.put_nowait({
                        "type": "subagent_progress",
                        "data": {"brief_id": brief.brief_id, "tool": name, "path": path},
                    })
                except Exception:
                    pass

            # Inject scoped architecture memory (signatures of the brief's seed
            # files + the backend's hot modules) so the sub-agent starts oriented (WI-5).
            try:
                from app.debug.orchestrator.arch_context import build_arch_context
                arch = build_arch_context(brief.context_files, brief.target_project)
            except Exception:
                arch = None
            brief_context = (
                (extra_context + "\n\n" + arch) if (extra_context and arch)
                else (arch or extra_context)
            )

            try:
                report = await run_subagent(
                    brief=brief, model=model,
                    extra_context=brief_context, on_tool_call=_on_tool,
                )
            except Exception as e:  # defensive: run_subagent is documented never-raise
                logger.exception("[spawn_tool] brief=%s crashed: %s", brief.brief_id, e)
                report = SubagentReport(
                    brief_id=brief.brief_id, role=brief.role,
                    status=StepStatus.FAILED, error=f"{type(e).__name__}: {e}",
                )

            reports[brief.brief_id] = report
            queue.put_nowait({
                "type": "subagent_complete",
                "data": {
                    "brief_id": brief.brief_id,
                    "role": brief.role.value,
                    "status": report.status.value,
                    "findings_count": len(report.findings),
                    # Short finding summaries for the click-through agent "views"
                    # panel (bounded: top 6, 200 chars each).
                    "findings": [
                        {"summary": str(f.summary)[:200], "confidence": round(float(f.confidence), 2)}
                        for f in report.findings[:6]
                    ],
                    "files_modified": list(report.files_modified or []),
                    "tool_call_count": report.tool_call_count,
                    "error": report.error,
                },
            })

    async def _run_all() -> None:
        await asyncio.gather(*(_run_one(b, m) for b, m in briefs))

    sentinel = object()
    gather_task = asyncio.create_task(_run_all())
    gather_task.add_done_callback(lambda _t: queue.put_nowait(sentinel))

    # Drain: events are FIFO and the sentinel is enqueued only after gather
    # finishes, so by the time we see it every real event has been yielded.
    while True:
        ev = await queue.get()
        if ev is sentinel:
            break
        yield ev

    if gather_task.done() and gather_task.exception():
        logger.error("[spawn_tool] fan-out gather crashed: %s", gather_task.exception())

    # Cost: record each sub-agent's token spend (per-agent model) on the ledger.
    try:
        from app.cost.cost_recorder import record_llm_cost
        for b, m in briefs:
            r = reports.get(b.brief_id)
            if r and (getattr(r, "tokens_in", 0) or getattr(r, "tokens_out", 0)):
                record_llm_cost("openai", m, r.tokens_in, r.tokens_out, stage="debug_subagent")
    except Exception as e:
        logger.debug("[spawn_tool] cost record failed: %s", e)

    ordered = [reports[b.brief_id] for b, _ in briefs if b.brief_id in reports]
    yield {
        "type": "subagent_spawn_complete",
        "data": {
            "count": len(ordered),
            "passed": sum(1 for r in ordered if r.status == StepStatus.PASSED),
            "failed": sum(1 for r in ordered if r.status == StepStatus.FAILED),
            "findings_count": sum(len(r.findings) for r in ordered),
            "files_modified": sorted({p for r in ordered for p in (r.files_modified or [])}),
        },
    }
    # Compile the N reports into ONE short brief BEFORE the brain sees them
    # (phase-narration spec): keeps the orchestrator's context clean so its reflection
    # line is crisp. compile_reports falls back to the raw serialised reports if the
    # compiler model is unavailable, so the brain always receives the fan-out outcome.
    serialised = _serialise_reports(ordered, warnings)
    try:
        from app.debug.orchestrator.report_compiler import compile_reports
        brief = await compile_reports(serialised)
    except Exception as e:  # defensive: compile_reports is documented never-raise
        logger.debug("[spawn_tool] report compile failed: %s", e)
        brief = serialised
    yield {"type": "result", "content": brief}


async def run_spawn_agents_blocking(
    input_dict: Dict[str, Any],
    *,
    extra_context: Optional[str] = None,
) -> str:
    """Non-streaming variant: run the fan-out and return only the serialised reports.
    Used by execute_chat_tool as a fallback for non-streaming call paths."""
    result = ""
    async for ev in spawn_agents_streaming(input_dict, extra_context=extra_context):
        if ev.get("type") == "result":
            result = ev.get("content", "")
    return result or json.dumps({"reports": []})
