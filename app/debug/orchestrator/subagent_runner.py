# FILE: app/debug/orchestrator/subagent_runner.py
# Purpose: Subagent runner — executes a single SubagentBrief against an LLM tool loop.
# Called-by: app.debug.orchestrator.loop_controller, app.debug.orchestrator.spawn_tool
# Depends-on: app.debug.action_executor, app.debug.orchestrator.schemas,
#             app.debug.orchestrator.provider_calls, app.debug.tool_definitions
# Last-renovated: 2026-07-02
"""
Subagent runner — executes a single SubagentBrief against an LLM tool loop.

Reuses:
  - app.debug.tool_definitions for tool schemas (role-scoped here)
  - app.debug.action_executor for tool execution (host/sandbox routing is
    already handled transparently inside action_executor)
  - app.debug.orchestrator.provider_calls for the LLM call: the loop keeps its
    history in the OpenAI chat shape; provider_calls owns the OpenAI/Anthropic
    callers, message adapters, and uniform extraction (DEBUG_PROVIDER toggle,
    2026-07-02). run_subagent/run_subagents_parallel take a `provider` arg
    (default "openai" — byte-identical legacy behaviour).

This is deliberately separate from app.pipeline_v2.llm_tools.run_tool_loop,
which is scoped to the build pipeline (3 tools: read/write/run_shell routed
through the sandbox). Debug subagents need a richer, role-scoped tool set.

v1.0 (2026-04-13): Initial subagent loop for Debug Orchestrator v1.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Callable, Dict, List, Optional, Set, Tuple

from app.debug.action_executor import execute_tool as _execute_tool
from app.debug.orchestrator.provider_calls import (
    call_llm_with_tools,
    extract_assistant_message,
    extract_final_text,
    extract_tool_calls,
)
from app.debug.orchestrator.schemas import (
    Finding,
    StepStatus,
    SubagentBrief,
    SubagentReport,
    SubagentRole,
)
from app.debug.tool_definitions import (
    READ_FILE_TOOL,
    LIST_FILES_TOOL,
    SEARCH_FILES_TOOL,
    READ_LOGS_TOOL,
    READ_PIPELINE_STATE_TOOL,
    WRITE_FILE_TOOL,
    EDIT_FILE_TOOL,
    RUN_COMMAND_TOOL,
    SCREENSHOT_TOOL,
    UI_DUMP_TOOL,
    EMULATOR_TAP_TOOL,
    EMULATOR_TYPE_TOOL,
    EMULATOR_KEY_TOOL,
    GRADLE_BUILD_TOOL,
    GRADLE_INSTALL_TOOL,
    APP_RESTART_TOOL,
    CRASH_LOG_TOOL,
    DESKTOP_SCREENSHOT_TOOL,
    DESKTOP_CLICK_TOOL,
    DESKTOP_TYPE_TOOL,
    DESKTOP_KEY_TOOL,
    DESKTOP_READ_SCREEN_TOOL,
)

logger = logging.getLogger(__name__)

# Hard safety caps — never trust the brief alone.
# Generous: these are last-resort runaway guards, not normal operating limits.
# A well-behaved subagent will stop long before these.
_ABS_MAX_TOOL_CALLS = 500
_ABS_MAX_ITERATIONS = 400
_TOOL_RESULT_TRUNCATE_AT = 30_000

# Reasoning effort per role.
# Philosophy: big models with reasoning plan and synthesise; mini models
# execute concrete instructions without needing to reason about strategy.
# Investigators are sent off with specific tasks by the decomposer — they
# don't need to reason, they need to gather evidence. Same for executors:
# the planner tells them exactly what to change.
_ROLE_REASONING_EFFORT = {
    "investigator": None,       # mini: just go look
    "pattern_matcher": None,    # mini: just go search
    "executor": None,           # mini: just apply the change the planner specified
    "code_verifier": None,      # mini: run command, check output
    "behaviour_verifier": None, # mini: run scripted steps
}


# ---------------------------------------------------------------------------
# Role -> tool scope (single source of truth)
# ---------------------------------------------------------------------------

_READ_ONLY_TOOLS = [
    READ_FILE_TOOL, LIST_FILES_TOOL, SEARCH_FILES_TOOL,
    READ_LOGS_TOOL, READ_PIPELINE_STATE_TOOL,
]

_EXECUTOR_TOOLS = _READ_ONLY_TOOLS + [
    WRITE_FILE_TOOL, EDIT_FILE_TOOL, RUN_COMMAND_TOOL,
]

_CODE_VERIFIER_TOOLS = _READ_ONLY_TOOLS + [RUN_COMMAND_TOOL]

_BEHAVIOUR_VERIFIER_TOOLS = _READ_ONLY_TOOLS + [
    SCREENSHOT_TOOL, UI_DUMP_TOOL,
    EMULATOR_TAP_TOOL, EMULATOR_TYPE_TOOL, EMULATOR_KEY_TOOL,
    GRADLE_BUILD_TOOL, GRADLE_INSTALL_TOOL,
    APP_RESTART_TOOL, CRASH_LOG_TOOL,
    DESKTOP_SCREENSHOT_TOOL, DESKTOP_CLICK_TOOL,
    DESKTOP_TYPE_TOOL, DESKTOP_KEY_TOOL,
    DESKTOP_READ_SCREEN_TOOL,
    RUN_COMMAND_TOOL,
]

ROLE_TOOL_MAP: Dict[SubagentRole, List[dict]] = {
    SubagentRole.INVESTIGATOR: _READ_ONLY_TOOLS,
    SubagentRole.PATTERN_MATCHER: _READ_ONLY_TOOLS,
    SubagentRole.EXECUTOR: _EXECUTOR_TOOLS,
    SubagentRole.CODE_VERIFIER: _CODE_VERIFIER_TOOLS,
    SubagentRole.BEHAVIOUR_VERIFIER: _BEHAVIOUR_VERIFIER_TOOLS,
}


def _tools_for_role(role: SubagentRole) -> List[dict]:
    return [dict(t) for t in ROLE_TOOL_MAP.get(role, _READ_ONLY_TOOLS)]


# ---------------------------------------------------------------------------
# Role-specific system prompts
# ---------------------------------------------------------------------------

_ROLE_PROMPTS: Dict[SubagentRole, str] = {
    SubagentRole.INVESTIGATOR: (
        "You are a Debug Investigator subagent. You have been given a SCOPED "
        "TASK by the planner. Your job is to execute it — fully — and return "
        "specific, actionable findings.\n\n"
        "How to work:\n"
        "1. Read the scoped task carefully. Identify every concrete question "
        "it asks.\n"
        "2. Use read-only tools to gather evidence. Read the suggested "
        "starting files first, then follow imports and references.\n"
        "3. For each finding, cite exact file path + line number, and give "
        "a confidence score 0.0-1.0.\n"
        "4. Do NOT propose fixes or modify anything — that is the executor's "
        "job. Your findings feed the planner, which decides what to change.\n\n"
        "Stopping criteria:\n"
        "- Stop when you have SPECIFIC findings for every question in the "
        "scoped task, OR\n"
        "- Stop when your tool budget is exhausted.\n"
        "- Do NOT stop early with vague findings like 'might need review'. "
        "Either cite exact code or say 'no evidence found' for that question.\n\n"
        "Output: one JSON block. No prose outside it."
    ),
    SubagentRole.PATTERN_MATCHER: (
        "You are a Debug Pattern-Matcher subagent. Your job is to find prior "
        "art: similar bugs, similar fixes, similar code patterns in the "
        "codebase that relate to the current task.\n\n"
        "Use read-only tools. Cite exactly where each pattern was found.\n\n"
        "When done, stop calling tools and respond with a JSON block of findings."
    ),
    SubagentRole.EXECUTOR: (
        "You are a Debug Executor subagent. Your job is to apply a SINGLE "
        "scoped fix step given by the planner. Stay within your target_files "
        "scope. Do NOT make fixes outside what the brief asks for.\n\n"
        "Use edit_file for targeted changes, write_file only for new files. "
        "After writing, briefly verify with run_command (syntax check / "
        "quick test) where sensible.\n\n"
        "When done, stop calling tools and respond with a JSON block noting "
        "what you modified and any concerns."
    ),
    SubagentRole.CODE_VERIFIER: (
        "You are a Debug Code Verifier subagent. Your job is to confirm the "
        "applied fixes compile/lint/pass tests. Run the necessary commands "
        "and report PASS or FAIL with evidence.\n\n"
        "When done, stop calling tools and respond with a JSON block."
    ),
    SubagentRole.BEHAVIOUR_VERIFIER: (
        "You are a Debug Behaviour Verifier subagent. Your job is to run "
        "scripted user-level scenarios (Android via adb, desktop via pyautogui) "
        "and confirm the fix actually produces the expected user-facing "
        "behaviour.\n\n"
        "Capture screenshots or logcat excerpts as evidence.\n\n"
        "When done, stop calling tools and respond with a JSON block."
    ),
}

_JSON_FINDINGS_HINT = (
    "\n\n---\nRESPOND-FORMAT: When you are done calling tools, output a single "
    'JSON object with this exact shape:\n'
    '{\n'
    '  "findings": [ {"summary": str, "confidence": 0.0-1.0, '
    '"evidence": [{"file": str, "line": int, "note": str}], '
    '"tags": [str]} ],\n'
    '  "dead_ends": [str],\n'
    '  "files_modified": [str]\n'
    '}\n'
    "No prose outside the JSON. This is machine-parsed."
)


# ---------------------------------------------------------------------------
# Brief -> initial user prompt
# ---------------------------------------------------------------------------

def _build_initial_user_prompt(brief: SubagentBrief, extra_context: Optional[str] = None) -> str:
    parts = [
        f"## Scoped Task\n{brief.task}",
        f"\n## Target Project\n{brief.target_project}",
    ]
    if brief.related_bugs:
        parts.append("\n## Related Bugs\n" + "\n".join(f"- {b}" for b in brief.related_bugs))
    if brief.context_files:
        parts.append("\n## Suggested Starting Files\n" + "\n".join(f"- {p}" for p in brief.context_files))
    if extra_context:
        parts.append(f"\n## Additional Context\n{extra_context}")
    parts.append(
        f"\n## Tool Budget\nYou have up to {min(brief.max_tool_calls, _ABS_MAX_TOOL_CALLS)} tool calls. "
        "Use them efficiently. Stop when you have enough evidence."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# JSON extraction from final text
# ---------------------------------------------------------------------------

def _extract_json_block(text: str) -> Optional[dict]:
    """Pull the final JSON object out of a response. Tolerant to prose/markdown."""
    if not text:
        return None
    t = text.strip()

    # Strip common markdown code fences
    for fence in ("```json", "```JSON", "```"):
        if fence in t:
            parts = t.split(fence)
            # Find the chunk that looks like JSON
            for chunk in parts:
                chunk = chunk.strip()
                if chunk.startswith("{") and chunk.rstrip().endswith("}"):
                    t = chunk
                    break

    # Last-resort: find outermost braces
    if not t.startswith("{"):
        start = t.find("{")
        end = t.rfind("}")
        if start >= 0 and end > start:
            t = t[start:end + 1]

    try:
        return json.loads(t)
    except Exception as e:
        logger.warning("[subagent_runner] JSON parse failed: %s", e)
        return None


def _parse_findings(raw: Optional[dict]) -> Tuple[List[Finding], List[str], List[str]]:
    """Convert raw JSON from the model into typed Findings + dead_ends + files_modified."""
    if not raw:
        return [], [], []

    findings: List[Finding] = []
    for f in raw.get("findings", []) or []:
        try:
            findings.append(Finding(
                summary=str(f.get("summary", ""))[:500],
                confidence=float(f.get("confidence", 0.5)),
                evidence=f.get("evidence", []) or [],
                tags=[str(t) for t in (f.get("tags", []) or [])],
            ))
        except Exception as e:
            logger.warning("[subagent_runner] Could not parse finding: %s", e)

    dead_ends = [str(d) for d in (raw.get("dead_ends", []) or [])]
    files_modified = [str(p) for p in (raw.get("files_modified", []) or [])]
    return findings, dead_ends, files_modified


# ---------------------------------------------------------------------------
# Main entry point: run_subagent
# ---------------------------------------------------------------------------

async def run_subagent(
    brief: SubagentBrief,
    model: str,
    provider: str = "openai",
    extra_context: Optional[str] = None,
    on_tool_call: Optional[Callable[[str, Dict], None]] = None,
    on_text: Optional[Callable[[str], None]] = None,
) -> SubagentReport:
    """Execute a single subagent tool loop for the given brief.

    The loop's history stays in the OpenAI chat shape regardless of provider;
    provider_calls adapts per call. Returns a SubagentReport with findings,
    files touched, and cost/time stats. Never raises — errors are captured in
    the report.error field.
    """
    t_start = time.time()
    max_calls = min(brief.max_tool_calls, _ABS_MAX_TOOL_CALLS)
    tools = _tools_for_role(brief.role)

    system_prompt = _ROLE_PROMPTS.get(brief.role, _ROLE_PROMPTS[SubagentRole.INVESTIGATOR])
    system_prompt += _JSON_FINDINGS_HINT

    user_prompt = _build_initial_user_prompt(brief, extra_context=extra_context)

    messages: List[Dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    total_in = 0
    total_out = 0
    tool_call_count = 0
    files_read: Set[str] = set()
    files_modified_observed: Set[str] = set()
    final_text = ""
    status = StepStatus.RUNNING
    err: Optional[str] = None

    effort = _ROLE_REASONING_EFFORT.get(brief.role.value)

    try:
        for iteration in range(_ABS_MAX_ITERATIONS):
            resp, in_tok, out_tok = await call_llm_with_tools(
                provider=provider, model=model, messages=messages, tools=tools,
                reasoning_effort=effort,
            )
            total_in += in_tok
            total_out += out_tok

            tcs = extract_tool_calls(resp, provider)
            messages.append(extract_assistant_message(resp, provider))

            if not tcs:
                final_text = extract_final_text(resp, provider)
                if on_text and final_text:
                    try:
                        on_text(final_text)
                    except Exception:
                        pass
                break

            # Execute tool calls (sequentially — parallelism lives at the brief level)
            for tc in tcs:
                if tool_call_count >= max_calls:
                    messages.append({
                        "role": "tool", "tool_call_id": tc["id"],
                        "content": f"ERROR: tool budget exhausted ({max_calls} calls). Stop and return findings now.",
                    })
                    continue

                tool_call_count += 1
                name = tc["name"]
                args = tc["args"]

                if on_tool_call:
                    try:
                        on_tool_call(name, args)
                    except Exception:
                        pass

                # Track files read/modified for the report
                if name == "read_file" and "path" in args:
                    files_read.add(str(args["path"]))
                if name in ("write_file", "edit_file") and "path" in args:
                    files_modified_observed.add(str(args["path"]))

                try:
                    result = await _execute_tool(name, args)
                except Exception as te:
                    result = f"ERROR: tool '{name}' raised: {te}"

                if result and len(result) > _TOOL_RESULT_TRUNCATE_AT:
                    result = (
                        f"[{len(result)} chars — truncated]\n"
                        + result[:_TOOL_RESULT_TRUNCATE_AT // 2]
                        + "\n...\n"
                        + result[-_TOOL_RESULT_TRUNCATE_AT // 2:]
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result or "",
                })

            if tool_call_count >= max_calls:
                # Force one last model call to let it produce findings
                messages.append({
                    "role": "user",
                    "content": "Tool budget exhausted. Produce your findings JSON now.",
                })
        else:
            logger.warning("[subagent_runner] brief=%s hit max iterations", brief.brief_id)

        status = StepStatus.PASSED

    except Exception as e:
        logger.exception("[subagent_runner] brief=%s failed: %s", brief.brief_id, e)
        err = f"{type(e).__name__}: {e}"
        status = StepStatus.FAILED

    # Parse the final JSON block
    raw = _extract_json_block(final_text)
    findings, dead_ends, files_modified_json = _parse_findings(raw)

    # Union observed modifications with whatever the model declared
    all_modified = sorted(files_modified_observed.union(set(files_modified_json)))

    return SubagentReport(
        brief_id=brief.brief_id,
        role=brief.role,
        status=status,
        findings=findings,
        dead_ends=dead_ends,
        files_read=sorted(files_read),
        files_modified=all_modified,
        tool_call_count=tool_call_count,
        tokens_used=total_in + total_out,
        tokens_in=total_in,
        tokens_out=total_out,
        elapsed_ms=int((time.time() - t_start) * 1000),
        error=err,
    )


# ---------------------------------------------------------------------------
# Parallel fan-out helper
# ---------------------------------------------------------------------------

async def run_subagents_parallel(
    briefs: List[SubagentBrief],
    model: str,
    provider: str = "openai",
    max_parallel: int = 5,
    extra_context: Optional[str] = None,
    on_tool_call: Optional[Callable[[str, Dict], None]] = None,
) -> List[SubagentReport]:
    """Run N briefs concurrently with a semaphore cap.

    Briefs with depends_on are NOT handled here — pass only independent briefs.
    The caller (loop_controller) handles dependency ordering.
    """
    sem = asyncio.Semaphore(max(1, max_parallel))

    async def _one(b: SubagentBrief) -> SubagentReport:
        async with sem:
            logger.info("[subagent_runner] Starting brief=%s role=%s", b.brief_id, b.role.value)
            r = await run_subagent(
                brief=b, model=model, provider=provider,
                extra_context=extra_context,
                on_tool_call=on_tool_call,
            )
            logger.info(
                "[subagent_runner] Done brief=%s status=%s tools=%d tokens=%d",
                b.brief_id, r.status.value, r.tool_call_count, r.tokens_used,
            )
            return r

    return await asyncio.gather(*(_one(b) for b in briefs))


# ---------------------------------------------------------------------------
# Regression signature helper (used by loop_controller)
# ---------------------------------------------------------------------------

def signature_for_failure(report: SubagentReport) -> str:
    """Stable hash of a subagent failure for regression detection."""
    key = json.dumps({
        "role": report.role.value,
        "error": report.error or "",
        "findings": sorted([f.summary for f in report.findings]),
    }, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]
