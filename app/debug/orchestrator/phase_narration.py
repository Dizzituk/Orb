# FILE: app/debug/orchestrator/phase_narration.py
# Purpose: Phase-level narration for the Debug-tab orchestrator -- the forcing function
#          behind "the brain says its plan before each move". Owns the `narration`
#          tool-schema contract (required intent + gated reflection), the validation
#          gate, the narration SSE events, AND the phase-aware executor the OpenAI tool
#          loop delegates each tool call to (kept here so chat_tool_loop stays small).
# Called-by: app.llm.chat_tool_loop (get_chat_tools schema inject + the OpenAI
#            tool-call execution in _stream_with_tools_openai)
# Depends-on: app.debug.orchestrator.spawn_tool, app.llm.chat_tool_loop (both lazy)
# Last-renovated: 2026-06-17
"""Phase-level narration (debug-visibility spec).

The Debug brain narrates at the MOVE boundary, not per leaf tool. A "move" is a
phase-level dispatch: spawn_agents (delegate to workers) or a sandbox boot/inspect
action. Each such tool carries a required `narration` object:

    narration: { intent: <required>, reflection: <required before a follow-up move> }

* ``intent``     -- one line on the move about to be made, emitted BEFORE the phase runs.
* ``reflection`` -- one line on what the LAST phase returned, required before the brain
  is allowed to dispatch again. Enforced by a runtime gate (not a prompt plea): a
  dispatch missing its required narration is REFUSED -- the code feeds an error back as
  the tool result instead of executing, and the model must re-issue the call.

Leaf tools (read_file, run_command, search, write/edit, ...) carry no narration and run
silently -- they are just work.

This module also hosts ``run_openai_tool_call``, the per-call executor the OpenAI tool
loop iterates: it validates + narrates phase tools, streams spawn_agents activity, runs
leaf tools, and yields a terminal sentinel carrying the tool-result string for the loop
to feed back to the model.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# The phase-level dispatch tools (Taz, 2026-06-17): spawn_agents + the sandbox
# boot/inspect moves. Everything else is leaf work and narrates nothing.
PHASE_TOOLS = frozenset({
    "spawn_agents",
    "start_sandbox_clone",
    "stop_sandbox_clone",
    "check_sandbox_status",
    "read_sandbox_boot",
    "inspect_sandbox_boot",
    "selfheal_sandbox_boot",
})

# Sentinel event type: the loop appends this content as the `tool` role message and
# reads `dispatched_phase` for the reflection gate. Never forwarded to the client.
TOOL_MESSAGE = "__tool_message__"

_NARRATION_PROPERTY: Dict[str, Any] = {
    "type": "object",
    "description": (
        "REQUIRED narration for this phase-level move (it renders to the user as your "
        "plan, before and after the move). `intent` is required on every call; "
        "`reflection` is required before any follow-up move in the same turn."
    ),
    "properties": {
        "intent": {
            "type": "string",
            "description": (
                "One first-person line on the move you are about to make and WHY, e.g. "
                "'To test that, I need to boot the clone and watch how it comes up.' "
                "Shown to the user before this phase runs."
            ),
        },
        "reflection": {
            "type": "string",
            "description": (
                "One line on what the PREVIOUS phase returned / what you learned, e.g. "
                "'Clone booted clean, backend healthy.' Required before a follow-up move; "
                "omit only on your first move of the turn."
            ),
        },
    },
    "required": ["intent"],
}


def is_phase_tool(name: str) -> bool:
    """True if `name` is a phase-level dispatch tool (carries narration)."""
    return name in PHASE_TOOLS


def inject_narration_schema(anthropic_tool: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of an Anthropic-format tool with the `narration` object added to
    its input_schema and marked required. Deep-copies the schema so module-level tool
    constants are never mutated. No-op (returns the tool unchanged) for non-phase tools.
    """
    if anthropic_tool.get("name") not in PHASE_TOOLS:
        return anthropic_tool
    tool = dict(anthropic_tool)
    schema = copy.deepcopy(tool.get("input_schema") or {"type": "object", "properties": {}})
    props = schema.setdefault("properties", {})
    props["narration"] = copy.deepcopy(_NARRATION_PROPERTY)
    required = schema.setdefault("required", [])
    if "narration" not in required:
        required.append("narration")
    tool["input_schema"] = schema
    return tool


def pop_narration(tool_input: Dict[str, Any]) -> Tuple[str, str]:
    """Remove the narration object from a phase tool's input (so it is not passed on to
    the executor) and return (intent, reflection) as stripped strings. Lenient: also
    accepts flat top-level intent/reflection if the model emitted them there."""
    intent = ""
    reflection = ""
    if isinstance(tool_input, dict):
        nar = tool_input.pop("narration", None)
        if isinstance(nar, dict):
            intent = str(nar.get("intent") or "").strip()
            reflection = str(nar.get("reflection") or "").strip()
        if not intent and "intent" in tool_input:
            intent = str(tool_input.pop("intent") or "").strip()
        if not reflection and "reflection" in tool_input:
            reflection = str(tool_input.pop("reflection") or "").strip()
    return intent, reflection


def dispatch_error(intent: str, reflection: str, *, require_reflection: bool) -> Optional[str]:
    """Return an error string if this phase dispatch must be refused, else None.

    `intent` is always required; `reflection` is required only once a prior phase has
    returned this turn (``require_reflection``)."""
    if not intent:
        return (
            "REFUSED: this is a phase-level move and requires narration.intent -- one "
            "line stating the move you are about to make and why. Re-issue the call "
            "with narration.intent set."
        )
    if require_reflection and not reflection:
        return (
            "REFUSED: a prior phase has already returned this turn, so you must include "
            "narration.reflection -- one line on what that phase returned / what you "
            "learned -- before dispatching the next move. Re-issue the call with "
            "narration.reflection set."
        )
    return None


def narration_events(intent: str, reflection: str, tool: str) -> List[Dict[str, Any]]:
    """Build the narration SSE events for an accepted dispatch: the reflection line
    (if present) THEN the intent line, so the stream reads reflect-then-act."""
    events: List[Dict[str, Any]] = []
    if reflection:
        events.append({"type": "narration", "kind": "reflection", "text": reflection, "tool": tool})
    events.append({"type": "narration", "kind": "intent", "text": intent, "tool": tool})
    return events


async def run_openai_tool_call(
    tc: Dict[str, Any],
    *,
    require_reflection: bool,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Execute ONE OpenAI tool call for the Debug loop, phase-aware.

    Yields client events (narration / subagent_* / tool_result), and finally a
    ``TOOL_MESSAGE`` sentinel:
        {"type": TOOL_MESSAGE, "content": <tool result string>, "dispatched_phase": bool}
    The loop appends `content` as the `tool` role message and uses `dispatched_phase`
    to drive the reflection gate. Phase tools validate + narrate first; a refused
    dispatch yields the error as the tool result and does NOT execute the move.
    """
    name = tc.get("name", "")
    tc_input = tc.get("input") or {}
    dispatched = False
    from app.llm.loop_guard import call_signature, is_failure_result

    if is_phase_tool(name):
        intent, reflection = pop_narration(tc_input)
        err = dispatch_error(intent, reflection, require_reflection=require_reflection)
        if err:
            logger.info("[phase_narration] refused %s dispatch (%s)", name, err.split(" -- ", 1)[0])
            yield {"type": "tool_result", "name": name, "result_preview": err[:200]}
            yield {"type": TOOL_MESSAGE, "content": err, "dispatched_phase": False,
                   "signature": call_signature(name, tc_input), "failed": True}
            return
        for ev in narration_events(intent, reflection, name):
            yield ev
        dispatched = True

    # --- execute the tool (phase or leaf) ---
    if name == "spawn_agents":
        # spawn_agents streams live sub-agent activity (orchestration spec 2.2):
        # forward its progress events, capture the terminal compiled result.
        from app.debug.orchestrator.spawn_tool import spawn_agents_streaming
        result_str = ""
        async for ev in spawn_agents_streaming(tc_input):
            if ev.get("type") == "result":
                result_str = ev.get("content", "")
            else:
                yield ev
    else:
        from app.llm.chat_tool_loop import execute_chat_tool
        result_str = await execute_chat_tool(name, tc_input)

    yield {"type": "tool_result", "name": name, "result_preview": result_str[:200]}
    yield {"type": TOOL_MESSAGE, "content": result_str, "dispatched_phase": dispatched,
           "signature": call_signature(name, tc_input), "failed": is_failure_result(name, result_str)}
