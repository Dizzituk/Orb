# FILE: app/debug/orchestrator/chat_stream_bridge.py
"""
Bridge that runs the Debug Orchestrator and emits its events as SSE bytes in
the same shape that `stream_debug_locked` already uses, so the existing Chat
tab UI renders orchestration runs without any frontend changes.

Event mapping (orchestrator -> chat SSE):
    phase_change           -> token: "## <phase>\n"
    decomposition_complete -> token: "- brief count + ids\n"
    subagent_progress      -> tool_call: {name}
    subagent_complete      -> token: short summary
    plan_complete          -> token: plan summary block
    execution_start        -> token: batch header
    execution_complete     -> token: "files modified: ..."
    verification_complete  -> token: pass/fail summary
    iteration_complete     -> token: iteration header
    resolution             -> token: final summary
    error                  -> error: message

v1.0 (2026-04-13): initial implementation.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator, Optional

from app.debug.orchestrator.loop_controller import run_orchestration
from app.debug.orchestrator.schemas import (
    DebugResolution,
    OrchestrationEvent,
    OrchestrationRequest,
)

logger = logging.getLogger(__name__)


def _sse(data: dict) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode("utf-8")


def _format_event_as_token(evt: OrchestrationEvent) -> Optional[str]:
    """Convert an orchestrator event to a user-facing markdown-ish line."""
    et = evt.event_type
    it = evt.iteration or 0
    phase = evt.phase.value if hasattr(evt.phase, "value") else str(evt.phase)
    msg = evt.message or ""
    data = evt.data or {}

    if et == "phase_change":
        emoji = {
            "decomposing": "🔍",
            "investigating": "🕵️",
            "planning": "🧠",
            "executing": "✏️",
            "verifying_code": "✅",
            "verifying_behaviour": "📱",
            "resolved": "🎉",
            "failed": "❌",
            "max_iterations": "⏱",
        }.get(phase, "•")
        return f"\n{emoji} **{phase.replace('_', ' ').title()}** (iter {it}) — {msg}\n"

    if et == "decomposition_complete":
        ids = data.get("brief_ids", [])
        return f"  Briefs: {len(ids)} → {', '.join(ids)}\n"

    if et == "subagent_complete":
        n = data.get("findings_total")
        return f"  ✓ {msg}{' — findings=' + str(n) if n is not None else ''}\n"

    if et == "plan_complete":
        steps = data.get("steps", []) or []
        contra = data.get("contradictions", []) or []
        out = f"  📋 Plan: {len(steps)} step(s)"
        if contra:
            out += f", ⚠ {len(contra)} contradiction(s)"
        out += f" — {msg}\n"
        return out

    if et == "execution_start":
        return f"  ▶ {msg}\n"

    if et == "execution_complete":
        files = data.get("files_modified", []) or []
        out = f"  ✏ Execution done — {len(files)} file(s) modified"
        if files and len(files) <= 6:
            out += "\n    " + "\n    ".join(files)
        out += "\n"
        return out

    if et == "verification_complete":
        return f"  🔎 {msg}\n"

    if et == "iteration_complete":
        return f"  ─── end of iteration {it} ───\n"

    if et == "resolution":
        return f"\n**Orchestration complete.** {msg}\n"

    if et in ("subagent_progress", "subagent_start"):
        return None  # surfaced as tool_call events elsewhere

    return None


async def stream_orchestration_as_chat(
    project_id: str,
    bug_list: str,
    target_project: Optional[str] = None,
    max_iterations: int = 2,
    max_subagents_parallel: int = 5,
    enable_behaviour_verify: bool = False,
) -> AsyncGenerator[bytes, None]:
    """Run the orchestrator and yield SSE bytes compatible with the Chat tab.

    Consumers receive the same `{"type": "token" | "tool_call" | "done" | "error"}`
    events as the normal chat tool loop, so no frontend changes are required.
    """
    yield _sse({
        "type": "token",
        "content": f"🚀 **Auto-routing to Debug Orchestrator** — parallel pipeline starting.\n\n",
    })
    yield _sse({"type": "metadata", "provider": "openai", "model": "orchestrator"})

    request = OrchestrationRequest(
        project_id=project_id,
        bug_list=bug_list,
        target_project=target_project,
        max_iterations=max_iterations,
        max_subagents_parallel=max_subagents_parallel,
        enable_behaviour_verify=enable_behaviour_verify,
    )

    queue: asyncio.Queue = asyncio.Queue()

    def _on_event(evt: OrchestrationEvent) -> None:
        try:
            queue.put_nowait(evt)
        except Exception as e:
            logger.warning("[chat_stream_bridge] queue.put_nowait failed: %s", e)

    async def _runner() -> DebugResolution:
        try:
            return await run_orchestration(request=request, on_event=_on_event)
        finally:
            await queue.put(None)

    task = asyncio.create_task(_runner())
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            # Map progress (tool calls) to the UI's tool_call event
            if item.event_type == "subagent_progress":
                tool_name = item.data.get("tool") if isinstance(item.data, dict) else None
                if tool_name:
                    yield _sse({
                        "type": "tool_call",
                        "name": tool_name,
                        "tool_use_id": f"orch-{item.iteration}-{tool_name}",
                        "input": {},
                    })
                continue

            line = _format_event_as_token(item)
            if line:
                yield _sse({"type": "token", "content": line})
    except asyncio.CancelledError:
        task.cancel()
        raise

    try:
        resolution = await task
    except Exception as e:
        logger.exception("[chat_stream_bridge] runner crashed: %s", e)
        yield _sse({"type": "error", "error": f"{type(e).__name__}: {e}"})
        return

    # Persist the resolution to the debug project activity timeline so it
    # shows up in the Info tab and survives across sessions.
    try:
        from app.debug.orchestrator.activity_store import record_orchestration
        record_orchestration(
            debug_project_id=project_id,
            resolution=resolution.model_dump(mode="json"),
        )
    except Exception as _act_err:
        logger.debug("[chat_stream_bridge] activity log (orchestration) failed: %s", _act_err)

    # Final block: concise summary
    lines = [
        "\n---",
        f"**Result:** {'✓ Resolved' if resolution.resolved else '✗ ' + resolution.final_phase.value}",
        f"**Iterations:** {len(resolution.iterations)}",
        f"**Tokens:** {resolution.total_tokens:,}",
        f"**Elapsed:** {resolution.total_elapsed_ms // 1000}s",
    ]
    if resolution.unresolved_bugs:
        lines.append(f"**Unresolved:** {len(resolution.unresolved_bugs)}")
        for b in resolution.unresolved_bugs[:5]:
            lines.append(f"  - {b}")
    if resolution.surfaced_contradictions:
        lines.append(f"**Contradictions flagged:** {len(resolution.surfaced_contradictions)}")
    if resolution.summary:
        lines.append("")
        lines.append(resolution.summary)

    yield _sse({"type": "token", "content": "\n".join(lines) + "\n"})
    yield _sse({"type": "done", "provider": "openai", "model": "orchestrator"})
