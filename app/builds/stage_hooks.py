# FILE: app/builds/stage_hooks.py
"""
Stage Lifecycle Hooks — wraps pipeline stream handlers to report
build project stage transitions and log output to per-project messages.

This module provides a generator wrapper that:
1. Before the stream starts: creates/finds a build project, notifies stage start
2. During streaming: collects output tokens and forwards them
3. After the stream ends: notifies stage passed/failed, persists output

Usage: Called from command_dispatch.py around handler invocation.
"""

import json
import logging
import time
from typing import AsyncGenerator, Dict, Optional, Tuple

from sqlalchemy.orm import Session

from app.builds.pipeline_bridge import (
    get_or_create_build_project,
    resolve_build_target,
    notify_stage_start,
    notify_stage_passed,
    notify_stage_failed,
    notify_stage_awaiting,
)
from app.transparency.collector import ReasoningCollector
from app.transparency.io_tracker import IOTracker

logger = logging.getLogger(__name__)

# Map dispatch table stage names → pipeline stage enum values
_STAGE_MAP = {
    "weaver": "weaver",
    "spec_gate": "spec_gate",
    "pipeline": "critical_pipeline",
    "critical_pipeline": "critical_pipeline",
    "segment_loop": "critical_pipeline",
    "overwatcher": "overwatcher",
    "implementation": "implementer",
    "implementer": "implementer",
    "final_checkout": "final_checkout",
}

# Stages that should trigger build project tracking
_TRACKED_STAGES = set(_STAGE_MAP.keys())

# ══════════════════════════════════════════════════════════════════
# v2.2 (2026-04-18) Dispatch deduplication
# ══════════════════════════════════════════════════════════════════
#
# Both the auto-advance SSE event (emitted by stage_hooks when a prior
# stage passes) and explicit user confirmations (voice/typed "Send to
# Spec Gate") route to the same stage handler. When both arrive in
# quick succession, the stage runs twice — two Claude Opus calls with
# extended thinking, two ~9-minute inspections, double cost, and a
# confused build project state.
#
# This module-level registry tracks in-flight stage dispatches keyed by
# (chat_project_id, dispatch_stage). Additions and checks are atomic
# within asyncio's single-threaded event loop (no await between check
# and add). A timestamp-based cooldown covers rare cases where the
# first invocation completes so fast that the second arrives after the
# tracking key was already removed.

_IN_FLIGHT_STAGES: Dict[Tuple[int, str], float] = {}
# Window during which a second identical dispatch is treated as a
# duplicate even if the first one is no longer "in flight". Covers the
# race where the auto-advance event and the user's voice command
# overlap by just a few seconds after a fast stage completion.
_DUPLICATE_DISPATCH_WINDOW_SEC = 15.0


def _is_duplicate_dispatch(chat_project_id: int, dispatch_stage: str) -> bool:
    """Check whether this (chat_project_id, stage) pair is already in
    flight or completed within the cooldown window. Atomic in the
    asyncio sense — no await between read and write.
    """
    key = (chat_project_id, dispatch_stage)
    now = time.monotonic()
    # Purge stale entries that fell outside the dedup window entirely.
    # Keeps the dict small over long-running processes.
    for k in list(_IN_FLIGHT_STAGES.keys()):
        if now - _IN_FLIGHT_STAGES[k] > _DUPLICATE_DISPATCH_WINDOW_SEC * 10:
            _IN_FLIGHT_STAGES.pop(k, None)
    ts = _IN_FLIGHT_STAGES.get(key)
    if ts is not None and (now - ts) < _DUPLICATE_DISPATCH_WINDOW_SEC:
        return True
    _IN_FLIGHT_STAGES[key] = now
    return False


def _release_dispatch_slot(chat_project_id: int, dispatch_stage: str) -> None:
    """Remove the in-flight marker once the stage has finished streaming.
    The timestamp stays in _IN_FLIGHT_STAGES only until the cooldown
    window expires — see _is_duplicate_dispatch for the purge logic.
    """
    key = (chat_project_id, dispatch_stage)
    # Refresh timestamp so the cooldown window starts from stage end, not
    # stage start. This is what catches the 'user said the thing right
    # after auto-advance finished' race.
    _IN_FLIGHT_STAGES[key] = time.monotonic()

# v2.1 Auto-advance: when a stage passes, which intent to fire next.
# This creates the one-button flow: Weaver → SpecGate → Builder (auto).
_AUTO_ADVANCE_MAP = {
    "weaver": "SEND_TO_SPEC_GATE",
    "spec_gate": "RUN_CRITICAL_PIPELINE_FOR_JOB",
    # critical_pipeline → final_checkout is handled inside the v2.1 orchestrator
}


def is_tracked_stage(stage_name: str) -> bool:
    """Check if a dispatch stage should trigger build project tracking."""
    return stage_name in _TRACKED_STAGES


def get_pipeline_stage(dispatch_stage: str) -> Optional[str]:
    """Map a dispatch table stage name to a pipeline stage enum value."""
    return _STAGE_MAP.get(dispatch_stage)


def _extract_brief_from_conversation(db: Session, chat_project_id: int) -> Optional[str]:
    """
    Extract the user's ramble/brief from recent conversation history.
    The command message itself ("how does that look all together") isn't useful
    as a project name — we need the actual descriptive messages before it.
    """
    try:
        from app.memory import service as memory_service
        history = memory_service.get_message_history(db, chat_project_id, limit=10)
        messages = history.messages if hasattr(history, 'messages') else history
        # Walk backwards through messages to find the last substantive user message
        # Skip the command trigger message itself
        for msg in reversed(messages):
            role = msg.role if hasattr(msg, 'role') else msg.get('role', '')
            content = (msg.content if hasattr(msg, 'content') else msg.get('content', '')).strip()
            if role == "user":
                # Skip short/command/confirmation messages
                if (
                    len(content) > 30
                    and "command:" not in content.lower()
                    and not content.lower().startswith("confirmed:")
                    and "confirm:" not in content.lower()[:20]
                ):
                    return content
        # If no long message found, try any user message that isn't a command
        for msg in reversed(messages):
            role = msg.role if hasattr(msg, 'role') else msg.get('role', '')
            content = (msg.content if hasattr(msg, 'content') else msg.get('content', '')).strip()
            if role == "user":
                if (
                    "command:" not in content.lower()
                    and not content.lower().startswith("confirmed:")
                    and len(content) > 5
                ):
                    return content
    except Exception as e:
        logger.debug("[stage_hooks] Failed to extract brief from conversation: %s", e)
    return None


def _make_sse_event(data: dict) -> bytes:
    """Create an SSE event as bytes (matching Weaver/SpecGate format)."""
    return f"data: {json.dumps(data)}\n\n".encode("utf-8")


def _parse_sse_chunk(chunk) -> Optional[dict]:
    """Parse an SSE chunk (str or bytes) into a dict, or None if unparseable."""
    try:
        if isinstance(chunk, bytes):
            text = chunk.decode("utf-8", errors="replace")
        elif isinstance(chunk, str):
            text = chunk
        else:
            return None

        text = text.strip()
        if text.startswith("data: "):
            return json.loads(text[6:])
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass
    return None


def _extract_weaver_intent(weaver_output: str) -> Optional[dict]:
    """Extract key requirements, design preferences, and constraints from Weaver output.

    The Weaver produces bold-bullet structured outlines like:
        - **Key requirements**
          - Transform Debug Assistant from...
          - **Debug tab UX**
            - Clicking the Debug tab...
        - **Design preferences**
          - Debug Projects should look...
        - **Constraints**
          - Must be grounded in...

    This parser finds each top-level bold section header, then collects
    all indented sub-bullets beneath it as the section content.
    """
    import re

    extraction: dict = {
        "key_requirements": [],
        "design_preferences": [],
        "constraints": [],
    }

    # Category detection patterns
    _CATEGORY_MAP = [
        (re.compile(r"key\s*requirement|requirement|feature|function", re.I), "key_requirements"),
        (re.compile(r"design\s*preference|\bdesign\b|\bstyle\b|\bui\b|\bux\b|\bvisual\b|\baesthetic\b", re.I), "design_preferences"),
        (re.compile(r"constraint|rule|limit|tech.*stack|restriction|must.*not", re.I), "constraints"),
    ]

    lines = weaver_output.split("\n")
    current_category: Optional[str] = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Measure indentation to distinguish root-level from nested bullets
        leading_spaces = len(line) - len(line.lstrip(" "))

        # Detect top-level section headers: "- **Key requirements**"
        # Must be at root level (0-1 spaces indent) with bold text
        if leading_spaces <= 1:
            top_match = re.match(r"^-\s+\*\*(.+?)\*\*\s*$", stripped)
            if top_match:
                heading_text = top_match.group(1).strip()
                matched = False
                for pattern, category in _CATEGORY_MAP:
                    if pattern.search(heading_text):
                        current_category = category
                        matched = True
                        break
                if not matched:
                    current_category = None
                continue
            # Non-bold root bullet — new unknown section, stop collecting
            if re.match(r"^-\s+", stripped):
                current_category = None
                continue

        # If we're inside a known category, collect ALL indented content
        if current_category is not None and leading_spaces >= 2:
            sub_match = re.match(r"^\s{2,}-\s+(.+)", line)
            if sub_match:
                item_text = sub_match.group(1).strip()
                # Strip bold wrapper if present (e.g. "**Debug tab UX**")
                item_text = re.sub(r"^\*\*(.+?)\*\*:?$", r"\1", item_text)
                if len(item_text) > 5:
                    extraction[current_category].append(item_text)

    # Fallback: if the parser found nothing (different Weaver format), grab
    # all substantial bullets as requirements
    total = sum(len(v) for v in extraction.values())
    if total == 0:
        fallback = re.findall(r"^\s*-\s+(.{15,})", weaver_output, re.MULTILINE)
        if fallback:
            extraction["key_requirements"] = [
                re.sub(r"^\*\*(.+?)\*\*:?\s*", r"\1: ", item.strip())
                for item in fallback[:15]
                if not re.match(r"^\*\*(?:Organizing|Analyzing)\b", item.strip())
            ]

    if any(v for v in extraction.values()):
        return extraction
    return None


async def wrap_with_build_tracking(
    stream: AsyncGenerator,
    db: Session,
    chat_project_id: int,
    dispatch_stage: str,
    message: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> AsyncGenerator:
    """
    Wrap a pipeline stream handler with build project lifecycle hooks.

    Yields all events from the inner stream while:
    - Creating/finding the build project on first call
    - Notifying stage start
    - Collecting output for per-project message log
    - Notifying stage passed/failed on completion
    - Emitting a build_project_update SSE event so the frontend can refresh
    """
    pipeline_stage = get_pipeline_stage(dispatch_stage)
    if not pipeline_stage:
        async for chunk in stream:
            yield chunk
        return

    # v2.2 (2026-04-18): Dispatch deduplication — reject concurrent or
    # near-concurrent duplicate invocations for the same (chat_project,
    # stage) pair. Fixes the auto-advance + voice-command collision that
    # caused two Claude Opus spec_gate calls in the previous run.
    if _is_duplicate_dispatch(chat_project_id, dispatch_stage):
        logger.info(
            "[stage_hooks] v2.2 DUPLICATE DISPATCH suppressed: stage=%s "
            "chat_project=%s — already in flight or within %.0fs cooldown",
            dispatch_stage, chat_project_id, _DUPLICATE_DISPATCH_WINDOW_SEC,
        )
        print(
            f"[STAGE_HOOKS] v2.2 DUPLICATE SUPPRESSED: "
            f"{dispatch_stage} for chat_project={chat_project_id}"
        )
        # Emit a minimal SSE event so the frontend knows something arrived
        # but was no-op'd, then return without invoking the inner stream.
        yield _make_sse_event({
            "type": "duplicate_dispatch_suppressed",
            "stage": dispatch_stage,
            "chat_project_id": chat_project_id,
            "reason": "already_in_flight_or_cooldown",
        })
        return

    # Extract the actual brief from conversation history, not the command text
    brief = _extract_brief_from_conversation(db, chat_project_id) or message

    # Get or create build project
    try:
        build_project = get_or_create_build_project(
            db, chat_project_id, brief=brief,
        )
        build_project_id = build_project.id

        # v2.2: Resolve build target (which project are we building into?)
        resolve_build_target(db, build_project, message=message, chat_project_id=chat_project_id)

        logger.info(
            "[stage_hooks] Tracking stage '%s' on build project '%s' (%s) target=%s",
            pipeline_stage, build_project.name, build_project_id, build_project.build_target_id,
        )
    except Exception as e:
        logger.warning("[stage_hooks] Failed to get/create build project: %s", e)
        async for chunk in stream:
            yield chunk
        return

    # Notify stage start
    notify_stage_start(db, build_project_id, pipeline_stage, detail="Started via command dispatch")
    print(f"[STAGE_HOOKS] Stage '{pipeline_stage}' STARTED for build project '{build_project.name}' ({build_project_id})")

    # Initialise transparency collector for this stage
    stage_index_map = {"weaver": 0, "spec_gate": 1, "critical_pipeline": 2, "implementer": 3, "final_checkout": 4}
    transparency = ReasoningCollector(
        job_id=build_project.job_id or "",
        run_id=f"run_{build_project_id[:8]}",
        build_project_id=build_project_id,
    )
    transparency.start_stage(
        stage_name=pipeline_stage,
        stage_index=stage_index_map.get(pipeline_stage, 0),
        metadata={"provider": provider, "model": model, "message_preview": (message or "")[:200]},
    )

    # Emit reasoning event start via SSE (guard against None)
    if transparency._current_event:
        yield _make_sse_event(transparency._current_event.to_sse_dict())

    # Emit a build project update event so frontend can refresh the grid
    yield _make_sse_event({
        "type": "build_project_update",
        "build_project_id": build_project_id,
        "stage": pipeline_stage,
        "status": "running",
    })

    # IO Tracker: set context var so all sandbox_fs / SandboxClient calls
    # within the inner stream handlers are automatically logged.
    io_tracker = IOTracker(collector=transparency, stage_name=pipeline_stage)

    # Stream through, collecting tokens for the per-project message log
    output_parts = []
    had_error = False
    is_awaiting = False
    spec_id = None
    # v17 (2026-05-20): track hard-stop signals from spec_gate_stream.
    # When SpecGate hard-stops (e.g. sandbox unreachable) it emits the
    # signal inside the 'done' event payload rather than as an 'error',
    # so without explicit detection the stage falsely reports as passed.
    hard_stopped = False
    hard_stop_reason = ""
    chunk_count = 0

    print(f"[STAGE_HOOKS] About to iterate inner stream for stage '{pipeline_stage}'")
    try:
        with io_tracker:
            async for chunk in stream:
                chunk_count += 1
                if chunk_count == 1:
                    print(f"[STAGE_HOOKS] First chunk received from inner stream (type={type(chunk).__name__}, len={len(chunk) if chunk else 0})")
                # Forward the chunk as-is (preserving bytes or str type)
                yield chunk

                # Parse to collect output and detect status
                parsed = _parse_sse_chunk(chunk)
                if parsed:
                    event_type = parsed.get("type", "")
                    if event_type == "token":
                        content = parsed.get("content", "")
                        if content:
                            output_parts.append(content)
                    elif event_type == "error":
                        had_error = True
                    elif event_type in ("spec_blocked", "spec_questions"):
                        is_awaiting = True
                    elif event_type in ("spec_ready", "spec_segmented"):
                        spec_id = parsed.get("spec_id")
                    elif event_type == "done" and parsed.get("hard_stopped"):
                        # v17: SpecGate signals hard-stop in the done event.
                        hard_stopped = True
                        # The reason is the most recent token containing 'HARD STOP'.
                        for _part in reversed(output_parts):
                            if "HARD STOP" in _part:
                                hard_stop_reason = _part.replace("HARD STOP", "").replace("**", "").strip("*\n :")
                                break

    except Exception as e:
        had_error = True
        logger.warning("[stage_hooks] Stream error after %d chunks: %s", chunk_count, e)

    print(f"[STAGE_HOOKS] Stage '{pipeline_stage}' stream FINISHED: {chunk_count} chunks, {sum(len(p) for p in output_parts)} output chars, error={had_error}, awaiting={is_awaiting}, hard_stopped={hard_stopped}")

    # Finish transparency trace for this stage
    try:
        t_status = "failed" if (had_error or hard_stopped) else ("warning" if is_awaiting else "passed")
        t_summary = f"{pipeline_stage} {'failed' if had_error else 'awaiting input' if is_awaiting else 'completed'}: {chunk_count} chunks, {sum(len(p) for p in output_parts)} chars output"
        finished_event = await transparency.finish_stage(
            status=t_status,
            model_used=model or "",
            reasoning_summary=t_summary,
        )
        # Emit final reasoning event via SSE so frontend sees the completed state
        if finished_event:
            yield _make_sse_event(finished_event.to_sse_dict())
    except Exception as te:
        logger.debug("[stage_hooks] Transparency finish failed: %s", te)
    logger.info(
        "[stage_hooks] Stage '%s' stream finished: %d chunks, %d output chars, error=%s, awaiting=%s",
        pipeline_stage, chunk_count, sum(len(p) for p in output_parts), had_error, is_awaiting,
    )

    # Save output to per-project message log
    full_output = "".join(output_parts)
    if full_output:
        try:
            from app.builds.messages import add_message
            add_message(
                db, build_project_id,
                role="assistant",
                content=full_output,
                stage=pipeline_stage,
                provider=provider,
                model=model,
            )
        except Exception as e:
            logger.warning("[stage_hooks] Failed to save pipeline message: %s", e)

    # Extract and save Weaver structured data (Brief Tab enrichment)
    if pipeline_stage == "weaver" and full_output and not had_error:
        try:
            extraction = _extract_weaver_intent(full_output)
            if extraction:
                from app.builds.pipeline_bridge import save_weaver_extraction
                save_weaver_extraction(db, build_project_id, extraction)
                print(f"[STAGE_HOOKS] Saved Weaver extraction: {list(extraction.keys())}")
        except Exception as we:
            logger.debug("[stage_hooks] Weaver extraction failed: %s", we)

    # Link spec if we got one
    if spec_id:
        try:
            from app.builds import service as build_service
            build_service.link_spec(db, build_project_id, spec_id)
        except Exception:
            pass

    # Notify stage completion. Hard-stop is checked first because it's a
    # subtype of failure with a specific reason worth surfacing to the UI.
    if hard_stopped:
        _detail = f"Hard stop: {hard_stop_reason}" if hard_stop_reason else "Hard stop"
        notify_stage_failed(db, build_project_id, pipeline_stage, detail=_detail)
    elif had_error:
        notify_stage_failed(db, build_project_id, pipeline_stage, detail="Stream error")
    elif is_awaiting:
        notify_stage_awaiting(db, build_project_id, pipeline_stage, detail="Awaiting user input")
    else:
        notify_stage_passed(db, build_project_id, pipeline_stage)

    # Emit final build project update event
    final_status = "failed" if (had_error or hard_stopped) else ("awaiting_input" if is_awaiting else "passed")
    yield _make_sse_event({
        "type": "build_project_update",
        "build_project_id": build_project_id,
        "stage": pipeline_stage,
        "status": final_status,
    })

    # ----------------------------------------------------------------
    # v2.1 Auto-advance: when a stage passes, trigger the next stage
    # automatically so the pipeline flows end-to-end without manual clicks.
    # ----------------------------------------------------------------
    if final_status == "passed" and pipeline_stage in _AUTO_ADVANCE_MAP:
        next_intent = _AUTO_ADVANCE_MAP[pipeline_stage]
        logger.info("[stage_hooks] Auto-advance: %s passed → triggering %s", pipeline_stage, next_intent)
        print(f"[STAGE_HOOKS] Auto-advance: {pipeline_stage} → {next_intent}")
        yield _make_sse_event({
            "type": "auto_advance",
            "from_stage": pipeline_stage,
            "next_intent": next_intent,
            "build_project_id": build_project_id,
            "chat_project_id": chat_project_id,
        })

    # v2.2 (2026-04-18): Refresh the dispatch-dedup timestamp to NOW so the
    # cooldown window starts from stage completion, not stage entry. This
    # is what catches the common race: auto-advance fires immediately on
    # stage pass, and the user's follow-up voice/typed confirmation
    # arrives moments later — inside the cooldown window.
    _release_dispatch_slot(chat_project_id, dispatch_stage)
