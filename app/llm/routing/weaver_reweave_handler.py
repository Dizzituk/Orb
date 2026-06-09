# FILE: app/llm/routing/weaver_reweave_handler.py
"""
Weaver auto-reweave routing handler.

Extracted from stream_router.py (2026-05-24) as part of breaking down
the monolithic stream_chat function. Single responsibility: detect when
the user is mid-weaver-flow and route their reply back into Weaver
UPDATE mode, unless they've issued an explicit exit command.

v5.2 behaviour (from original location, unchanged):

    Weaver finishes → flow enters AWAITING_SPEC_GATE_CONFIRM.
    If the user replies with anything that isn't an explicit command
    ('send to spec gate', 'run critical pipeline', etc.), assume they're
    adding requirements or answering questions, auto-route back to
    Weaver UPDATE. Creates a natural loop until the user says they're
    ready to ship the spec.

This module is a pure handler — it takes the request and trace context,
returns either a StreamingResponse to short-circuit further routing, or
None to let stream_router.py fall through to normal routing.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)


def handle_weaver_design_questions(
    req: Any,
    db: Any,
    trace: Any,
    stage_trace: Any,
    translation_result: Any = None,
) -> Optional[StreamingResponse]:
    """Handle auto-reweave: route user replies back to Weaver UPDATE.

    Returns:
        StreamingResponse if the turn was intercepted and routed to Weaver.
        None if no interception is appropriate — caller should fall through.
    """
    # Lazy imports to avoid circular dependency with stream_router and to
    # keep this module light on cold start.
    from app.llm.routing.handler_registry import (
        _FLOW_STATE_AVAILABLE,
        _WEAVER_AVAILABLE,
        get_active_flow,
        SpecFlowStage,
        generate_weaver_stream,
    )
    from app.llm._stream_router_utils import (
        _EXPLICIT_COMMAND_INTENTS,
        _WEAVER_EXIT_INTENTS,
    )
    from app.llm.translation_routing import _get_weaver_config
    from app.llm.routing.command_dispatch import log_routing_failure

    if not _FLOW_STATE_AVAILABLE or not get_active_flow:
        return None

    active_flow = get_active_flow(req.project_id)
    if not active_flow:
        return None

    # Only intercept in weaver-active stages
    if active_flow.stage not in (
        SpecFlowStage.WEAVER_DESIGN_QUESTIONS,
        SpecFlowStage.AWAITING_SPEC_GATE_CONFIRM,
    ):
        return None

    # If translation resolved to a weaver-exit intent, don't intercept —
    # the user wants to leave the weaver flow (proceed to spec gate, etc.)
    if translation_result is not None:
        intent = translation_result.resolved_intent
        if intent and intent in _WEAVER_EXIT_INTENTS:
            logger.info(
                "[weaver_reweave] User issued exit intent '%s' — leaving weaver flow",
                intent.value,
            )
            print(f"[WEAVER_REWEAVE] Exit intent '{intent.value}' — NOT auto-reweaving")
            return None
        # Same for any explicit command intent (architecture, sandbox, etc.)
        if intent and intent in _EXPLICIT_COMMAND_INTENTS:
            logger.info(
                "[weaver_reweave] Explicit command '%s' — bypassing auto-reweave",
                intent.value,
            )
            print(f"[WEAVER_REWEAVE] Explicit command '{intent.value}' — NOT auto-reweaving")
            return None

    # Auto-route to Weaver UPDATE
    if not _WEAVER_AVAILABLE:
        log_routing_failure(
            stage_trace,
            "Weaver handler not available for auto-reweave routing",
            "generate_weaver_stream",
            "falling through to normal routing",
        )
        return None

    weaver_provider, weaver_model = _get_weaver_config()
    if stage_trace:
        stage_trace.enter_stage(
            "weaver_auto_reweave",
            provider=weaver_provider,
            model=weaver_model,
        )

    logger.info(
        "[weaver_reweave] Auto-routing to Weaver UPDATE (flow stage: %s)",
        active_flow.stage.value,
    )
    print("[WEAVER_REWEAVE] Auto-routing to Weaver UPDATE (user replied in active weaver flow)")

    # v4.1.0: Pass req.message as pending_user_message to fix race condition.
    # The user's reply may not be in the DB yet when Weaver reads messages,
    # so hash-based dedup would otherwise see "nothing new". Passing the
    # message directly ensures the reply is always visible to the weaver
    # regardless of persistence timing.
    generator = generate_weaver_stream(
        project_id=req.project_id,
        message=req.message,
        db=db,
        trace=trace,
        conversation_id=str(req.project_id),
        is_continuation=True,
        captured_answers=None,
        pending_user_message=req.message,
    )

    # v9.0: Wrap re-weave in build tracking so the builds tab updates.
    # Previously the re-weave bypassed wrap_with_build_tracking, leaving
    # the build project with stale first-weave output.
    try:
        from app.builds.stage_hooks import is_tracked_stage, wrap_with_build_tracking
        if is_tracked_stage("weaver"):
            generator = wrap_with_build_tracking(
                stream=generator,
                db=db,
                chat_project_id=req.project_id,
                dispatch_stage="weaver",
                message=req.message,
                provider=weaver_provider,
                model=weaver_model,
            )
    except ImportError:
        # Build tracking not available — stream still works without it.
        pass

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
