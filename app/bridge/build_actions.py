# FILE: app/bridge/build_actions.py
# Purpose: Voice-triggered Android build command for the Astra Bridge — detect a
#          build request, confirm by voice, run the existing build flow async, and
#          deliver the result via the missed-replies poller.
# Called-by: app.bridge.router (bridge_chat, bridge_chat_and_speak)
# Depends-on: app.cloud.build_and_deploy, app.pipeline_v2.target_registry,
#             app.llm.routing.chat_intent_detection, app.llm.routing.confirmation_gate
# Last-renovated: 2026-06-14
"""
Bridge build actions.

The Bridge is a thin transport over the same backend brain. When the user says
"build the APK" from the phone, that must do exactly what the desktop does: run
the proven build_and_deploy flow on the host. The phone cannot block on a
multi-minute Gradle build, so the contract here is:

  1. Acknowledge instantly with a spoken line.
  2. Run the build asynchronously (build_and_deploy -> Gradle -> APK -> cloud).
  3. Write the outcome as an assistant message on the project so the existing
     /bridge/missed-replies poller delivers and speaks it (survives dead zones).

Confirmation is voice-friendly: the first time, ASTRA asks "Shall I build...?"
and accepts the next "yes". The shared confirmation_gate learns after 5 approvals
with 0 rejections and then stops asking — the same JSONL history the desktop
writes, so the learning is shared across both surfaces.

Release builds are gated honestly: the Bridge's app/build.gradle.kts has no
signingConfig and the reused executor only runs assembleDebug, so we say so out
loud and offer a debug build rather than produce something that can't be
installed. We never invent signing keys.

v1.0 (2026-06-14): Phase 1 of the Bridge capability-parity job.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Default build target for a phone-originated build request that names no
# project. On the Bridge, "build the apk" overwhelmingly means the Bridge app.
_DEFAULT_TARGET_ID = "astra-bridge"

# Pending voice confirmations, keyed by project_id. A build request that isn't
# auto-approved yet parks here until the user's next "yes"/"no". Short-lived.
_PENDING: dict[int, dict] = {}
_PENDING_TTL_SEC = 600  # 10 minutes

# Strong refs to in-flight background build tasks so the event loop doesn't GC
# them mid-build (asyncio holds only weak refs to tasks).
_TASKS: set = set()

_AFFIRMATIVE_RE = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|sure|ok|okay|go(?:\s+ahead)?|do\s+it|"
    r"please\s+do|go\s+for\s+it|build\s+it|confirm(?:ed)?|affirmative|"
    r"sounds?\s+good|aye)\b",
    re.IGNORECASE,
)
_NEGATIVE_RE = re.compile(
    r"^\s*(?:no|nope|nah|don'?t|do\s+not|cancel|stop|never\s*mind|"
    r"negative|forget\s+it|not\s+now)\b",
    re.IGNORECASE,
)
_RELEASE_RE = re.compile(r"\brelease\b", re.IGNORECASE)


@dataclass
class BuildTurnResult:
    """The Bridge's spoken/text reply for a build-related turn."""
    reply: str
    model: str = "build-gate"


def is_build_request(message: str) -> bool:
    """True if the message is an imperative APK build request."""
    try:
        from app.llm.routing.chat_intent_detection import detect_build_deploy_intent
        return detect_build_deploy_intent(message or "")
    except Exception as e:
        logger.debug("[bridge.build] intent detect failed: %s", e)
        return False


def _is_affirmative(message: str) -> bool:
    return bool(_AFFIRMATIVE_RE.match((message or "").strip()))


def _is_negative(message: str) -> bool:
    return bool(_NEGATIVE_RE.match((message or "").strip()))


def _wants_release(message: str) -> bool:
    return bool(_RELEASE_RE.search(message or ""))


def _resolve_target(message: str) -> tuple[str, str]:
    """Resolve (target_id, target_name) from the message, default to the Bridge."""
    try:
        from app.pipeline_v2.target_registry import (
            resolve_project_from_message, get_profile,
        )
        profile = resolve_project_from_message(message or "")
        if profile is not None:
            return profile.project_id, profile.project_name
        default = get_profile(_DEFAULT_TARGET_ID)
        if default is not None:
            return default.project_id, default.project_name
    except Exception as e:
        logger.debug("[bridge.build] target resolve failed: %s", e)
    return _DEFAULT_TARGET_ID, "Astra Bridge"


def _signing_configured(target_id: str) -> bool:
    """Best-effort check for a release signingConfig in the target's app gradle."""
    try:
        from app.pipeline_v2.target_registry import get_profile
        profile = get_profile(target_id)
        if profile is None:
            return False
        gradle = os.path.join(
            profile.project_root.replace("/", os.sep), "app", "build.gradle.kts"
        )
        if not os.path.isfile(gradle):
            return False
        with open(gradle, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        return "signingConfig" in text
    except Exception as e:
        logger.debug("[bridge.build] signing check failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Confirmation gate (shared JSONL history with the desktop)
# ---------------------------------------------------------------------------

def _gate_pattern_key() -> str:
    from app.llm.routing.confirmation_gate import _make_pattern_key
    # message is ignored for intent_routing keys, so "" is fine and keeps the
    # key identical to the desktop's intent:BUILD_AND_DEPLOY (shared learning).
    return _make_pattern_key("intent_routing", "BUILD_AND_DEPLOY", "")


def _should_auto_approve() -> bool:
    try:
        from app.llm.routing.confirmation_gate import _load_pattern_history
        return _load_pattern_history(_gate_pattern_key()).should_auto_approve
    except Exception as e:
        logger.debug("[bridge.build] gate history load failed: %s", e)
        return False


def _log_decision(approved: bool, message: str) -> None:
    try:
        from app.llm.routing.confirmation_gate import process_confirmation_response
        process_confirmation_response(
            pattern_key=_gate_pattern_key(),
            gate_type="intent_routing",
            proposed_action="BUILD_AND_DEPLOY",
            approved=approved,
            original_message=message,
            confidence=0.8,
        )
    except Exception as e:
        logger.debug("[bridge.build] decision log failed: %s", e)


# ---------------------------------------------------------------------------
# Async build + completion delivery
# ---------------------------------------------------------------------------

def _prune_pending() -> None:
    now = time.time()
    stale = [
        pid for pid, p in _PENDING.items()
        if now - p.get("ts", 0) > _PENDING_TTL_SEC
    ]
    for pid in stale:
        _PENDING.pop(pid, None)


def _start_build(message: str, project_id: int, target_id: str, target_name: str) -> None:
    """Schedule the async build + completion-report task on the event loop."""
    task = asyncio.create_task(
        _run_and_report(message, project_id, target_id, target_name)
    )
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)


async def _run_and_report(
    message: str, project_id: int, target_id: str, target_name: str,
) -> None:
    """Run build_and_deploy and write the outcome to the project for delivery."""
    logger.info(
        "[bridge.build] Build starting: target=%s project=%s", target_id, project_id,
    )
    try:
        from app.cloud.build_and_deploy import build_and_deploy
        result = await build_and_deploy(text=message, target_id=target_id)
    except Exception as e:
        logger.error("[bridge.build] Build crashed: %s", e)
        result = {"success": False, "error": str(e)}

    reply = _format_completion(target_name, result)
    _write_assistant_message(project_id, reply)


def _format_completion(target_name: str, result: dict) -> str:
    """Build the spoken/text completion line from build_and_deploy's result dict."""
    if not result:
        return f"The {target_name} build finished but returned no result."
    if not result.get("success"):
        err = result.get("error") or "an unknown error"
        return f"The {target_name} build failed: {err}"
    # Success — build_and_deploy already crafts the right wording (Drive link,
    # local path, or partial upload-failure note). Prefer its message verbatim
    # so the phone hears the APK location / cloud link, not a bracketed marker.
    msg = result.get("message")
    if msg:
        return msg
    apk_kb = (result.get("build") or {}).get("apk_size", 0) // 1024
    return f"Built the {target_name} APK ({apk_kb}KB)."


def _write_assistant_message(project_id: int, content: str) -> None:
    """Persist the completion as an assistant message (own session, request is gone)."""
    from app.db import get_db_session
    from app.memory.service import create_message
    from app.memory.schemas import MessageCreate
    db = get_db_session()
    try:
        create_message(db, MessageCreate(
            project_id=project_id, role="assistant",
            content=content, provider="bridge", model="build-complete",
        ))
        logger.info("[bridge.build] Completion delivered to project %s", project_id)
    except Exception as e:
        logger.error("[bridge.build] Failed to write completion message: %s", e)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def maybe_handle_build_turn(
    message: str, project_id: int, db=None,
) -> Optional[BuildTurnResult]:
    """Intercept build-command turns on the Bridge.

    Returns a BuildTurnResult (spoken ack / question / honest gate) when this
    turn is build-related, or None to let normal chat handle it. The caller
    persists the reply as the assistant message and returns it to the phone.

    db is accepted for signature symmetry with the rest of the bridge layer;
    the build write happens on its own session in the background task.
    """
    _prune_pending()
    text = (message or "").strip()

    # 1. Resolve an outstanding voice confirmation first.
    pending = _PENDING.get(project_id)
    if pending:
        if _is_affirmative(text):
            _PENDING.pop(project_id, None)
            _log_decision(True, pending.get("message", text))
            tid = pending["target_id"]
            tname = pending["target_name"]
            _start_build(pending.get("message", text), project_id, tid, tname)
            return BuildTurnResult(
                reply=(
                    f"Starting the debug build of the {tname} now — "
                    f"I'll let you know when it's ready."
                ),
                model="build-deploy",
            )
        if _is_negative(text):
            _PENDING.pop(project_id, None)
            _log_decision(False, pending.get("message", text))
            return BuildTurnResult(reply="Okay, I won't build it.", model="build-gate")
        # Neither yes nor no — abandon the stale confirmation and fall through
        # so this message is handled on its own merits.
        _PENDING.pop(project_id, None)

    # 2. Fresh build request?
    if not is_build_request(text):
        return None

    target_id, target_name = _resolve_target(text)

    # 3. Release needs a signing key — be honest, offer debug. The reused
    #    executor (build_and_deploy) only runs assembleDebug, so we never
    #    silently produce an unsigned/release artifact.
    if _wants_release(text):
        if _signing_configured(target_id):
            reason = (
                "release signing is configured, but release builds aren't wired "
                "through the Bridge yet"
            )
        else:
            reason = (
                "a release build needs a signing key set up first, and that "
                "isn't configured yet"
            )
        _PENDING[project_id] = {
            "target_id": target_id, "target_name": target_name,
            "message": text, "ts": time.time(),
        }
        return BuildTurnResult(
            reply=(
                f"I can build a debug {target_name} APK right now, but {reason}. "
                f"Shall I build the debug APK instead? Say yes to go ahead."
            ),
            model="build-gate",
        )

    # 4. Auto-approve if the gate has learned to trust this, else ask once.
    if _should_auto_approve():
        _log_decision(True, text)
        _start_build(text, project_id, target_id, target_name)
        return BuildTurnResult(
            reply=(
                f"Starting the debug build of the {target_name} now — "
                f"I'll let you know when it's ready."
            ),
            model="build-deploy",
        )

    _PENDING[project_id] = {
        "target_id": target_id, "target_name": target_name,
        "message": text, "ts": time.time(),
    }
    return BuildTurnResult(
        reply=f"Shall I build and upload the {target_name} APK? Say yes to go ahead.",
        model="build-gate",
    )
