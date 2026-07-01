# FILE: app/web_automation/coursera_reader.py
# Purpose: Coursera login health check + vision-first course-progress composite.
# Called-by: app.web_automation.register, app.debug.executors.web_automation, app.debug.web_tool_definitions
# Depends-on: app.web_automation.bridge, app.web_automation.tool_handlers, app.web_automation.action_queue
# Last-renovated: 2026-07-01
"""
Coursera login health check + vision-first course-progress composite.

Why this exists (2026-07-01 Coursera fix):
  * A logged-out/expired Coursera session silently serves the PUBLIC
    course page, which always looks like the user is starting from
    module one. Reading it as truth is worse than failing.
  * Module completion state (green ticks, progress bars) is rendered
    visually — extract_text / dom_snapshot return titles WITHOUT done
    state, so text reads produce confident wrong answers.

Two LLM tools live here:
  web_coursera_health   — cheap logged-in/logged-out verdict.
  web_coursera_progress — health check, then screenshot + vision model
                          answer for "where am I up to?".

Both return a `state` of ok / logged_out / desktop_offline / unknown /
error plus a ready-to-relay `message`, so the agent never has to invent
its own explanation (and never reports module one from a public page).
"""
from __future__ import annotations

import logging
from typing import Optional

from app.web_automation import bridge
from app.web_automation.action_queue import DESKTOP_OFFLINE_PREFIX
from app.web_automation.tool_handlers import (
    _resolve_session,
    _dispatch,
    vision_check_handler,
)

logger = logging.getLogger(__name__)

COURSERA_PLATFORM = "coursera"

# The logged-in "My Learning" dashboard. Logged out, Coursera 302s this
# to the public coursera.org home — that bounce is itself the strongest
# logged-out signal we have. Keep in sync with seed.py's coursera row.
MY_LEARNING_URL = "https://www.coursera.org/my-learning"

# DOM text markers (case-insensitive substring match on dom_snapshot
# element text). Logged-out markers only ever render for anonymous
# visitors; learner markers only for authenticated ones.
LOGGED_OUT_MARKERS = ("log in", "join for free", "forgot password", "welcome back")
LOGGED_IN_MARKERS = ("my learning", "continue learning", "recently viewed")

LOGIN_NEEDED_MESSAGE = (
    "Coursera needs a fresh login on the desktop — the saved browser session "
    "has expired, so Coursera is showing its public site (which always looks "
    "like the course is starting from module one). No course progress was "
    "read. Open the ASTRA desktop Browser tab, log in to Coursera, then ask "
    "again."
)

DESKTOP_OFFLINE_MESSAGE = (
    "The desktop browser is offline — the ASTRA desktop app isn't running, "
    "so Coursera can't be opened at all. Start the ASTRA desktop app, then "
    "ask again."
)

DEFAULT_PROGRESS_QUESTION = (
    "This should be a logged-in Coursera page. Where is the user up to? "
    "Report exactly what is shown: course name(s), overall progress "
    "(percentage, or modules/weeks completed vs total — count green ticks "
    "or filled progress bars), and the next lesson or module to do. If the "
    "page is a login page or the public marketing site (Join for Free "
    "buttons, no personal progress anywhere), reply with exactly: "
    "LOGGED_OUT. Do not guess progress that is not visibly shown."
)


def _offline(error: str) -> bool:
    return (error or "").startswith(DESKTOP_OFFLINE_PREFIX)


def _result_dict(
    state: str,
    *,
    message: str = "",
    answer: str = "",
    current_url: str = "",
    screenshot_path: str = "",
    error: str = "",
) -> dict:
    return {
        "ok": state == "ok",
        "state": state,
        "logged_in": state == "ok",
        "answer": answer,
        "message": message,
        "current_url": current_url,
        "screenshot_path": screenshot_path,
        "error": error,
    }


async def _login_state(ref: str) -> dict:
    """Open the session, land on My Learning, classify login state.

    Returns a _result_dict with state:
      ok              — logged in, sitting on the My Learning dashboard
      logged_out      — bounced to the public site / login markers found
      desktop_offline — Electron never picked the work up
      unknown         — page didn't classify either way (caller decides)
      error           — a step failed outright
    """
    sess = _resolve_session(ref)
    if not sess:
        return _result_dict("error", error=f"no session matches '{ref}'")

    opened = await bridge.ensure_session_open(sess.id)
    if not opened.get("ok"):
        err = opened.get("error") or "open failed"
        if _offline(err):
            return _result_dict(
                "desktop_offline", message=DESKTOP_OFFLINE_MESSAGE, error=err
            )
        return _result_dict("error", error=f"could not open session: {err}")

    # Always (re)navigate to My Learning: ensure_view only loads the
    # landing URL for a brand-new view, and an old view may be parked on
    # a lecture page — or worse, on the public home after a logout.
    nav = await _dispatch(ref, "navigate", {"url": MY_LEARNING_URL}, timeout_seconds=25.0)
    if not nav.get("ok"):
        err = nav.get("error") or "navigate failed"
        if _offline(err):
            return _result_dict(
                "desktop_offline", message=DESKTOP_OFFLINE_MESSAGE, error=err
            )
        return _result_dict("error", error=f"could not reach My Learning: {err}")

    current_url = ((nav.get("result") or {}).get("current_url") or "").lower()

    # Give the SPA a moment to hydrate before reading the DOM — course
    # cards and the learner nav render after the document load event.
    await _dispatch(ref, "wait", {"ms": 1500}, timeout_seconds=10.0)

    snap = await _dispatch(ref, "dom_snapshot", {}, timeout_seconds=15.0)
    texts = ""
    if snap.get("ok"):
        elements = (snap.get("result") or {}).get("elements") or []
        texts = " | ".join(
            str(el.get("text") or "") for el in elements if el.get("text")
        ).lower()

    bounced = "my-learning" not in current_url
    saw_logged_out = any(m in texts for m in LOGGED_OUT_MARKERS)
    saw_logged_in = any(m in texts for m in LOGGED_IN_MARKERS)

    if saw_logged_out or (bounced and not saw_logged_in):
        return _result_dict(
            "logged_out", message=LOGIN_NEEDED_MESSAGE, current_url=current_url
        )
    if saw_logged_in:
        return _result_dict(
            "ok",
            message="Logged in — My Learning dashboard is showing.",
            current_url=current_url,
        )
    # DOM said nothing useful (snapshot failed / empty page). Don't
    # declare either way — the caller (or vision) arbitrates.
    return _result_dict(
        "unknown",
        message=(
            "Could not confirm login state from the page structure. Verify "
            "visually (web_coursera_progress / web_vision_check) before "
            "trusting any course content."
        ),
        current_url=current_url,
    )


# ── LLM tool handlers ────────────────────────────────────────────────

async def coursera_health_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Cheap login health check for the Coursera session (no vision call)."""
    ref = str(input_data.get("session") or COURSERA_PLATFORM)
    try:
        return await _login_state(ref)
    except Exception as e:  # never let a health check take the tool loop down
        logger.exception("[coursera_reader] health check raised")
        return _result_dict("error", error=f"health check failed: {e}")


async def coursera_progress_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Health check, then screenshot + vision read of course progress."""
    ref = str(input_data.get("session") or COURSERA_PLATFORM)
    question = str(input_data.get("question") or "").strip() or DEFAULT_PROGRESS_QUESTION

    try:
        health = await _login_state(ref)
    except Exception as e:
        logger.exception("[coursera_reader] progress health step raised")
        return _result_dict("error", error=f"health check failed: {e}")

    if health["state"] in ("logged_out", "desktop_offline", "error"):
        return health

    # state is ok or unknown → let the vision model read the live page.
    # The default question instructs it to answer LOGGED_OUT if it sees
    # the public/marketing site, which covers the 'unknown' residue.
    vision = await vision_check_handler({"session": ref, "question": question}, context)
    if not vision.get("ok"):
        err = vision.get("error") or "vision check failed"
        if _offline(err):
            return _result_dict(
                "desktop_offline", message=DESKTOP_OFFLINE_MESSAGE, error=err
            )
        return _result_dict(
            "error", current_url=health.get("current_url", ""), error=err
        )

    answer = (vision.get("answer") or "").strip()
    if "logged_out" in answer.lower().replace(" ", "_"):
        return _result_dict(
            "logged_out",
            message=LOGIN_NEEDED_MESSAGE,
            current_url=health.get("current_url", ""),
            screenshot_path=vision.get("screenshot_path", ""),
        )

    return _result_dict(
        "ok",
        answer=answer,
        message="Read from the logged-in Coursera dashboard.",
        current_url=health.get("current_url", ""),
        screenshot_path=vision.get("screenshot_path", ""),
    )


# ── Registry wiring (merged in by register.py / tool layers) ─────────

_SESSION_FIELD = {
    "type": "string",
    "description": "Session UUID or platform key. Defaults to 'coursera'.",
    "minLength": 1,
    "maxLength": 128,
}

_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["ok", "state"],
    "properties": {
        "ok":              {"type": "boolean"},
        "state":           {"type": "string",
                            "enum": ["ok", "logged_out", "desktop_offline", "unknown", "error"]},
        "logged_in":       {"type": "boolean"},
        "answer":          {"type": "string"},
        "message":         {"type": "string"},
        "current_url":     {"type": "string"},
        "screenshot_path": {"type": "string"},
        "error":           {"type": "string"},
    },
}

COURSERA_SCHEMAS = {
    "web_coursera_health": {
        "input": {
            "type": "object",
            "properties": {"session": _SESSION_FIELD},
        },
        "output": _OUTPUT_SCHEMA,
    },
    "web_coursera_progress": {
        "input": {
            "type": "object",
            "properties": {
                "session": _SESSION_FIELD,
                "question": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 2000,
                    "description": "Optional override for the vision question. "
                                   "Default asks where the user is up to.",
                },
            },
        },
        "output": _OUTPUT_SCHEMA,
    },
}

COURSERA_DESCRIPTIONS = {
    "web_coursera_health":
        "Check whether the Coursera browser session is actually logged in. Opens the "
        "session, lands on the My Learning dashboard and classifies the result: "
        "state='ok' (logged in), 'logged_out' (expired session — relay the message, "
        "do NOT read the public page as course state), 'desktop_offline' (ASTRA "
        "desktop app not running), or 'unknown'. Run this before reading any "
        "Coursera content with extract_text/dom_snapshot.",
    "web_coursera_progress":
        "Answer 'where am I up to on my Coursera course?' the RIGHT way: verifies "
        "login, navigates to the logged-in My Learning dashboard, screenshots it and "
        "asks the vision model to read progress. Use this INSTEAD of web_extract_text "
        "or web_dom_snapshot for any Coursera progress/position question — module "
        "completion ticks are visual and invisible to text extraction. Returns "
        "state + answer, or an actionable message when Coursera needs login "
        "(state='logged_out') or the desktop app is closed (state='desktop_offline'). "
        "Relay `message` to the user verbatim in those cases.",
}

COURSERA_HANDLERS = {
    "web_coursera_health":   coursera_health_handler,
    "web_coursera_progress": coursera_progress_handler,
}


# ── Chat-facing tool defs (imported by app.debug.web_tool_definitions) ─
# The chat loop uses a "parameters" shape, not the registry's
# input/output shape. Kept here so the whole Coursera surface lives in
# this one file and web_tool_definitions.py stays under its size cap.

_CHAT_SESSION_PARAM = {
    "type": "string",
    "description": "Session platform key or UUID. Omit for the default 'coursera'.",
}

WEB_COURSERA_HEALTH_TOOL = {
    "name": "web_coursera_health",
    "description": COURSERA_DESCRIPTIONS["web_coursera_health"],
    "parameters": {
        "type": "object",
        "properties": {"session": _CHAT_SESSION_PARAM},
        "required": [],
    },
}

WEB_COURSERA_PROGRESS_TOOL = {
    "name": "web_coursera_progress",
    "description": COURSERA_DESCRIPTIONS["web_coursera_progress"],
    "parameters": {
        "type": "object",
        "properties": {
            "session": _CHAT_SESSION_PARAM,
            "question": {
                "type": "string",
                "description": (
                    "Optional override for the vision question, e.g. to focus "
                    "on one named course. Default asks where the user is up to."
                ),
            },
        },
        "required": [],
    },
}
