# FILE: app/web_automation/coursera_reader.py
# Purpose: Coursera login health check + vision-first course-progress composite.
# Called-by: app.web_automation.register, app.debug.executors.web_automation, app.debug.web_tool_definitions
# Depends-on: app.web_automation.bridge, app.web_automation.tool_handlers, app.web_automation.action_queue
# Last-renovated: 2026-07-03
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
from urllib.parse import urlsplit

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


def _on_my_learning(url: str) -> bool:
    """True only when the URL's PATH is the My Learning dashboard.

    Path-based on purpose: the logged-out login page carries
    `?redirectTo=%2Fmy-learning`, so a substring test on the full URL
    would call the login page 'on my-learning'.
    """
    try:
        return urlsplit(url or "").path.rstrip("/").lower().startswith("/my-learning")
    except (ValueError, TypeError):
        return False


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

    opened_r = opened.get("result") or {}
    current_url = (opened_r.get("current_url") or "").lower()
    view_was_fresh = bool(opened_r.get("view_was_fresh"))

    # Get to My Learning WITHOUT double-loading. A fresh view has just
    # loaded the landing URL — issuing another navigate to the same URL
    # races Coursera's own client-side redirect and dies with Chromium
    # ERR_ABORTED (-3), wedging the session's serial action queue (live
    # incident, 2026-07-01 23:51). So:
    #   already there + fresh view  → nothing to load
    #   already there + old view    → reload (freshens a stale page with
    #                                 no competing loadURL to race)
    #   somewhere else              → navigate as before
    if _on_my_learning(current_url):
        if not view_was_fresh:
            reload_r = await _dispatch(ref, "reload", {}, timeout_seconds=20.0)
            if not reload_r.get("ok") and _offline(reload_r.get("error") or ""):
                return _result_dict(
                    "desktop_offline", message=DESKTOP_OFFLINE_MESSAGE,
                    error=reload_r.get("error") or "",
                )
            # Any other reload failure: proceed and let the DOM markers /
            # vision arbitrate — a stale read is caught by them, and
            # 'unknown' is an honest answer.
    else:
        nav = await _dispatch(
            ref, "navigate", {"url": MY_LEARNING_URL}, timeout_seconds=25.0
        )
        if nav.get("ok"):
            current_url = ((nav.get("result") or {}).get("current_url") or "").lower()
        else:
            err = nav.get("error") or "navigate failed"
            if _offline(err):
                return _result_dict(
                    "desktop_offline", message=DESKTOP_OFFLINE_MESSAGE, error=err
                )
            # A failed navigate is NOT always a failed arrival: aborted
            # loads (redirect races, ERR_ABORTED) and slow reporting can
            # fail the action while the page lands fine. Ask the view
            # where it actually is before giving up.
            state = await _dispatch(ref, "current_state", {}, timeout_seconds=8.0)
            state_url = ((state.get("result") or {}).get("url") or "").lower()
            if state.get("ok") and state_url:
                current_url = state_url
                if not _on_my_learning(current_url):
                    # Genuinely somewhere else (could be the logged-out
                    # bounce to the public home) — fall through and let
                    # the classification below decide; it treats a
                    # non-my-learning URL as the logged-out signal.
                    pass
            else:
                if _offline(state.get("error") or ""):
                    return _result_dict(
                        "desktop_offline", message=DESKTOP_OFFLINE_MESSAGE,
                        error=state.get("error") or "",
                    )
                return _result_dict(
                    "error", error=f"could not reach My Learning: {err}"
                )

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

    bounced = not _on_my_learning(current_url)
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
    # DOM said nothing useful (snapshot failed / empty page / SPA
    # rendered differently). Fall back to screenshot + vision model
    # instead of returning "unknown" and pushing verification to the
    # caller (which creates verification loops — 2026-07-03 fix).
    logger.info("[coursera_reader] DOM markers inconclusive — falling back to visual check")
    try:
        vision_result = await vision_check_handler(
            {
                "session": ref,
                "question": (
                    "Is this a logged-in Coursera page? Look for personal "
                    "course cards, 'My Learning' heading, user avatar/profile "
                    "icon, or any personal course progress. If you see "
                    "'Join for Free', 'Log In', 'Sign Up' buttons, a public "
                    "marketing page, or a login form, reply LOGGED_OUT. "
                    "If you see personal course progress or a user dashboard, "
                    "reply LOGGED_IN."
                ),
            },
            None,
        )
        answer = (vision_result.get("answer") or "").strip().upper().replace(" ", "_")
        if "LOGGED_OUT" in answer:
            return _result_dict(
                "logged_out", message=LOGIN_NEEDED_MESSAGE, current_url=current_url
            )
        # Vision says logged in or inconclusive — no explicit logout
        # indicators found by either DOM or vision, so proceed.
        return _result_dict(
            "ok",
            message="Visual verification passed — no login issues detected.",
            current_url=current_url,
        )
    except Exception as e:
        logger.warning("[coursera_reader] Visual fallback failed: %s", e)
        # Even the visual check failed. No explicit logout markers were
        # found by DOM, so proceed optimistically rather than looping.
        return _result_dict(
            "ok",
            message=(
                "Login verification inconclusive (DOM and visual check both "
                "unclear), but no logged-out indicators found. Proceeding — "
                "downstream content reads will catch a real logout."
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


async def _dashboard_courses(ref: str) -> list:
    """Course titles readable from the My Learning DOM — no vision.

    Card links point at /learn/<slug> and their text is the course title.
    Order-preserving dedupe, capped at 6. Used as the degraded answer when
    the visual read is unavailable (e.g. the browser view is hidden, so
    capturePage returns an empty image while every DOM tool still works —
    live incident 2026-07-03 00:15).
    """
    snap = await _dispatch(ref, "dom_snapshot", {}, timeout_seconds=15.0)
    if not snap.get("ok"):
        return []
    titles: list = []
    for el in ((snap.get("result") or {}).get("elements")) or []:
        href = el.get("href") or ""
        text = " ".join(str(el.get("text") or "").split())
        if "/learn/" in href and len(text) >= 8 and text not in titles:
            titles.append(text)
        if len(titles) >= 6:
            break
    return titles


async def coursera_progress_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Health check, then screenshot + vision read of course progress.
    Falls back to a DOM-only course list when the visual read is
    unavailable (hidden view) — clearly labelled, ticks excluded."""
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
        # Degraded DOM answer (2026-07-03): a hidden browser view cannot be
        # screenshotted, but the dashboard DOM still names the courses.
        # Answer what the DOM knows and say plainly what it can't (ticks).
        titles = await _dashboard_courses(ref)
        if titles:
            return _result_dict(
                "ok",
                answer=(
                    "Visual progress read unavailable (" + err + "). "
                    "From the dashboard DOM, active course(s): "
                    + "; ".join(titles) + "."
                ),
                message=(
                    "DOM-only read — completion ticks and percentages need "
                    "the desktop Browser tab visible on screen; ask again "
                    "with it open for the full visual read."
                ),
                current_url=health.get("current_url", ""),
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
