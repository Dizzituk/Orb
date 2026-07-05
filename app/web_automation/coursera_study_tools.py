# FILE: app/web_automation/coursera_study_tools.py
# Purpose: LLM tool surface for the hands-free Coursera study loop (resume / read lesson / next item).
# Called-by: app.web_automation.register, app.debug.web_tool_definitions, app.debug.executors.web_automation
# Depends-on: app.content.distribution.posting_drivers.coursera_driver, app.web_automation.study_state, app.web_automation.tool_handlers
# Last-renovated: 2026-07-03
"""
Chat + registry surface for the Coursera study driver (Amendment A
Job 10). Mirrors coursera_reader.py's pattern: schemas, descriptions,
handlers AND chat-facing defs all live next to each other here, merged
in by register.py / web_tool_definitions.py, so neither of those files
grows.

One tool call per user intent (small-local-model friendly):
  coursera_resume      — "resume my course"
  coursera_read_lesson — "read me the lesson"
  coursera_next_item   — "next lesson"

All three relay coursera_reader's logged_out / desktop_offline messages
verbatim and announce-but-never-attempt quizzes.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.content.distribution.posting_drivers import coursera_driver
from app.web_automation.study_state import mark_study_activity

logger = logging.getLogger(__name__)


def _mark_from(result: dict) -> dict:
    """Refresh the mid-study marker with whatever the tool learned.
    Marked at tool entry too — even a failed resume means the user is
    trying to study, so bare 'continue' keeps meaning the course."""
    if isinstance(result, dict) and result.get("ok"):
        mark_study_activity(
            course=str(result.get("course") or ""),
            item_title=str(result.get("item_title") or ""),
            item_type=str(result.get("item_type") or ""),
        )
    return result


class _UnknownSession(Exception):
    pass


def _session_kw(input_data: dict) -> dict:
    """Optional session override -> driver kwargs. Default: the driver
    resolves the 'coursera' platform session itself. Raises
    _UnknownSession for a ref that matches nothing."""
    ref = str(input_data.get("session") or "").strip()
    if not ref or ref == coursera_driver.PLATFORM:
        return {}
    from app.web_automation.tool_handlers import _resolve_session
    sess = _resolve_session(ref)
    if not sess:
        raise _UnknownSession(f"no session matches '{ref}'")
    return {"session_id": sess.id}


async def coursera_resume_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Dashboard (health-gated) -> current lesson item."""
    try:
        mark_study_activity()
        course = str(input_data.get("course") or "").strip() or None
        return _mark_from(
            await coursera_driver.resume_course(course=course, **_session_kw(input_data))
        )
    except _UnknownSession as e:
        return {"ok": False, "state": "error", "error": str(e)}
    except Exception as e:  # never take the tool loop down
        logger.exception("[coursera_study] resume raised")
        return {"ok": False, "state": "error", "error": f"resume failed: {e}"}


async def coursera_read_lesson_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Full transcript / article body of the current lesson, plain DOM text."""
    try:
        mark_study_activity()
        return _mark_from(
            await coursera_driver.read_transcript(**_session_kw(input_data))
        )
    except _UnknownSession as e:
        return {"ok": False, "state": "error", "error": str(e)}
    except Exception as e:
        logger.exception("[coursera_study] read_lesson raised")
        return {"ok": False, "state": "error", "error": f"read_lesson failed: {e}"}


async def coursera_next_item_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Advance to the next lesson item; quizzes announced, never attempted."""
    try:
        mark_study_activity()
        return _mark_from(
            await coursera_driver.next_item(**_session_kw(input_data))
        )
    except _UnknownSession as e:
        return {"ok": False, "state": "error", "error": str(e)}
    except Exception as e:
        logger.exception("[coursera_study] next_item raised")
        return {"ok": False, "state": "error", "error": f"next_item failed: {e}"}


# ── Registry wiring (merged in by register.py) ───────────────────────

_SESSION_FIELD = {
    "type": "string",
    "description": "Session UUID or platform key. Defaults to 'coursera'.",
    "minLength": 1,
    "maxLength": 128,
}

_STUDY_OUTPUT = {
    "type": "object",
    "required": ["ok", "state"],
    "properties": {
        "ok":              {"type": "boolean"},
        "state":           {"type": "string",
                            "enum": ["ok", "logged_out", "desktop_offline", "unknown", "error"]},
        "course":          {"type": "string"},
        "item_title":      {"type": "string"},
        "item_type":       {"type": "string",
                            "description": "video | reading | quiz | assignment | activity | course_home"},
        "transcript_text": {"type": "string"},
        "truncated":       {"type": "boolean"},
        "current_url":     {"type": "string"},
        "message":         {"type": "string"},
        "error":           {"type": "string"},
    },
}

COURSERA_STUDY_SCHEMAS = {
    "coursera_resume": {
        "input": {
            "type": "object",
            "properties": {
                "session": _SESSION_FIELD,
                "course": {
                    "type": "string",
                    "description": "Optional course name to open (matched against My Learning cards). Omit for the most recent course.",
                },
            },
        },
        "output": _STUDY_OUTPUT,
    },
    "coursera_read_lesson": {
        "input": {
            "type": "object",
            "properties": {"session": _SESSION_FIELD},
        },
        "output": _STUDY_OUTPUT,
    },
    "coursera_next_item": {
        "input": {
            "type": "object",
            "properties": {"session": _SESSION_FIELD},
        },
        "output": _STUDY_OUTPUT,
    },
}

COURSERA_STUDY_DESCRIPTIONS = {
    "coursera_resume":
        "Resume the user's Coursera course hands-free (desktop can be unattended). "
        "Verifies login on the My Learning dashboard first, clicks Resume / Go To "
        "Course (optionally a named `course`), hops through the course home if "
        "needed, and lands on the current lesson item. Returns course, item_title "
        "and item_type. Use for 'resume my course' / 'open my course' / 'where "
        "were we'. If state is 'logged_out' or 'desktop_offline', relay `message` "
        "to the user VERBATIM. If item_type is 'quiz' or 'assignment' it has "
        "stopped on the landing page and will NOT attempt it. One call does the "
        "whole job — no manual navigate/click sequence needed.",
    "coursera_read_lesson":
        "Read the current Coursera lesson so you can discuss it or read it aloud: "
        "returns the FULL video transcript (plain DOM text — no vision, no "
        "screenshots) or the article body for reading items, plus item_title. If "
        "the browser isn't on a lesson it resumes the course first. Use for 'read "
        "me the lesson' / 'what does this lesson say' / 'let's study'. Relay "
        "`message` verbatim on logged_out / desktop_offline. Do NOT drive "
        "web_extract_text or web_vision_check for lesson text — this tool is the "
        "whole job.",
    "coursera_next_item":
        "Advance to the next Coursera lesson item and report its title and type. "
        "If the next item is a quiz or assignment it stops on the landing page, "
        "announces it in `message`, and never attempts answers — quizzes are the "
        "user's. Use for 'next lesson' / 'move on' / 'continue the course'. Relay "
        "`message` verbatim on logged_out / desktop_offline.",
}

COURSERA_STUDY_HANDLERS = {
    "coursera_resume":      coursera_resume_handler,
    "coursera_read_lesson": coursera_read_lesson_handler,
    "coursera_next_item":   coursera_next_item_handler,
}


# ── Chat-facing tool defs (imported by app.debug.web_tool_definitions) ─

_CHAT_SESSION_PARAM = {
    "type": "string",
    "description": "Session platform key or UUID. Omit for the default 'coursera'.",
}

COURSERA_RESUME_TOOL = {
    "name": "coursera_resume",
    "description": COURSERA_STUDY_DESCRIPTIONS["coursera_resume"],
    "parameters": {
        "type": "object",
        "properties": {
            "session": _CHAT_SESSION_PARAM,
            "course": {
                "type": "string",
                "description": (
                    "Optional course name, e.g. 'the economics course'. Omit "
                    "to resume the most recent course."
                ),
            },
        },
        "required": [],
    },
}

COURSERA_READ_LESSON_TOOL = {
    "name": "coursera_read_lesson",
    "description": COURSERA_STUDY_DESCRIPTIONS["coursera_read_lesson"],
    "parameters": {
        "type": "object",
        "properties": {"session": _CHAT_SESSION_PARAM},
        "required": [],
    },
}

COURSERA_NEXT_ITEM_TOOL = {
    "name": "coursera_next_item",
    "description": COURSERA_STUDY_DESCRIPTIONS["coursera_next_item"],
    "parameters": {
        "type": "object",
        "properties": {"session": _CHAT_SESSION_PARAM},
        "required": [],
    },
}
