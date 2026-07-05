# FILE: app/web_automation/study_state.py
# Purpose: In-memory "user is mid-study" marker + its [STUDY SESSION] chat context block.
# Called-by: app.web_automation.coursera_study_tools, app.llm.routing.prompt_builders
# Depends-on: stdlib only
# Last-renovated: 2026-07-03
"""
Study-session state (2026-07-03, follow-up to the Amendment A study loop).

When a Coursera study tool has run recently, bare continuation phrasings
("continue", "carry on", "what's next") mean THE COURSE — not the
conversation. This module holds that fact and renders it as a context
block, so the chat model treats course-continuation as a certainty
instead of an inference from tool descriptions.

In-memory by design: resets on restart (a fresh boot is not mid-study),
no DB, stdlib-only so the prompt path stays import-light.
"""
from __future__ import annotations

import time
from typing import Optional

# 30 minutes: long enough to survive a lesson plus its discussion, short
# enough that last night's session doesn't hijack tomorrow's "continue".
STUDY_ACTIVE_WINDOW_S = 30 * 60

_state = {"ts": 0.0, "course": "", "item_title": "", "item_type": ""}


def mark_study_activity(
    course: str = "", item_title: str = "", item_type: str = ""
) -> None:
    """Record that a study tool ran just now.

    Called at tool ENTRY (so even a failed resume keeps the bias — the
    user is still trying to study) and again with details on success.
    Empty fields keep the last known values.
    """
    _state["ts"] = time.time()
    if course:
        _state["course"] = str(course)
    if item_title:
        _state["item_title"] = str(item_title)
    if item_type:
        _state["item_type"] = str(item_type)


def clear_study_activity() -> None:
    """Reset the marker (tests; a future explicit 'done studying')."""
    _state.update({"ts": 0.0, "course": "", "item_title": "", "item_type": ""})


def active_study_session() -> Optional[dict]:
    """State dict while inside the activity window, else None."""
    ts = float(_state["ts"] or 0.0)
    if ts <= 0:
        return None
    age = time.time() - ts
    if age > STUDY_ACTIVE_WINDOW_S:
        return None
    return {**_state, "age_s": int(max(0, age))}


def build_study_session_block() -> Optional[str]:
    """[STUDY SESSION] context block, or None when not mid-study.

    Injected by prompt_builders next to the other context blocks; kept
    short — it is an intent disambiguator, not a playbook.
    """
    s = active_study_session()
    if not s:
        return None
    where = s["item_title"] or "the current lesson"
    course = f" of '{s['course']}'" if s["course"] else ""
    kind = f" ({s['item_type']})" if s["item_type"] else ""
    mins = max(1, s["age_s"] // 60)
    return (
        "[STUDY SESSION]\n"
        f"The user is mid-study on Coursera — last activity ~{mins} min ago; "
        f"current item: '{where}'{kind}{course}.\n"
        "Bare continuation phrasings — 'continue', 'carry on', 'what's next', "
        "'next', 'keep going', 'move on' — mean the COURSE:\n"
        "  • advance to the next item  -> call coursera_next_item\n"
        "  • read/discuss this lesson  -> call coursera_read_lesson\n"
        "Do not ask which thing to continue, and do not improvise with "
        "web_click/web_navigate on Coursera — the coursera_* tools are the "
        "whole job.\n"
        "[/STUDY SESSION]"
    )


__all__ = [
    "STUDY_ACTIVE_WINDOW_S",
    "mark_study_activity",
    "clear_study_activity",
    "active_study_session",
    "build_study_session_block",
]
