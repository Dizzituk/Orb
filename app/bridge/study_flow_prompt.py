# FILE: app/bridge/study_flow_prompt.py
# Purpose: STUDY FLOW system-prompt block for bridge/voice sources — study turns act-then-answer, never announce.
# Called-by: app.bridge.capability_layer (_prepare_astra_chat)
# Depends-on: stdlib only
# Last-renovated: 2026-07-03
"""
Hands-free study flow (2026-07-03). Live transcript problem: three user
turns per lecture transition — Astra ANNOUNCED the resume instead of doing
it, then acted without content, then needed a third prompt to summarise;
it also asked which browser session when the coursera_* tools already
default to the coursera session.

Bridge sources only (source startswith "bridge" — /chat and /chat-and-speak).
The desktop prompt and the Room ("room" source) are untouched; the
[STUDY SESSION] state block (app/web_automation/study_state.py) stays the
mid-study disambiguator — this block is the standing playbook.
"""
from __future__ import annotations

STUDY_FLOW_BLOCK = (
    "\n\n## STUDY FLOW (Coursera, hands-free)\n"
    "- \"continue my course\" / \"resume the course\" / \"next lecture\" -> CALL the tool\n"
    "  (coursera_resume or coursera_next_item) IMMEDIATELY in this turn. Never reply\n"
    "  with an announcement of what you are about to do. Never ask which browser\n"
    "  session — the tools default to the coursera session.\n"
    "- After the tool lands on a video or reading item, ALSO call coursera_read_lesson\n"
    "  in the SAME turn, then answer in one reply: where we are (course + item title)\n"
    "  followed by a natural spoken summary of the transcript, ready to discuss.\n"
    "- If item_type is quiz/assignment: relay the tool's message verbatim and stop —\n"
    "  assessments are the user's to take.\n"
    "- Never end a turn having only announced an action.\n"
)


def study_flow_block(source: str | None) -> str:
    """The block for bridge surfaces, '' for everything else (desktop, room)."""
    if source and str(source).startswith("bridge"):
        return STUDY_FLOW_BLOCK
    return ""


__all__ = ["STUDY_FLOW_BLOCK", "study_flow_block"]
