# FILE: tests/test_weave_enrichment.py
# Purpose: live19 — compaction never swallows pasted documents; named design values carried verbatim (prompt pins).
# Called-by: pytest
# Depends-on: app.llm.weaver_stream, app.llm._weaver_prompts
# Last-renovated: 2026-07-05
"""The 13:14 weave admitted its own loss ("4,422-character document is
referenced, but its text is not present") — progressive-memory compaction
summarized the user's pasted brief away. And Astra's named palette (oxblood,
cocoa, cream, muted amber, avoid neon) was generalized to "burgundy + brown"."""

from app.llm._weaver_prompts import (
    WEAVER_CREATE_SYSTEM_PROMPT,
    WEAVER_UPDATE_SYSTEM_PROMPT,
)


def test_both_prompts_pin_design_values_as_requirements():
    for prompt in (WEAVER_CREATE_SYSTEM_PROMPT, WEAVER_UPDATE_SYSTEM_PROMPT):
        assert "DESIGN VALUES ARE REQUIREMENTS" in prompt
        assert "VERBATIM" in prompt
        assert "oxblood" in prompt  # the real incident palette, kept as example
        assert "NEVER generalize" in prompt


def test_document_reappend_source_is_wired():
    """Pin the compaction exemption in weaver_stream source: after
    format_for_weaver, large user messages re-append verbatim."""
    import inspect
    from app.llm import weaver_stream
    src = inspect.getsource(weaver_stream)
    assert "PASTED USER DOCUMENTS" in src
    assert ">= 2500" in src  # the size threshold for document-class messages
    assert "exempt from" in src
