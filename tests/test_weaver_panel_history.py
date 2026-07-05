# FILE: tests/test_weaver_panel_history.py
# Purpose: live7 — Weaver weaves the VISIBLE conversation: panel_history merges with DB rows.
# Called-by: pytest
# Depends-on: app.llm._weaver_stream_prepare, app.llm._weaver_stream_utils_15, app.llm.image_refs
# Last-renovated: 2026-07-04
"""The 22:41 project-orphan incident (2026-07-04): a stack restart rotated the
panel onto a fresh chat project, so 'Astra, send that to the weaver' ran
against a project holding only the command while the real thread (8k-char
paste + Astra's game replies) sat orphaned in the old project — the weave saw
'Analyzing 1 messages' and produced an empty brief. The Weaver now merges the
request's panel_history into its DB gather: it weaves what's ON SCREEN."""

from types import SimpleNamespace

import pytest

from app.llm import _weaver_stream_utils_15 as utils_15
from app.llm._weaver_stream_prepare import prepare_weaver_messages
from app.llm._weaver_stream_utils_14 import _hash_message
from app.llm._weaver_stream_utils_15 import _merge_panel_history
from app.llm.image_refs import extract_image_refs, image_ref_marker
from app.llm.spec_flow_state import (
    clear_flow_state,
    save_weave_checkpoint,
    save_woven_user_hashes,
)

WEAVE_COMMAND = "Astra, send that to the weaver."

PASTE = (
    "I want a standalone retro Tetris PC game. It needs neon block colours "
    "(pink, cyan, purple) on a glowing dark grid, a toggleable CRT scanline "
    "filter, and 8-bit sounds through the Web Audio API. Controls: A and D "
    "slide the piece, arrow keys rotate, down arrow fast-drops, P pauses. "
    "Ghost piece, high score tracker and level progression all included. "
    "Target folder/location: D:/Games/Tazzas Tetris"
)

# >=700 chars => substantive by default (live6 filter)
ASTRA_REPLY = (
    "Yeah man, I just checked the folder and there's a fully self-contained "
    "retro Tetris game sitting in there now, seriously clean at 23.3 KB. "
) + ("It has a glowing dark grid, neon block colours, a CRT scanline filter "
     "toggle, chiptune blips for movement and rotation, a thud on landing "
     "and an ascending chime on line clears. ") * 6


class _FakeMsg:
    def __init__(self, role, content):
        self.role = role
        self.content = content


def _patch_db(monkeypatch, chronological_rows):
    """Point the weaver's gather at a fake memory service.

    list_messages returns newest-first (the gather reverses it), so feed it
    the chronological rows reversed.
    """
    rows = [_FakeMsg(r, c) for r, c in chronological_rows]

    def fake_list_messages(db, project_id, limit=50):
        return list(reversed(rows))

    monkeypatch.setattr(
        utils_15, "memory_service", SimpleNamespace(list_messages=fake_list_messages)
    )


@pytest.fixture
def fresh_project():
    """Unique project id with clean flow state before and after."""
    project_id = 987_421
    clear_flow_state(project_id)
    yield project_id
    clear_flow_state(project_id)


# ---------------------------------------------------------------------------
# _merge_panel_history unit behaviour
# ---------------------------------------------------------------------------

def test_no_panel_history_returns_db_unchanged():
    db_msgs = [{"role": "user", "content": PASTE}]
    assert _merge_panel_history(db_msgs, None) is db_msgs
    assert _merge_panel_history(db_msgs, []) is db_msgs
    assert _merge_panel_history(db_msgs, "not-a-list") is db_msgs


def test_panel_only_messages_prepend_in_order():
    db_msgs = [{"role": "user", "content": WEAVE_COMMAND}]
    panel = [
        {"role": "user", "content": PASTE},
        {"role": "assistant", "content": ASTRA_REPLY},
    ]
    merged = _merge_panel_history(db_msgs, panel)
    assert [m["content"] for m in merged] == [PASTE, ASTRA_REPLY, WEAVE_COMMAND]


def test_exact_duplicates_not_doubled():
    db_msgs = [
        {"role": "user", "content": PASTE},
        {"role": "user", "content": WEAVE_COMMAND},
    ]
    panel = [{"role": "user", "content": PASTE}]
    merged = _merge_panel_history(db_msgs, panel)
    assert merged == db_msgs


def test_db_row_with_marker_beats_panel_copy(tmp_path):
    """DB rows carry [image_ref] markers the panel copy lacks — containment
    dedup must keep the DB version, not duplicate the text."""
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"\x89PNG fake")
    db_content = PASTE + "\n\n" + image_ref_marker(str(shot), "shot.png")
    db_msgs = [{"role": "user", "content": db_content}]
    panel = [{"role": "user", "content": PASTE}]
    merged = _merge_panel_history(db_msgs, panel)
    assert merged == db_msgs
    assert extract_image_refs(merged) == [str(shot)]


def test_short_affirmative_survives_containment_guard():
    """'yes' appears inside longer rows everywhere — short panel messages must
    dedup by exact hash only, never by containment."""
    db_msgs = [{"role": "user", "content": "So yes we should add the CRT filter to the game build."}]
    panel = [{"role": "user", "content": "yes"}]
    merged = _merge_panel_history(db_msgs, panel)
    assert {"role": "user", "content": "yes"} in merged


def test_control_and_junk_panel_entries_filtered():
    db_msgs = [{"role": "user", "content": WEAVE_COMMAND}]
    panel = [
        {"role": "system", "content": "internal system line"},
        # Same markers _is_control_message strips from DB rows (parity):
        {"role": "assistant", "content": "🧵 Weaving spec from conversation..."},
        {"role": "assistant", "content": "Ready for Spec Gate — say yes to proceed."},
        {"role": "user", "content": "how does that look all together"},
        {"role": "user", "content": ""},
        "not-a-dict",
    ]
    merged = _merge_panel_history(db_msgs, panel)
    assert merged == db_msgs


# ---------------------------------------------------------------------------
# prepare_weaver_messages end-to-end (the incident scenario)
# ---------------------------------------------------------------------------

def test_fresh_project_with_panel_history_weaves_full_conversation(monkeypatch, fresh_project):
    """The 22:41 incident, fixed: DB project holds only the weave command, the
    on-screen conversation arrives via panel_history — the weave must cover
    the paste AND Astra's substantive reply."""
    _patch_db(monkeypatch, [("user", WEAVE_COMMAND)])
    panel = [
        {"role": "user", "content": PASTE},
        {"role": "assistant", "content": ASTRA_REPLY},
    ]

    prep = prepare_weaver_messages(None, fresh_project, None, None, panel_history=panel)

    assert prep.early_exit_message == ""
    assert prep.total_message_count == 3  # paste + reply + command, not "1 messages"
    assert any(PASTE[:60] in m["content"] for m in prep.relevant_messages if m["role"] == "user")
    assert any(m["role"] == "assistant" for m in prep.relevant_messages)  # live6 stays fixed
    assert PASTE[:60] in prep.ramble_text
    assert ASTRA_REPLY[:60] in prep.ramble_text
    assert prep.is_update_mode is False


def test_no_panel_history_regression(monkeypatch, fresh_project):
    """Without panel_history the prepare path is byte-for-byte the old one."""
    _patch_db(monkeypatch, [("user", PASTE), ("assistant", ASTRA_REPLY)])

    prep = prepare_weaver_messages(None, fresh_project, None, None)

    assert prep.total_message_count == 2
    assert PASTE[:60] in prep.ramble_text


def test_update_mode_and_dedup_with_panel_history(monkeypatch, fresh_project):
    """Woven-hash dedup keeps working on the merged history: already-woven
    panel messages stay out of new_user_messages, genuinely new ones enter,
    and checkpoint presence flips update mode."""
    new_request = "Actually make the blocks twenty percent bigger please."
    _patch_db(monkeypatch, [("user", WEAVE_COMMAND)])
    panel = [
        {"role": "user", "content": PASTE},
        {"role": "assistant", "content": ASTRA_REPLY},
        {"role": "user", "content": new_request},
    ]

    save_woven_user_hashes(fresh_project, {_hash_message({"role": "user", "content": PASTE})})
    save_weave_checkpoint(fresh_project, 2, "previous woven job description")

    prep = prepare_weaver_messages(None, fresh_project, None, None, panel_history=panel)

    # is_update_mode is and-chained upstream (truthy checkpoint string, not
    # literal True) — assert truthiness, matching how callers consume it.
    assert bool(prep.is_update_mode) is True
    # Meta-mode cleanup strips trailing punctuation from user messages,
    # so match on prefix rather than the exact raw string.
    new_contents = [m["content"] for m in prep.new_user_messages]
    assert any(c.startswith("Actually make the blocks twenty percent bigger") for c in new_contents)
    assert not any(c.startswith(PASTE[:60]) for c in new_contents)


def test_image_refs_from_db_rows_survive_merge(monkeypatch, fresh_project, tmp_path):
    """Marker-bearing DB rows must still feed extract_image_refs after the
    merge (panel copies never carry markers)."""
    shot = tmp_path / "screen.png"
    shot.write_bytes(b"\x89PNG fake")
    db_paste = PASTE + "\n\n" + image_ref_marker(str(shot), "screen.png")
    _patch_db(monkeypatch, [("user", db_paste), ("user", WEAVE_COMMAND)])
    panel = [
        {"role": "user", "content": PASTE},
        {"role": "assistant", "content": ASTRA_REPLY},
    ]

    prep = prepare_weaver_messages(None, fresh_project, None, None, panel_history=panel)

    assert extract_image_refs(prep.filtered_messages) == [str(shot)]
    # and the paste text is not duplicated by its markerless panel copy
    paste_hits = [m for m in prep.all_messages if m["content"].startswith(PASTE[:60])]
    assert len(paste_hits) == 1
