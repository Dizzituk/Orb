# FILE: tests/test_weaver_target_line.py
# Purpose: live8 — Target line upgrades to the conversation's stated absolute path deterministically.
# Called-by: pytest
# Depends-on: app.llm._weaver_target_line, app.llm.greenfield_autoscope
# Last-renovated: 2026-07-04
"""23:23 incident (2026-07-04): the weave restated the verbal chain
"Documents/Games/Tazza's Tetris" although Astra's own reply had named
C:\\Users\\dizzi\\OneDrive\\Documents\\Games\\Tazza's Tetris. The verbal form
resolved only thanks to the registry mapping — one drift away from another
phantom folder. upgrade_target_line makes the precise form win without
trusting the model."""

from app.llm._weaver_target_line import upgrade_target_line
from app.llm.greenfield_autoscope import extract_greenfield_target

ABS = "C:\\Users\\dizzi\\OneDrive\\Documents\\Games\\Tazza's Tetris"

# The actual Gemini reply that named the path (condensed, incl. backtick wrap
# and the file-path mention whose PARENT is the same folder).
GEMINI_REPLY = (
    "I looked at your screenshot and mapped out the exact path: "
    "`C:\\Users\\dizzi\\OneDrive\\Documents\\Games\\Tazza's Tetris`.\n"
    "I've written a complete job specification file directly to that folder at "
    "`C:\\Users\\dizzi\\OneDrive\\Documents\\Games\\Tazza's Tetris\\weaver_job_spec.md`. "
    "It locks in the Python/Pygame stack and the strict 20KB/30KB modular file limits."
)

WOVEN = (
    "* What is being built: Python standalone Tetris game\n"
    "* Job class: greenfield_new_app\n"
    "Target folder/location: Documents/Games/Tazza's Tetris\n"
)


def _msgs(*contents, role="assistant"):
    return [{"role": role, "content": c} for c in contents]


def test_incident_verbal_line_upgrades_to_stated_absolute():
    new_text, resolved = upgrade_target_line(WOVEN, _msgs(GEMINI_REPLY))
    assert resolved == ABS
    assert f"Target folder/location: {ABS}" in new_text
    assert "Target folder/location: Documents/Games" not in new_text
    # everything else untouched
    assert new_text.startswith("* What is being built: Python standalone Tetris game")


def test_upgraded_line_still_extracts_downstream():
    new_text, resolved = upgrade_target_line(WOVEN, _msgs(GEMINI_REPLY))
    got = extract_greenfield_target(new_text)
    assert got is not None
    assert got["root"] == "C:/Users/dizzi/OneDrive/Documents/Games/Tazza's Tetris"
    assert got["name"] == "Tazza's Tetris"


def test_already_absolute_line_untouched():
    text = WOVEN.replace("Documents/Games/Tazza's Tetris", ABS)
    new_text, resolved = upgrade_target_line(text, _msgs(GEMINI_REPLY))
    assert resolved is None
    assert new_text == text


def test_curly_apostrophes_still_match():
    curly_woven = WOVEN.replace("Tazza's", "Tazza’s")
    curly_reply = GEMINI_REPLY.replace("Tazza's", "Tazza’s")
    new_text, resolved = upgrade_target_line(curly_woven, _msgs(curly_reply))
    assert resolved is not None
    assert resolved.endswith("Tazza's Tetris")  # cleaner normalises to disk form


def test_file_path_only_names_its_parent_folder():
    reply = (
        "The spec lives at "
        "`C:\\Users\\dizzi\\OneDrive\\Documents\\Games\\Tazza's Tetris\\weaver_job_spec.md` now."
    )
    new_text, resolved = upgrade_target_line(WOVEN, _msgs(reply))
    assert resolved == ABS


def test_non_matching_leaf_leaves_line_alone():
    reply = "I put it in C:\\Users\\dizzi\\OneDrive\\Documents\\Games\\Space Invaders yesterday."
    new_text, resolved = upgrade_target_line(WOVEN, _msgs(reply))
    assert resolved is None
    assert new_text == WOVEN


def test_two_distinct_matches_are_ambiguous():
    reply_a = "Path A: C:\\Users\\dizzi\\OneDrive\\Documents\\Games\\Tazza's Tetris"
    reply_b = "Path B: D:\\Backup\\Documents\\Games\\Tazza's Tetris"
    new_text, resolved = upgrade_target_line(WOVEN, _msgs(reply_a, reply_b))
    assert resolved is None
    assert new_text == WOVEN


def test_user_and_assistant_messages_both_count():
    new_text, resolved = upgrade_target_line(
        WOVEN, [{"role": "user", "content": f"put it in {ABS}."}]
    )
    assert resolved == ABS


def test_prose_tail_inside_leaf_is_conservative():
    """'...Tazza's Tetris please' swallows the prose into the leaf segment —
    genuinely ambiguous (could be a folder named that), so no upgrade; the
    autoscope's registry resolution of the verbal line remains the fallback."""
    new_text, resolved = upgrade_target_line(
        WOVEN, [{"role": "user", "content": f"put it in {ABS} please"}]
    )
    assert resolved is None
    assert new_text == WOVEN


def test_arrow_form_verbal_chain_matches():
    woven = WOVEN.replace(
        "Target folder/location: Documents/Games/Tazza's Tetris",
        "Target folder/location: Documents → Games → Tazza's Tetris",
    )
    new_text, resolved = upgrade_target_line(woven, _msgs(GEMINI_REPLY))
    assert resolved == ABS


def test_no_target_line_or_empty_input():
    assert upgrade_target_line("", _msgs(GEMINI_REPLY)) == ("", None)
    text = "* What is being built: game\n"
    assert upgrade_target_line(text, _msgs(GEMINI_REPLY)) == (text, None)


def test_traversal_and_drive_roots_never_win():
    woven = WOVEN.replace("Documents/Games/Tazza's Tetris", "Games/Tazza's Tetris")
    reply = "Check C:\\Games\\Tazza's Tetris\\..\\..\\Windows and also C:\\ itself."
    new_text, resolved = upgrade_target_line(woven, _msgs(reply))
    assert resolved is None
