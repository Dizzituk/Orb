# FILE: tests/test_weaver_substantive_filter.py
# Purpose: live6 — Weaver must treat Astra's conversational build replies as substantive (read assistant msgs).
# Called-by: pytest
# Depends-on: app.llm._weaver_substantive_filter
# Last-renovated: 2026-07-04
"""Taz found the Weaver ignored Astra's own replies. Astra's reply describing
the built game is load-bearing (it holds the real spec) — it must now weave in,
while genuine pleasantries stay out."""

from app.llm._weaver_substantive_filter import _is_substantive_assistant_content

# The actual 22:41 Astra reply (condensed), prose not code — was DROPPED before.
ASTRA_TETRIS_REPLY = (
    "Yeah man, that screenshot was taken right when the folder was empty. But I "
    "just checked C:\\Users\\dizzi\\OneDrive\\Documents\\Games\\Tazza's Tetris and "
    "there's a 23.8 KB index.html sitting in there now. It's a fully self-contained "
    "retro Tetris game, seriously clean at 23.3 KB. Retro Glow & CRT Filter: glowing "
    "dark grid, neon block colors (pink, cyan, purple), a toggleable CRT scanline "
    "filter. 8-Bit Sound Synth via the Web Audio API — blips for movement and "
    "rotation, a thud on landing, an ascending chime on line clear. Controls: A and D "
    "slide the piece left and right, Arrow Left/Right spin the piece, Arrow Down for a "
    "fast drop, P to pause. There's a ghost piece, high score tracker, and level "
    "progression."
)


def test_astra_conversational_build_reply_is_substantive():
    assert _is_substantive_assistant_content(ASTRA_TETRIS_REPLY) is True


def test_long_reply_substantive_by_default():
    long_prose = "So basically " + ("the app has a lovely retro feel and " * 40)
    assert len(long_prose) >= 700
    assert _is_substantive_assistant_content(long_prose) is True


def test_pleasantry_still_dropped():
    assert _is_substantive_assistant_content("Sure! Happy to help with that.") is False
    assert _is_substantive_assistant_content("Got it, done.") is False


def test_mid_length_two_hit_build_reply_included():
    # 250-700 char band: needs 2+ build-vocab hits (this has app/game/controls/
    # arrow/retro/colour/file) but is not code-heavy.
    msg = (
        "Right, so I built the game as a standalone app with arrow-key controls "
        "and a retro colour theme. Left and right arrows slide the piece, up "
        "rotates it clockwise, and down does a soft drop. It's all bundled into "
        "one index.html file so you can double-click it, with a little arcade "
        "feel and some chiptune beeps on the side. The whole thing is nicely "
        "self-contained and comes in well under your size budget."
    )
    assert 250 <= len(msg) < 700
    assert _is_substantive_assistant_content(msg) is True


def test_too_short_dropped():
    # under the 250 floor even if it mentions the build
    assert _is_substantive_assistant_content(
        "I built the app with arrow controls."
    ) is False


def test_empty_and_none():
    assert _is_substantive_assistant_content("") is False
    assert _is_substantive_assistant_content(None) is False
