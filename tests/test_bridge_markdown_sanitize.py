# FILE: tests/test_bridge_markdown_sanitize.py
"""Bridge markdown sanitiser (2026-07-03).

The phone renders reply text raw and the TTS speaks what it is given —
live incident: "**goals, intelligence, and agency**" showed its asterisks
in the bubble and Chatterbox voiced an "asterisk" noise. for_display feeds
the bubble/history payloads, for_speech feeds the synthesis input; the DB
rows stay raw (never sanitised in place).
"""
from app.bridge.markdown_sanitize import for_display, for_speech

MIXED_FIXTURE = (
    "## Study notes\n"
    "\n"
    "**goals, intelligence, and agency** drive the *whole* framing.\n"
    "\n"
    "- Physical stance = micro\n"
    "- Teleological stance = macro\n"
    "1. First takeaway\n"
    "\n"
    "See [the lecture page](https://coursera.org/lecture/42) and the\n"
    "`run_astra_chat` entry point.\n"
    "\n"
    "```python\n"
    "x = 1\n"
    "```\n"
    "\n"
    "> quoted aside\n"
    "\n"
    "---\n"
    "| a | b |\n"
)


class TestForDisplay:
    def test_bold_unwrapped_exact(self):
        assert for_display("**goals, intelligence, and agency**") == (
            "goals, intelligence, and agency"
        )

    def test_bullets_become_dots_and_newlines_survive(self):
        src = "- Physical stance = micro\n- Teleological stance = macro"
        assert for_display(src) == (
            "• Physical stance = micro\n• Teleological stance = macro"
        )

    def test_numbered_lists_become_dots(self):
        assert for_display("1. First takeaway") == "• First takeaway"

    def test_links_keep_text_drop_url(self):
        out = for_display(MIXED_FIXTURE)
        assert "the lecture page" in out
        assert "https://coursera.org" not in out

    def test_headers_flattened_and_code_markers_gone(self):
        out = for_display(MIXED_FIXTURE)
        assert "Study notes" in out
        assert "#" not in out
        assert "`" not in out
        assert "run_astra_chat" in out  # inline-code text kept
        assert "x = 1" in out           # fenced code text kept, fences gone

    def test_plain_text_untouched(self):
        plain = "Right, so 2 plus 2 is 4 — nothing fancy here.\nSecond line."
        assert for_display(plain) == plain

    def test_literal_asterisk_between_digits_untouched(self):
        # "2*3" is arithmetic, not emphasis — display must not eat it.
        assert for_display("about 2*3 metres") == "about 2*3 metres"


class TestForSpeech:
    def test_bullets_join_into_sentence_flow(self):
        src = "- Physical stance = micro\n- Teleological stance = macro"
        out = for_speech(src)
        assert "•" not in out
        assert out == "Physical stance = micro. Teleological stance = macro."

    def test_no_markdown_chars_survive_mixed_fixture(self):
        out = for_speech(MIXED_FIXTURE)
        for ch in ("*", "#", "`", "[", "]", "|"):
            assert ch not in out, f"banned char {ch!r} survived: {out!r}"

    def test_content_words_survive_mixed_fixture(self):
        out = for_speech(MIXED_FIXTURE)
        assert "goals, intelligence, and agency" in out
        assert "Physical stance = micro." in out
        assert "the lecture page" in out
        assert "quoted aside" in out

    def test_blank_runs_collapse_to_one(self):
        assert for_speech("first paragraph\n\n\n\nsecond paragraph") == (
            "first paragraph\n\nsecond paragraph"
        )

    def test_hr_and_blockquote_dropped(self):
        out = for_speech("above\n---\n> aside\nbelow")
        assert "---" not in out
        assert ">" not in out
        assert "aside" in out and "above" in out and "below" in out

    def test_idempotent_on_own_output(self):
        once = for_speech(MIXED_FIXTURE)
        assert for_speech(once) == once
