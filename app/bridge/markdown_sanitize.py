# FILE: app/bridge/markdown_sanitize.py
# Purpose: Markdown -> plain text for the bridge surface: for_display (phone bubble/history), for_speech (TTS input).
# Called-by: app.bridge.chat_speak_stream, app.bridge.chat_endpoints, app.bridge.router (history GET)
# Depends-on: stdlib re only
# Last-renovated: 2026-07-03
"""
The phone renders reply text raw (no markdown engine) and the TTS speaks
what it is given — live incident 2026-07-03: "**goals, intelligence, and
agency**" showed its asterisks in the bubble and Chatterbox voiced an
"asterisk" noise. The desktop renders markdown, so this is bridge-only.

for_display(text): markdown -> clean plain text, newlines preserved.
for_speech(text):  for_display PLUS bullets -> sentence flow, blank-run
                   collapse, and a hard guarantee that no *, #, `, [, ]
                   (or table pipes / blockquote markers / --- rules)
                   survives to be spoken.

Surface-side only: callers apply these on the way OUT (headers, response
fields, history payloads). Message rows in the DB keep the raw markdown so
artifact markers and any future renderer are not destroyed. Both functions
are no-ops on plain text, so double-sanitising is harmless.
"""
from __future__ import annotations

import re

# ── display-pass patterns ────────────────────────────────────────────────
_FENCE_LINE = re.compile(r"^[ \t]*```[^\n]*\n?", re.MULTILINE)  # fence lines out, code kept
_IMAGE_LINK = re.compile(r"!\[([^\]]*)\]\(([^)]*)\)")           # ![alt](url) -> alt
_INLINE_LINK = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")           # [text](url) -> text
_HEADER = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.MULTILINE)  # "## Title" -> "Title"
_BULLET = re.compile(r"^([ \t]*)[-*+][ \t]+", re.MULTILINE)     # "- item" -> "• item"
_NUMBERED = re.compile(r"^([ \t]*)\d{1,3}[.)][ \t]+", re.MULTILINE)
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_BOLD_UNDER = re.compile(r"__(.+?)__", re.DOTALL)
# Single-marker emphasis: only when it wraps non-space content and is not
# glued to a word/marker on the outside — "2*3" and "a * b" stay untouched.
_EMPH = re.compile(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])")
_EMPH_UNDER = re.compile(r"(?<![\w_])_(?!\s)([^_\n]+?)(?<!\s)_(?![\w_])")
_INLINE_CODE = re.compile(r"`([^`\n]*)`")

# ── speech-pass patterns ─────────────────────────────────────────────────
_HR_LINE = re.compile(r"^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$\n?", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^[ \t]*>[ \t]?", re.MULTILINE)
_SPOKEN_BANNED = re.compile(r"[*#`\[\]|]")                      # the hard no-speak set
_MULTI_BLANK = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")           # 2+ blank lines -> 1
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT = re.compile(r" +([.,;:!?])")


def for_display(text: str) -> str:
    """Markdown -> plain text for the phone bubble: emphasis unwrapped,
    headers flattened, [text](url) -> text, list markers -> '• ',
    newlines preserved."""
    if not text:
        return ""
    out = text.replace("\r\n", "\n")
    out = _FENCE_LINE.sub("", out)
    out = _IMAGE_LINK.sub(r"\1", out)
    out = _INLINE_LINK.sub(r"\1", out)
    out = _HEADER.sub("", out)
    out = _BULLET.sub(r"\1• ", out)
    out = _NUMBERED.sub(r"\1• ", out)
    out = _BOLD.sub(r"\1", out)
    out = _BOLD_UNDER.sub(r"\1", out)
    out = _EMPH.sub(r"\1", out)
    out = _EMPH_UNDER.sub(r"\1", out)
    out = _INLINE_CODE.sub(r"\1", out)
    return out.strip()


def _bullets_to_flow(text: str) -> str:
    """'• item' lines -> spoken sentence flow: each item is closed as a
    sentence and consecutive items join onto one line ('. ' rhythm)."""
    out: list[str] = []
    run: list[str] = []

    def _flush() -> None:
        if run:
            out.append(" ".join(run))
            run.clear()

    for raw in text.split("\n"):
        stripped = raw.strip()
        if stripped.startswith("•"):
            item = stripped.lstrip("•").strip()
            if item:
                if item[-1] not in ".!?:;":
                    item += "."
                run.append(item)
        else:
            _flush()
            out.append(raw)
    _flush()
    return "\n".join(out)


def for_speech(text: str) -> str:
    """for_display plus spoken-flow shaping. Guarantee: no *, #, `, [, ]
    reaches the TTS engine (they render as literal noise); table pipes,
    blockquote markers and --- rules are dropped too."""
    out = for_display(text)
    out = _HR_LINE.sub("", out)
    out = _BLOCKQUOTE.sub("", out)
    out = _bullets_to_flow(out)
    out = _SPOKEN_BANNED.sub(" ", out)
    out = _MULTI_BLANK.sub("\n\n", out)
    out = _MULTI_SPACE.sub(" ", out)
    out = _SPACE_BEFORE_PUNCT.sub(r"\1", out)
    return out.strip()


__all__ = ["for_display", "for_speech"]
