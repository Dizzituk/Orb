# FILE: app/bridge/debug_speech.py
# Purpose: Map decoded debug-session SSE chunks (both the chat-tool-loop
#          narration path and the orchestrator's markdown-token path) to a
#          stream of speakable plain-text sentences for /bridge/debug-and-speak.
#          Suppresses tool-call chrome, reasoning tokens, and cost lines so the
#          road narration stays tight (spec §2.3 + Taz's §4.2 verbosity
#          decision: intent + reflection + final answer, minimal sub-agent
#          chatter). SpeechAccumulator owns the sentence-buffering state so the
#          endpoint just calls feed()/flush() and gets back ready-to-synthesise
#          strings.
# Called-by: app.bridge.debug_and_speak_stream
# Depends-on: app.bridge.chat_and_speak (_split_into_sentences, reused so
#             sentence-boundary behaviour matches the existing chat-and-speak
#             pipeline exactly)
# Last-renovated: 2026-07-01
from __future__ import annotations

import re
from typing import List, Optional

from app.bridge.chat_and_speak import _split_into_sentences

# Pure chrome on every path -- never spoken. "error" is deliberately NOT here:
# a silently-dying run with no spoken explanation is worse on the road than a
# short "ran into an error" line (see _speak_error below).
_SILENT_TYPES = {"tool_call", "tool_result", "reasoning", "metadata", "done"}

_MD_STRIP_RE = re.compile(r"[*_`#>]+|^\s*[-•]\s+", re.MULTILINE)
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]+"
)


def clean_for_speech(text: str) -> str:
    """Strip markdown decoration and emoji from an orchestrator token line so
    Chatterbox doesn't try to pronounce '###' or a rocket emoji out loud."""
    text = _EMOJI_RE.sub("", text)
    text = _MD_STRIP_RE.sub("", text)
    return " ".join(text.split())


def _speak_subagent(chunk_type: str, data: dict) -> Optional[str]:
    """Minimal sub-agent chatter (Taz §4.2): announce the fan-out starting and
    finishing only, not every individual agent's progress ticks."""
    if chunk_type == "subagent_spawn":
        count = data.get("count", 0)
        return f"Spawning {count} agent{'s' if count != 1 else ''}." if count else None
    if chunk_type == "subagent_spawn_complete":
        passed, failed = data.get("passed", 0), data.get("failed", 0)
        if failed:
            return f"Agents finished: {passed} passed, {failed} failed."
        return f"Agents finished, {passed} passed." if passed else None
    return None  # subagent_start / _progress / _complete: silent


def _speak_error(chunk: dict) -> Optional[str]:
    detail = (chunk.get("error") or "").strip()
    if not detail:
        return "Ran into an error and stopped."
    return f"Ran into an error: {detail[:200]}"


def speakable_line_from_event(chunk: dict) -> Optional[str]:
    """Complete, standalone speakable lines only (narration / sub-agent
    summaries / errors) -- NOT 'token' fragments, which need sentence
    buffering (see SpeechAccumulator.feed). None means suppress."""
    chunk_type = chunk.get("type", "")
    if chunk_type in _SILENT_TYPES:
        return None
    if chunk_type == "narration":
        text = (chunk.get("text") or "").strip()
        return text or None
    if chunk_type.startswith("subagent"):
        return _speak_subagent(chunk_type, chunk.get("data") or {})
    if chunk_type == "error":
        return _speak_error(chunk)
    return None


class SpeechAccumulator:
    """Stateful sentence buffer driven by feed(chunk). 'token' fragments (the
    model's streamed prose -- the real answer on the chat-tool-loop path, or
    markdown-formatted phase lines on the orchestrator path) are buffered and
    released at sentence boundaries via the same splitter the chat-and-speak
    pipeline uses, so mid-sentence audio never reaches Chatterbox. Every other
    speakable chunk type is already a complete line and passes straight
    through feed() with no buffering."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, chunk: dict) -> List[str]:
        if chunk.get("type") == "token":
            text = clean_for_speech(chunk.get("content") or "")
            if not text:
                return []
            self._buffer += text
            sentences = _split_into_sentences(self._buffer)
            if len(sentences) > 1:
                self._buffer = sentences[-1]
                return sentences[:-1]
            return []
        line = speakable_line_from_event(chunk)
        return [line] if line else []

    def flush(self) -> List[str]:
        """Call once the source stream ends to release any trailing fragment
        that never hit a sentence boundary (e.g. a reply with no final '.')."""
        tail = self._buffer.strip()
        self._buffer = ""
        return [tail] if tail else []
