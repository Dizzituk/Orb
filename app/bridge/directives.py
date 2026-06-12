# FILE: app/bridge/directives.py
# Purpose: Extract + strip [[astra:...]] action directives from assistant replies for the phone to execute.
# Called-by: app.bridge.router (chat + chat-and-speak responses), app.bridge.chat_speak_stream (replay)
# Depends-on: stdlib re/json only
# Last-renovated: 2026-06-11
"""
Directive convention (added 2026-06-11, Phase 4b of the voice-loop polish).

Any backend component that wants the PHONE to perform an action embeds a
marker in the assistant reply text:

    [[astra:directive_name]]          e.g. [[astra:sync_health]]
    [[astra:directive_name:arg]]      e.g. [[astra:sync_health:48]]

The bridge strips the marker before the text reaches the screen or the TTS
voice, and forwards the parsed directives to the app:
  - voice path  -> X-Directives response header (URL-encoded JSON list)
  - text path   -> `directives` field on BridgeChatResponse

Currently recognised by AstraBridge:
    sync_health[:hours]  — fire a background Health Connect sync of the last
                           N hours (default 48). Used when the user says yes
                           to a sync nudge in chat.

Unknown directive names are forwarded anyway (the app ignores what it does
not know), so new directives need only a backend emitter and an app handler.
"""

from __future__ import annotations

import json
import re

DIRECTIVE_RE = re.compile(r"\[\[astra:([a-z0-9_]+)(?::([^\]]*))?\]\]", re.IGNORECASE)


def extract_directives(text: str) -> list[dict]:
    """Return [{'name': ..., 'arg': ...}] for every directive marker in text."""
    if not text or "[[astra:" not in text:
        return []
    out: list[dict] = []
    for match in DIRECTIVE_RE.finditer(text):
        out.append({"name": match.group(1).lower(), "arg": match.group(2)})
    return out


def strip_directives(text: str) -> str:
    """Remove directive markers (and any whitespace they leave dangling)."""
    if not text or "[[astra:" not in text:
        return text
    stripped = DIRECTIVE_RE.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", stripped).strip()


def directives_payload(directives: list[dict]) -> str:
    """JSON-serialise for the X-Directives header (caller URL-encodes)."""
    return json.dumps(directives)
