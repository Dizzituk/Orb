# FILE: app/email/provider.py
# Purpose: Provider-agnostic email contract — EmailProvider ABC + dataclasses
#          + plain-text/body hygiene helpers. No provider-specific fields.
# Called-by: app.email (factory), app.email.gmail_provider, app.email.proton_provider,
#            app.tools.email_tools
# Depends-on: stdlib only
# Last-renovated: 2026-06-12
"""
The email contract.

Intents and UI code see ONLY these types and methods; Gmail/Proton/anything
else implements them. Labels/folders map to a generic tags list. Bodies are
plain text — providers strip HTML before returning.

The draft-confirm-send loop is part of the contract: create_draft() returns
a draft_id, send_draft(draft_id) commits it. Sending without a confirmed
draft is intentionally awkward — the safety rail lives in the seam itself.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from html import unescape
from typing import List, Optional


@dataclass
class ThreadSummary:
    id: str
    sender: str
    subject: str
    snippet: str
    unread: bool
    date: str                      # ISO 8601 where the provider allows
    tags: List[str] = field(default_factory=list)


@dataclass
class Message:
    id: str
    sender: str
    to: str
    date: str
    body: str                      # plain text, HTML stripped, signature trimmed
    tags: List[str] = field(default_factory=list)


@dataclass
class Thread:
    id: str
    subject: str
    messages: List[Message] = field(default_factory=list)


class EmailProvider(ABC):
    """Contract every mail backend implements. All methods are synchronous —
    callers (async tool handlers) wrap them in asyncio.to_thread."""

    name: str = "abstract"

    @abstractmethod
    def list_threads(self, query: Optional[str] = None, label: Optional[str] = None,
                     max_results: int = 10) -> List[ThreadSummary]:
        """Newest-first thread summaries. label is a generic tag name —
        'UNREAD' must work on every provider (others may be provider-specific)."""

    @abstractmethod
    def read_thread(self, thread_id: str) -> Thread:
        """Full thread with plain-text bodies."""

    @abstractmethod
    def send(self, to: str, subject: str, body: str,
             reply_to_thread: Optional[str] = None) -> dict:
        """Direct send — intents should prefer the draft-confirm-send loop.
        Returns {ok, id?, error?}."""

    @abstractmethod
    def create_draft(self, to: str, subject: str, body: str,
                     reply_to_thread: Optional[str] = None) -> dict:
        """Create (never send) a draft. Returns {ok, draft_id, to, subject, body}."""

    @abstractmethod
    def send_draft(self, draft_id: str) -> dict:
        """Send a previously created draft. Returns {ok, id?, error?}."""

    def search(self, query: str, max_results: int = 10) -> List[ThreadSummary]:
        """Default: providers with a query-capable list reuse it."""
        return self.list_threads(query=query, max_results=max_results)


# ── body hygiene (shared by providers) ─────────────────────────────────────

_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_BR_RE = re.compile(r"<\s*(br|/p|/div|/tr)\s*/?>", re.IGNORECASE)
_ANY_TAG_RE = re.compile(r"<[^>]+>")
_BLANK_RUNS_RE = re.compile(r"\n{3,}")

# Common signature openers; everything from the marker down is trimmed.
_SIGNATURE_MARKERS = (
    "\n-- \n", "\n--\n", "\nsent from my", "\nget outlook for",
    "\nkind regards\n", "\nbest regards\n",
)


def strip_html(html: str) -> str:
    """HTML -> readable plain text (good enough for TTS, not a renderer)."""
    text = _TAG_RE.sub("", str(html or ""))
    text = _BR_RE.sub("\n", text)
    text = _ANY_TAG_RE.sub("", text)
    text = unescape(text)
    lines = [ln.strip() for ln in text.splitlines()]
    return _BLANK_RUNS_RE.sub("\n\n", "\n".join(lines)).strip()


def trim_signature(body: str) -> str:
    """Cut obvious signatures so TTS doesn't read legal footers aloud."""
    lowered = (body or "").lower()
    cut = len(body or "")
    for marker in _SIGNATURE_MARKERS:
        idx = lowered.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return (body or "")[:cut].rstrip()


def clean_body(raw: str, *, is_html: bool = False) -> str:
    """The provider-side pipeline: optional HTML strip, then signature trim."""
    text = strip_html(raw) if is_html else str(raw or "")
    return trim_signature(text).strip()
