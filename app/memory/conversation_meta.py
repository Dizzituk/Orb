# FILE: app/memory/conversation_meta.py
# Purpose: Detect WHOLE-conversation meta questions ("summarise everything",
#          "what were the very first messages", "full lowdown of the day") that
#          the topic-tagged spine cannot serve because they are topicless.
# Called-by: app.memory.conversation_overview, app.memory.spine_service,
#            app.endpoints.chat
# Depends-on: app.memory.recall_intent (stdlib otherwise)
# Last-renovated: 2026-07-01 (Lane B Task 1: meta-question detector)
"""
Whole-conversation meta-question detection (Lane B, 2026-07-01).

is_recall_question (app.memory.recall_intent) is shared with the web-search and
grounding gates, where over-firing SUPPRESSES a real search — so its bar is
deliberately high and it misses phrasings like "give me a full lowdown of the
whole day" or "what were the very first messages". This detector is a SUPERSET
used only to gate extra in-chat context injection (the conversation overview/
timeline), where the cost of over-firing is a few hundred wasted prompt chars,
never a lost search. recall_intent.py is intentionally NOT modified.
"""
from __future__ import annotations

import re
from typing import List

from app.memory.recall_intent import is_recall_question

# Additional whole-conversation phrasings beyond is_recall_question. Each is
# still conversation-leaning (lowdown/day/chat/first-messages framing) so plain
# lookups ("lowdown on quantum computing") don't fire.
_META_PATTERNS: List[str] = [
    # "lowdown / low-down" WITH a conversation-leaning tail (day/chat/we/...).
    # A bare verb-led "give me the lowdown on X" is the idiomatic phrasing for a
    # genuine topic lookup, so it must NOT fire (review 2026-07-01: an anchor-less
    # verb+lowdown pattern was dropped for exactly that over-fire).
    r"\b(?:full|whole|complete|proper|quick)?\s*low[\s-]?down\b.{0,50}"
    r"\b(?:day|chat|conversation|convo|session|everything|we\b|talked|discussed)",
    # "what were the/our/my (very) first messages" — covers the bare-"the" form
    # AND the plural possessives (recall_intent's possessive pattern is singular
    # 'message' only, so plurals fired nowhere; review 2026-07-01).
    # Negative lookahead keeps internet-trivia ("first message ever sent") out.
    r"\bwhat (?:was|were|is|are) (?:the|our|my|your) (?:very )?first (?:few )?messages?\b"
    r"(?!\s+(?:ever|in history|sent on|on the internet))",
    r"\b(?:very )?first (?:few )?messages? (?:of|in|from) (?:this|our|the|today)"
    r"(?:'s)? (?:chat|conversation|convo|session|day|morning)\b",
    # "how did this chat start / begin / kick off / open / get started"
    r"\bhow did (?:this|our|the|today's) (?:chat|conversation|convo|session|day|"
    r"discussion|thread) (?:start|begin|kick off|open|get going|get started)\b",
    r"\bhow (?:did|has) (?:this|our|the) (?:chat|conversation|convo|day) (?:started|"
    r"begun|opened|gone|been going)\b",
    # indirect form: "tell me how this chat started" / "how this conversation began"
    r"\bhow (?:this|our|the|today's) (?:chat|conversation|convo|session|day|"
    r"discussion|thread) (?:started|began|begun|opened|kicked off|got started|"
    r"got going)\b",
    # "the start/beginning/opening of this chat/conversation/day"
    r"\b(?:start|beginning|opening) of (?:this|our|the|today's) (?:chat|"
    r"conversation|convo|session|discussion|thread|day)\b",
    # "what did I/we open with", "first thing I/we/you said/asked"
    r"\bwhat did (?:i|we|you) (?:open|start|begin|kick) (?:with|off with)\b",
    r"\b(?:the )?first thing (?:i|we|you) (?:said|asked|sent|talked about|"
    r"mentioned|brought up)\b",
    # whole-day sweep: "everything (from) the whole/entire day", "the full day"
    r"\b(?:whole|entire|full) day\b.{0,40}\b(?:summar|recap|rundown|low[\s-]?down|"
    r"cover|review|go (?:over|through))",
    r"\b(?:summar|recap|rundown|review|go (?:over|through))\w*\b.{0,40}"
    r"\b(?:whole|entire|full) day\b",
]

_META_RE = re.compile("|".join(_META_PATTERNS), re.IGNORECASE)


def is_conversation_meta_question(text: str) -> bool:
    """True when the user asks about the conversation/day AS A WHOLE — a
    summary, the opener, or the arc — rather than about one topic in it.

    Superset of is_recall_question: everything that gate catches counts as a
    meta question here, plus the topicless whole-conversation phrasings above.
    """
    if not text:
        return False
    return bool(_META_RE.search(text)) or is_recall_question(text)


__all__ = ["is_conversation_meta_question"]
