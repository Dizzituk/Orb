# FILE: app/memory/recall_intent.py
# Purpose: ONE shared "is the user asking me to recall OUR conversation?" predicate.
#          Consumed by (a) coverage -> fire the [CONVERSATION_COVERAGE] block,
#          (b) the Tier-0 web-search staleness gate and (c) the grounding gate ->
#          SUPPRESS a web search. So "what was I mentioning to you about anthropic"
#          / "what have we talked about today" recall the chat instead of searching
#          Brave for the company/the news.
# Called-by: app.memory.nat_jobs.coverage, app.translation._tier0_web_search,
#            app.grounding.grounding_gate
# Depends-on: stdlib only (deliberately dependency-free to avoid import cycles)
# Last-renovated: 2026-06-24
"""
Shared recall-intent detection.

Conversation-ANCHORED on purpose: every pattern requires a reference to *this*
chat or to what the user/assistant said ("we", "I told you", "our conversation",
"my first message"). It must never match a genuine lookup like "what's the latest
on anthropic" — otherwise it would wrongly suppress a real web search. Over-firing
on coverage is cheap (an extra DB-read block); wrongly suppressing a search is not,
so the bar is "clearly about our own conversation".
"""
from __future__ import annotations

import re
from typing import List

# Each entry is conversation-anchored. Broadened 2026-06-24 to cover the real
# phrasings that were being force-routed to web search (noun-fronted "what topics
# have we...", first-person "what was I mentioning...", "my first message",
# "did we talk about...", "read back").
_RECALL_PATTERNS: List[str] = [
    # "what did/have/were/was we talk/discuss/say/cover..."
    r"\bwhat (?:did|have|were|was) we (?:talk|chat|discuss|cover|go over|gone over|"
    r"say|said|speak|spoke|spoken|been (?:talking|chatting|discussing|on about))",
    r"\bwhat (?:was|were) (?:we|this|the) (?:talking|chat|conversation)",
    # summarise/recap/rundown/read-back OF the chat/conversation/today
    r"\bsummar(?:y|ise|ize)\b.{0,40}\b(?:chat|conversation|convo|discussion|talk|"
    r"call|session|thread|today|day)\b",
    r"\b(?:recap|run through|go over|rundown|read[\s-]?back|lay[\s-]?down)\b.{0,40}"
    r"\b(?:chat|conversation|convo|discussion|talk|session|thread|everything|"
    r"topics?|today|day|we)\b",
    r"\b(?:give|gimme|get|getting|do|read) (?:me )?(?:a |the )?(?:solid )?(?:rundown|"
    r"recap|run[\s-]?down|read[\s-]?back|read back|lay[\s-]?down|summary|catch[\s-]?up)\b",
    r"\bremind me (?:what|about what) we\b",
    r"\beverything (?:that )?we(?:'ve| have)?\s*(?:talked|discussed|covered|gone over|"
    r"spoke|spoken|said|been (?:over|on about))",
    r"\bwhat have we (?:been )?(?:talking|discussing|covering|chatting|speaking)\b",
    r"\bour (?:whole|entire) (?:chat|conversation|convo|discussion|day|thread)\b",
    r"\bwhat all (?:did|have) we\b",
    r"\bcatch me up\b",
    # noun-fronted: "what topics/things/subjects have we spoken about"
    r"\bwhat (?:topics?|things?|stuff|subjects?|else) (?:have|did|were|was|do) we\b",
    # recurring-preoccupation questions ("what do I keep going on about",
    # "what do I keep coming back to", "my recurring themes")
    r"\bwhat (?:do|did) i (?:keep|always|often|usually|constantly|tend to|repeatedly) "
    r"(?:talk|talking|go|going|come|coming|bring|bringing|mention|mentioning|say|"
    r"saying|on about)\b",
    r"\bwhat (?:topics?|themes?|things?|subjects?) do i (?:keep|always|often|usually)\b",
    r"\bkeep (?:talking about|going on about|coming back to|on about|bringing (?:it |this )?up)\b",
    r"\bmy (?:recurring|usual|common|main) (?:themes?|topics?|subjects?|preoccupations?)\b",
    # first-person recall: "what was I mentioning to you about anthropic"
    r"\bwhat (?:did|was|were|have|do) i (?:say|said|saying|mention|mentioning|mentioned|"
    r"tell|telling|told|ask|asked|asking|talk|talking|bring up|brought up|on about|"
    r"going on about)\b",
    # message recall: "what's my very first message about". Possessive-anchored
    # (my|your|our, NOT bare "the") so trivia like "the first message ever sent on
    # the internet" still grounds instead of being read as conversation recall.
    r"\b(?:my|your|our) (?:very )?(?:first|last|previous|earlier|opening|original) "
    r"(?:message|question|point|thing i (?:said|asked))\b",
    # "did we (ever) talk/speak/discuss/mention about ..."
    r"\bdid we (?:ever |even )?(?:talk|speak|discuss|cover|mention|chat|go over|touch on)\b",
    # conversation-anchored remembering. "do you remember when/how/why" needs a
    # conversational subject after it; bare how|why|when|that were dropped
    # 2026-06-24 because they anchored to nothing and matched trivia like
    # "do you remember how photosynthesis works".
    r"\bdo you remember (?:what|us|our|me|earlier|yesterday|last time|talking|"
    r"saying|we|i)\b",
    r"\bdo you remember (?:when|how|why) (?:we|i|you|us|our)\b",
    r"\bremember (?:what|when) (?:we|i|you)\b",
    # referential to prior turns IN this chat. Bare "we were talking about X" was
    # DROPPED 2026-06-24: it over-matched mixed-intent messages ("we were talking
    # about the shops, find me one nearby") and wrongly suppressed the search. The
    # question forms ("what were we talking about") are covered above.
    r"\b(?:earlier|back) (?:in|on) (?:this|our|the) (?:chat|conversation|convo|thread|"
    r"session|discussion)\b",
]

_RECALL_RE = re.compile("|".join(_RECALL_PATTERNS), re.IGNORECASE)


def is_recall_question(text: str) -> bool:
    """True when the user is asking ASTRA to recall THIS conversation / what was
    said, rather than to look something up online. Conversation-anchored so it
    never suppresses a genuine "what's the latest on X" web search."""
    return bool(text) and bool(_RECALL_RE.search(text))


__all__ = ["is_recall_question"]
