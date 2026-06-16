# FILE: app/llm/routing/_capability_guard.py
# Purpose: Detect BARE capability-probe questions ("can you make images?") so
#          they route to a text answer instead of (costly) image generation.
# Called-by: app.llm.routing.chat_model_selection (select_chat_model image branch)
# Depends-on: re (stdlib); app.astra_memory.topic_tagger (optional, non-fatal)
# Last-renovated: 2026-06-16
"""
Capability-question guard (P0 cost bug J1).

A message like "can you make images, can you make 3d worlds, can you edit
social media posts, can you edit videos" matches ``detect_image_gen_intent``
(the substring "make images" trips the image regex) and therefore wrongly
fired IMAGE GENERATION on the phone — bypassing the LLM and costing money per
call. It is really a capability *question* and should be answered in text.

The disambiguation boundary (decided by Taz):

  An interrogative ("can you" / "could you" / "are you able to" /
  "do you support" / "what can you" / "what are you capable of") routes to
  CHAT only when it is a BARE capability probe — i.e. it does NOT name a
  concrete subject. If the interrogative names a concrete subject (markers
  " of ", " me a ", " me an " after the verb, e.g. "can you make me an image
  of a red surfboard"), it STILL routes to image.

So the LOAD-BEARING signal is the interrogative/capability phrasing regex.
NOTE: extract_tags() does NOT tag these as 'identity_capability' (it returns
['content'] / ['general']) — the tag is only a secondary OR-signal, wrapped in
try/except so a tagger failure can never break routing.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Concrete-subject markers. If any of these follow the verb, the interrogative
# is a real request ("make me an image of a red surfboard") — NOT a bare probe
# — so we must NOT suppress the image route. Spaces are intentional so we match
# " of " / " me a " / " me an " as whole tokens, not substrings inside words.
# ---------------------------------------------------------------------------
_CONCRETE_SUBJECT = re.compile(r"\bof\s|\bme\san?\s", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Capability-interrogative phrasings. Anchored to the START of the message OR
# to a clause boundary (. , ;) so we catch the lead clause of a multi-probe
# list ("can you make images, can you make 3d worlds, ..."). Forms covered:
#   - can / could / are you (optionally "able to" / "ever") ...
#   - do you support ...
#   - what can you ...
#   - what are you (able to / capable of) ...
# ---------------------------------------------------------------------------
_CAPABILITY_INTERROGATIVE = re.compile(
    r"(?:^|[.,;]\s*)"
    r"(?:"
    r"(?:can|could|are)\s+you(?:\s+(?:able\s+to|ever))?\b"
    r"|do\s+you\s+support\b"
    r"|what\s+can\s+you\b"
    r"|what\s+are\s+you(?:\s+(?:able\s+to|capable\s+of))?\b"
    r")",
    re.IGNORECASE,
)


def is_capability_question(message: str) -> bool:
    """
    True when ``message`` is a BARE capability probe that should be answered in
    text rather than triggering image generation.

    Two independent True-signals (OR):
      1. It matches a capability-interrogative pattern AND lacks a
         concrete-subject marker (" of " / " me a " / " me an "). This is the
         load-bearing signal and the boundary Taz chose.
      2. extract_tags(message) contains 'identity_capability' (secondary
         signal; import + call wrapped in try/except, non-fatal).
    """
    if not message:
        return False
    text = message.strip().lower()

    # Signal 1 — interrogative phrasing without a concrete subject.
    if _CAPABILITY_INTERROGATIVE.search(text) and not _CONCRETE_SUBJECT.search(text):
        return True

    # Signal 2 — topic tagger says identity_capability (best-effort only).
    # Mirrors the _detect_codebase_question pattern in capability_layer.py:
    # any tagger import/call failure is swallowed so routing never breaks.
    try:
        from app.astra_memory.topic_tagger import extract_tags

        if "identity_capability" in (extract_tags(message) or []):
            return True
    except Exception:
        pass

    return False
