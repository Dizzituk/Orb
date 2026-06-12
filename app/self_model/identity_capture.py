# FILE: app/self_model/identity_capture.py
# Purpose: Real-time identity capture — regex-based pattern matching for
# Called-by: app.bridge.identity_hook, app.memory.integration
# Depends-on: app.self_model.identity, app.self_model.write_arbiter
# Last-renovated: 2026-06-11
"""
Real-time identity capture — regex-based pattern matching for
biographical hard facts in user messages.

Writes directly to the Tier 1 identity store. No SQL, no LLM, no
confidence scoring — regex-matched biographical statements are
treated as explicit user declarations.

Design:
  - Each pattern maps a regex group to an identity field name
  - First match wins per field per message
  - Normalisers clean captured values (strip trailing punctuation,
    canonicalise date formats, etc.)
  - Result logged + returned so callers can show confirmation UI
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─── Value normalisers ─────────────────────────────────────

def _strip_punct(v: str) -> str:
    return v.strip().rstrip(".,!?;:'\"")


def _normalise_date(v: str) -> Optional[str]:
    """Try hard to produce an ISO yyyy-mm-dd from various inputs."""
    s = _strip_punct(v)
    # Remove ordinal suffixes (1st, 2nd, 3rd, 4th)
    s = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", s, flags=re.IGNORECASE)
    # "16 of May 1981" → "16 May 1981"
    s = re.sub(r"\b(\d+)\s+of\s+([A-Za-z]+)", r"\1 \2", s, flags=re.IGNORECASE)
    # Collapse double spaces and tidy commas
    s = re.sub(r"\s+", " ", s).strip().rstrip(",")
    # Try a bunch of formats
    formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d",
        "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y",
        "%d %B, %Y", "%d %b, %Y",
    ]
    for f in formats:
        try:
            return datetime.strptime(s, f).date().isoformat()
        except ValueError:
            continue
    # Last-ditch: if string contains a year, try extracting D M Y loosely
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", s)
    if m:
        for f in ("%d %B %Y", "%d %b %Y"):
            try:
                return datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", f).date().isoformat()
            except ValueError:
                continue
    return None


def _normalise_location(v: str) -> str:
    s = _strip_punct(v)
    # Drop trailing "now", "currently", etc.
    s = re.sub(r"\s+(now|currently|these\s+days|at\s+the\s+moment)$", "", s, flags=re.IGNORECASE)
    return s.strip()


def _normalise_name(v: str) -> str:
    return _strip_punct(v).strip()


def _normalise_age(v: str) -> Optional[int]:
    try:
        n = int(_strip_punct(v))
        if 5 <= n <= 120:
            return n
    except ValueError:
        pass
    return None


# ─── Validation stoplists ──────────────────────────────────
# Phase 1 audit (2026-05-02): added after `birthplace = "basketball"`
# corruption. Voice-to-text can fragment a sentence like "I was born
# in Barnstaple, basketball is something I love…" into "basketball"
# being captured as the birthplace value because the regex stopped
# at the comma. The fix is a defence-in-depth stoplist for capture
# values that are clearly not place names / occupations.

# Common English single words that are NEVER valid as biographical
# values. If a capture matches one of these (case-insensitively) AND
# is a single word, reject it. Keep this list focused on the words
# that have actually shown up as voice-to-text mishears or sentence
# fragments.
_NON_PLACE_STOPLIST: set = {
    # sports / activities
    "basketball", "football", "soccer", "cricket", "tennis", "hockey",
    "baseball", "rugby", "golf", "swimming", "running", "surfing",
    "skiing", "climbing", "cycling", "boxing",
    # generic nouns that get captured by accident
    "house", "home", "work", "office", "school", "college", "hospital",
    "room", "place", "area", "world", "country", "city", "town", "village",
    "street", "road", "building", "flat", "apartment",
    # affirmations / interjections
    "yes", "no", "maybe", "sure", "ok", "okay", "yeah", "nope",
    # location pronouns
    "there", "here", "somewhere", "nowhere", "anywhere", "everywhere",
    # time markers
    "today", "tomorrow", "yesterday", "now", "later", "recently",
    # generic concepts
    "way", "time", "day", "year", "life", "thing", "stuff", "people",
    "person", "family", "friend", "work", "job",
}

# Verb tokens whose presence in a captured location value indicates a
# sentence fragment rather than a place name. "Lagos, Algarve" is a
# place; "grew up in Lagos, Algarve" is a fragment that begins with a
# verb. Real place names never contain these tokens.
_LOCATION_FRAGMENT_VERBS: set = {
    "grew", "grow", "grown", "growing",
    "lived", "living",
    "moved", "moving",
    "spent", "spending",
    "raised", "raising",
    "stayed", "staying",
    "born",
    "visited", "visiting",
    "went", "going", "goes", "gone",
    "came", "coming", "comes",
    "left", "leaving", "leaves",
    "travelled", "travelling", "travels",
}


# ─── Patterns ──────────────────────────────────────────────
# Each tuple: (regex, field, normaliser)
# Patterns are checked in order; first match per field wins.

_PATTERNS: List[Tuple[re.Pattern, str, Any]] = [
    # ── date of birth ──
    (re.compile(
        r"(?:my\s+(?:date\s+of\s+birth|dob|birthday|birth\s*date|birth)\s+(?:is|was(?:\s+on)?)|"
        r"i\s+was\s+born\s+on)\s+"
        r"(?:the\s+)?"
        r"(\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?[A-Za-z]+,?\s+\d{2,4}|"
        r"[A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}|"
        r"\d{1,4}[/\-]\d{1,2}[/\-]\d{1,4}|"
        r"\d{4}-\d{2}-\d{2})"
        r"(?=[.,;]|\s+and\b|\s+in\b|$)",
        re.IGNORECASE,
    ), "date_of_birth", _normalise_date),

    # ── birthplace ──
    (re.compile(
        r"i\s+was\s+born\s+in\s+([A-Za-z][A-Za-z ,.'\-]{2,60}?)(?:\.|,|;|\s+(?:and|but|though|however|on|while|before|after)\b|$)",
        re.IGNORECASE,
    ), "birthplace", _normalise_location),

    (re.compile(
        r"(?:my\s+(?:birthplace|hometown|home\s*town)\s+is|i\s+am\s+from\s+originally|"
        r"originally\s+(?:from|i(?:'m|\s+am)\s+from))\s+([A-Za-z][A-Za-z ,.'\-]{2,60}?)(?:\.|,|;|\s+and\b|$)",
        re.IGNORECASE,
    ), "birthplace", _normalise_location),

    # "I was born on <date> in <place>" — complementary to the DOB pattern
    (re.compile(
        r"i\s+was\s+born\s+on\s+[A-Za-z0-9 ,/\-]+?\s+in\s+([A-Za-z][A-Za-z ,.'\-]{2,60}?)(?:\.|,|;|\s+and\b|$)",
        re.IGNORECASE,
    ), "birthplace", _normalise_location),

    # ── current location ──
    # Only "live in", "based in", "now in" etc. — anything with ONLY "moved
    # to" is historical unless paired with "now" or "where I live/am now".
    # Explicit "I live in X" / "I am based in X" / "I now live in X" wins.
    (re.compile(
        r"(?:i\s+(?:now\s+)?live\s+in|"
        r"i(?:'m|\s+am)\s+(?:based|living|located)\s+(?:in|at)|"
        r"i(?:'ve|\s+have)\s+(?:just\s+|recently\s+)?moved\s+(?:down\s+|up\s+)?to|"
        r"i\s+just\s+moved\s+(?:down\s+|up\s+)?to|"
        r"my\s+(?:home|address|current\s+location)\s+is|"
        r"i(?:'m|\s+am)\s+now\s+(?:living\s+)?in|"
        r"i\s+(?:currently\s+)?stay\s+in|"
        r"where\s+i\s+(?:live|am)\s+(?:now|currently))\s+"
        r"([A-Za-z][A-Za-z ,.'\-]{2,60}?)"
        r"(?:\.|,|;|\s+(?:and|but|though|however|for|with|because|since|while|now)\b|$)",
        re.IGNORECASE,
    ), "current_location", _normalise_location),

    # Secondary pattern for tail-of-timeline moves: "... moved to X, which
    # is where I am now." Capture X from the where-I-am-now clause.
    (re.compile(
        r"(?:moved\s+(?:down\s+|up\s+|back\s+)?to\s+)"
        r"([A-Za-z][A-Za-z ,.'\-]{2,60}?)"
        r",?\s*which\s+is\s+where\s+i\s+am\s+now",
        re.IGNORECASE,
    ), "current_location", _normalise_location),

    # ── preferred name / nickname ──
    # Capture a single name token (allow hyphen/apostrophe). Stop at
    # the first whitespace-followed-by-lowercase-filler to avoid
    # swallowing "by the way", "for now", etc.
    (re.compile(
        r"(?:call\s+me|you\s+can\s+call\s+me|i\s+go\s+by|i\s+prefer\s+(?:to\s+be\s+called|"
        r"being\s+called)|i(?:'m|\s+am)\s+called)\s+([A-Z][A-Za-z'\-]{1,30})\b",
        re.IGNORECASE,
    ), "preferred_name", _normalise_name),

    # ── full name (legal, capitalised) ──
    (re.compile(
        r"(?:my\s+(?:full\s+)?name\s+is|i(?:'m|\s+am))\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b",
    ), "name", _normalise_name),

    # ── age ──
    (re.compile(
        r"i(?:'m|\s+am)\s+(\d{2,3})\s+(?:years?\s+old|yrs?\s+old)",
        re.IGNORECASE,
    ), "age", _normalise_age),

    # ── occupation ──
    # -- occupation --
    # Tightened: "i'm a" is too broad for voice-to-text. Require the
    # captured value to start with a letter and contain only plausible
    # job-title words (no filler words like "very", "sort", "kind").
    (re.compile(
        r"(?:i\s+work\s+(?:at|for|as\s+a|as\s+an)|my\s+job\s+is|"
        r"my\s+occupation\s+is|my\s+role\s+is|"
        r"i(?:'m|\s+am)\s+a(?:n)?\s+(?=(?:delivery|driver|developer|engineer|"
        r"teacher|nurse|doctor|designer|manager|chef|trainer|builder|"
        r"plumber|electrician|mechanic|freelance|self-employed|"
        r"consultant|analyst|writer|artist|musician|photographer|"
        r"accountant|lawyer|solicitor|therapist|carer|receptionist|"
        r"barista|bartender|courier|cleaner|gardener|farmer|"
        r"programmer|architect|pilot|paramedic|firefighter|"
        r"personal|fitness|software|data|web|full|front|back|"
        r"senior|junior|lead|head|chief|assistant)\b))"
        r"\s*([A-Za-z][A-Za-z /,&'\-]{2,60}?)"
        r"(?:\.|,|;|\s+(?:and|but|though|however|while|who|which|that|building|working|living)\b|$)",
        re.IGNORECASE,
    ), "occupation", _strip_punct),
    # ── nationality ──
    (re.compile(
        r"i(?:'m|\s+am)\s+(British|English|Welsh|Scottish|Irish|American|Portuguese|"
        r"Spanish|French|German|Italian|Dutch|Polish|Australian|Canadian|"
        r"South\s+African|Indian)(?:\s|\.|,|$)",
        re.IGNORECASE,
    ), "nationality", _strip_punct),

    # ── email ──
    (re.compile(
        r"my\s+email\s+(?:is|address\s+is)\s+([a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-.]+)",
        re.IGNORECASE,
    ), "email", _strip_punct),

    # ── phone ──
    (re.compile(
        r"my\s+(?:phone|number|mobile|cell)\s+(?:is|number\s+is)\s+([\d\s\+\-()]{8,20})",
        re.IGNORECASE,
    ), "phone", _strip_punct),
]


def scan_message(message: str) -> Dict[str, Any]:
    """
    Scan a user message for identity facts. Returns a dict of
    {field: normalised_value} for every pattern that matched.
    Does not write — caller decides whether to persist.
    """
    if not message or len(message.strip()) < 4:
        return {}

    out: Dict[str, Any] = {}
    seen_fields: set = set()

    for pattern, field, normaliser in _PATTERNS:
        if field in seen_fields:
            continue
        m = pattern.search(message)
        if not m:
            continue
        raw = m.group(1)
        value = normaliser(raw) if callable(normaliser) else _strip_punct(raw)
        if value in (None, "", []):
            continue
        # Guard against absurdly short names/locations
        if isinstance(value, str) and len(value) < 2:
            continue
        out[field] = value
        seen_fields.add(field)

    return out


def capture_and_persist(message: str, source: str = "chat_capture") -> Dict[str, Any]:
    """
    Scan a message and persist any matched facts to the identity store.

    Returns: {
      "captured":   {field: value, ...},
      "written":    [ {action, field, ...}, ... ],
      "skipped":    [ {field, reason}, ... ],
    }
    """
    captured = scan_message(message)
    result = {"captured": captured, "written": [], "skipped": []}

    if not captured:
        return result

    # Phase 3 (2026-05-02): hoist the arbiter import once per call.
    try:
        from app.self_model.write_arbiter import (
            propose as _arbiter_propose, get_arbiter_mode,
        )
        _arbiter_available = True
    except Exception as _arb_imp_err:
        logger.debug("[identity_capture] arbiter import failed: %s", _arb_imp_err)
        _arbiter_propose = None
        get_arbiter_mode = lambda: "shadow"
        _arbiter_available = False

    try:
        from app.self_model.identity import get_identity_store
        store = get_identity_store()
    except Exception as exc:
        logger.warning("[identity_capture] store unavailable: %s", exc)
        for field in captured:
            result["skipped"].append({"field": field, "reason": "store_unavailable"})
        return result

    for field, value in captured.items():
        # Phase 3 (2026-05-02): observe via arbiter BEFORE the existing
        # validation gates run. This means the arbiter sees the raw capture
        # and gets to record "would have queued / rejected as contradiction /
        # validation-failed" alongside whatever the live validators decide.
        # In shadow mode (default) this is observation only; in enforce mode
        # it eventually replaces the validators below.
        if _arbiter_available:
            try:
                _arbiter_propose(
                    field_name=field,
                    proposed_value=value,
                    source=f"identity_capture:{source}",
                    evidence={"raw_message": message[:200]},
                )
                # Phase 4 (2026-05-02): if the arbiter is in enforce mode, the
                # proposal has been queued for human confirmation. Do NOT also
                # write directly to identity.json — that would defeat the whole
                # point of the arbiter. Skip to next captured field.
                if get_arbiter_mode() == "enforce":
                    result["skipped"].append({
                        "field": field, "reason": "arbiter_enforce_mode_queued",
                    })
                    continue
            except Exception as _arb_err:
                logger.debug("[identity_capture] arbiter observe failed: %s", _arb_err)

        # Validation gate: reject obviously bad captures from voice-to-text
        if isinstance(value, str):
            _v_lower = value.lower()
            # Reject filler words that indicate conversational fragment
            _filler_starts = ("very ", "really ", "quite ", "sort of", "kind of",
                              "just ", "actually ", "basically ", "pretty ",
                              "a bit ", "not ", "more ", "less ")
            if any(_v_lower.startswith(f) for f in _filler_starts):
                logger.info("[identity_capture] Rejected noisy capture for %s: %r", field, value[:40])
                result["skipped"].append({"field": field, "reason": "filler_word_rejection"})
                continue

            # Reject question-shaped captures (voice-to-text runs sentences
            # together without punctuation, so "I live in Redruth where do I
            # live" can capture "Redruth where do I live"). Any question word
            # inside the captured value means we grabbed a sentence boundary.
            _question_words = (" where ", " what ", " when ", " why ", " who ",
                               " how ", "?")
            if any(qw in f" {_v_lower} " for qw in _question_words):
                logger.info("[identity_capture] Rejected question-shaped capture for %s: %r", field, value[:60])
                result["skipped"].append({"field": field, "reason": "question_shape_rejection"})
                continue

            # Location fields: reject if too long or if it contains multiple
            # location-like tokens strung together ("the UK somewhere in
            # Cornwall where I was" type fragments).
            if field in ("current_location", "birthplace", "address"):
                # Strict cap: real locations are short
                if len(value) > 60:
                    logger.info("[identity_capture] Rejected long location: %r", value[:60])
                    result["skipped"].append({"field": field, "reason": "location_too_long"})
                    continue
                # Word cap: "Redruth, Cornwall" = 2 words; "North Yorkshire, UK"
                # = 3. Anything over 6 is almost certainly a fragment.
                if len(value.split()) > 6:
                    logger.info("[identity_capture] Rejected multi-phrase location: %r", value[:60])
                    result["skipped"].append({"field": field, "reason": "location_too_many_words"})
                    continue
                # Reject if it starts with "the " followed by a country/generic
                # noun ("the UK Where...", "the country somewhere") -- real
                # addresses don't start this way even though individual words
                # might appear mid-string.
                if _v_lower.startswith("the ") and len(value.split()) > 2:
                    logger.info("[identity_capture] Rejected 'the X Y...' location: %r", value[:60])
                    result["skipped"].append({"field": field, "reason": "location_generic_phrase"})
                    continue
                # Phase 1 audit: reject sentence-fragment verbs in location
                # captures ("grew up in Lagos", "lived in London" etc.).
                # These come from voice-to-text running sentences together.
                _value_tokens = set(re.findall(r"\b[a-z]+\b", _v_lower))
                _verb_hits = _value_tokens & _LOCATION_FRAGMENT_VERBS
                if _verb_hits:
                    logger.info(
                        "[identity_capture] Rejected verb-fragment location for %s: %r (verbs=%s)",
                        field, value[:60], sorted(_verb_hits),
                    )
                    result["skipped"].append({
                        "field": field, "reason": "location_verb_fragment",
                        "verbs": sorted(_verb_hits),
                    })
                    continue
                # Phase 1 audit: single-word location captures must start
                # with a capital letter (real place names are proper nouns)
                # and must not be in the common-noun stoplist. This is what
                # would have stopped "basketball" landing as birthplace.
                _tokens = value.split()
                if len(_tokens) == 1:
                    _single = _tokens[0]
                    if _single[:1].islower():
                        logger.info(
                            "[identity_capture] Rejected uncapitalised single-word %s: %r",
                            field, value,
                        )
                        result["skipped"].append({
                            "field": field, "reason": "single_word_uncapitalised",
                        })
                        continue
                    if _v_lower in _NON_PLACE_STOPLIST:
                        logger.info(
                            "[identity_capture] Rejected stoplisted single-word %s: %r",
                            field, value,
                        )
                        result["skipped"].append({
                            "field": field, "reason": "single_word_in_stoplist",
                        })
                        continue

            # Occupation: reject if the captured value matches the stoplist
            # (e.g. "basketball" wouldn't normally hit occupation, but a
            # single common-noun capture is a smell regardless of field).
            if field == "occupation":
                if _v_lower in _NON_PLACE_STOPLIST:
                    logger.info(
                        "[identity_capture] Rejected stoplisted occupation: %r", value,
                    )
                    result["skipped"].append({
                        "field": field, "reason": "occupation_in_stoplist",
                    })
                    continue

            # Reject captures that look like sentence fragments (too many words)
            if field == "occupation" and len(value.split()) > 8:
                logger.info("[identity_capture] Rejected long occupation: %r", value[:40])
                result["skipped"].append({"field": field, "reason": "too_long_for_field"})
                continue
        try:
            r = store.set(field, value, source=source)
            result["written"].append(r)
            logger.info(
                "[identity_capture] %s %s = %r (source=%s)",
                r.get("action", "?"), field, value, source,
            )
        except Exception as exc:
            logger.warning("[identity_capture] write failed for %s: %s", field, exc)
            result["skipped"].append({"field": field, "reason": str(exc)})

    return result