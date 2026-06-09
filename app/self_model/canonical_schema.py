# FILE: app/self_model/canonical_schema.py
"""
Canonical schema for Tier 1 identity fields.

Single source of truth for what fields exist, what tier they live in,
what shape they take, and who is allowed to write them. The arbiter
(write_arbiter.py) and the reconciler (reconciler.py, Phase 6) both
read from here so we never have two copies of the same constraint
that can drift out of sync.

This module is the executable form of section 5 of
docs/memory_canonical_schema.md. If you change the schema doc, change
this file in the same commit.

Authoritative since Phase 3 (2026-05-02).

Health coaching fields added 2026-05-25 to support the unified ASTRA
coaching model: every chat session has access to these facts so any
conversation that touches food, training, or wellbeing can engage
coherently without re-asking the user about their baseline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional


# =============================================================================
# Tiers
# =============================================================================

class Tier(str, Enum):
    """
    Storage tier classification.

    T1 — immutable-once-confirmed identity facts (always-injected).
    T2 — preferences with confidence + decay (retrieved on demand).
    T3 — ephemeral working context (TTL-driven, HotIndex-backed).
    """
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"


# =============================================================================
# Validators (small, pure functions returning (ok, reason))
# =============================================================================

ValidatorResult = tuple  # (ok: bool, reason: str)


def _v_nonempty_string(v) -> ValidatorResult:
    if not isinstance(v, str):
        return (False, f"expected str, got {type(v).__name__}")
    if not v.strip():
        return (False, "empty string")
    return (True, "")


def _v_iso_date(v) -> ValidatorResult:
    if not isinstance(v, str):
        return (False, "not a string")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v.strip()):
        return (False, "not ISO yyyy-mm-dd")
    return (True, "")


def _v_email(v) -> ValidatorResult:
    if not isinstance(v, str):
        return (False, "not a string")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v.strip()):
        return (False, "not RFC-shaped email")
    return (True, "")


def _v_phone(v) -> ValidatorResult:
    if not isinstance(v, str):
        return (False, "not a string")
    s = v.strip()
    if not re.fullmatch(r"[\d\s\+\-()]{8,20}", s):
        return (False, "not phone-shaped (digits/+/-/space, 8-20 chars)")
    return (True, "")


def _v_address(v) -> ValidatorResult:
    """Full postal address — must contain a digit OR a UK postcode pattern."""
    if not isinstance(v, str):
        return (False, "not a string")
    s = v.strip()
    if not (10 <= len(s) <= 200):
        return (False, f"length {len(s)} outside 10-200")
    has_digit = bool(re.search(r"\d", s))
    has_uk_postcode = bool(re.search(
        r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", s, re.IGNORECASE,
    ))
    if not (has_digit or has_uk_postcode):
        return (False, "no digit or UK postcode pattern")
    return (True, "")


# Common-noun stoplist for single-word place captures (mirrors
# identity_capture._NON_PLACE_STOPLIST). Kept here too so the arbiter
# applies the same gate even if a writer doesn't.
_PLACE_STOPLIST = frozenset({
    "basketball", "football", "soccer", "cricket", "tennis", "hockey",
    "baseball", "rugby", "golf", "swimming", "running", "surfing",
    "house", "home", "work", "office", "school", "college", "hospital",
    "room", "place", "area", "world", "country", "city", "town", "village",
    "yes", "no", "maybe", "sure", "ok", "okay", "yeah", "nope",
    "there", "here", "somewhere", "today", "tomorrow", "yesterday",
    "way", "time", "day", "year", "life", "thing", "stuff",
})


def _v_short_place(v) -> ValidatorResult:
    """Short place name for birthplace. 2-60 chars, must look proper."""
    if not isinstance(v, str):
        return (False, "not a string")
    s = v.strip()
    if not (2 <= len(s) <= 60):
        return (False, f"length {len(s)} outside 2-60")
    if len(s.split()) > 6:
        return (False, "more than 6 words (likely a fragment)")
    if s.lower() in _PLACE_STOPLIST:
        return (False, f"matches non-place stoplist: {s!r}")
    tokens = s.split()
    if len(tokens) == 1 and tokens[0][:1].islower():
        return (False, "single-word location must start with capital")
    # Reject sentence-fragment verbs
    fragment_verbs = {"grew", "lived", "moved", "spent", "raised", "stayed",
                      "born", "visited", "went", "going", "came", "left"}
    value_tokens = set(re.findall(r"\b[a-z]+\b", s.lower()))
    if value_tokens & fragment_verbs:
        return (False, f"contains fragment verbs: {sorted(value_tokens & fragment_verbs)}")
    return (True, "")


def _v_short_name(v) -> ValidatorResult:
    """Personal name. 1-80 chars, alphabetic + spaces + hyphens + apostrophes."""
    if not isinstance(v, str):
        return (False, "not a string")
    s = v.strip()
    if not (1 <= len(s) <= 80):
        return (False, f"length {len(s)} outside 1-80")
    if not re.fullmatch(r"[A-Za-z][A-Za-z\s\-']*", s):
        return (False, "non-name characters present")
    return (True, "")


def _v_nationality(v) -> ValidatorResult:
    if not isinstance(v, str):
        return (False, "not a string")
    s = v.strip()
    known = {
        "british", "english", "welsh", "scottish", "irish", "american",
        "portuguese", "spanish", "french", "german", "italian", "dutch",
        "polish", "australian", "canadian", "south african", "indian",
    }
    if s.lower() not in known:
        return (False, f"not in known list (extend canonical_schema if needed)")
    return (True, "")


def _v_short_string(v, max_len: int = 80) -> ValidatorResult:
    if not isinstance(v, str):
        return (False, "not a string")
    s = v.strip()
    if not (1 <= len(s) <= max_len):
        return (False, f"length {len(s)} outside 1-{max_len}")
    return (True, "")


def _v_text_paragraph(v, max_len: int = 500) -> ValidatorResult:
    """
    A short free-form paragraph — for narrative fields like a training
    goal summary. Allows multi-sentence content but caps length so the
    Tier 1 injection budget doesn't blow up.
    """
    if not isinstance(v, str):
        return (False, "not a string")
    s = v.strip()
    if not (10 <= len(s) <= max_len):
        return (False, f"length {len(s)} outside 10-{max_len}")
    return (True, "")


def _v_short_string_list(v, max_items: int = 12, max_item_len: int = 80) -> ValidatorResult:
    """
    Ordered list of short strings. Used for prioritised lists like
    training_priorities where ordering carries meaning.

    Rejects: non-list, empty list, non-string elements, blanks, items
    over the per-item length cap, or more than max_items entries.
    """
    if not isinstance(v, list):
        return (False, "not a list")
    if not v:
        return (False, "empty list")
    if len(v) > max_items:
        return (False, f"more than {max_items} items")
    for i, item in enumerate(v):
        if not isinstance(item, str):
            return (False, f"item {i} is not a string ({type(item).__name__})")
        s = item.strip()
        if not s:
            return (False, f"item {i} is empty/whitespace")
        if len(s) > max_item_len:
            return (False, f"item {i} length {len(s)} exceeds {max_item_len}")
    return (True, "")


def _v_consideration_list(v) -> ValidatorResult:
    """
    List of short health consideration strings. Each item is one bullet
    (e.g. "ADHD - affects eating timing", "low-impact knee-friendly only").
    Allows slightly longer items than _v_short_string_list since each item
    may carry a brief explanation.
    """
    return _v_short_string_list(v, max_items=15, max_item_len=200)


# =============================================================================
# Field schema records
# =============================================================================

@dataclass(frozen=True)
class FieldSchema:
    """Schema for one Tier 1 field."""
    name: str
    tier: Tier
    description: str
    validator: Callable
    canonical_writer: str
    aliases: tuple = ()         # other names this field might appear under
    deprecated: bool = False    # if True, accept reads but reject writes


# Registry of all known Tier 1 fields. The arbiter consults this for
# every proposal. Phase 5 migration also reads this to drive its
# user_facts.json → identity.json moves.
_TIER_1_FIELDS: Dict[str, FieldSchema] = {
    "name": FieldSchema(
        name="name",
        tier=Tier.T1,
        description="Full legal name",
        validator=_v_short_name,
        canonical_writer="arbiter:confirmed",
    ),
    "preferred_name": FieldSchema(
        name="preferred_name",
        tier=Tier.T1,
        description="Name ASTRA addresses the user by",
        validator=lambda v: _v_short_string(v, max_len=40),
        canonical_writer="arbiter:confirmed",
    ),
    "date_of_birth": FieldSchema(
        name="date_of_birth",
        tier=Tier.T1,
        description="Date of birth in ISO yyyy-mm-dd",
        validator=_v_iso_date,
        canonical_writer="arbiter:confirmed",
        aliases=("dob",),
    ),
    "birthplace": FieldSchema(
        name="birthplace",
        tier=Tier.T1,
        description="Place physically born in (NOT same as residence_history[0])",
        validator=_v_short_place,
        canonical_writer="arbiter:confirmed",
    ),
    "address": FieldSchema(
        name="address",
        tier=Tier.T1,
        description="Full postal address (canonical 'where you live')",
        validator=_v_address,
        canonical_writer="arbiter:confirmed",
    ),
    "current_location": FieldSchema(
        name="current_location",
        tier=Tier.T1,
        description="Short-form town/region (DEPRECATED 2026-05-02 — use address)",
        validator=_v_short_place,
        canonical_writer="arbiter:confirmed",
        deprecated=True,
    ),
    "nationality": FieldSchema(
        name="nationality",
        tier=Tier.T1,
        description="Nationality (e.g. 'British')",
        validator=_v_nationality,
        canonical_writer="arbiter:confirmed",
    ),
    "email": FieldSchema(
        name="email",
        tier=Tier.T1,
        description="Primary email address",
        validator=_v_email,
        canonical_writer="arbiter:confirmed",
    ),
    "phone": FieldSchema(
        name="phone",
        tier=Tier.T1,
        description="Primary phone number",
        validator=_v_phone,
        canonical_writer="arbiter:confirmed",
    ),
    "current_occupation": FieldSchema(
        name="current_occupation",
        tier=Tier.T1,
        description="Current primary occupation (short string)",
        validator=lambda v: _v_short_string(v, max_len=80),
        canonical_writer="arbiter:confirmed",
        aliases=("occupation",),
    ),

    # ── Health coaching fields (added 2026-05-25) ───────────────────────────
    # These exist so any ASTRA chat — not just a "health tab" — has the
    # baseline facts to engage coherently when food, training, or wellbeing
    # come up. Tier 1 because they are stable about the person and need to
    # be injected on every conversation.
    "primary_sport": FieldSchema(
        name="primary_sport",
        tier=Tier.T1,
        description="The user's primary sporting pursuit (e.g. 'bodyboarding')",
        validator=lambda v: _v_short_string(v, max_len=40),
        canonical_writer="arbiter:confirmed",
    ),
    "training_goal_summary": FieldSchema(
        name="training_goal_summary",
        tier=Tier.T1,
        description=(
            "Short narrative summary of what the user is training toward "
            "(e.g. 'improve bodyboarding endurance and paddle power, "
            "gradual controlled mass reduction over 12 months')"
        ),
        validator=_v_text_paragraph,
        canonical_writer="arbiter:confirmed",
    ),
    "nutrition_approach": FieldSchema(
        name="nutrition_approach",
        tier=Tier.T1,
        description=(
            "The user's stated nutritional approach in their own words "
            "(e.g. 'broadly low-carb, managing sugar cravings, batch-cooks "
            "three meals at a time')"
        ),
        validator=lambda v: _v_short_string(v, max_len=300),
        canonical_writer="arbiter:confirmed",
    ),
}

# Structured list fields (separate write tools — arbiter doesn't gate these
# the same way, but it does observe).
_TIER_1_LIST_FIELDS: Dict[str, FieldSchema] = {
    "residence_history": FieldSchema(
        name="residence_history",
        tier=Tier.T1,
        description="Timeline of places lived (list of {place, from, to, notes})",
        validator=lambda v: (isinstance(v, list), "must be a list"),
        canonical_writer="tool:save_residence",
    ),
    "occupation_history": FieldSchema(
        name="occupation_history",
        tier=Tier.T1,
        description="Career timeline (list of {role, employer, from, to, notes})",
        validator=lambda v: (isinstance(v, list), "must be a list"),
        canonical_writer="tool:save_occupation",
    ),

    # ── Health coaching list fields (added 2026-05-25) ──────────────────────
    "training_priorities": FieldSchema(
        name="training_priorities",
        tier=Tier.T1,
        description=(
            "Ordered list of training priorities, most important first. "
            "Each item is a short string (e.g. ['endurance', 'flexibility', "
            "'controlled mass reduction', 'light explosive power'])"
        ),
        validator=_v_short_string_list,
        canonical_writer="arbiter:confirmed",
    ),
    "health_considerations": FieldSchema(
        name="health_considerations",
        tier=Tier.T1,
        description=(
            "List of short health-relevant considerations ASTRA should keep "
            "in mind across all conversations. Each item is one bullet, "
            "optionally with a brief note "
            "(e.g. ['ADHD — affects eating timing', 'manages sugar cravings'])"
        ),
        validator=_v_consideration_list,
        canonical_writer="arbiter:confirmed",
    ),
}


# =============================================================================
# Public API
# =============================================================================

def get_field_schema(field: str) -> Optional[FieldSchema]:
    """Look up a field's schema by name OR alias. Returns None if unknown."""
    if field in _TIER_1_FIELDS:
        return _TIER_1_FIELDS[field]
    if field in _TIER_1_LIST_FIELDS:
        return _TIER_1_LIST_FIELDS[field]
    for f in _TIER_1_FIELDS.values():
        if field in f.aliases:
            return f
    for f in _TIER_1_LIST_FIELDS.values():
        if field in f.aliases:
            return f
    return None


def is_tier_1_field(field: str) -> bool:
    """True if `field` is a known Tier 1 scalar or list field (or alias)."""
    return get_field_schema(field) is not None


def all_tier_1_field_names() -> List[str]:
    """Canonical names of all Tier 1 fields (no aliases)."""
    return list(_TIER_1_FIELDS.keys()) + list(_TIER_1_LIST_FIELDS.keys())


def is_deprecated(field: str) -> bool:
    """True if the field is marked deprecated (reads ok, writes proposed-only)."""
    schema = get_field_schema(field)
    return bool(schema and schema.deprecated)


def validate_value(field: str, value) -> tuple:
    """
    Run the field's validator. Returns (ok, reason).
    Unknown fields: (False, "unknown field").
    """
    schema = get_field_schema(field)
    if not schema:
        return (False, f"unknown field {field!r}")
    return schema.validator(value)
