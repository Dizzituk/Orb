# FILE: app/self_model/identity_format.py
"""
Format the Tier 1 identity store into a system-prompt block.

Single responsibility: turn the IdentityStore into a [USER IDENTITY]
block ready to prepend to the LLM context. Computes derived values
where useful (e.g. age from date_of_birth).
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Display order and pretty labels for known fields. Health coaching fields
# appended 2026-05-25 — placed at the end so the biographical block reads
# first, then the coaching context for any chat that touches food, training,
# or wellbeing.
_DISPLAY_ORDER = [
    ("name",                  "Name"),
    ("preferred_name",        "Preferred name"),
    ("date_of_birth",         "Date of birth"),
    ("birthplace",            "Birthplace"),
    ("current_location",      "Current location"),
    ("address",               "Full address"),
    ("residence_history",     "Places lived"),
    ("nationality",           "Nationality"),
    ("languages",             "Languages"),
    ("occupation",            "Occupation"),
    ("email",                 "Email"),
    ("phone",                 "Phone"),
    # Health coaching block — visible in every chat so any conversation
    # can engage as a coach without re-asking baseline questions.
    ("primary_sport",         "Primary sport"),
    ("training_goal_summary", "Training goal"),
    ("training_priorities",   "Training priorities (in order)"),
    ("nutrition_approach",    "Nutrition approach"),
    ("health_considerations", "Health considerations"),
]

# Fields whose list-of-strings value should render as a bulleted block
# rather than a comma-joined inline. Used where item ordering or item
# length makes the bulleted form much more readable for the LLM.
_BULLET_RENDER_FIELDS = {
    "training_priorities",
    "health_considerations",
}


def _compute_age(dob_value: str) -> Optional[int]:
    """Parse a DOB string and return current age in years, or None."""
    if not dob_value:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            d = datetime.strptime(str(dob_value).strip(), fmt).date()
            today = date.today()
            return today.year - d.year - (
                (today.month, today.day) < (d.month, d.day)
            )
        except ValueError:
            continue
    return None


def _format_residence_entry(entry: dict) -> str:
    """Render one residence_history entry as a compact one-liner."""
    place = entry.get("place", "?")
    frm = entry.get("from")
    to = entry.get("to")
    years = entry.get("duration_years")
    notes = entry.get("notes")

    bits = [str(place)]
    if frm and to:
        bits.append(f"{frm}–{to}")
    elif frm and not to:
        bits.append(f"from {frm}")
    elif years:
        bits.append(f"{years} years")
    if notes:
        bits.append(f"({notes})")
    return " — ".join(bits) if len(bits) > 1 else bits[0]


def _format_string_list_bullets(items: list) -> str:
    """
    Render a list of strings as an indented bulleted block. Used for
    ordered priority lists where the bullet form makes ordering visible
    to the model.
    """
    return "\n    - " + "\n    - ".join(str(x) for x in items)


def _format_value(v, field_name: Optional[str] = None) -> str:
    """
    Format a value for the identity block.

    field_name is optional and used to route specific fields to a
    bulleted render even when the value is a list of plain strings
    (e.g. training_priorities, where ordering matters).
    """
    if isinstance(v, list):
        if not v:
            return ""
        # List of dicts (structured) → multi-line sub-entries.
        if isinstance(v[0], dict):
            return "\n    - " + "\n    - ".join(
                _format_residence_entry(item) if "place" in item else str(item)
                for item in v
            )
        # List of scalars where the field opts into bullet rendering.
        if field_name in _BULLET_RENDER_FIELDS:
            return _format_string_list_bullets(v)
        # Default scalar-list render: comma-joined.
        return ", ".join(str(x) for x in v)
    return str(v)


def build_identity_block() -> Optional[str]:
    """
    Load the identity store and produce the injection block.
    Returns None if the store is empty.
    """
    try:
        from app.self_model.identity import get_identity_store
        store = get_identity_store()
        fields = store.all()
    except Exception as exc:
        logger.debug("[identity_format] load failed: %s", exc)
        return None

    if not fields:
        return None

    lines = ["[USER IDENTITY]"]

    # Known fields in display order
    for key, label in _DISPLAY_ORDER:
        f = fields.get(key)
        if f is None or f.value in (None, "", []):
            continue
        lines.append(f"  {label}: {_format_value(f.value, field_name=key)}")
        # Append derived age right after DOB
        if key == "date_of_birth":
            age = _compute_age(f.value)
            if age is not None:
                lines.append(f"  Age: {age}")

    # Any unknown / custom fields tacked on at the end
    for key, f in fields.items():
        if key in {k for k, _ in _DISPLAY_ORDER}:
            continue
        if f.value in (None, "", []):
            continue
        lines.append(f"  {key}: {_format_value(f.value, field_name=key)}")

    if len(lines) == 1:
        return None  # no fields actually had values

    lines.append("[/USER IDENTITY]")
    return "\n".join(lines)
