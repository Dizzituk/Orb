# FILE: app/self_model/fragments/injection.py
"""
Build the prompt-injection block for HOT themes.

Only HOT themes (effective weight >= HOT_THRESHOLD) appear in the
injected block. WARM/COLD themes are available via surfacing but
don't shape day-to-day behaviour until they earn promotion.

Behaviour themes inject as explicit style guidance. Interest/
Preference themes inject as "known about the user" context.
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import List, Optional

from app.self_model.fragments.models import ClaimType
from app.self_model.fragments.weights import (
    theme_weight, tier_for_weight, Tier,
)

logger = logging.getLogger(__name__)

MAX_BEHAVIOUR_LINES = 6
MAX_INTEREST_LINES  = 8


def _dominant_claim_type(fragments) -> str:
    if not fragments:
        return ClaimType.UNKNOWN.value
    counts = Counter(f.claim_type for f in fragments)
    return counts.most_common(1)[0][0]


def _representative_text(fragments) -> str:
    """Pick the highest-weighted recent fragment's text as the theme caption."""
    if not fragments:
        return ""
    # Prefer non-passive signals, most recent
    def sort_key(f):
        priority = {"correction": 3, "declaration": 2, "reinforce": 2, "passive": 1}
        return (priority.get(f.signal_type, 0), f.created_at)
    best = max(fragments, key=sort_key)
    t = (best.text or "").strip()
    return t[:120]


def build_fragments_block() -> Optional[str]:
    """
    Produce the [LEARNED PATTERNS] block from hot themes, split into
    behavioural guidance and interest/preference context.
    """
    try:
        from app.self_model.fragments.store import get_fragment_store
    except Exception as exc:
        logger.debug("[fragments.injection] store import failed: %s", exc)
        return None

    store = get_fragment_store()
    themes = store.all_themes()
    if not themes:
        return None

    now = datetime.now(timezone.utc)

    behaviours: List[tuple] = []
    interests: List[tuple] = []

    for theme_id, theme_meta in themes.items():
        members = store.by_theme(theme_id)
        if not members:
            continue
        w = theme_weight(members, now=now)
        if tier_for_weight(w) != Tier.HOT:
            continue
        ct = _dominant_claim_type(members)
        rep = _representative_text(members)
        item = (w, rep, len(members))
        if ct == ClaimType.BEHAVIOUR.value:
            behaviours.append(item)
        elif ct in (ClaimType.INTEREST.value, ClaimType.PREFERENCE.value, ClaimType.STANCE.value):
            interests.append(item)

    if not behaviours and not interests:
        return None

    behaviours.sort(key=lambda x: x[0], reverse=True)
    interests.sort(key=lambda x: x[0], reverse=True)

    lines = ["[LEARNED PATTERNS]"]

    if behaviours:
        lines.append("  Behaviour guidance (reinforced over time):")
        for w, rep, n in behaviours[:MAX_BEHAVIOUR_LINES]:
            lines.append(f"    - {rep}  (weight {w:.1f}, {n} mentions)")

    if interests:
        lines.append("  Durable interests / preferences:")
        for w, rep, n in interests[:MAX_INTEREST_LINES]:
            lines.append(f"    - {rep}  (weight {w:.1f}, {n} mentions)")

    lines.append("[/LEARNED PATTERNS]")
    return "\n".join(lines)