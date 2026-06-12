# FILE: app/self_model/fragments/weights.py
# Purpose: Weight decay + current effective weight computation.
# Called-by: app.self_model.fragments.injection, app.self_model.fragments.router, app.self_model.fragments.surfacing
# Depends-on: app.self_model.fragments.models
# Last-renovated: 2026-06-11
"""
Weight decay + current effective weight computation.

Design:
  - Each fragment has a fixed `base_weight` and `created_at` (never change)
  - Current effective weight = base_weight * decay(age_days)
  - Decay is EXPONENTIAL with a half-life of DEFAULT_HALF_LIFE_DAYS
  - Dismissed fragments decay 2x faster
  - Theme weight = sum of member fragment weights, with a diversity multiplier
    (themes with fragments from many sessions/dates/contexts weigh more)

Tiering is then:
  - HOT   : theme weight >= HOT_THRESHOLD   → inject into prompt
  - WARM  : theme weight >= WARM_THRESHOLD  → known, can surface on match
  - COLD  : below WARM_THRESHOLD            → dormant, surface-only
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Iterable, List, Optional

from app.self_model.fragments.models import Fragment


DEFAULT_HALF_LIFE_DAYS      = 90.0
DISMISSED_HALF_LIFE_DAYS    = 45.0   # dismissed fragments decay twice as fast

HOT_THRESHOLD   = 25.0
WARM_THRESHOLD  = 3.0


class Tier(str, Enum):
    HOT  = "hot"
    WARM = "warm"
    COLD = "cold"


def _parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _age_days(created_at: str, now: Optional[datetime] = None) -> float:
    now = now or datetime.now(timezone.utc)
    then = _parse_iso(created_at)
    if not then:
        return 0.0
    delta = now - then
    return max(0.0, delta.total_seconds() / 86400.0)


def effective_weight(
    fragment: Fragment,
    now: Optional[datetime] = None,
) -> float:
    """
    Current decayed weight of a single fragment.
    Uses half-life exponential decay.
    """
    half_life = DISMISSED_HALF_LIFE_DAYS if fragment.dismissed else DEFAULT_HALF_LIFE_DAYS
    age = _age_days(fragment.created_at, now=now)
    # weight = base * (0.5 ^ (age / half_life))
    return fragment.base_weight * math.pow(0.5, age / half_life)


def diversity_multiplier(fragments: Iterable[Fragment]) -> float:
    """
    Reward themes whose fragments are spread across sessions/dates/projects.
    A cluster with ten fragments all from one session is less durable than
    one with ten fragments across ten different sessions.

    Returns a multiplier in roughly [0.4, 1.5]:
      - single-session, single-day burst    → 0.4
      - balanced spread across many sessions → 1.0
      - rare wide spread                      → up to 1.5
    """
    frags = list(fragments)
    if not frags:
        return 1.0

    sessions = {f.session_id for f in frags if f.session_id is not None}
    projects = {f.project_id for f in frags if f.project_id is not None}
    dates = set()
    for f in frags:
        dt = _parse_iso(f.created_at)
        if dt:
            dates.add(dt.date().isoformat())

    # Session diversity: 1 session = poor, >=5 = good
    s = min(len(sessions) or 1, 10) / 5.0       # 0.2..2.0
    # Date diversity
    d = min(len(dates) or 1, 10) / 5.0
    # Project diversity (optional — default 1.0 if none recorded)
    p = min(len(projects) or 1, 5) / 3.0 if projects else 1.0

    raw = (s + d + p) / 3.0
    # Clamp to a sane range
    return max(0.4, min(1.5, raw))


def theme_weight(
    fragments: Iterable[Fragment],
    now: Optional[datetime] = None,
) -> float:
    """
    Aggregate effective weight of a theme (cluster of fragments).
    """
    frags = list(fragments)
    if not frags:
        return 0.0
    total = sum(effective_weight(f, now=now) for f in frags)
    return total * diversity_multiplier(frags)


def tier_for_weight(w: float) -> Tier:
    if w >= HOT_THRESHOLD:
        return Tier.HOT
    if w >= WARM_THRESHOLD:
        return Tier.WARM
    return Tier.COLD