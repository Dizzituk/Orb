# FILE: app/self_model/fragments/surfacing.py
# Purpose: Cold-memory surfacing.
# Called-by: app.self_model.fragments.router
# Depends-on: app.embeddings.gemini_provider, app.self_model.fragments.cluster, app.self_model.fragments.store, app.self_model.fragments.weights
# Last-renovated: 2026-06-11
"""
Cold-memory surfacing.

Given a fresh user message, find dormant themes whose embeddings
match it strongly enough to justify Astra bringing up "you mentioned
this a while back".

Rules:
  - Only surface themes in COLD tier (effective weight < WARM_THRESHOLD)
  - Cosine similarity >= SURFACE_THRESHOLD
  - Theme has not been surfaced in last SURFACE_COOLDOWN_DAYS
  - No theme currently flagged as dismissed

Returned candidates are informational — downstream decides whether
to actually phrase them into the assistant's reply.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from app.self_model.fragments.cluster import cosine_similarity
from app.self_model.fragments.weights import theme_weight, tier_for_weight, Tier

logger = logging.getLogger(__name__)

SURFACE_THRESHOLD       = 0.72
SURFACE_COOLDOWN_DAYS   = 30
MAX_SURFACE_CANDIDATES  = 3


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def find_surface_candidates(message: str) -> List[Dict[str, Any]]:
    """
    Embed the message, compare against all theme centroids, return
    candidates that should be surfaced.
    """
    if not message or len(message.strip()) < 4:
        return []

    # Embed the incoming message
    try:
        from app.embeddings.gemini_provider import generate_embedding
        query_emb = generate_embedding(message)
    except Exception as exc:
        logger.debug("[surfacing] embed query failed: %s", exc)
        return []
    if not query_emb:
        return []

    from app.self_model.fragments.store import get_fragment_store
    store = get_fragment_store()
    now = datetime.now(timezone.utc)

    candidates: List[Dict[str, Any]] = []

    for theme_id, theme in store.all_themes().items():
        c = theme.get("centroid")
        if not c:
            continue
        sim = cosine_similarity(query_emb, c)
        if sim < SURFACE_THRESHOLD:
            continue

        members = store.by_theme(theme_id)
        if not members:
            continue

        w = theme_weight(members, now=now)
        tier = tier_for_weight(w)

        # Only surface COLD themes (hot/warm already inject or are in active use)
        if tier != Tier.COLD:
            continue

        # Cooldown check: last surface on any member
        last_surfaced = None
        for m in members:
            ts = _parse_iso(m.last_surfaced_at)
            if ts and (last_surfaced is None or ts > last_surfaced):
                last_surfaced = ts
        if last_surfaced and (now - last_surfaced) < timedelta(days=SURFACE_COOLDOWN_DAYS):
            continue

        # Dismissed-flag check: skip themes where MOST members are dismissed
        dismissed_count = sum(1 for m in members if m.dismissed)
        if dismissed_count * 2 > len(members):
            continue

        candidates.append({
            "theme_id": theme_id,
            "label": theme.get("label", "(untitled)"),
            "similarity": sim,
            "weight": w,
            "tier": tier.value,
            "member_count": len(members),
            "oldest_mention": min((m.created_at for m in members), default=""),
            "newest_mention": max((m.created_at for m in members), default=""),
        })

    candidates.sort(key=lambda c: c["similarity"], reverse=True)
    return candidates[:MAX_SURFACE_CANDIDATES]


def mark_theme_surfaced(theme_id: str) -> int:
    """Mark every fragment in a theme as just surfaced. Returns count."""
    from app.self_model.fragments.store import get_fragment_store
    store = get_fragment_store()
    now = datetime.now(timezone.utc).isoformat()
    members = store.by_theme(theme_id)
    for m in members:
        store.mark_surfaced(m.fragment_id, now)
    return len(members)


def mark_theme_dismissed(theme_id: str) -> int:
    """Mark every fragment in a theme as dismissed (accelerates decay)."""
    from app.self_model.fragments.store import get_fragment_store
    store = get_fragment_store()
    members = store.by_theme(theme_id)
    for m in members:
        store.mark_dismissed(m.fragment_id)
    return len(members)