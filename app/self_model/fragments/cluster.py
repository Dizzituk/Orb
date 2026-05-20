# FILE: app/self_model/fragments/cluster.py
"""
Theme clustering — group semantically-similar fragments into themes.

Simple greedy agglomerative clustering:
  - For each fragment with an embedding:
    - find the existing theme whose centroid is closest (cosine)
    - if similarity >= CLUSTER_JOIN_THRESHOLD → join that theme
    - otherwise → create a new theme seeded by this fragment
  - Theme centroid = mean of member fragment embeddings (refreshed on join)
  - Orphan fragments (no embedding) stay orphans — clustering re-runs when
    they get embedded later

Thresholds are conservative: we'd rather have more themes than mis-cluster.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from app.self_model.fragments.models import Fragment

logger = logging.getLogger(__name__)


CLUSTER_JOIN_THRESHOLD = 0.82   # raised from 0.72 — voice messages cluster too aggressively
THEME_LABEL_MAX_LEN    = 80


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def centroid(embeddings: List[List[float]]) -> Optional[List[float]]:
    if not embeddings:
        return None
    dim = len(embeddings[0])
    sums = [0.0] * dim
    for e in embeddings:
        for i, v in enumerate(e):
            sums[i] += v
    return [s / len(embeddings) for s in sums]


def _theme_label(fragments: List[Fragment]) -> str:
    """Best-effort human-readable theme label from member fragment texts."""
    if not fragments:
        return "(empty theme)"
    # Use the longest fragment as the representative — often the most informative
    rep = max(fragments, key=lambda f: len(f.text or ""))
    label = (rep.text or "").strip()
    if len(label) > THEME_LABEL_MAX_LEN:
        label = label[:THEME_LABEL_MAX_LEN - 1] + "…"
    return label


def recompute_themes() -> Dict[str, int]:
    """
    Full re-cluster pass. Reads all fragments with embeddings from the
    store, builds themes from scratch, writes theme metadata + updates
    fragment.theme_id pointers.

    Returns summary: {"total_fragments": n, "themes_created": m, "orphans": k}
    """
    from app.self_model.fragments.store import get_fragment_store
    store = get_fragment_store()

    # Wipe existing theme assignments first (fragments kept, metadata reset)
    existing_theme_ids = list(store.all_themes().keys())
    for tid in existing_theme_ids:
        store.delete_theme(tid)

    embedded = [f for f in store.all() if f.embedding]
    orphans = len(store.all()) - len(embedded)

    themes: Dict[str, Dict] = {}   # theme_id -> {"centroid": [...], "members": [frag]}

    for frag in embedded:
        best_theme: Optional[str] = None
        best_sim = 0.0
        for tid, t in themes.items():
            sim = cosine_similarity(frag.embedding, t["centroid"])
            if sim > best_sim:
                best_sim = sim
                best_theme = tid

        if best_theme is not None and best_sim >= CLUSTER_JOIN_THRESHOLD:
            # Join existing theme
            themes[best_theme]["members"].append(frag)
            members = themes[best_theme]["members"]
            themes[best_theme]["centroid"] = centroid([m.embedding for m in members])
            store.assign_theme(frag.fragment_id, best_theme)
        else:
            # New theme seeded by this fragment
            new_id = f"theme_{uuid4().hex[:10]}"
            themes[new_id] = {
                "centroid": list(frag.embedding),
                "members": [frag],
            }
            store.assign_theme(frag.fragment_id, new_id)

    # Persist theme metadata
    for tid, data in themes.items():
        members = data["members"]
        store.save_theme(tid, {
            "theme_id": tid,
            "label": _theme_label(members),
            "member_count": len(members),
            "centroid": data["centroid"],
            "member_ids": [m.fragment_id for m in members],
        })

    result = {
        "total_fragments": len(store.all()),
        "embedded_fragments": len(embedded),
        "themes_created": len(themes),
        "orphans": orphans,
    }
    logger.info("[cluster] recompute complete: %s", result)
    return result


def assign_fragment_to_theme(fragment: Fragment) -> Optional[str]:
    """
    Incremental assignment for a single freshly-added fragment.
    Finds the best existing theme to join, or creates a new one.
    Returns the theme_id it ended up in.
    """
    if not fragment.embedding:
        return None

    from app.self_model.fragments.store import get_fragment_store
    store = get_fragment_store()

    best_theme: Optional[str] = None
    best_sim = 0.0
    for tid, t in store.all_themes().items():
        c = t.get("centroid")
        if not c:
            continue
        sim = cosine_similarity(fragment.embedding, c)
        if sim > best_sim:
            best_sim = sim
            best_theme = tid

    if best_theme is not None and best_sim >= CLUSTER_JOIN_THRESHOLD:
        existing_members = store.by_theme(best_theme)
        new_members = existing_members + [fragment]
        embs = [m.embedding for m in new_members if m.embedding]
        new_centroid = centroid(embs) if embs else None
        t = store.get_theme(best_theme)
        if t and new_centroid:
            t["centroid"] = new_centroid
            t["member_count"] = len(new_members)
            t["member_ids"] = [m.fragment_id for m in new_members]
            t["label"] = _theme_label(new_members)
            store.save_theme(best_theme, t)
        store.assign_theme(fragment.fragment_id, best_theme)
        return best_theme

    # Create a new theme seeded by this fragment
    new_id = f"theme_{uuid4().hex[:10]}"
    store.save_theme(new_id, {
        "theme_id": new_id,
        "label": _theme_label([fragment]),
        "member_count": 1,
        "centroid": list(fragment.embedding),
        "member_ids": [fragment.fragment_id],
    })
    store.assign_theme(fragment.fragment_id, new_id)
    return new_id