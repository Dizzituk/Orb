# FILE: app/self_model/fragments/router.py
# Purpose: FastAPI endpoints for the fragment / theme layer.
# Called-by: main
# Depends-on: app.self_model.fragments.cluster, app.self_model.fragments.injection, app.self_model.fragments.store, app.self_model.fragments.surfacing (+1 more)
# Last-renovated: 2026-06-11
"""
FastAPI endpoints for the fragment / theme layer.

Routes:
  GET    /self-model/fragments              — all fragments (paginated summary)
  GET    /self-model/fragments/{id}         — single fragment
  POST   /self-model/fragments/dismiss      — mark a theme as dismissed (accelerates decay)
  GET    /self-model/themes                 — all themes with current weights + tiers
  GET    /self-model/themes/{id}            — theme detail + members
  POST   /self-model/themes/recompute       — full re-cluster pass
  GET    /self-model/fragments/_/preview    — what the hot-theme injection block looks like now
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/self-model", tags=["self-model-fragments"])


class DismissRequest(BaseModel):
    theme_id: str


@router.get("/fragments")
def list_fragments(limit: int = 200) -> Dict[str, Any]:
    from app.self_model.fragments.store import get_fragment_store
    store = get_fragment_store()
    all_f = store.all()
    return {
        "total": len(all_f),
        "summary": store.summary(),
        "fragments": [f.to_dict() for f in all_f[-limit:]],
    }


@router.get("/fragments/_/preview")
def preview_injection() -> Dict[str, Any]:
    from app.self_model.fragments.injection import build_fragments_block
    block = build_fragments_block()
    return {"block": block or "", "present": block is not None}


@router.get("/fragments/{fragment_id}")
def get_fragment(fragment_id: str) -> Dict[str, Any]:
    from app.self_model.fragments.store import get_fragment_store
    store = get_fragment_store()
    f = store.get(fragment_id)
    if not f:
        raise HTTPException(status_code=404, detail="fragment not found")
    return f.to_dict()


@router.post("/fragments/dismiss")
def dismiss_theme(req: DismissRequest) -> Dict[str, Any]:
    from app.self_model.fragments.surfacing import mark_theme_dismissed
    n = mark_theme_dismissed(req.theme_id)
    return {"dismissed_fragments": n}


@router.get("/themes")
def list_themes() -> Dict[str, Any]:
    from app.self_model.fragments.store import get_fragment_store
    from app.self_model.fragments.weights import theme_weight, tier_for_weight
    store = get_fragment_store()
    themes = store.all_themes()
    now = datetime.now(timezone.utc)
    out: List[Dict[str, Any]] = []
    for tid, meta in themes.items():
        members = store.by_theme(tid)
        w = theme_weight(members, now=now)
        out.append({
            "theme_id": tid,
            "label": meta.get("label", ""),
            "member_count": len(members),
            "weight": round(w, 3),
            "tier": tier_for_weight(w).value,
        })
    out.sort(key=lambda t: t["weight"], reverse=True)
    return {"total": len(out), "themes": out}


@router.get("/themes/{theme_id}")
def get_theme(theme_id: str) -> Dict[str, Any]:
    from app.self_model.fragments.store import get_fragment_store
    from app.self_model.fragments.weights import theme_weight, tier_for_weight, effective_weight
    store = get_fragment_store()
    t = store.get_theme(theme_id)
    if not t:
        raise HTTPException(status_code=404, detail="theme not found")
    members = store.by_theme(theme_id)
    now = datetime.now(timezone.utc)
    return {
        "theme": {
            "theme_id": theme_id,
            "label": t.get("label", ""),
            "member_count": len(members),
            "weight": round(theme_weight(members, now=now), 3),
            "tier": tier_for_weight(theme_weight(members, now=now)).value,
        },
        "members": [
            {
                **f.to_dict(),
                "current_weight": round(effective_weight(f, now=now), 3),
            }
            for f in members
        ],
    }


@router.post("/themes/recompute")
def recompute_all() -> Dict[str, Any]:
    from app.self_model.fragments.cluster import recompute_themes
    return recompute_themes()