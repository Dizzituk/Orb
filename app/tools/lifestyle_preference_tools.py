# FILE: app/tools/lifestyle_preference_tools.py
# Purpose: Food-preference tool handlers (CRUD) for ASTRA's health-coaching capability.
# Called-by: app.tools.lifestyle_tools
# Depends-on: app.lifestyle.service, app.tools.lifestyle_tools
# Last-renovated: 2026-06-21
"""
Food-preference tool handlers - add / read / delete / revise the soft, evolving
notes about how the user eats. Kept out of lifestyle_tools.py for the size
budget; re-exported from there so the Gemini executor adapters import them from
the usual place. Session/dict helpers come from lifestyle_tools (defined above
its re-export line -> no cycle).
"""
from __future__ import annotations

import logging
from typing import Optional

from app.tools.lifestyle_tools import _db_session, _to_dict

logger = logging.getLogger(__name__)


async def add_food_preference_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Record/reinforce one thing about how the user eats. Soft + evolving, not absolute."""
    category = input_data.get("category") or "note"
    content = (input_data.get("content") or "").strip()
    if not content:
        return {"ok": False, "error": "content is required"}
    stability = input_data.get("stability")
    conf = input_data.get("confidence")
    try:
        conf = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        conf = None
    try:
        from app.lifestyle.service import add_food_preference
        with _db_session() as db:
            d = _to_dict(add_food_preference(
                db, category, content, source="chat",
                stability=stability, confidence=conf,
            ))
    except Exception as exc:
        logger.exception("[lifestyle_tools] add_food_preference failed")
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "pref_id": d.get("id"), "category": d.get("category"),
            "content": d.get("content"), "stability": d.get("stability"),
            "confidence": d.get("confidence")}


async def get_food_preferences_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Read everything Astra knows about how the user eats."""
    try:
        from app.lifestyle.service import get_food_preferences
        with _db_session() as db:
            prefs = [_to_dict(p) for p in get_food_preferences(db)]
    except Exception as exc:
        logger.exception("[lifestyle_tools] get_food_preferences failed")
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "preferences": prefs, "count": len(prefs)}


async def delete_food_preference_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Delete a food preference/habit by id."""
    raw = input_data.get("pref_id")
    try:
        pref_id = int(raw)
    except (TypeError, ValueError):
        return {"ok": False, "error": "pref_id must be an integer"}
    try:
        from app.lifestyle.service import delete_food_preference
        with _db_session() as db:
            ok = delete_food_preference(db, pref_id)
    except Exception as exc:
        logger.exception("[lifestyle_tools] delete_food_preference failed")
        return {"ok": False, "error": str(exc)}
    if not ok:
        return {"ok": False, "pref_id": pref_id, "error": f"no preference with id {pref_id}"}
    return {"ok": True, "pref_id": pref_id, "deleted": True}


async def update_food_preference_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Revise an existing preference in place (evolve it) by id."""
    raw = input_data.get("pref_id")
    try:
        pref_id = int(raw)
    except (TypeError, ValueError):
        return {"ok": False, "error": "pref_id must be an integer"}
    try:
        from app.lifestyle.service import update_food_preference
        with _db_session() as db:
            out = update_food_preference(
                db, pref_id,
                content=input_data.get("content"),
                category=input_data.get("category"),
                stability=input_data.get("stability"),
                confidence=input_data.get("confidence"),
            )
            d = _to_dict(out) if out else None
    except Exception as exc:
        logger.exception("[lifestyle_tools] update_food_preference failed")
        return {"ok": False, "error": str(exc)}
    if not d:
        return {"ok": False, "pref_id": pref_id, "error": f"no preference with id {pref_id}"}
    return {"ok": True, "pref_id": pref_id, "content": d.get("content"),
            "category": d.get("category"), "stability": d.get("stability")}
