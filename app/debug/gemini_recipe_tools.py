# FILE: app/debug/gemini_recipe_tools.py
# Purpose: Gemini executors for the saved-recipe tools (food overhaul Phase 6).
# Called-by: app.debug.gemini_lifestyle_tools (merged into LIFESTYLE_TOOL_EXECUTORS)
# Depends-on: app.tools.lifestyle_nutrition_tools
# Last-renovated: 2026-06-14
"""
Gemini tool executors for save_recipe / log_recipe (food overhaul Phase 6).

Split out of gemini_lifestyle_tools.py to respect its size ceiling; merged into
LIFESTYLE_TOOL_EXECUTORS exactly like the energy executors.
"""
from __future__ import annotations


async def _exec_save_recipe(args: dict) -> str:
    from app.tools.lifestyle_nutrition_tools import save_recipe_handler
    r = await save_recipe_handler(args, context=None)
    if not r.get("ok"):
        return f"ERROR: {r.get('error', 'unknown error')}"
    rec = r.get("recipe") or {}
    n = len(rec.get("ingredients") or [])
    return f"Saved recipe '{rec.get('name')}' with {n} ingredient(s)."


async def _exec_log_recipe(args: dict) -> str:
    from app.tools.lifestyle_nutrition_tools import log_recipe_handler
    r = await log_recipe_handler(args, context=None)
    if not r.get("ok"):
        return r.get("message") or f"ERROR: {r.get('error', 'not found')}"
    return f"Logged '{r.get('recipe')}' as {r.get('logged')} item(s)."


RECIPE_TOOL_EXECUTORS = {
    "save_recipe": _exec_save_recipe,
    "log_recipe": _exec_log_recipe,
}
