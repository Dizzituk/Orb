# FILE: app/lifestyle/nutrition_copy.py
# Purpose: Copy a day's food diary onto another day — voice-first nutrition reuse
# Called-by: app.llm.stream_router, app.tools.lifestyle_nutrition_tools, main
# Depends-on: app.db, app.lifestyle.models, app.lifestyle.service
# Last-renovated: 2026-06-11
"""
Copy a day's food diary onto another day — voice-first nutrition reuse
(2026-06-11, built from Taz's request: "copy yesterday's food over into
today, as I'm eating the same thing").

Three entry points share one engine:
  - copy_day_nutrition(db, source, target)  — the service function
  - POST /lifestyle/nutrition/copy-day      — HTTP for desktop/bridge
  - try_quick_nutrition_command(text, db)   — deterministic voice parser
    used by the stream router's lifestyle domain branch, so the command
    executes instantly with no LLM round-trip and no web-search misroute.

The Gemini tool layer also exposes this as `copy_nutrition_day` (see
app/tools/lifestyle_nutrition_tools.py) for conversational/multimodal
paths with flexible dates.

Copies are idempotent: rows already present on the target day with the
same description, meal type and calories are skipped, so saying it twice
doesn't double the diary.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


# =============================================================================
# Date words
# =============================================================================

def resolve_day_word(value: Optional[str], *, default: str, future_bias: bool = False) -> date:
    """Turn 'yesterday'/'today'/'tomorrow'/weekday/'YYYY-MM-DD' into a date.

    Weekdays resolve to the most recent occurrence (today or earlier) for
    sources; with future_bias=True they resolve to today or the next
    occurrence (for targets like 'duplicate it for Friday')."""
    token = (str(value).strip().lower() if value else "") or default
    today = date.today()

    if token == "today":
        return today
    if token == "yesterday":
        return today - timedelta(days=1)
    if token == "tomorrow":
        return today + timedelta(days=1)

    if token in _WEEKDAYS:
        target_dow = _WEEKDAYS.index(token)
        delta = (today.weekday() - target_dow) % 7
        if future_bias:
            forward = (target_dow - today.weekday()) % 7
            return today + timedelta(days=forward)
        return today - timedelta(days=delta)

    try:
        return datetime.strptime(token[:10], "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(
            f"Couldn't understand the day '{value}'. Use yesterday/today/tomorrow, "
            f"a weekday name, or YYYY-MM-DD."
        )


# =============================================================================
# Engine
# =============================================================================

def copy_day_nutrition(
    db,
    source_date: date,
    target_date: date,
    *,
    skip_duplicates: bool = True,
) -> Dict[str, Any]:
    """Duplicate every NutritionLog row from source_date onto target_date,
    preserving time-of-day, macros and confidence. Marks rows source='copy'.
    Refreshes the target DailySummary. Returns counts, totals and items."""
    from app.lifestyle.models import NutritionLog

    if source_date == target_date:
        return {"ok": False, "error": "Source and target are the same day — nothing to copy."}

    day_start = datetime.combine(source_date, time.min)
    day_end = day_start + timedelta(days=1)
    source_rows: List[NutritionLog] = (
        db.query(NutritionLog)
        .filter(NutritionLog.logged_at >= day_start, NutritionLog.logged_at < day_end)
        .order_by(NutritionLog.logged_at.asc())
        .all()
    )
    if not source_rows:
        return {
            "ok": False,
            "error": f"No food logged on {source_date.isoformat()} — nothing to copy.",
        }

    existing_keys = set()
    if skip_duplicates:
        t_start = datetime.combine(target_date, time.min)
        t_end = t_start + timedelta(days=1)
        for row in (
            db.query(NutritionLog)
            .filter(NutritionLog.logged_at >= t_start, NutritionLog.logged_at < t_end)
            .all()
        ):
            existing_keys.add((row.description, row.meal_type, round(row.calories or 0)))

    copied: List[Dict[str, Any]] = []
    skipped = 0
    totals = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "sugar_g": 0.0}

    for row in source_rows:
        key = (row.description, row.meal_type, round(row.calories or 0))
        if key in existing_keys:
            skipped += 1
            continue
        clone = NutritionLog(
            logged_at=datetime.combine(target_date, row.logged_at.time()),
            meal_type=row.meal_type,
            description=row.description,
            calories=row.calories,
            protein_g=row.protein_g,
            carbs_g=row.carbs_g,
            fat_g=row.fat_g,
            fibre_g=row.fibre_g,
            sugar_g=row.sugar_g,
            is_estimate=row.is_estimate,
            confidence=row.confidence,
            source="copy",
        )
        db.add(clone)
        copied.append({"description": row.description, "meal_type": row.meal_type,
                       "calories": row.calories})
        for field in totals:
            value = getattr(row, field, None)
            if value:
                totals[field] += value

    db.flush()
    try:
        from app.lifestyle.service import _update_daily_summary
        _update_daily_summary(db, target_date)
    except Exception as exc:
        logger.warning("[nutrition_copy] daily summary refresh failed: %s", exc)
    db.commit()

    logger.info(
        "[nutrition_copy] %s -> %s: %d copied, %d skipped (%.0f kcal)",
        source_date, target_date, len(copied), skipped, totals["calories"],
    )
    return {
        "ok": True,
        "source_date": source_date.isoformat(),
        "target_date": target_date.isoformat(),
        "copied": len(copied),
        "skipped_duplicates": skipped,
        "totals": {k: round(v, 1) for k, v in totals.items()},
        "items": copied,
    }


# =============================================================================
# Deterministic voice command (no LLM)
# =============================================================================

_FOOD_NOUN = r"(?:food|meals?|nutrition|diet|diary|eating|intake|macros)"
_DAY_TOKEN = r"(?:yesterday|today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{4}-\d{2}-\d{2})"

_COPY_RE = re.compile(
    rf"\b(?:copy|duplicate|repeat|mirror|carry)\b.{{0,60}}?\b{_FOOD_NOUN}\b",
    re.IGNORECASE,
)
_SAME_RE = re.compile(
    rf"\b(?:eat(?:ing)?|having|had)\b.{{0,40}}?\bsame\b.{{0,50}}?\b(?:as\s+)?({_DAY_TOKEN})\b",
    re.IGNORECASE,
)
_TARGET_RE = re.compile(rf"\b(?:in)?to\s+({_DAY_TOKEN})\b|\bfor\s+({_DAY_TOKEN})\b", re.IGNORECASE)
_DAY_RE = re.compile(rf"\b({_DAY_TOKEN})\b", re.IGNORECASE)


def try_quick_nutrition_command(message: str, db) -> Optional[str]:
    """If the message is a copy-my-food command, execute it and return a
    spoken-style result string. Returns None when the message isn't one,
    so the caller falls through to normal domain chat."""
    if not message:
        return None
    text = message.strip()

    source_token: Optional[str] = None
    target_token: Optional[str] = None
    matched = False

    if _COPY_RE.search(text):
        matched = True
        target_match = _TARGET_RE.search(text)
        if target_match:
            target_token = target_match.group(1) or target_match.group(2)
        day_tokens = [m.group(1).lower() for m in _DAY_RE.finditer(text)]
        if target_token:
            remaining = [t for t in day_tokens if t != target_token.lower()]
        else:
            remaining = day_tokens
        if remaining:
            source_token = remaining[0]
    else:
        same_match = _SAME_RE.search(text)
        if same_match:
            matched = True
            source_token = same_match.group(1)
            target_token = "today"

    if not matched:
        return None

    try:
        source_day = resolve_day_word(source_token, default="yesterday")
        target_day = resolve_day_word(target_token, default="today", future_bias=True)
    except ValueError as exc:
        return str(exc)

    result = copy_day_nutrition(db, source_day, target_day)
    if not result.get("ok"):
        return result.get("error", "Couldn't copy that day's food.")

    totals = result["totals"]
    bits = [f"{totals['calories']:.0f} kcal"]
    if totals.get("protein_g"):
        bits.append(f"{totals['protein_g']:.0f}g protein")
    if totals.get("sugar_g"):
        bits.append(f"{totals['sugar_g']:.0f}g sugar")
    skipped_note = (
        f" Skipped {result['skipped_duplicates']} already on there."
        if result["skipped_duplicates"] else ""
    )
    return (
        f"Done — copied {result['copied']} item"
        f"{'s' if result['copied'] != 1 else ''} from {result['source_date']} "
        f"into {result['target_date']}: {', '.join(bits)}.{skipped_note} "
        f"If you end up eating something different, just tell me and I'll adjust it."
    )


# =============================================================================
# HTTP API
# =============================================================================

router = APIRouter(prefix="/lifestyle/nutrition", tags=["lifestyle-nutrition"])


class CopyDayIn(BaseModel):
    source: str = "yesterday"
    target: str = "today"
    skip_duplicates: bool = True


@router.post("/copy-day")
def copy_day_endpoint(body: CopyDayIn) -> Dict[str, Any]:
    from app.db import SessionLocal
    try:
        source_day = resolve_day_word(body.source, default="yesterday")
        target_day = resolve_day_word(body.target, default="today", future_bias=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db = SessionLocal()
    try:
        result = copy_day_nutrition(db, source_day, target_day,
                                    skip_duplicates=body.skip_duplicates)
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    finally:
        db.close()
