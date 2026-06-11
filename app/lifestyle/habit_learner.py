# FILE: app/lifestyle/habit_learner.py
"""
Nightly eating-habit learner — Job 4 of the memory roadmap (2026-06-10).

The lifestyle_food_preference table existed but had no writer — habits
were never learned. This module is that writer:

  1. Scans recent NutritionLog entries and derives habits: staple foods
     (eaten on several distinct days), meal timing, where the shopping
     happens, sugar pattern vs the active limit.
  2. Upserts them into FoodPreference rows (confidence rises when a habit
     keeps showing up; last_seen refreshed — feeds the coach/Plan tab).
  3. Mirrors staples into the Tier 2 ASTRA preference layer, where they
     inherit the full evidence/reinforcement/decay machinery for free.
  4. Indexes the last week of daily summaries and the strongest habits
     into the hot index with lifestyle tags, so chat retrieval can finally
     see what Taz actually eats and does — the cross-silo bridge.

Run nightly by app/lifestyle/scheduler.py, or manually:
  python -m app.lifestyle.habit_learner
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_STOPWORDS: Set[str] = {
    "a", "an", "and", "the", "with", "of", "on", "in", "some", "bit", "few",
    "my", "for", "to", "from", "at", "had", "have", "ate", "eat", "eating",
    "meal", "lunch", "dinner", "breakfast", "snack", "small", "big", "large",
    "g", "kg", "ml", "x", "two", "three", "one", "half", "portion", "plate",
    "bowl", "cup", "glass", "fresh", "grilled", "fried", "cooked", "raw",
}

_SHOP_NAMES = ("aldi", "lidl", "tesco", "sainsbury", "asda", "morrisons", "co-op", "coop", "iceland")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _phrases(description: str) -> Set[str]:
    """Extract candidate food phrases (1-2 word, stopword-filtered)."""
    cleaned = re.sub(r"\(.*?\)", " ", (description or "").lower())
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
    tokens = [t for t in cleaned.split() if len(t) > 2 and t not in _STOPWORDS]
    found: Set[str] = set(tokens)
    for i in range(len(tokens) - 1):
        found.add(f"{tokens[i]} {tokens[i + 1]}")
    return found


# =============================================================================
# Analysis
# =============================================================================

def analyse_recent_nutrition(db: Session, days: int = 21) -> Dict[str, Any]:
    """Derive habit candidates from the last `days` of nutrition logs."""
    from app.lifestyle.models import NutritionLog

    since = _now() - timedelta(days=days)
    logs: List[NutritionLog] = (
        db.query(NutritionLog).filter(NutritionLog.logged_at >= since).all()
    )

    phrase_days: Dict[str, Set[str]] = defaultdict(set)
    first_log_hours: List[int] = []
    day_meal_counts: Dict[str, int] = defaultdict(int)
    day_sugar: Dict[str, float] = defaultdict(float)
    shops_seen: Set[str] = set()

    day_first_seen: Dict[str, datetime] = {}
    for log in logs:
        if not log.logged_at:
            continue
        day_key = log.logged_at.strftime("%Y-%m-%d")
        day_meal_counts[day_key] += 1
        if log.sugar_g:
            day_sugar[day_key] += log.sugar_g
        if day_key not in day_first_seen or log.logged_at < day_first_seen[day_key]:
            day_first_seen[day_key] = log.logged_at
        for phrase in _phrases(log.description or ""):
            if phrase in _SHOP_NAMES:
                shops_seen.add(phrase)
            else:
                phrase_days[phrase].add(day_key)

    for first in day_first_seen.values():
        first_log_hours.append(first.hour)

    # Prefer 2-word phrases over their contained single words.
    staples = {p: d for p, d in phrase_days.items() if len(d) >= 4}
    for phrase in list(staples):
        if " " in phrase:
            for word in phrase.split():
                if word in staples and staples[word] <= staples[phrase] | staples[word]:
                    if len(staples[word]) <= len(staples[phrase]) + 1:
                        staples.pop(word, None)

    total_days = len(day_meal_counts)
    avg_meals = round(sum(day_meal_counts.values()) / total_days, 1) if total_days else 0
    median_first_hour = sorted(first_log_hours)[len(first_log_hours) // 2] if first_log_hours else None
    avg_sugar = round(sum(day_sugar.values()) / total_days, 1) if total_days else None

    return {
        "window_days": days,
        "logged_days": total_days,
        "total_logs": len(logs),
        "staples": {p: len(d) for p, d in sorted(staples.items(), key=lambda kv: -len(kv[1]))[:12]},
        "shops": sorted(shops_seen),
        "avg_meals_per_day": avg_meals,
        "median_first_log_hour": median_first_hour,
        "avg_sugar_g_per_day": avg_sugar,
    }


# =============================================================================
# FoodPreference upserts (direct model — confidence rises on reinforcement)
# =============================================================================

def _upsert_food_pref(db: Session, category: str, content: str, key_token: str) -> str:
    """Insert or reinforce a FoodPreference. Returns 'created'|'reinforced'."""
    from app.lifestyle.models import FoodPreference

    existing = (
        db.query(FoodPreference)
        .filter(FoodPreference.category == category)
        .filter(FoodPreference.content.like(f"%{key_token}%"))
        .first()
    )
    if existing:
        existing.content = content
        existing.confidence = min(1.0, (existing.confidence or 0.6) + 0.1)
        existing.last_seen = _now()
        existing.updated_at = _now()
        if existing.confidence >= 0.85 and existing.stability == "evolving":
            existing.stability = "core"
        return "reinforced"

    db.add(FoodPreference(
        category=category, content=content, stability="evolving",
        confidence=0.6, last_seen=_now(), source="habit_learner",
    ))
    return "created"


def write_habits(db: Session, analysis: Dict[str, Any]) -> Dict[str, int]:
    counts = {"created": 0, "reinforced": 0}
    window = analysis["window_days"]

    for phrase, day_count in analysis["staples"].items():
        outcome = _upsert_food_pref(
            db, "staple",
            f"Regularly eats {phrase} — on {day_count} of the last {window} days",
            phrase,
        )
        counts[outcome] += 1

    for shop in analysis["shops"]:
        outcome = _upsert_food_pref(db, "shop", f"Shops at {shop.title()}", shop)
        counts[outcome] += 1

    if analysis["median_first_log_hour"] is not None:
        hour = analysis["median_first_log_hour"]
        outcome = _upsert_food_pref(
            db, "timing",
            f"Typically eats first around {hour:02d}:00 and logs ~{analysis['avg_meals_per_day']} meals/day",
            "eats first around",
        )
        counts[outcome] += 1

    db.commit()
    return counts


# =============================================================================
# Tier 2 mirror — staples inherit reinforcement + decay machinery
# =============================================================================

def mirror_to_tier2(db: Session, analysis: Dict[str, Any]) -> int:
    mirrored = 0
    try:
        from app.astra_memory.preference_service import create_preference, reinforce_preference, get_preference
        from app.astra_memory.preference_models import SignalType

        for phrase in list(analysis["staples"].keys())[:8]:
            slug = re.sub(r"\s+", "_", phrase)
            key = f"lifestyle.food.staple.{slug}"
            if get_preference(db, key):
                reinforce_preference(db, key, signal_type=SignalType.IMPLICIT,
                                     context_pointer="habit_learner")
            else:
                create_preference(
                    db, preference_key=key, preference_value=phrase,
                    source="habit_learner", applies_to="lifestyle",
                    namespace="lifestyle",
                )
            mirrored += 1
    except Exception as exc:
        logger.warning("[habit_learner] Tier 2 mirror skipped: %s", exc)
    return mirrored


# =============================================================================
# Hot-index bridge — lifestyle becomes retrievable in chat
# =============================================================================

def index_lifestyle_to_hot(db: Session) -> int:
    """Index last 7 daily summaries + strongest food prefs into the hot index."""
    indexed = 0
    try:
        from app.astra_memory.retrieval import upsert_hot_index, RetrievalCost
        from app.lifestyle.models import DailySummary, FoodPreference

        recent = (
            db.query(DailySummary).order_by(DailySummary.date.desc()).limit(7).all()
        )
        for day in recent:
            bits = []
            if day.total_calories: bits.append(f"{day.total_calories:.0f} kcal")
            if day.total_protein_g: bits.append(f"{day.total_protein_g:.0f}g protein")
            if day.total_sugar_g: bits.append(f"{day.total_sugar_g:.0f}g sugar")
            if day.activity_mins: bits.append(f"{day.activity_mins}m {day.activity_types or 'activity'}")
            if day.weight_kg: bits.append(f"{day.weight_kg:.1f}kg")
            if day.sleep_minutes: bits.append(f"{day.sleep_minutes // 60}h{day.sleep_minutes % 60:02d} sleep")
            one_liner = ", ".join(bits) or "no data logged"
            upsert_hot_index(
                db=db, record_type="lifestyle_day", record_id=str(day.date),
                title=f"Health & food summary {day.date}", one_liner=one_liner,
                tags=["lifestyle"], entities=[],
                retrieval_priority=0.45, retrieval_cost=RetrievalCost.TINY,
            )
            indexed += 1

        strong_prefs = (
            db.query(FoodPreference)
            .filter(FoodPreference.confidence >= 0.6)
            .order_by(FoodPreference.confidence.desc()).limit(20).all()
        )
        for pref in strong_prefs:
            upsert_hot_index(
                db=db, record_type="food_pref", record_id=str(pref.id),
                title=f"Eating habit ({pref.category})", one_liner=pref.content[:200],
                tags=["lifestyle"], entities=[],
                retrieval_priority=0.55, retrieval_cost=RetrievalCost.TINY,
            )
            indexed += 1
    except Exception as exc:
        logger.warning("[habit_learner] hot-index bridge failed: %s", exc)
    return indexed


# =============================================================================
# Orchestrator
# =============================================================================

def run_habit_learning(db: Optional[Session] = None, days: int = 21) -> Dict[str, Any]:
    from app.db import SessionLocal
    close = False
    if db is None:
        db = SessionLocal()
        close = True
    try:
        analysis = analyse_recent_nutrition(db, days=days)
        pref_counts = write_habits(db, analysis)
        mirrored = mirror_to_tier2(db, analysis)
        indexed = index_lifestyle_to_hot(db)
        result = {
            "analysis": analysis,
            "food_prefs": pref_counts,
            "tier2_mirrored": mirrored,
            "hot_indexed": indexed,
            "ran_at": _now().isoformat(),
        }
        logger.info("[habit_learner] %s", {k: v for k, v in result.items() if k != "analysis"})
        return result
    finally:
        if close:
            db.close()


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run_habit_learning(), indent=2, default=str))
