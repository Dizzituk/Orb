# FILE: app/lifestyle/coaching.py
"""
Coaching layer for the Lifestyle Engine: food preferences (soft / evolving /
tiered memory) and the single written meal plan.

Split out of service.py to keep that file under the size ceiling. service.py
re-exports these names, so existing `from app.lifestyle.service import
add_food_preference` imports and `service.add_food_preference(...)` attribute
access keep working unchanged.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.lifestyle.models import FoodPreference, MealPlan
from app.lifestyle.schemas import (
    FoodPreferenceOut, MealPlanOut,
)

logger = logging.getLogger(__name__)


def _naive_utc(dt):
    """Coerce any datetime to naive UTC so it compares safely with DB-stored
    (naive) datetimes (SQLite drops tzinfo on round-trip)."""
    if dt is None:
        return None
    if getattr(dt, "tzinfo", None) is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


_PREF_CATEGORIES = {
    "like", "dislike", "staple", "shop", "timing", "habit", "constraint", "note",
}
_PREF_STABILITIES = {"core", "evolving", "passing"}
_PREF_STALE_DAYS = 14


def _is_stale(p: FoodPreference) -> bool:
    """Passing items not reinforced in a while are 'stale' (shown faded, prunable)."""
    if (p.stability or "evolving") != "passing":
        return False
    seen = _naive_utc(p.last_seen or p.created_at)
    if not seen:
        return False
    return seen < _naive_utc(datetime.now(timezone.utc)) - timedelta(days=_PREF_STALE_DAYS)


def _pref_to_out(p: FoodPreference) -> FoodPreferenceOut:
    return FoodPreferenceOut(
        id=p.id,
        category=p.category or "note",
        content=p.content,
        stability=p.stability or "evolving",
        confidence=p.confidence if p.confidence is not None else 0.6,
        last_seen=p.last_seen.isoformat() if p.last_seen else None,
        stale=_is_stale(p),
        source=p.source or "chat",
        created_at=p.created_at.isoformat() if p.created_at else None,
        updated_at=p.updated_at.isoformat() if p.updated_at else None,
    )


def add_food_preference(db: Session, category: str, content: str,
                        source: str = "chat", stability: Optional[str] = None,
                        confidence: Optional[float] = None) -> FoodPreferenceOut:
    """Record a food preference/habit.

    Soft, not absolute: if the same thing is stated again it is *reinforced*
    (confidence rises, last_seen refreshed, stability upgraded if a stronger
    one is given) rather than duplicated. New items default to 'evolving'.
    """
    cat = (category or "note").strip().lower()
    if cat not in _PREF_CATEGORIES:
        cat = "note"
    text = (content or "").strip()
    stab = (stability or "").strip().lower()
    if stab not in _PREF_STABILITIES:
        stab = ""

    now = datetime.now(timezone.utc)
    existing = (
        db.query(FoodPreference)
        .filter(FoodPreference.category == cat,
                func.lower(FoodPreference.content) == text.lower())
        .first()
    )
    if existing:
        # Reinforce rather than duplicate.
        existing.confidence = min(1.0, (existing.confidence or 0.6) + 0.15)
        existing.last_seen = now
        existing.updated_at = now
        if stab:
            existing.stability = stab
        db.commit()
        db.refresh(existing)
        logger.info(f"[lifestyle] Food preference reinforced: [{cat}] {text[:60]}")
        return _pref_to_out(existing)

    pref = FoodPreference(
        category=cat, content=text, source=source,
        stability=stab or "evolving",
        confidence=confidence if confidence is not None else 0.6,
        last_seen=now,
    )
    db.add(pref)
    db.commit()
    db.refresh(pref)
    logger.info(f"[lifestyle] Food preference added: [{cat}] {text[:60]}")
    return _pref_to_out(pref)


def update_food_preference(db: Session, pref_id: int, *, content: Optional[str] = None,
                           category: Optional[str] = None, stability: Optional[str] = None,
                           confidence: Optional[float] = None) -> Optional[FoodPreferenceOut]:
    """Revise a preference in place so it can evolve instead of duplicating."""
    pref = db.query(FoodPreference).filter(FoodPreference.id == pref_id).first()
    if not pref:
        return None
    if content is not None and content.strip():
        pref.content = content.strip()
    if category is not None:
        cat = category.strip().lower()
        if cat in _PREF_CATEGORIES:
            pref.category = cat
    if stability is not None:
        stab = stability.strip().lower()
        if stab in _PREF_STABILITIES:
            pref.stability = stab
    if confidence is not None:
        try:
            pref.confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            pass
    now = datetime.now(timezone.utc)
    pref.last_seen = now
    pref.updated_at = now
    db.commit()
    db.refresh(pref)
    logger.info(f"[lifestyle] Food preference {pref_id} updated")
    return _pref_to_out(pref)


def get_food_preferences(db: Session) -> List[FoodPreferenceOut]:
    """All food preferences. Core first, then evolving, then passing; newest within."""
    order = {"core": 0, "evolving": 1, "passing": 2}
    prefs = db.query(FoodPreference).all()
    prefs.sort(key=lambda p: (
        order.get(p.stability or "evolving", 1),
        p.category or "note",
        -(p.created_at.timestamp() if p.created_at else 0),
    ))
    return [_pref_to_out(p) for p in prefs]


def delete_food_preference(db: Session, pref_id: int) -> bool:
    """Delete a food preference by id."""
    pref = db.query(FoodPreference).filter(FoodPreference.id == pref_id).first()
    if not pref:
        return False
    db.delete(pref)
    db.commit()
    logger.info(f"[lifestyle] Food preference {pref_id} deleted")
    return True


# ═════════════════════════════════════════
# COACHING: WRITTEN PLAN
# ═════════════════════════════════════════

def get_meal_plan(db: Session) -> MealPlanOut:
    """Return the single active plan (or an empty placeholder)."""
    plan = db.query(MealPlan).order_by(MealPlan.id.asc()).first()
    if not plan:
        return MealPlanOut(title="My Plan", body=None, updated_at=None, exists=False)
    return MealPlanOut(
        title=plan.title or "My Plan",
        body=plan.body,
        updated_at=plan.updated_at.isoformat() if plan.updated_at else None,
        exists=True,
    )


def set_meal_plan(db: Session, body: str, title: Optional[str] = None) -> MealPlanOut:
    """Create or update the single plan row (get-or-create)."""
    plan = db.query(MealPlan).order_by(MealPlan.id.asc()).first()
    if not plan:
        plan = MealPlan()
        db.add(plan)
    plan.body = body
    if title is not None:
        plan.title = title.strip() or "My Plan"
    plan.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(plan)
    logger.info("[lifestyle] Meal plan updated")
    return MealPlanOut(
        title=plan.title or "My Plan",
        body=plan.body,
        updated_at=plan.updated_at.isoformat() if plan.updated_at else None,
        exists=True,
    )

