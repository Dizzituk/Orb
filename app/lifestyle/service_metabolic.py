# FILE: app/lifestyle/service_metabolic.py
# Purpose: Body profile + BMR/TDEE metabolic summary (calorie/protein/macro targets).
# Called-by: app.lifestyle.service (re-export facade)
# Depends-on: app.lifestyle.models, app.lifestyle.schemas, app.lifestyle.service_activity
# Last-renovated: 2026-06-21
"""
Metabolic layer for the Lifestyle Engine: the single body-profile row and the
metabolic summary that turns profile + latest weight + rolling burn + weight
goal into BMR / TDEE / sustainable calorie + protein + macro targets (via
metabolic.py and burn.py). Split out of service.py; the facade re-exports every
public name here for back-compat. The active-goal lookup lives in
service_activity and is imported from there.
"""
import logging
from datetime import datetime, date, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.lifestyle.models import LifestyleProfile, WeightEntry
from app.lifestyle.schemas import ProfileOut, MetabolicSummary

from app.lifestyle.service_activity import _get_active_goal_value

logger = logging.getLogger(__name__)


def _profile_to_out(p: LifestyleProfile) -> ProfileOut:





    return ProfileOut(





        sex=p.sex,





        height_cm=p.height_cm,





        dob=p.dob.isoformat() if p.dob else None,





        age_years=p.age_years,





        activity_level=p.activity_level or "moderate",





        goal_pace=p.goal_pace or "moderate",





    )


def get_profile(db: Session) -> Optional[ProfileOut]:





    """Return the single profile row, or None if it hasn't been set up yet."""





    p = db.query(LifestyleProfile).order_by(LifestyleProfile.id.asc()).first()





    return _profile_to_out(p) if p else None


def upsert_profile(db: Session, *, sex: Optional[str] = None,





                   height_cm: Optional[float] = None, dob: Optional[str] = None,





                   age_years: Optional[int] = None,





                   activity_level: Optional[str] = None,





                   goal_pace: Optional[str] = None) -> ProfileOut:





    """Create or partially update the single profile row. Only set fields change."""





    p = db.query(LifestyleProfile).order_by(LifestyleProfile.id.asc()).first()





    if not p:





        p = LifestyleProfile()





        db.add(p)











    if sex is not None:





        p.sex = sex.strip().lower()





    if height_cm is not None:





        p.height_cm = float(height_cm)





    if dob is not None:





        p.dob = _parse_date(dob)





    if age_years is not None:





        p.age_years = int(age_years)





    if activity_level is not None:





        p.activity_level = activity_level.strip().lower()





    if goal_pace is not None:





        p.goal_pace = goal_pace.strip().lower()





    p.updated_at = datetime.now(timezone.utc)











    db.commit()





    db.refresh(p)





    logger.info("[lifestyle] Profile updated")





    return _profile_to_out(p)


def _parse_date(value: str):





    """Parse an ISO date string (YYYY-MM-DD) to a date; None on failure."""





    try:





        return datetime.strptime(value.strip()[:10], "%Y-%m-%d").date()





    except (ValueError, AttributeError):





        return None


def _effective_age(p: LifestyleProfile) -> Optional[float]:





    """Age in years from dob if present, else the stored age_years."""





    if p.dob:





        today = date.today()





        years = today.year - p.dob.year - (





            (today.month, today.day) < (p.dob.month, p.dob.day)





        )





        return float(years)





    return float(p.age_years) if p.age_years else None


def get_metabolic_summary(db: Session) -> MetabolicSummary:





    """





    Gather profile + latest weight + recent wearable burn + weight goal, then





    compute BMR / TDEE / sustainable calorie + protein targets via metabolic.py.





    """





    from app.lifestyle import metabolic











    p = db.query(LifestyleProfile).order_by(LifestyleProfile.id.asc()).first()











    latest_weight = (





        db.query(WeightEntry).order_by(WeightEntry.recorded_at.desc()).first()





    )





    weight_kg = latest_weight.weight_kg if latest_weight else None





    goal_weight = _get_active_goal_value(db, "weight")











    # Daily burn from a rolling average of real active calories (Garmin +





    # hand-logged workouts), only trusted past a minimum sample. See burn.py.





    from app.lifestyle.burn import compute_burn_signal





    burn = compute_burn_signal(db)





    active_cals = burn.avg_active_calories if burn.trusted else None











    plan = metabolic.build_plan(





        weight_kg=weight_kg,





        height_cm=p.height_cm if p else None,





        age_years=_effective_age(p) if p else None,





        sex=p.sex if p else None,





        activity_level=(p.activity_level if p else None) or metabolic.DEFAULT_ACTIVITY,





        pace=(p.goal_pace if p else None) or metabolic.DEFAULT_PACE,





        active_calories=active_cals,





        goal_weight_kg=goal_weight,





    )











    height_cm = p.height_cm if p else None





    bmi = None





    bmi_category = None





    if weight_kg and height_cm:





        h_m = height_cm / 100.0





        if h_m > 0:





            bmi = round(weight_kg / (h_m * h_m), 1)





            bmi_category = _bmi_category(bmi)











    return MetabolicSummary(





        complete=plan.get("complete", False),





        missing=plan.get("missing", []),





        bmr=plan.get("bmr"),





        tdee=plan.get("tdee"),





        tdee_source=burn.source_label,





        activity_level=plan.get("activity_level"),





        active_calories_used=plan.get("active_calories_used"),





        target_calories=plan.get("target_calories"),





        deficit_kcal=plan.get("deficit_kcal"),





        floored_at_bmr=plan.get("floored_at_bmr", False),





        pace=plan.get("pace"),





        protein_g=plan.get("protein_g"),





        carbs_g=plan.get("carbs_g"),





        fat_g=plan.get("fat_g"),





        current_weight_kg=plan.get("current_weight_kg"),





        goal_weight_kg=plan.get("goal_weight_kg"),





        height_cm=height_cm,





        bmi=bmi,





        bmi_category=bmi_category,





    )


def _bmi_category(bmi: float) -> str:





    if bmi < 18.5:





        return "underweight"





    if bmi < 25:





        return "healthy"





    if bmi < 30:





        return "overweight"





    return "obese"
