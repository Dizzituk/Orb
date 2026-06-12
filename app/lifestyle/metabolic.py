# FILE: app/lifestyle/metabolic.py
# Purpose: Metabolic maths for the Lifestyle Engine — BMR, daily burn (TDEE), and
# Called-by: app.lifestyle.service
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Metabolic maths for the Lifestyle Engine — BMR, daily burn (TDEE), and
sustainable calorie / protein targets.

Pure functions, no database. The service layer (service.get_metabolic_summary)
gathers the inputs — profile, latest weight, recent wearable active-calories,
active weight goal — and calls build_plan() here. Keeping the maths separate
makes it trivially testable and keeps the file small.

BMR uses Mifflin-St Jeor, the most accurate of the common predictive equations:
    male:   10*kg + 6.25*cm - 5*age + 5
    female: 10*kg + 6.25*cm - 5*age - 161

Daily burn (TDEE) has two routes:
    estimated — BMR * activity factor (used when there's no wearable data)
    measured  — BMR + measured active calories from the wearable (preferred
                once Garmin / Health Connect data is flowing, because it
                reflects the actual day rather than a guessed multiplier)
"""
from __future__ import annotations

from typing import Optional, Dict, Any


# Activity multipliers for estimated TDEE (Mifflin-St Jeor convention).
ACTIVITY_FACTORS: Dict[str, float] = {
    "sedentary": 1.2,      # desk-bound, little movement
    "light": 1.375,        # light activity / 1-3 sessions a week
    "moderate": 1.55,      # moderate / 3-5 sessions a week
    "active": 1.725,       # hard exercise or a physical job
    "very_active": 1.9,    # very hard exercise + physical job
}
DEFAULT_ACTIVITY = "moderate"

# Daily calorie deficit (kcal) per loss pace. ~7700 kcal per kg of fat, so
# moderate ~= 0.5 kg/week. Deliberately conservative — a smaller deficit is
# far easier to hold and keeps hunger down, which is the whole point.
PACE_DEFICITS: Dict[str, int] = {
    "gentle": 300,      # ~0.27 kg/week
    "moderate": 550,    # ~0.5 kg/week
    "aggressive": 800,  # ~0.73 kg/week
}
DEFAULT_PACE = "moderate"

# Protein target: grams per kg of reference bodyweight. High end of the
# evidence range because protein is the strongest lever for satiety and for
# holding muscle in a deficit — directly serves "I don't want to be hungry".
PROTEIN_G_PER_KG = 1.8


def compute_bmr(weight_kg: float, height_cm: float, age_years: float,
                sex: str) -> Optional[float]:
    """Mifflin-St Jeor BMR in kcal/day. Returns None if any input is missing."""
    if not weight_kg or not height_cm or not age_years or not sex:
        return None
    base = 10.0 * weight_kg + 6.25 * height_cm - 5.0 * age_years
    s = (sex or "").strip().lower()
    if s in ("m", "male", "man"):
        bmr = base + 5
    elif s in ("f", "female", "woman"):
        bmr = base - 161
    else:
        return None
    return round(bmr, 0)


def estimated_tdee(bmr: float, activity_level: str = DEFAULT_ACTIVITY) -> float:
    """BMR * activity factor."""
    factor = ACTIVITY_FACTORS.get(
        (activity_level or "").strip().lower(), ACTIVITY_FACTORS[DEFAULT_ACTIVITY]
    )
    return round(bmr * factor, 0)


def measured_tdee(bmr: float, active_calories: float) -> float:
    """BMR + measured active calories from the wearable (the actual day)."""
    return round(bmr + max(0.0, active_calories), 0)


def recommend_calories(tdee: float, bmr: float,
                       pace: str = DEFAULT_PACE) -> Dict[str, Any]:
    """
    Sustainable daily calorie target = TDEE - deficit, floored at BMR.

    Never recommends eating below BMR (the resting requirement); if the deficit
    would push below it, the target is raised back to BMR and floored_at_bmr is
    flagged so the caller can suggest a gentler pace or more movement instead.
    """
    deficit = PACE_DEFICITS.get(
        (pace or "").strip().lower(), PACE_DEFICITS[DEFAULT_PACE]
    )
    raw = tdee - deficit
    floored = raw < bmr
    target = max(raw, bmr)
    return {
        "target_calories": round(target, 0),
        "deficit_kcal": int(round(tdee - target)),
        "requested_deficit_kcal": deficit,
        "floored_at_bmr": floored,
        "pace": (pace or DEFAULT_PACE).strip().lower(),
    }


def recommend_protein_g(reference_weight_kg: float) -> float:
    """Daily protein target in grams for satiety + muscle retention."""
    if not reference_weight_kg:
        return 0.0
    return round(PROTEIN_G_PER_KG * reference_weight_kg, 0)


# Fraction of total daily calories allocated to fat. 30% is a balanced,
# sustainable level — comfortably above the ~20% floor needed for hormones
# and fat-soluble vitamin absorption, and nowhere near a ketogenic split.
# Protein is fixed by bodyweight; fat takes this share; carbs fill the rest.
FAT_CALORIE_FRACTION = 0.30

# Calories per gram, for converting between grams and kcal.
KCAL_PER_G_PROTEIN = 4.0
KCAL_PER_G_CARB = 4.0
KCAL_PER_G_FAT = 9.0


def recommend_macros(target_calories: float, protein_g: float) -> Dict[str, Any]:
    """Split the calorie target into a balanced protein/carb/fat gram target.

    Protein is already fixed (bodyweight-driven). Fat takes FAT_CALORIE_FRACTION
    of total calories. Carbs fill whatever calories remain. This yields a
    balanced mix (typically ~40/30/30 cals carb/protein/fat for a cut), never
    keto. Carbs are floored at 0 so a very high protein target can't push them
    negative.
    """
    if not target_calories or target_calories <= 0:
        return {"carbs_g": None, "fat_g": None}
    protein_cals = (protein_g or 0) * KCAL_PER_G_PROTEIN
    fat_cals = target_calories * FAT_CALORIE_FRACTION
    carb_cals = max(0.0, target_calories - protein_cals - fat_cals)
    return {
        "carbs_g": round(carb_cals / KCAL_PER_G_CARB, 0),
        "fat_g": round(fat_cals / KCAL_PER_G_FAT, 0),
    }


def build_plan(
    *,
    weight_kg: Optional[float],
    height_cm: Optional[float],
    age_years: Optional[float],
    sex: Optional[str],
    activity_level: str = DEFAULT_ACTIVITY,
    pace: str = DEFAULT_PACE,
    active_calories: Optional[float] = None,
    goal_weight_kg: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Tie the maths together into one structured plan. Reports clearly what's
    missing rather than guessing, so the coach can ask for the gap.
    """
    missing = []
    if not weight_kg:
        missing.append("weight")
    if not height_cm:
        missing.append("height")
    if not age_years:
        missing.append("age")
    if not sex:
        missing.append("sex")

    bmr = compute_bmr(weight_kg, height_cm, age_years, sex) if not missing else None
    if bmr is None:
        return {
            "ok": True,
            "complete": False,
            "missing": missing or ["sex"],
            "bmr": None,
            "tdee": None,
            "tdee_source": None,
            "target_calories": None,
            "protein_g": None,
            "current_weight_kg": weight_kg,
            "goal_weight_kg": goal_weight_kg,
        }

    if active_calories is not None and active_calories > 0:
        tdee = measured_tdee(bmr, active_calories)
        tdee_source = "measured"
    else:
        tdee = estimated_tdee(bmr, activity_level)
        tdee_source = "estimated"

    # Reference weight for protein: the goal weight when cutting toward one
    # (avoids over-counting protein against a high starting bodyweight), else
    # current weight.
    ref_weight = goal_weight_kg if (goal_weight_kg and goal_weight_kg < weight_kg) else weight_kg

    cals = recommend_calories(tdee, bmr, pace)
    protein = recommend_protein_g(ref_weight)
    macros = recommend_macros(cals["target_calories"], protein)

    return {
        "ok": True,
        "complete": True,
        "missing": [],
        "bmr": bmr,
        "tdee": tdee,
        "tdee_source": tdee_source,
        "activity_level": (activity_level or DEFAULT_ACTIVITY).strip().lower(),
        "active_calories_used": round(active_calories, 0) if active_calories else None,
        "target_calories": cals["target_calories"],
        "deficit_kcal": cals["deficit_kcal"],
        "floored_at_bmr": cals["floored_at_bmr"],
        "pace": cals["pace"],
        "protein_g": protein,
        "carbs_g": macros["carbs_g"],
        "fat_g": macros["fat_g"],
        "protein_basis_kg": ref_weight,
        "goal_weight_kg": goal_weight_kg,
        "current_weight_kg": weight_kg,
    }
