# FILE: app/lifestyle/nutrition.py
"""
Nutrition estimation engine.

Handles:
- Text-based food description → estimated macros (using LLM)
- Common food database lookups (local, no API needed)
- Keto-awareness: flags high-carb items, suggests alternatives
- Sugar tracking with addiction management context

Future (phone bridge phase):
- Barcode scanning via Open Food Facts API
- Photo-based meal recognition via Gemini Vision
"""
import json
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════
# COMMON FOODS DATABASE (offline, no API)
# ═══════════════════════════════════════════
# Per 100g values — used for quick lookups before hitting LLM

COMMON_FOODS: Dict[str, Dict] = {
    # Proteins
    "chicken breast": {"calories": 165, "protein": 31, "carbs": 0, "fat": 3.6, "sugar": 0, "fibre": 0},
    "salmon fillet": {"calories": 208, "protein": 20, "carbs": 0, "fat": 13, "sugar": 0, "fibre": 0},
    "eggs": {"calories": 155, "protein": 13, "carbs": 1.1, "fat": 11, "sugar": 1.1, "fibre": 0},
    "tuna": {"calories": 132, "protein": 28, "carbs": 0, "fat": 1.3, "sugar": 0, "fibre": 0},
    "steak": {"calories": 271, "protein": 26, "carbs": 0, "fat": 18, "sugar": 0, "fibre": 0},
    "bacon": {"calories": 541, "protein": 37, "carbs": 1.4, "fat": 42, "sugar": 0, "fibre": 0},
    "prawns": {"calories": 99, "protein": 24, "carbs": 0.2, "fat": 0.3, "sugar": 0, "fibre": 0},
    "pork chop": {"calories": 231, "protein": 25, "carbs": 0, "fat": 14, "sugar": 0, "fibre": 0},

    # Carbs
    "white rice": {"calories": 130, "protein": 2.7, "carbs": 28, "fat": 0.3, "sugar": 0, "fibre": 0.4},
    "brown rice": {"calories": 112, "protein": 2.3, "carbs": 24, "fat": 0.8, "sugar": 0, "fibre": 1.8},
    "pasta": {"calories": 131, "protein": 5, "carbs": 25, "fat": 1.1, "sugar": 0.6, "fibre": 1.8},
    "bread": {"calories": 265, "protein": 9, "carbs": 49, "fat": 3.2, "sugar": 5, "fibre": 2.7},
    "potato": {"calories": 77, "protein": 2, "carbs": 17, "fat": 0.1, "sugar": 0.8, "fibre": 2.2},
    "sweet potato": {"calories": 86, "protein": 1.6, "carbs": 20, "fat": 0.1, "sugar": 4.2, "fibre": 3},

    # Vegetables
    "broccoli": {"calories": 34, "protein": 2.8, "carbs": 7, "fat": 0.4, "sugar": 1.7, "fibre": 2.6},
    "spinach": {"calories": 23, "protein": 2.9, "carbs": 3.6, "fat": 0.4, "sugar": 0.4, "fibre": 2.2},
    "avocado": {"calories": 160, "protein": 2, "carbs": 9, "fat": 15, "sugar": 0.7, "fibre": 7},
    "peppers": {"calories": 31, "protein": 1, "carbs": 6, "fat": 0.3, "sugar": 4.2, "fibre": 2.1},
    "onion": {"calories": 40, "protein": 1.1, "carbs": 9.3, "fat": 0.1, "sugar": 4.2, "fibre": 1.7},
    "mushrooms": {"calories": 22, "protein": 3.1, "carbs": 3.3, "fat": 0.3, "sugar": 2, "fibre": 1},

    # Fats & dairy
    "butter": {"calories": 717, "protein": 0.9, "carbs": 0.1, "fat": 81, "sugar": 0.1, "fibre": 0},
    "olive oil": {"calories": 884, "protein": 0, "carbs": 0, "fat": 100, "sugar": 0, "fibre": 0},
    "cheese": {"calories": 402, "protein": 25, "carbs": 1.3, "fat": 33, "sugar": 0.5, "fibre": 0},
    "cream": {"calories": 340, "protein": 2, "carbs": 3, "fat": 36, "sugar": 3, "fibre": 0},
    "greek yogurt": {"calories": 97, "protein": 9, "carbs": 3.6, "fat": 5, "sugar": 3.2, "fibre": 0},

    # Nuts & seeds (keto staples)
    "almonds": {"calories": 579, "protein": 21, "carbs": 22, "fat": 50, "sugar": 4.4, "fibre": 12},
    "peanut butter": {"calories": 588, "protein": 25, "carbs": 20, "fat": 50, "sugar": 9, "fibre": 6},
    "walnuts": {"calories": 654, "protein": 15, "carbs": 14, "fat": 65, "sugar": 2.6, "fibre": 6.7},

    # Common sugar items (for sugar addiction tracking)
    "chocolate bar": {"calories": 546, "protein": 5, "carbs": 60, "fat": 31, "sugar": 48, "fibre": 3},
    "coca cola": {"calories": 42, "protein": 0, "carbs": 11, "fat": 0, "sugar": 11, "fibre": 0},
    "biscuits": {"calories": 502, "protein": 6, "carbs": 64, "fat": 25, "sugar": 27, "fibre": 2},
    "cake": {"calories": 347, "protein": 4, "carbs": 52, "fat": 14, "sugar": 34, "fibre": 1},
    "sweets": {"calories": 381, "protein": 3, "carbs": 93, "fat": 1, "sugar": 70, "fibre": 0},
}


def lookup_common_food(food_name: str) -> Optional[Dict]:
    """
    Try to find a food in the common database.
    Returns per-100g macros or None if not found.
    """
    name_lower = food_name.strip().lower()

    # Direct match
    if name_lower in COMMON_FOODS:
        return COMMON_FOODS[name_lower]

    # Partial match (e.g. "grilled chicken breast" matches "chicken breast")
    for key, data in COMMON_FOODS.items():
        if key in name_lower or name_lower in key:
            return data

    return None


def is_keto_friendly(carbs_g: float, sugar_g: float = 0) -> bool:
    """Check if a food/meal fits keto macros (under 20g net carbs per meal)."""
    return carbs_g <= 20


def get_sugar_warning(sugar_g: float, daily_limit: float = 25) -> Optional[str]:
    """
    Generate a sugar warning message for addiction management.
    Returns None if within limits.
    """
    if sugar_g <= 0:
        return None

    if sugar_g > daily_limit:
        return f"⚠️ This item alone has {sugar_g:.0f}g sugar — over your {daily_limit:.0f}g daily limit."

    if sugar_g > daily_limit * 0.5:
        return f"🟡 {sugar_g:.0f}g sugar — that's over half your {daily_limit:.0f}g daily budget."

    return None


def build_estimation_prompt(description: str) -> str:
    """
    Build the LLM prompt for estimating macros from a text description.

    This will be called by the lifestyle intelligence layer when the common
    food database doesn't have a match.
    """
    return f"""Estimate the nutritional content of this meal/food as accurately as possible.
Assume standard UK portion sizes unless specified otherwise.

Food description: "{description}"

Respond with ONLY a JSON object, no other text:
{{
  "calories": <number>,
  "protein_g": <number>,
  "carbs_g": <number>,
  "fat_g": <number>,
  "fibre_g": <number>,
  "sugar_g": <number>,
  "confidence": <0.0-1.0>,
  "notes": "<brief explanation of your estimate>"
}}
"""
