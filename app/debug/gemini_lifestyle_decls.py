# FILE: app/debug/gemini_lifestyle_tools.py
"""
Lifestyle tool declarations and executors for the Gemini multimodal
tool loop (app.debug.gemini_tool_loop).

Why this lives here, separate from app/tools/lifestyle_tools.py:
    The Gemini multimodal path (used for image-bearing turns) has its
    own native function-calling format and its own executor map. It
    cannot call into the generic app.tools registry directly because
    that registry uses JSON-Schema input/output validation and an async
    executor wrapper this code path doesn't go through.

    Rather than duplicate the BUSINESS logic, this module's executors
    delegate to the already-tested handlers in app.tools.lifestyle_tools,
    which call the lifestyle service layer. So the path is:

        Gemini -> _lifestyle_*_executor (here)
              -> handler (app.tools.lifestyle_tools)
              -> service (app.lifestyle.service)
              -> DB

    No duplicated business logic. This module is purely the adapter
    between Gemini's function-calling shape and the existing handlers.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Function declarations for Gemini
# ---------------------------------------------------------------------------
# These follow the same dict shape as the file-tool declarations in
# gemini_tool_loop.py — Gemini converts both via _convert_schema().

LIFESTYLE_TOOL_DECLARATIONS: List[Dict[str, Any]] = [
    {
        "name": "log_nutrition",
        "description": (
            "Log a meal, snack, drink, or food item into the user's "
            "lifestyle/nutrition database. USE THIS \u2014 NOT write_file \u2014 "
            "when the user wants to log what they're eating or have eaten. "
            "The entry appears on their nutrition dashboard immediately.\n"
            "\n"
            "HARD RULE \u2014 complete entries only. Only call this once you "
            "have ALL of: the portion/quantity, calories, protein_g, carbs_g "
            "and fat_g. If any are missing, do NOT call it yet \u2014 first ask "
            "the user for the portion and any label values, or use web_search "
            "to estimate the macros as closely as you can, THEN call it once "
            "with every value filled. Never log a partial entry and call again "
            "to refine it \u2014 that creates duplicate rows. Include sugar_g "
            "too whenever you can (it's tracked separately). The tool rejects "
            "incomplete calls, so gather first, then log once.\n"
            "\n"
            "ITEMISE \u2014 when the user ate several distinct foods in a meal "
            "(e.g. 'chicken, rice and broccoli'), pass them as the `items` "
            "array: ONE object per food, each with its OWN description and "
            "macros. Do NOT merge several foods into a single combined entry. "
            "Itemising lets the user see each food's macros and check your "
            "estimates. Use the single description+macros form only for a "
            "genuinely single item (e.g. one Snickers bar). meal_type applies "
            "to the whole call.\n"
            "\n"
            "meal_type: snack | breakfast | lunch | dinner | drink | "
            "supplement | meal (default)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": (
                        "Preferred for meals with multiple foods: one object "
                        "per food, each with its own description + calories, "
                        "protein_g, carbs_g, fat_g (and sugar_g if known). "
                        "Each is logged as its own row."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string", "description": "This food, e.g. 'grilled chicken breast (200g)'."},
                            "calories": {"type": "number", "description": "Calories in kcal"},
                            "protein_g": {"type": "number", "description": "Protein in grams"},
                            "carbs_g": {"type": "number", "description": "Carbohydrates in grams"},
                            "fat_g": {"type": "number", "description": "Fat in grams"},
                            "sugar_g": {"type": "number", "description": "Sugars in grams"},
                        },
                    },
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Free-text description of the food/drink, "
                        "e.g. 'Snickers bar (58.7g)' or "
                        "'home-made chicken salad with rice'"
                    ),
                },
                "meal_type": {
                    "type": "string",
                    "description": (
                        "snack | breakfast | lunch | dinner | drink "
                        "| supplement | meal (default)"
                    ),
                },
                "calories": {
                    "type": "number",
                    "description": "Calories in kcal",
                },
                "protein_g": {
                    "type": "number",
                    "description": "Protein in grams",
                },
                "carbs_g": {
                    "type": "number",
                    "description": "Total carbohydrates in grams",
                },
                "fat_g": {
                    "type": "number",
                    "description": "Total fat in grams",
                },
                "sugar_g": {
                    "type": "number",
                    "description": "Sugars in grams (subset of carbs)",
                },
            },
            "required": ["description"],
        },
    },
    {
        "name": "log_workout",
        "description": (
            "Log a workout, training session, or physical activity "
            "into the lifestyle database. USE THIS when the user "
            "describes an activity they've done or are doing — "
            "surf session, gym, walk, stretch, bodyboarding, etc.\n"
            "\n"
            "activity_type is required and should be one of the "
            "short tags the lifestyle UI recognises: gym, surf, "
            "stretch, walk, bodyboard, run, swim_pool, swim_open_water, "
            "yoga, hiit, calisthenics, paddle, other. Everything else "
            "is optional."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "activity_type": {
                    "type": "string",
                    "description": (
                        "Short activity tag: gym | surf | stretch | "
                        "walk | bodyboard | run | swim_pool | "
                        "swim_open_water | yoga | hiit | calisthenics "
                        "| paddle | other"
                    ),
                },
                "duration_mins": {
                    "type": "integer",
                    "description": "Duration in minutes",
                },
                "title": {
                    "type": "string",
                    "description": "Optional short title for the session",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional longer notes",
                },
                "calories_burned": {
                    "type": "number",
                    "description": "Estimated calories burned",
                },
                "surf_location": {
                    "type": "string",
                    "description": "Surf-specific: where the session was",
                },
                "surf_conditions": {
                    "type": "string",
                    "description": "Surf-specific: brief conditions note",
                },
            },
            "required": ["activity_type"],
        },
    },
    {
        "name": "log_weight",
        "description": (
            "Log the user's body weight in kilograms when they state it "
            "(e.g. 'I'm 112 today', 'I weighed 123 kg this morning'). USE "
            "THIS \u2014 NOT save_to_memory \u2014 to record a weight "
            "measurement. It writes to the lifestyle weight table the Health "
            "dashboard reads from, so the entry and the trend appear "
            "immediately. Returns the change since the previous entry so you "
            "can comment on progress.\n"
            "\n"
            "Required: weight_kg (number, kilograms). Optional: notes. If the "
            "user gives pounds or stone, convert to kilograms first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "weight_kg": {
                    "type": "number",
                    "description": "Body weight in kilograms.",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional note, e.g. 'morning, fasted'.",
                },
            },
            "required": ["weight_kg"],
        },
    },
    {
        "name": "set_goal",
        "description": (
            "Set or update one of the user's lifestyle targets. USE THIS "
            "when the user asks to change a goal \u2014 e.g. 'set my calorie "
            "target to 2800', 'my goal weight is 95 kg', 'cap my sugar at "
            "25 g'. Replaces any existing active goal of the same type and "
            "updates what the Health dashboard shows as the target.\n"
            "\n"
            "goal_type is one of: weight, calories, protein, sugar_limit, "
            "activity. target_value (number) and unit are required. Typical "
            "units: calories -> kcal, weight -> kg, protein -> g, "
            "sugar_limit -> g, activity -> sessions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal_type": {
                    "type": "string",
                    "description": (
                        "weight | calories | protein | sugar_limit | activity"
                    ),
                },
                "target_value": {
                    "type": "number",
                    "description": "The numeric target.",
                },
                "unit": {
                    "type": "string",
                    "description": "Unit for the target (kcal, kg, g, sessions).",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional note about the goal.",
                },
            },
            "required": ["goal_type", "target_value", "unit"],
        },
    },
    {
        "name": "get_recent_nutrition",
        "description": (
            "Read the user's recent nutrition logs. Returns per-day "
            "totals (calories, protein, carbs, fat, sugar), meal "
            "count, and full log entries. Use this BEFORE giving "
            "advice on food or before answering 'what have I eaten' "
            "questions, so your reply is grounded in the actual data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "days_back": {
                    "type": "integer",
                    "description": (
                        "How many days of history to return, 1-14. "
                        "Default 1 (today only)."
                    ),
                },
            },
        },
    },
    {
        "name": "get_weight_trend",
        "description": (
            "Read the user's weight history and trend deltas. "
            "Returns the weight points over the requested period "
            "plus current weight, 7-day change, 30-day change, and "
            "active weight target if set. Use before coaching on "
            "weight or body composition."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": (
                        "How many days of history to return, 7-365. "
                        "Default 30."
                    ),
                },
            },
        },
    },
    {
        "name": "get_recent_workouts",
        "description": (
            "Read the user's most recent workout sessions, newest "
            "first. Each session includes activity_type, duration, "
            "title, notes, calories, and surf-specific fields. Use "
            "before coaching on training or answering 'when did I "
            "last train' questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max sessions to return, 1-50. Default 10.",
                },
            },
        },
    },
    {
        "name": "set_profile",
        "description": (
            "Set or update the user's body profile used to work out BMR and "
            "daily calorie burn: height, sex, age (or date of birth), activity "
            "level, and weight-loss pace. Call this when the user gives any of "
            "these (e.g. 'I'm 6 foot 2', 'I'm 41', 'I'm fairly active', 'take "
            "it steady'). Only the fields you pass are changed. After updating, "
            "call get_metabolic_summary to read the numbers back.\n"
            "\n"
            "height_cm in centimetres (6'2\" = 188). sex is 'male' or 'female'. "
            "Give age_years OR dob (YYYY-MM-DD); dob is better as it stays "
            "current. activity_level: sedentary | light | moderate | active | "
            "very_active. goal_pace: gentle | moderate | aggressive (how fast "
            "to lose - moderate is about 0.5 kg/week and the easiest to hold)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "height_cm": {"type": "number", "description": "Height in centimetres."},
                "sex": {"type": "string", "description": "male | female (for the BMR formula)."},
                "age_years": {"type": "integer", "description": "Age in years (use if no dob)."},
                "dob": {"type": "string", "description": "Date of birth YYYY-MM-DD (preferred over age)."},
                "activity_level": {
                    "type": "string",
                    "description": "sedentary | light | moderate | active | very_active",
                },
                "goal_pace": {
                    "type": "string",
                    "description": "gentle | moderate | aggressive (weight-loss pace)",
                },
            },
        },
    },
    {
        "name": "get_metabolic_summary",
        "description": (
            "Read the user's metabolic picture: BMR (resting burn), TDEE "
            "(total daily burn - measured from wearable active calories when "
            "available, otherwise estimated from activity level), plus a "
            "sustainable recommended calorie target and protein target based "
            "on their weight goal. Use this BEFORE advising how much to eat, "
            "setting a calorie target, or coaching toward a weight goal, so the "
            "advice is built on their real numbers. If the profile is "
            "incomplete it returns which inputs are missing - ask for those and "
            "call set_profile. Takes no parameters."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "delete_nutrition",
        "description": (
            "Delete a single nutrition log entry by its id. Use this to remove "
            "a duplicate or a wrong entry the user wants gone (e.g. 'delete the "
            "duplicate pork bites', 'remove that last food log'). Get the id "
            "from get_recent_nutrition, which lists each entry's id. Deleting "
            "recomputes the day's totals. To remove several duplicates, call "
            "this once per id."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "log_id": {
                    "type": "integer",
                    "description": "The nutrition log id to delete (from get_recent_nutrition).",
                },
            },
            "required": ["log_id"],
        },
    },
    {
        "name": "add_food_preference",
        "description": (
            "Record or reinforce one thing about how the user eats, so the "
            "coach builds a picture over time. PREFER asking the user and "
            "logging what they say over leaving it blank \u2014 they shouldn't "
            "have to fill anything in. Call this whenever they reveal a "
            "preference or habit \u2014 'I shop at Aldi', 'I love spicy food', "
            "'I can't stand olives', 'I usually skip breakfast'. One fact per "
            "call. This is SOFT memory, not a fixed record: if they restate or "
            "refine something, just call this again with the same content and "
            "it's reinforced (confidence rises) rather than duplicated; use "
            "update_food_preference to change wording or tier.\n"
            "\n"
            "category: like | dislike | staple (keeps in) | shop (where they "
            "buy) | timing (when they eat) | habit | constraint (allergy/diet "
            "rule) | note. stability: core (rarely changes \u2014 allergies, "
            "diet rules, regular shop), evolving (default \u2014 current tastes "
            "and habits), passing (short-term, fades if not repeated). content: "
            "the fact in a short phrase."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "like | dislike | staple | shop | timing | habit | constraint | note",
                },
                "content": {
                    "type": "string",
                    "description": "The preference/habit in a short phrase, e.g. 'Shops at Aldi'.",
                },
                "stability": {
                    "type": "string",
                    "description": "core | evolving (default) | passing \u2014 how settled this is.",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "get_food_preferences",
        "description": (
            "Read everything the coach knows about how the user eats \u2014 "
            "likes, dislikes, staples, where they shop, meal timing, habits and "
            "constraints. Use this BEFORE building a plan or suggesting what to "
            "eat, so advice fits what they actually like and buy. No parameters."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "delete_food_preference",
        "description": (
            "Remove a food preference/habit by its id (use when the user says "
            "it's wrong or no longer true). Get the id from get_food_preferences."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pref_id": {
                    "type": "integer",
                    "description": "The preference id to delete (from get_food_preferences).",
                },
            },
            "required": ["pref_id"],
        },
    },
    {
        "name": "update_food_preference",
        "description": (
            "Revise an existing preference in place so it can evolve instead of "
            "piling up duplicates. Use when the user changes or refines "
            "something already known, or to move an item between tiers (e.g. "
            "lock it as core, or mark it passing). Get the id from "
            "get_food_preferences. Only pass the fields you want changed.\n"
            "\n"
            "stability: core | evolving | passing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pref_id": {
                    "type": "integer",
                    "description": "The preference id to update (from get_food_preferences).",
                },
                "content": {"type": "string", "description": "New wording for the preference."},
                "category": {"type": "string", "description": "New category, if it changed."},
                "stability": {"type": "string", "description": "core | evolving | passing."},
            },
            "required": ["pref_id"],
        },
    },
    {
        "name": "set_meal_plan",
        "description": (
            "Write or update the user's visible written plan \u2014 the "
            "nutrition/activity plan of action shown in the Plan tab. Use when "
            "the user asks you to work out or change a plan. FIRST gather what "
            "you need: call get_metabolic_summary (BMR, daily burn, calorie + "
            "protein targets), get_food_preferences (what they like and buy), "
            "and get_recent_nutrition if useful; THEN write a clear, practical "
            "plan that fits their numbers and tastes. If you don't yet know "
            "their key eating details \u2014 where they shop, what they usually "
            "buy, foods they avoid, when they eat \u2014 ASK a couple of quick "
            "questions and log the answers with add_food_preference first; "
            "don't expect them to have filled anything in. Plain text with "
            "simple headings and lines is fine. Calling this replaces the whole "
            "plan, so include everything you want kept.\n"
            "\n"
            "body: the full plan text. title: optional short title."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "body": {
                    "type": "string",
                    "description": "The full plan text (plain text / light markdown).",
                },
                "title": {
                    "type": "string",
                    "description": "Optional short title, e.g. 'Cut to 105kg - high protein'.",
                },
            },
            "required": ["body"],
        },
    },
    {
        "name": "get_meal_plan",
        "description": (
            "Read the user's current written plan. Use before updating it so "
            "you build on what's there rather than overwriting blindly. No "
            "parameters."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_today_summary",
        "description": (
            "Where the user is at TODAY: calories and macros eaten so far "
            "against their targets, how much is left, and current weight. Use "
            "this to answer 'how many calories have I had', 'what's my protein "
            "at', or 'I'm peckish, what can I eat' \u2014 check what's left "
            "before suggesting food. No parameters."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
]

from app.debug.gemini_lifestyle_decls2 import LIFESTYLE_TOOL_DECLARATIONS_MORE  # noqa: E402
LIFESTYLE_TOOL_DECLARATIONS.extend(LIFESTYLE_TOOL_DECLARATIONS_MORE)
