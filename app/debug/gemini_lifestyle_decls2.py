# FILE: app/debug/gemini_lifestyle_decls2.py
"""
Overflow lifestyle tool declarations (strength + meal-prep), split out of
gemini_lifestyle_decls.py to keep each file under the size ceiling.
gemini_lifestyle_decls.py imports MORE and appends it to the main list.
"""
from typing import Any, Dict, List

LIFESTYLE_TOOL_DECLARATIONS_MORE: List[Dict[str, Any]] = [
    {
        "name": "log_exercise",
        "description": (
            "Log strength-training sets \u2014 the weights and reps the user "
            "actually did. USE THIS when they say what they lifted, e.g. "
            "'squats, 3 sets of 5 at 100 kilos', 'benched 80 for 8, 8, 6', "
            "'did pull-ups, 3 sets of 10'. The sets appear in their Fitness "
            "tab workout log. One exercise per call; call again for the next "
            "exercise.\n"
            "\n"
            "Give exercise_name plus EITHER a `sets` list of {weight_kg, reps} "
            "(use this when sets differ, e.g. 80x8, 80x8, 80x6), OR a single "
            "weight_kg + reps with sets_count to repeat an identical set. "
            "Bodyweight moves: omit weight_kg. If they give pounds, convert to "
            "kg first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "exercise_name": {
                    "type": "string",
                    "description": "The exercise, e.g. 'squat', 'bench press', 'pull-up'.",
                },
                "sets": {
                    "type": "array",
                    "description": "List of sets, each {weight_kg, reps}. Use when sets differ.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "weight_kg": {"type": "number", "description": "Weight in kg (omit for bodyweight)."},
                            "reps": {"type": "integer", "description": "Reps in the set."},
                        },
                    },
                },
                "weight_kg": {"type": "number", "description": "Single-set form: weight in kg."},
                "reps": {"type": "integer", "description": "Single-set form: reps."},
                "sets_count": {"type": "integer", "description": "Single-set form: how many identical sets (default 1)."},
                "notes": {"type": "string", "description": "Optional note, e.g. 'felt easy', 'last set to failure'."},
            },
            "required": ["exercise_name"],
        },
    },
    {
        "name": "get_exercise_history",
        "description": (
            "Read the user's history for ONE exercise so you can coach "
            "progressive overload. Returns recent sessions (newest first) with "
            "the sets done each day, plus last top weight, all-time best "
            "weight, and estimated 1RM. USE THIS before suggesting what to lift "
            "\u2014 e.g. they say 'I'm doing legs, what should I squat?' \u2192 "
            "look up 'squat' and base the suggestion on what they did last "
            "time. exercise_name is required."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "exercise_name": {
                    "type": "string",
                    "description": "The exercise to look up, e.g. 'squat'.",
                },
                "limit_sessions": {
                    "type": "integer",
                    "description": "How many recent sessions to return, 1-20. Default 8.",
                },
            },
            "required": ["exercise_name"],
        },
    },
    {
        "name": "get_workout_log",
        "description": (
            "Read the user's day-by-day strength log across ALL exercises "
            "(newest day first), plus the list of exercises they've trained "
            "recently. USE THIS to answer 'what have I been training', 'what "
            "did I do this week', or when they ask 'what should I do today' and "
            "you want to see what they've hit lately and what's due. Returns "
            "each day with its exercises and sets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "How many days back to include, 1-120. Default 21.",
                },
            },
        },
    },
    {
        "name": "delete_exercise_set",
        "description": (
            "Delete a single logged set by its id (remove a mistake or "
            "duplicate). Get the id from get_workout_log or "
            "get_exercise_history."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "set_id": {
                    "type": "integer",
                    "description": "The set id to delete.",
                },
            },
            "required": ["set_id"],
        },
    },
    {
        "name": "prep_split",
        "description": (
            "Split a batch cook / meal prep across several days. USE THIS when "
            "the user says they've cooked a batch and want it divided forward "
            "\u2014 e.g. 'I made a big chicken curry, split it over the next 3 "
            "days', 'meal prepped 5 portions'. You give the WHOLE batch's total "
            "macros and the number of days; it divides evenly and writes one "
            "portion into each upcoming day's diary, so those days show the "
            "prepped meal ahead of time.\n"
            "\n"
            "FIRST get the total macros for everything they cooked \u2014 ask, "
            "or web_search to estimate the batch total (calories, protein, "
            "carbs, fat; sugar too if you can). Pass them in `total`. days = "
            "how many days to divide across. start_date is optional "
            "(YYYY-MM-DD); defaults to tomorrow. Do NOT also call log_nutrition "
            "for the same food \u2014 prep_split does the logging."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "What was cooked, e.g. 'Chicken curry with rice'.",
                },
                "days": {
                    "type": "integer",
                    "description": "How many days to split the batch across (1-14).",
                },
                "total": {
                    "type": "object",
                    "description": "The WHOLE batch's totals (not per-portion).",
                    "properties": {
                        "calories": {"type": "number", "description": "Total kcal for the whole batch"},
                        "protein_g": {"type": "number", "description": "Total protein (g) for the whole batch"},
                        "carbs_g": {"type": "number", "description": "Total carbs (g) for the whole batch"},
                        "fat_g": {"type": "number", "description": "Total fat (g) for the whole batch"},
                        "sugar_g": {"type": "number", "description": "Total sugar (g) for the whole batch"},
                    },
                },
                "start_date": {
                    "type": "string",
                    "description": "Optional first day, YYYY-MM-DD. Defaults to tomorrow.",
                },
                "meal_type": {
                    "type": "string",
                    "description": "snack | breakfast | lunch | dinner | meal (default).",
                },
            },
            "required": ["description", "days", "total"],
        },
    },
    {
        "name": "get_health_history",
        "description": (
            "Read the full daily health series over a date RANGE for long-arc "
            "trend analysis \u2014 use this whenever the user asks how something "
            "has changed over time, e.g. 'how's my resting heart rate trended "
            "this month', 'have I lost weight', 'how's my training volume been', "
            "'am I sleeping more'. Returns one row per day (weight, calories, "
            "protein/carbs/fat/sugar, activity minutes, steps, floors, active "
            "calories burnt, resting heart rate, sleep) PLUS a summary block "
            "with averages and the weight change across the window.\n"
            "\n"
            "Pass `days` for a rolling window back from today (e.g. 30, 90), or "
            "explicit start_date/end_date (YYYY-MM-DD). days is capped at 365. "
            "This is the read for trends; for just today's totals use "
            "get_today_summary instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Rolling window size in days back from today (default 30, max 365). Ignored if start_date/end_date are given.",
                },
                "start_date": {
                    "type": "string",
                    "description": "Optional range start, YYYY-MM-DD.",
                },
                "end_date": {
                    "type": "string",
                    "description": "Optional range end, YYYY-MM-DD (defaults to today).",
                },
            },
        },
    },
]
