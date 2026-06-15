# FILE: app/debug/gemini_lifestyle_decls2.py
# Purpose: Overflow lifestyle tool declarations (strength + meal-prep), split out of
# Called-by: app.debug.gemini_lifestyle_decls
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
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
        "name": "copy_nutrition_day",
        "description": (
            "Copy one day's ENTIRE food diary onto another day. USE THIS when "
            "the user says they're eating the same as another day or wants a "
            "day's food duplicated \u2014 e.g. 'copy yesterday's food into "
            "today', 'I'm eating the same as yesterday', 'duplicate today's "
            "meals for tomorrow', 'repeat Monday's food today'. It clones every "
            "logged item with its macros and meal times onto the target day "
            "and refreshes that day's totals. Items already on the target day "
            "are skipped, so it's safe to call even if some food is logged.\n"
            "\n"
            "source_date and target_date each accept YYYY-MM-DD or the words "
            "yesterday / today / tomorrow / a weekday name. Defaults are "
            "source=yesterday, target=today. Do NOT also call log_nutrition "
            "for the same items \u2014 this tool does the logging."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_date": {
                    "type": "string",
                    "description": "Day to copy FROM: YYYY-MM-DD, yesterday, today, tomorrow, or a weekday. Default yesterday.",
                },
                "target_date": {
                    "type": "string",
                    "description": "Day to copy ONTO: YYYY-MM-DD, yesterday, today, tomorrow, or a weekday. Default today.",
                },
            },
        },
    },
    {
        "name": "resolve_nutrition",
        "description": (
            "Look up a branded / ready-made food's nutrition ONLINE and log it, "
            "instead of guessing macros. USE THIS for packaged or restaurant "
            "items — e.g. 'Aldi Chicken, Tomato & Basil Topped Pasta', a ready "
            "meal, a protein bar, a named takeaway. It runs a ladder: your local "
            "product library → Open Food Facts (barcode or name) → a retailer "
            "product page you pass in `url` → a SIMILAR same-type item (flagged "
            "as an estimate) → and only if nothing fits does it ask you to get "
            "the macros. It logs ONE row with the right portion, source and "
            "micronutrients.\n"
            "\n"
            "Give `name` (and `brand` if known) or a `barcode`. Add `grams` for "
            "the portion. If you've web_searched and found the exact product "
            "page, pass its `url` so it can scrape the panel. For a multi-item "
            "meal call this once PER branded item. Set log=false to only look up "
            "without logging. If it returns found=false, ask the user."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Food/product name, e.g. 'Chicken, Tomato & Basil Topped Pasta'."},
                "brand": {"type": "string", "description": "Brand/retailer if known, e.g. 'Aldi'."},
                "barcode": {"type": "string", "description": "Barcode digits if you have them (e.g. read off a photo)."},
                "grams": {"type": "number", "description": "Portion in grams (the amount eaten)."},
                "url": {"type": "string", "description": "A retailer product-page URL you found via web_search, to scrape the nutrition panel."},
                "meal_type": {"type": "string", "description": "snack | breakfast | lunch | dinner | meal (default)."},
                "log": {"type": "boolean", "description": "Whether to log it (default true)."},
            },
        },
    },
    {
        "name": "save_recipe",
        "description": (
            "Remember a meal/dish the user cooks repeatedly, with its ingredients "
            "and typical volumes, so it can be logged later in one shot. USE THIS "
            "when the user describes or photographs the ingredients of a dish they "
            "make often — e.g. 'save this as my beef chilli', 'this is my usual "
            "overnight oats'. Pass a name and the ingredient list (each with grams "
            "and, where you know them, its macros). Later, log_recipe expands it "
            "into one row per ingredient automatically."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Recipe name, e.g. 'Beef chilli'."},
                "ingredients": {
                    "type": "array",
                    "description": "One object per ingredient.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Ingredient, e.g. '500g beef mince'."},
                            "grams": {"type": "number", "description": "Typical amount in grams."},
                            "calories": {"type": "number"},
                            "protein_g": {"type": "number"},
                            "carbs_g": {"type": "number"},
                            "fat_g": {"type": "number"},
                            "sugar_g": {"type": "number"},
                        },
                    },
                },
                "meal_type": {"type": "string", "description": "snack | breakfast | lunch | dinner | meal (default)."},
                "notes": {"type": "string"},
            },
            "required": ["name", "ingredients"],
        },
    },
    {
        "name": "log_recipe",
        "description": (
            "Log a previously-saved recipe in one go — it expands into one row "
            "per ingredient at the saved volumes. USE THIS when the user says "
            "they're eating a dish you've saved before — e.g. 'log my beef "
            "chilli', 'had my usual overnight oats', or after photographing the "
            "same ingredients again. Match by `name`, or pass `ingredient_names` "
            "to find the recipe by its ingredients. If nothing matches, save it "
            "first with save_recipe."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The saved recipe's name."},
                "ingredient_names": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Detected ingredient names, to match a recipe by its ingredients.",
                },
                "meal_type": {"type": "string", "description": "Override meal type; defaults to the recipe's."},
                "on_date": {"type": "string", "description": "Optional day YYYY-MM-DD; defaults to today."},
            },
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
    # ── Energy engine (2026-06-11) ── executors in gemini_energy_tools.py ──
    {
        "name": "set_day_effort",
        "description": (
            "Records ONLY how physically heavy the day felt (for calorie-burn "
            "estimation). effort is light | normal | heavy | very_heavy. "
            "THIS TOOL LOGS NO WORK DATA. Never use it for end-of-day logging: "
            "miles, odometer, parcels, collections, pay, earnings, fuel or "
            "expenses MUST be logged with finish_work_day / log_expense / "
            "reconcile_work_day — call those FIRST, then optionally this if the "
            "user also described effort or bulk."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "effort": {
                    "type": "string",
                    "description": "light | normal | heavy | very_heavy (or a natural synonym like easy, busy, brutal).",
                },
                "date": {
                    "type": "string",
                    "description": "Optional day being described, YYYY-MM-DD. Defaults to today.",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional short note, e.g. '4 heavy parcels, rest light'.",
                },
            },
            "required": ["effort"],
        },
    },
    {
        "name": "get_energy_today",
        "description": (
            "Today's energy picture from the context-aware engine: best-evidence "
            "burn estimate (Garmin active kcal, or steps priced by work-day "
            "context, or logged sessions), the dynamic intake target derived "
            "from the weekly deficit ledger, and the week's running balance. "
            "Use for 'how many calories have I burnt / can I eat today', "
            "'how's my week going'. For BMR/TDEE plan numbers use "
            "get_metabolic_summary instead."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
]
