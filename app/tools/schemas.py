# FILE: app/tools/schemas.py
"""
Tool schemas (v1) for Orb local tools.

These are lightweight JSONSchema-like dicts used for basic validation inside ToolExecutor.
No external jsonschema dependency.
"""

from __future__ import annotations

# -------------------------
# http_fetch v1
# -------------------------

HTTP_FETCH_INPUT_V1 = {
    "type": "object",
    "required": ["url"],
    "properties": {
        "url": {"type": "string", "minLength": 1, "maxLength": 2048},
        "method": {"type": "string", "minLength": 1, "maxLength": 10},
        "headers": {"type": "object"},
        "max_bytes": {"type": "integer", "minimum": 1, "maximum": 5_000_000},
    },
}

HTTP_FETCH_OUTPUT_V1 = {
    "type": "object",
    "required": ["ok", "url", "status_code", "final_url", "headers", "text", "truncated", "error"],
    "properties": {
        "ok": {"type": "boolean"},
        "url": {"type": "string"},
        "status_code": {"type": "integer"},
        "final_url": {"type": "string"},
        "headers": {"type": "object"},
        "text": {"type": "string"},
        "truncated": {"type": "boolean"},
        "error": {"type": "string"},
    },
}

# -------------------------
# web_search v1
# -------------------------

WEB_SEARCH_INPUT_V1 = {
    "type": "object",
    "required": ["query"],
    "properties": {
        "query": {"type": "string", "minLength": 1, "maxLength": 512},
        "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
    },
}

WEB_SEARCH_OUTPUT_V1 = {
    "type": "object",
    "required": ["query", "provider", "results"],
    "properties": {
        "query": {"type": "string"},
        "provider": {"type": "string"},
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["title", "url", "snippet"],
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "snippet": {"type": "string"},
                },
            },
        },
    },
}

# -------------------------
# weather v1 (optional / stub)
# -------------------------

WEATHER_INPUT_V1 = {
    "type": "object",
    "required": ["location"],
    "properties": {
        "location": {"type": "string", "minLength": 1, "maxLength": 256},
        "days": {"type": "integer", "minimum": 1, "maximum": 14},
    },
}

WEATHER_OUTPUT_V1 = {
    "type": "object",
    "required": ["ok", "location", "forecast", "error"],
    "properties": {
        "ok": {"type": "boolean"},
        "location": {"type": "string"},
        "forecast": {"type": "object"},
        "error": {"type": "string"},
    },
}

# -------------------------
# Lifestyle tools v1 (health coaching — added 2026-05-25)
# Five tools that let any ASTRA chat read and write nutrition,
# workout, and weight data via the existing lifestyle service layer.
# -------------------------

GET_RECENT_NUTRITION_INPUT_V1 = {
    "type": "object",
    "properties": {
        "days_back": {
            "type": "integer",
            "minimum": 1,
            "maximum": 14,
        },
    },
}

GET_RECENT_NUTRITION_OUTPUT_V1 = {
    "type": "object",
    "required": ["days"],
    "properties": {
        "days": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["date", "totals", "meal_count", "logs"],
                "properties": {
                    "date": {"type": "string"},
                    "totals": {"type": "object"},
                    "meal_count": {"type": "integer"},
                    "logs": {"type": "array"},
                },
            },
        },
    },
}

GET_WEIGHT_TREND_INPUT_V1 = {
    "type": "object",
    "properties": {
        "days": {"type": "integer", "minimum": 7, "maximum": 365},
    },
}

GET_WEIGHT_TREND_OUTPUT_V1 = {
    "type": "object",
    "required": ["points", "current_kg", "change_7d_kg", "change_30d_kg", "target_kg"],
    "properties": {
        "points": {"type": "array"},
        "current_kg": {"type": ["number", "null"]},
        "change_7d_kg": {"type": ["number", "null"]},
        "change_30d_kg": {"type": ["number", "null"]},
        "target_kg": {"type": ["number", "null"]},
    },
}

GET_RECENT_WORKOUTS_INPUT_V1 = {
    "type": "object",
    "properties": {
        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
    },
}

GET_RECENT_WORKOUTS_OUTPUT_V1 = {
    "type": "object",
    "required": ["sessions"],
    "properties": {
        "sessions": {"type": "array"},
    },
}

LOG_NUTRITION_INPUT_V1 = {
    "type": "object",
    "required": ["description"],
    "properties": {
        "description": {"type": "string", "minLength": 1, "maxLength": 500},
        "meal_type": {"type": "string", "maxLength": 40},
        "calories": {"type": ["number", "null"]},
        "protein_g": {"type": ["number", "null"]},
        "carbs_g": {"type": ["number", "null"]},
        "fat_g": {"type": ["number", "null"]},
        "sugar_g": {"type": ["number", "null"]},
    },
}

LOG_NUTRITION_OUTPUT_V1 = {
    "type": "object",
    "required": ["ok", "log_id", "description"],
    "properties": {
        "ok": {"type": "boolean"},
        "log_id": {"type": ["integer", "null"]},
        "description": {"type": "string"},
        "calories": {"type": ["number", "null"]},
        "protein_g": {"type": ["number", "null"]},
        "carbs_g": {"type": ["number", "null"]},
        "fat_g": {"type": ["number", "null"]},
        "sugar_g": {"type": ["number", "null"]},
        "error": {"type": "string"},
    },
}

LOG_WORKOUT_INPUT_V1 = {
    "type": "object",
    "required": ["activity_type"],
    "properties": {
        "activity_type": {"type": "string", "minLength": 1, "maxLength": 40},
        "duration_mins": {"type": ["integer", "null"], "minimum": 1, "maximum": 1440},
        "title": {"type": ["string", "null"], "maxLength": 200},
        "notes": {"type": ["string", "null"], "maxLength": 1000},
        "calories_burned": {"type": ["number", "null"]},
        "surf_location": {"type": ["string", "null"], "maxLength": 120},
        "surf_conditions": {"type": ["string", "null"], "maxLength": 200},
    },
}

LOG_WORKOUT_OUTPUT_V1 = {
    "type": "object",
    "required": ["ok", "session_id", "activity_type"],
    "properties": {
        "ok": {"type": "boolean"},
        "session_id": {"type": ["integer", "null"]},
        "activity_type": {"type": "string"},
        "duration_mins": {"type": ["integer", "null"]},
        "title": {"type": ["string", "null"]},
        "error": {"type": "string"},
    },
}

# -------------------------
# Lifestyle write tools v1 (weight + goals) — added 2026-05-31
# -------------------------

LOG_WEIGHT_INPUT_V1 = {
    "type": "object",
    "required": ["weight_kg"],
    "properties": {
        "weight_kg": {"type": "number", "minimum": 20, "maximum": 500},
        "notes": {"type": ["string", "null"], "maxLength": 300},
    },
}

LOG_WEIGHT_OUTPUT_V1 = {
    "type": "object",
    "required": ["ok", "entry_id", "weight_kg"],
    "properties": {
        "ok": {"type": "boolean"},
        "entry_id": {"type": ["integer", "null"]},
        "weight_kg": {"type": ["number", "null"]},
        "change_since_last_kg": {"type": ["number", "null"]},
        "previous_weight_kg": {"type": ["number", "null"]},
        "previous_date": {"type": ["string", "null"]},
        "error": {"type": "string"},
    },
}

SET_GOAL_INPUT_V1 = {
    "type": "object",
    "required": ["goal_type", "target_value", "unit"],
    "properties": {
        "goal_type": {"type": "string", "minLength": 1, "maxLength": 40},
        "target_value": {"type": "number"},
        "unit": {"type": "string", "minLength": 1, "maxLength": 30},
        "notes": {"type": ["string", "null"], "maxLength": 300},
    },
}

SET_GOAL_OUTPUT_V1 = {
    "type": "object",
    "required": ["ok", "goal_type", "target_value", "unit"],
    "properties": {
        "ok": {"type": "boolean"},
        "goal_id": {"type": ["integer", "null"]},
        "goal_type": {"type": "string"},
        "target_value": {"type": ["number", "null"]},
        "unit": {"type": "string"},
        "error": {"type": "string"},
    },
}

# -------------------------
# Strength tools v1 (per-exercise progressive overload) — added 2026-05-31
# -------------------------

LOG_EXERCISE_SET_INPUT_V1 = {
    "type": "object",
    "required": ["exercise_name"],
    "properties": {
        "exercise_name": {"type": "string", "minLength": 1, "maxLength": 80},
        "weight_kg": {"type": ["number", "null"], "minimum": 0, "maximum": 1000},
        "reps": {"type": ["integer", "null"], "minimum": 0, "maximum": 1000},
        "set_index": {"type": ["integer", "null"], "minimum": 1, "maximum": 50},
        "unit": {"type": "string", "maxLength": 4},
        "notes": {"type": ["string", "null"], "maxLength": 300},
    },
}

LOG_EXERCISE_SET_OUTPUT_V1 = {
    "type": "object",
    "required": ["ok", "set_id", "exercise_name"],
    "properties": {
        "ok": {"type": "boolean"},
        "set_id": {"type": ["integer", "null"]},
        "exercise_name": {"type": "string"},
        "weight_kg": {"type": ["number", "null"]},
        "reps": {"type": ["integer", "null"]},
        "error": {"type": "string"},
    },
}

GET_EXERCISE_HISTORY_INPUT_V1 = {
    "type": "object",
    "required": ["exercise_name"],
    "properties": {
        "exercise_name": {"type": "string", "minLength": 1, "maxLength": 80},
        "limit_sessions": {"type": "integer", "minimum": 1, "maximum": 30},
    },
}

GET_EXERCISE_HISTORY_OUTPUT_V1 = {
    "type": "object",
    "required": ["exercise_name", "sessions"],
    "properties": {
        "exercise_name": {"type": "string"},
        "sessions": {"type": "array"},
        "last_performed": {"type": ["string", "null"]},
        "last_top_weight_kg": {"type": ["number", "null"]},
        "best_weight_kg": {"type": ["number", "null"]},
        "best_est_1rm_kg": {"type": ["number", "null"]},
        "error": {"type": "string"},
    },
}

# -------------------------
# Convenience registry
# -------------------------

TOOL_SCHEMAS = {
    "http_fetch": {"input": HTTP_FETCH_INPUT_V1, "output": HTTP_FETCH_OUTPUT_V1},
    "web_search": {"input": WEB_SEARCH_INPUT_V1, "output": WEB_SEARCH_OUTPUT_V1},
    "weather": {"input": WEATHER_INPUT_V1, "output": WEATHER_OUTPUT_V1},
    "get_recent_nutrition": {
        "input": GET_RECENT_NUTRITION_INPUT_V1,
        "output": GET_RECENT_NUTRITION_OUTPUT_V1,
    },
    "get_weight_trend": {
        "input": GET_WEIGHT_TREND_INPUT_V1,
        "output": GET_WEIGHT_TREND_OUTPUT_V1,
    },
    "get_recent_workouts": {
        "input": GET_RECENT_WORKOUTS_INPUT_V1,
        "output": GET_RECENT_WORKOUTS_OUTPUT_V1,
    },
    "log_nutrition": {
        "input": LOG_NUTRITION_INPUT_V1,
        "output": LOG_NUTRITION_OUTPUT_V1,
    },
    "log_workout": {
        "input": LOG_WORKOUT_INPUT_V1,
        "output": LOG_WORKOUT_OUTPUT_V1,
    },
    "log_weight": {
        "input": LOG_WEIGHT_INPUT_V1,
        "output": LOG_WEIGHT_OUTPUT_V1,
    },
    "set_goal": {
        "input": SET_GOAL_INPUT_V1,
        "output": SET_GOAL_OUTPUT_V1,
    },
    "log_exercise_set": {
        "input": LOG_EXERCISE_SET_INPUT_V1,
        "output": LOG_EXERCISE_SET_OUTPUT_V1,
    },
    "get_exercise_history": {
        "input": GET_EXERCISE_HISTORY_INPUT_V1,
        "output": GET_EXERCISE_HISTORY_OUTPUT_V1,
    },
}
