# FILE: app/tools/lifestyle_tools_registration.py
# Purpose: Registration of the lifestyle health-coaching tools with the global tool registry.
# Called-by: app.tools.lifestyle_tools, app.tools.registry
# Depends-on: app.tools.lifestyle_tools, app.tools.registry, app.tools.schemas
# Last-renovated: 2026-06-21
"""
Registration of the lifestyle health-coaching tools (ToolDefinition records)
with the global tool registry. Split out of lifestyle_tools.py for the size
budget; lifestyle_tools re-exports register_lifestyle_tools so registry.py keeps
importing it from the usual place. The 8 handlers registered here are imported
from lifestyle_tools, which defines them all before its bottom-of-file re-export
of this module (no import cycle).
"""
from __future__ import annotations

import logging

from app.tools.lifestyle_tools import (
    get_recent_nutrition_handler, get_weight_trend_handler,
    get_recent_workouts_handler, log_nutrition_handler,
    log_workout_handler, log_weight_handler,
    set_goal_handler, set_deficit_handler,
)

logger = logging.getLogger(__name__)


def register_lifestyle_tools() -> None:
    """
    Register all lifestyle tools with the global tool registry.
    Called once from registry._register_defaults at module load.
    """
    from app.tools.registry import ToolDefinition, register_tool
    from app.tools.schemas import TOOL_SCHEMAS

    register_tool(ToolDefinition(
        name="get_recent_nutrition",
        version="v1",
        description=(
            "Return per-day nutrition summaries for the last N days "
            "(1-14, default 1 = today). Each day has totals, meal_count, "
            "and the full list of logs. Use this to read what the user "
            "has eaten before coaching on food, calories, or macros."
        ),
        input_schema=TOOL_SCHEMAS["get_recent_nutrition"]["input"],
        output_schema=TOOL_SCHEMAS["get_recent_nutrition"]["output"],
        handler=get_recent_nutrition_handler,
    ))

    register_tool(ToolDefinition(
        name="get_weight_trend",
        version="v1",
        description=(
            "Return the user's weight history over the last N days "
            "(7-365, default 30) with current weight, 7-day change, "
            "30-day change, and active weight target if set."
        ),
        input_schema=TOOL_SCHEMAS["get_weight_trend"]["input"],
        output_schema=TOOL_SCHEMAS["get_weight_trend"]["output"],
        handler=get_weight_trend_handler,
    ))

    register_tool(ToolDefinition(
        name="get_recent_workouts",
        version="v1",
        description=(
            "Return the most recent workout/activity sessions, newest "
            "first (1-50, default 10). Sessions include activity_type "
            "(gym, surf, stretch, walk, bodyboard, etc), duration, "
            "title, notes, calories burned, and surf-specific fields."
        ),
        input_schema=TOOL_SCHEMAS["get_recent_workouts"]["input"],
        output_schema=TOOL_SCHEMAS["get_recent_workouts"]["output"],
        handler=get_recent_workouts_handler,
    ))

    register_tool(ToolDefinition(
        name="log_nutrition",
        version="v1",
        description=(
            "Log a meal or food item. description is required (free "
            "text). meal_type defaults to 'meal'. Macro fields "
            "(calories, protein_g, carbs_g, fat_g, sugar_g) are "
            "optional — pass them when you know them, omit when you "
            "don't and the entry is flagged as an estimate. Don't re-log "
            "foods already in today's diary (check get_today_summary first) — "
            "re-logging the same items creates duplicate rows."
        ),
        input_schema=TOOL_SCHEMAS["log_nutrition"]["input"],
        output_schema=TOOL_SCHEMAS["log_nutrition"]["output"],
        handler=log_nutrition_handler,
    ))

    register_tool(ToolDefinition(
        name="log_workout",
        version="v1",
        description=(
            "Log a workout / activity session. activity_type is "
            "required and should be a short tag like 'gym', 'surf', "
            "'stretch', 'walk', 'bodyboard'. duration_mins, title, "
            "notes, calories_burned, surf_location, surf_conditions "
            "are all optional. Use this when the user describes a "
            "workout they did or are doing now."
        ),
        input_schema=TOOL_SCHEMAS["log_workout"]["input"],
        output_schema=TOOL_SCHEMAS["log_workout"]["output"],
        handler=log_workout_handler,
    ))

    register_tool(ToolDefinition(
        name="log_weight",
        version="v1",
        description=(
            "Log the user's body weight in kg when they tell you it "
            "(e.g. 'I'm 112 today'). Returns the change since their previous "
            "entry so you can coach on progress. weight_kg is required; notes "
            "optional."
        ),
        input_schema=TOOL_SCHEMAS["log_weight"]["input"],
        output_schema=TOOL_SCHEMAS["log_weight"]["output"],
        handler=log_weight_handler,
    ))

    register_tool(ToolDefinition(
        name="set_goal",
        version="v1",
        description=(
            "Set or update a lifestyle goal/target. goal_type is one of "
            "'weight', 'calories', 'protein', 'sugar_limit', 'activity'; "
            "target_value and unit are required (e.g. calories/2800/kcal, "
            "weight/95/kg). Replaces any existing active goal of that type."
        ),
        input_schema=TOOL_SCHEMAS["set_goal"]["input"],
        output_schema=TOOL_SCHEMAS["set_goal"]["output"],
        handler=set_goal_handler,
    ))

    register_tool(ToolDefinition(
        name="set_deficit",
        version="v1",
        description=(
            "Set the user's daily calorie DEFICIT in kcal -- the live lever that "
            "decides how hard their cut is. Use this when they want to change how "
            "aggressive the cut is (e.g. 'make my deficit 300 instead of 550'). It "
            "overrides the gentle/moderate/aggressive pace default and flows straight "
            "into today's intake target, live, with no code change. Discuss the number "
            "with them first (~300 is a gentle, sustainable cut that's easy to hold; "
            "~550 moderate). Pass deficit_kcal=0 (or null) to clear it and fall back to "
            "the pace default. Returns today's recomputed target so you can confirm it."
        ),
        input_schema=TOOL_SCHEMAS["set_deficit"]["input"],
        output_schema=TOOL_SCHEMAS["set_deficit"]["output"],
        handler=set_deficit_handler,
    ))

    logger.info("[lifestyle_tools] registered 8 health coaching tools")
