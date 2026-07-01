# FILE: app/tools/reminder_tools_registration.py
# Purpose: Registration of the reminder tools (ToolDefinition records) with the global tool registry.
# Called-by: app.tools.registry
# Depends-on: app.tools.reminder_tools, app.tools.registry, app.tools.schemas
# Last-renovated: 2026-07-01
from __future__ import annotations

import logging

from app.tools.reminder_tools import (
    create_reminder_handler, list_reminders_handler, cancel_reminder_handler,
)

logger = logging.getLogger(__name__)


def register_reminder_tools() -> None:
    """
    Register the reminder tools with the global tool registry.
    Called once from registry._register_defaults at module load.
    """
    from app.tools.registry import ToolDefinition, register_tool
    from app.tools.schemas import TOOL_SCHEMAS

    register_tool(ToolDefinition(
        name="create_reminder",
        version="v1",
        description=(
            "Create a one-shot reminder that fires on BOTH desktop and phone at the "
            "given time. text is what to remind the user of; when is natural language "
            "('in 20 minutes', 'tomorrow 9am', 'friday 8am', '3pm', 'at 17:30') and is "
            "parsed by code, not estimated — always pass the user's own phrasing "
            "verbatim rather than converting it yourself."
        ),
        input_schema=TOOL_SCHEMAS["create_reminder"]["input"],
        output_schema=TOOL_SCHEMAS["create_reminder"]["output"],
        handler=create_reminder_handler,
    ))

    register_tool(ToolDefinition(
        name="list_reminders",
        version="v1",
        description=(
            "List upcoming (not yet due) and currently due-but-unacknowledged "
            "reminders, soonest first. Use this before cancelling a reminder "
            "referenced vaguely (e.g. 'the 3pm one') to resolve its id."
        ),
        input_schema=TOOL_SCHEMAS["list_reminders"]["input"],
        output_schema=TOOL_SCHEMAS["list_reminders"]["output"],
        handler=list_reminders_handler,
    ))

    register_tool(ToolDefinition(
        name="cancel_reminder",
        version="v1",
        description=(
            "Cancel a pending reminder by id. Call list_reminders first if the user "
            "referred to it by time/description rather than id."
        ),
        input_schema=TOOL_SCHEMAS["cancel_reminder"]["input"],
        output_schema=TOOL_SCHEMAS["cancel_reminder"]["output"],
        handler=cancel_reminder_handler,
    ))

    logger.info("[reminder_tools] registered 3 reminder tools")
