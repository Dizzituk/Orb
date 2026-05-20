# FILE: app/web_automation/register.py
"""
Register web-automation tools with the central tool registry.

Called once from main.py startup.
"""
from __future__ import annotations

import logging

from app.tools.registry import ToolDefinition, register_tool
from app.web_automation.tool_handlers import HANDLERS, TOOL_DESCRIPTIONS
from app.web_automation.tool_schemas import TOOL_SCHEMAS

logger = logging.getLogger(__name__)


def register_web_tools() -> int:
    """
    Register every handler in HANDLERS with the tool registry.
    Returns the number of tools registered.
    Idempotent: duplicate registrations overwrite the existing definition.
    """
    count = 0
    for name, handler in HANDLERS.items():
        schemas = TOOL_SCHEMAS[name]
        register_tool(
            ToolDefinition(
                name=name,
                version="v1",
                description=TOOL_DESCRIPTIONS[name],
                input_schema=schemas["input"],
                output_schema=schemas["output"],
                handler=handler,
            )
        )
        count += 1
    logger.info("[web_automation] registered %d web automation tools", count)
    return count
