# FILE: app/web_automation/register.py
# Purpose: Register web-automation tools with the central tool registry.
# Called-by: app.web_automation
# Depends-on: app.tools.registry, app.web_automation.coursera_reader, app.web_automation.coursera_study_tools, app.web_automation.tool_handlers, app.web_automation.tool_schemas
# Last-renovated: 2026-07-02
"""
Register web-automation tools with the central tool registry.

Called once from main.py startup.
"""
from __future__ import annotations

import logging

from app.tools.registry import ToolDefinition, register_tool
from app.web_automation.coursera_reader import (
    COURSERA_DESCRIPTIONS,
    COURSERA_HANDLERS,
    COURSERA_SCHEMAS,
)
from app.web_automation.coursera_study_tools import (
    COURSERA_STUDY_DESCRIPTIONS,
    COURSERA_STUDY_HANDLERS,
    COURSERA_STUDY_SCHEMAS,
)
from app.web_automation.tool_handlers import HANDLERS, TOOL_DESCRIPTIONS
from app.web_automation.tool_schemas import TOOL_SCHEMAS

logger = logging.getLogger(__name__)


def register_web_tools() -> int:
    """
    Register every handler in HANDLERS (+ the Coursera composites and the
    study-loop tools, which keep their definitions next to their logic in
    coursera_reader.py / coursera_study_tools.py) with the tool registry.
    Returns the number of tools registered. Idempotent: duplicate
    registrations overwrite the existing definition.
    """
    handlers = {**HANDLERS, **COURSERA_HANDLERS, **COURSERA_STUDY_HANDLERS}
    schemas_by_name = {**TOOL_SCHEMAS, **COURSERA_SCHEMAS, **COURSERA_STUDY_SCHEMAS}
    descriptions = {**TOOL_DESCRIPTIONS, **COURSERA_DESCRIPTIONS, **COURSERA_STUDY_DESCRIPTIONS}
    count = 0
    for name, handler in handlers.items():
        schemas = schemas_by_name[name]
        register_tool(
            ToolDefinition(
                name=name,
                version="v1",
                description=descriptions[name],
                input_schema=schemas["input"],
                output_schema=schemas["output"],
                handler=handler,
            )
        )
        count += 1
    logger.info("[web_automation] registered %d web automation tools", count)
    return count
