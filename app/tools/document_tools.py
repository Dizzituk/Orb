# FILE: app/tools/document_tools.py
# Purpose: Editor chat tools — read/edit the document open in the desktop
#          editor pane ("put the word total in A10", "read me the first
#          paragraph", "add a totals row"), plus save.
# Called-by: app.tools.registry (_register_defaults)
# Depends-on: app.documents.editor_client
# Last-renovated: 2026-06-12
"""
Document/editor chat tools (v1, 2026-06-12).

  editor_status        what's open in the editor pane
  read_document        full text (docs) for summarising / reading aloud
  edit_document_text   insert or replace text in the open doc
  read_sheet_range     values of an A1 range in the open sheet
  set_sheet_range      write values into an A1 range
  list_sheet_names     tabs of the open workbook
  save_document        run the editor's save path (atomic + .bak-once)

All of these act on the document OPEN IN THE PANE (the desktop command
centre's Editor tab). When nothing is open they say so honestly. After an
edit, confirm to the user WHAT changed ("Total now sits in A10") — the
results carry enough to do that.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
# Handlers
# ────────────────────────────────────────────────────────────────────────

async def editor_status_handler(input_data: dict, context: Optional[dict]) -> dict:
    try:
        from app.documents import editor_client
        return {"ok": True, "state": editor_client.state()}
    except Exception as exc:
        logger.exception("[document_tools] editor_status failed")
        return {"ok": False, "error": str(exc)}


async def read_document_handler(input_data: dict, context: Optional[dict]) -> dict:
    try:
        from app.documents import editor_client
        return await editor_client.doc_get_text()
    except Exception as exc:
        logger.exception("[document_tools] read_document failed")
        return {"ok": False, "error": str(exc)}


async def edit_document_text_handler(input_data: dict, context: Optional[dict]) -> dict:
    text = input_data.get("text")
    if text is None:
        return {"ok": False, "error": "text is required"}
    try:
        from app.documents import editor_client
        start = input_data.get("replace_start")
        end = input_data.get("replace_end")
        if start is not None and end is not None:
            return await editor_client.doc_replace_range(int(start), int(end), str(text))
        return await editor_client.doc_insert_text(
            str(text), str(input_data.get("position") or "end"))
    except Exception as exc:
        logger.exception("[document_tools] edit_document_text failed")
        return {"ok": False, "error": str(exc)}


async def read_sheet_range_handler(input_data: dict, context: Optional[dict]) -> dict:
    a1 = (input_data.get("a1") or "").strip()
    if not a1:
        return {"ok": False, "error": "a1 range is required (e.g. 'A1:C10')"}
    try:
        from app.documents import editor_client
        return await editor_client.sheet_get_range(a1)
    except Exception as exc:
        logger.exception("[document_tools] read_sheet_range failed")
        return {"ok": False, "error": str(exc)}


async def set_sheet_range_handler(input_data: dict, context: Optional[dict]) -> dict:
    a1 = (input_data.get("a1") or "").strip()
    values = input_data.get("values")
    if not a1 or values is None:
        return {"ok": False, "error": "a1 and values are required"}
    if not isinstance(values, list) or not all(isinstance(r, list) for r in values):
        return {"ok": False, "error": "values must be a 2D array (rows of cells)"}
    try:
        from app.documents import editor_client
        return await editor_client.sheet_set_range(a1, values)
    except Exception as exc:
        logger.exception("[document_tools] set_sheet_range failed")
        return {"ok": False, "error": str(exc)}


async def list_sheet_names_handler(input_data: dict, context: Optional[dict]) -> dict:
    try:
        from app.documents import editor_client
        return await editor_client.sheet_get_names()
    except Exception as exc:
        logger.exception("[document_tools] list_sheet_names failed")
        return {"ok": False, "error": str(exc)}


async def save_document_handler(input_data: dict, context: Optional[dict]) -> dict:
    try:
        from app.documents import editor_client
        return await editor_client.save(input_data.get("path"))
    except Exception as exc:
        logger.exception("[document_tools] save_document failed")
        return {"ok": False, "error": str(exc)}


# ────────────────────────────────────────────────────────────────────────
# Schemas (inline — self-contained file)
# ────────────────────────────────────────────────────────────────────────

_OK_ERR = {"ok": {"type": "boolean"}, "error": {"type": "string"}}

_RESULT_OUTPUT = {
    "type": "object",
    "required": ["ok"],
    "properties": {**_OK_ERR, "result": {}, "state": {"type": ["object", "null"]}},
}

EMPTY_INPUT = {"type": "object", "properties": {}}

EDIT_TEXT_INPUT = {
    "type": "object",
    "required": ["text"],
    "properties": {
        "text": {"type": "string"},
        "position": {"type": ["string", "null"],
                     "description": "'end' (default), 'start', or a character "
                                    "offset for inserts"},
        "replace_start": {"type": ["integer", "null"],
                          "description": "With replace_end: replace this "
                                         "character range instead of inserting"},
        "replace_end": {"type": ["integer", "null"]},
    },
}

READ_RANGE_INPUT = {
    "type": "object",
    "required": ["a1"],
    "properties": {"a1": {"type": "string", "description": "e.g. 'A1:C10' or 'A10'"}},
}

SET_RANGE_INPUT = {
    "type": "object",
    "required": ["a1", "values"],
    "properties": {
        "a1": {"type": "string",
               "description": "Top-left anchor or full range, e.g. 'A10' or 'A10:B12'"},
        "values": {"type": "array", "items": {"type": "array"},
                   "description": "2D rows of cell values; strings starting "
                                  "with '=' are formulas"},
    },
}

SAVE_INPUT = {
    "type": "object",
    "properties": {"path": {"type": ["string", "null"],
                            "description": "Defaults to the open file"}},
}


# ────────────────────────────────────────────────────────────────────────
# Registration
# ────────────────────────────────────────────────────────────────────────

def register_document_tools() -> None:
    """Register the editor tools with the global tool registry.
    Called once from registry._register_defaults at module load."""
    from app.tools.registry import ToolDefinition, register_tool

    register_tool(ToolDefinition(
        name="editor_status",
        version="v1",
        description=(
            "What's open in the desktop editor pane (command centre → "
            "Editor): path, kind (sheet/doc). Check before document edits "
            "when unsure; if nothing is open, ask the user to open the file "
            "from the Drive tab."
        ),
        input_schema=EMPTY_INPUT,
        output_schema=_RESULT_OUTPUT,
        handler=editor_status_handler,
    ))

    register_tool(ToolDefinition(
        name="read_document",
        version="v1",
        description=(
            "Full text of the OPEN document ('what does this say', 'read me "
            "the first paragraph', 'summarise this'). Speak only what was "
            "asked for — first paragraph means the first paragraph."
        ),
        input_schema=EMPTY_INPUT,
        output_schema=_RESULT_OUTPUT,
        handler=read_document_handler,
    ))

    register_tool(ToolDefinition(
        name="edit_document_text",
        version="v1",
        description=(
            "Insert text into the open document (position end/start/offset) "
            "or replace a character range. Visible immediately in the pane. "
            "Confirm to the user what changed; remind them it isn't on disk "
            "until save_document."
        ),
        input_schema=EDIT_TEXT_INPUT,
        output_schema=_RESULT_OUTPUT,
        handler=edit_document_text_handler,
    ))

    register_tool(ToolDefinition(
        name="read_sheet_range",
        version="v1",
        description=(
            "Values of an A1 range in the open spreadsheet ('what's in "
            "column B', 'read the totals row')."
        ),
        input_schema=READ_RANGE_INPUT,
        output_schema=_RESULT_OUTPUT,
        handler=read_sheet_range_handler,
    ))

    register_tool(ToolDefinition(
        name="set_sheet_range",
        version="v1",
        description=(
            "Write values into the open spreadsheet — 'put the word total "
            "in A10' = a1:'A10', values:[['total']]. '=' strings are live "
            "formulas ('add a totals row' = SUM formulas). Lands visibly in "
            "the pane; confirm what changed; not on disk until save_document."
        ),
        input_schema=SET_RANGE_INPUT,
        output_schema=_RESULT_OUTPUT,
        handler=set_sheet_range_handler,
    ))

    register_tool(ToolDefinition(
        name="list_sheet_names",
        version="v1",
        description="Tab names of the open workbook.",
        input_schema=EMPTY_INPUT,
        output_schema=_RESULT_OUTPUT,
        handler=list_sheet_names_handler,
    ))

    register_tool(ToolDefinition(
        name="save_document",
        version="v1",
        description=(
            "Save the open document to disk through the editor (atomic "
            "write; the original is kept as .bak on the first save). Use "
            "when the user says 'save it/that'."
        ),
        input_schema=SAVE_INPUT,
        output_schema=_RESULT_OUTPUT,
        handler=save_document_handler,
    ))
