# FILE: app/debug/tool_defs_documents.py
# Purpose: Document-creation tool schemas (docx, pdf, xlsx, html report).
# Called-by: app.debug.tool_definitions (facade); tool registry dispatch.
# Depends-on: none (pure schema constants).
# Last-renovated: 2026-06-11 (split from tool_definitions.py, Phase 4)
from __future__ import annotations

_CONTENT_SCHEMA_DESC = (
    "List of block objects. Each block has a 'type' field. Supported types:\n"
    "  - {type:'heading', level:1-4, text:'...'}\n"
    "  - {type:'paragraph', text:'...'}\n"
    "  - {type:'list', items:['...','...'], ordered:false}\n"
    "  - {type:'table', headers:['col1','col2'], rows:[['a','b'],['c','d']]}\n"
    "  - {type:'rule'}    (horizontal divider)\n"
    "  - {type:'spacer'}  (blank line)\n"
    "  - {type:'code', text:'...', language:'python'}\n"
    "Order in the list determines render order."
)


_THEME_DESC = (
    "Visual theme. 'auto' (default) inspects the filename for keywords like "
    "'legal', 'evidence', 'letter' and picks 'astra_minimal' (plain, formal); "
    "everything else gets 'astra_default' (modern, branded). Pass 'minimal' "
    "or 'default' to force a choice."
)


_SKILL_DESC = (
    "Optional skill playbook ID that guides document structure and tone. "
    "Available skills: 'formal_document' (legal, letters to MPs, formal "
    "reports - serif, restrained, no tables in body), 'casual_document' "
    "(personal writing, friendly updates - sans-serif, natural voice), "
    "'data_spreadsheet' (for xlsx: proper header row, evidence ref column, "
    "no inline totals). If omitted, the skill is auto-detected from the "
    "filename and title keywords. Pass an explicit value to override."
)


_BRIEF_DESC = (
    "Natural-language description of what the document should contain. When "
    "provided INSTEAD of pre-structured 'content', ASTRA runs a reasoning-"
    "tier structuring pass that follows the chosen skill\u2019s playbook to "
    "produce the final document. Use this when you do not want to hand-build "
    "content blocks yourself. You can combine brief with 'source_material' "
    "to provide facts, data, or prior text that should be incorporated."
)


_SOURCE_MATERIAL_DESC = (
    "Optional supporting data for the structuring pass. Can be a plain string "
    "(prior text to incorporate), a JSON object/array (structured facts, a "
    "dataset of records), or a JSON-encoded string. Ignored when 'content' "
    "or 'sheets' is provided directly."
)


CREATE_DOCX_TOOL = {
    "name": "create_docx",
    "description": (
        "Create a styled Microsoft Word document. Use this when the user "
        "wants a presentation-quality .docx for a report, summary, proposal, "
        "letter, or evidence pack. "
        "Two modes: (1) pass pre-structured 'content' blocks for exact "
        "control, or (2) pass a natural-language 'brief' and let ASTRA "
        "structure the document via a skill playbook using a reasoning-tier "
        "model. Mode 2 is preferred for most cases - you describe what the "
        "doc should say and ASTRA produces proper structure automatically. "
        "Theme auto-selects from filename keywords (legal/evidence/letter "
        "-> minimal plain style; everything else -> branded styled). Cover "
        "page is added automatically for the styled theme."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute output path. .docx extension added if missing."},
            "title": {"type": "string", "description": "Document title (used on cover page and Word metadata)."},
            "subtitle": {"type": "string", "description": "Optional subtitle."},
            "author": {"type": "string", "description": "Optional author name."},
            "theme": {"type": "string", "description": _THEME_DESC},
            "skill": {"type": "string", "description": _SKILL_DESC},
            "brief": {"type": "string", "description": _BRIEF_DESC},
            "source_material": {"description": _SOURCE_MATERIAL_DESC},
            "content": {"type": "array", "description": _CONTENT_SCHEMA_DESC, "items": {"type": "object"}},
        },
        "required": ["path", "title"],
    },
}


CREATE_PDF_TOOL = {
    "name": "create_pdf",
    "description": (
        "Create a styled PDF document. Same schema, skill, and brief modes "
        "as create_docx. Output is a clean A4 PDF with page numbers and "
        "generated-date footer. Good for documents that will be shared, "
        "printed, or attached to correspondence. For editable outputs prefer "
        "create_docx (the user can then export to PDF themselves)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute output path. .pdf extension added if missing."},
            "title": {"type": "string", "description": "Document title."},
            "subtitle": {"type": "string", "description": "Optional subtitle."},
            "author": {"type": "string", "description": "Optional author name."},
            "theme": {"type": "string", "description": _THEME_DESC},
            "skill": {"type": "string", "description": _SKILL_DESC},
            "brief": {"type": "string", "description": _BRIEF_DESC},
            "source_material": {"description": _SOURCE_MATERIAL_DESC},
            "content": {"type": "array", "description": _CONTENT_SCHEMA_DESC, "items": {"type": "object"}},
        },
        "required": ["path", "title"],
    },
}


CREATE_XLSX_TOOL = {
    "name": "create_xlsx",
    "description": (
        "Create a styled Excel workbook. Two modes: (1) pass pre-built "
        "'sheets' objects for exact control, or (2) pass a natural-language "
        "'brief' and optional 'source_material' and let ASTRA structure the "
        "workbook via the data_spreadsheet skill using a reasoning-tier "
        "model. Mode 2 is preferred when you have raw data that needs "
        "turning into a proper workbook. Header row gets theme fill and bold "
        "text, column widths auto-fit, freeze pane and auto-filter applied."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute output path. .xlsx extension added if missing."},
            "title": {"type": "string", "description": "Workbook title (used as Excel metadata)."},
            "theme": {"type": "string", "description": _THEME_DESC},
            "skill": {"type": "string", "description": _SKILL_DESC},
            "brief": {"type": "string", "description": _BRIEF_DESC},
            "source_material": {"description": _SOURCE_MATERIAL_DESC},
            "sheets": {
                "type": "array",
                "description": (
                    "List of sheet objects (mode 1): "
                    "{name:'Tab name', headers:['col1','col2'], rows:[[...],[...]], "
                    "freeze_header:true, auto_filter:true, column_widths:[12,30]}. "
                    "Only 'name' is mandatory; everything else has sensible defaults."
                ),
                "items": {"type": "object"}
            },
        },
        "required": ["path"],
    },
}


CREATE_HTML_REPORT_TOOL = {
    "name": "create_html_report",
    "description": (
        "Create a single-file HTML report with embedded CSS. Same schema, "
        "skill, and brief modes as create_docx. Renders cleanly in any "
        "browser, prints well, respects prefers-color-scheme. Use for "
        "shareable web reports or dashboards-as-documents."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute output path. .html extension added if missing."},
            "title": {"type": "string", "description": "Report title."},
            "subtitle": {"type": "string", "description": "Optional subtitle."},
            "author": {"type": "string", "description": "Optional author name."},
            "theme": {"type": "string", "description": _THEME_DESC},
            "skill": {"type": "string", "description": _SKILL_DESC},
            "brief": {"type": "string", "description": _BRIEF_DESC},
            "source_material": {"description": _SOURCE_MATERIAL_DESC},
            "content": {"type": "array", "description": _CONTENT_SCHEMA_DESC, "items": {"type": "object"}},
        },
        "required": ["path", "title"],
    },
}
