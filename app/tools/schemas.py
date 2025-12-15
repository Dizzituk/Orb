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
# Convenience registry
# -------------------------

TOOL_SCHEMAS = {
    "http_fetch": {"input": HTTP_FETCH_INPUT_V1, "output": HTTP_FETCH_OUTPUT_V1},
    "web_search": {"input": WEB_SEARCH_INPUT_V1, "output": WEB_SEARCH_OUTPUT_V1},
    "weather": {"input": WEATHER_INPUT_V1, "output": WEATHER_OUTPUT_V1},
}
