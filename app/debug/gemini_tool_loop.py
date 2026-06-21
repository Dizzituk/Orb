# FILE: app/debug/gemini_tool_loop.py
# Purpose: Gemini Tool Loop — native function-calling loop for debug lock mode.
# Called-by: app.llm.routing.image_chat_routing
# Depends-on: app.debug.gemini_finance_tools, app.debug.gemini_lifestyle_tools, app.llm.model_families
# Last-renovated: 2026-06-11
"""
Gemini Tool Loop — native function-calling loop for debug lock mode.

Bypasses the generic provider registry and calls the Google Generative AI
SDK directly with proper function declarations.

v1.0 (2026-03-07): Initial implementation.
v2.0 (2026-03-21): Streaming refactor — yields structured events in
    real-time instead of returning a final string. Enables the frontend
    to show tool calls, thinking text, and the final response as they
    happen rather than after the entire loop completes.
v2.1 (2026-05-26): Lifestyle tools added via gemini_lifestyle_tools.
    Five domain tools (log_nutrition, log_workout, get_recent_nutrition,
    get_weight_trend, get_recent_workouts) are merged in alongside the
    existing file tools so image-bearing turns can route "log this into
    nutrition" to the lifestyle database instead of writing HTML files.
    Existing file tools are byte-identical to v2.0.

Event types yielded:
  - tool_start:  ASTRA is about to call a tool (name + args summary)
  - tool_result: Tool execution completed (name + result preview)
  - thinking:    Intermediate model text between tool calls
  - final_text:  The model's final response (streamed in chunks)
  - error:       Something went wrong
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.debug.gemini_lifestyle_tools import (
    LIFESTYLE_TOOL_DECLARATIONS,
    LIFESTYLE_TOOL_EXECUTORS,
    summarise_lifestyle_call,
    summarise_lifestyle_result,
)
from app.debug.gemini_finance_tools import (
    FINANCE_TOOL_DECLARATIONS,
    FINANCE_TOOL_EXECUTORS,
    summarise_finance_call,
    summarise_finance_result,
)

logger = logging.getLogger(__name__)

SANDBOX_BASE = os.getenv("ASTRA_SANDBOX_URL", "http://192.168.250.2:8765")
MAX_TOOL_ROUNDS = 30

# Default model for the loop — from the model registry (ONE source,
# overridable via ASTRA_MODEL_ROLE_VISION_TOOLS in .env), not hardcoded.
from app.llm.model_families import resolve as _resolve_model
_DEFAULT_MODEL = _resolve_model("role_vision_tools")


# ---------------------------------------------------------------------------
# Host file/shell tools - extracted to gemini_file_tools.py (split 2026-06-20).
# Imported back so TOOL_DECLARATIONS / _TOOL_EXECUTORS below compose unchanged.
# ---------------------------------------------------------------------------
from app.debug.gemini_file_tools import (  # noqa: F401
    ALLOWED_WRITE_ROOTS,
    _is_path_allowed,
    _FILE_TOOL_DECLARATIONS,
    _exec_read_file,
    _exec_write_file,
    _exec_edit_file,
    _exec_run_shell,
    _exec_search_files,
    _exec_list_dir,
)


# Public: the full tool set Gemini sees. File tools first, then lifestyle.
TOOL_DECLARATIONS = _FILE_TOOL_DECLARATIONS + LIFESTYLE_TOOL_DECLARATIONS + FINANCE_TOOL_DECLARATIONS


# Compose the executor map: file tools first, then lifestyle tools merged in.
_TOOL_EXECUTORS = {
    "read_file": _exec_read_file,
    "write_file": _exec_write_file,
    "edit_file": _exec_edit_file,
    "run_shell": _exec_run_shell,
    "search_files": _exec_search_files,
    "list_dir": _exec_list_dir,
    **LIFESTYLE_TOOL_EXECUTORS,
    **FINANCE_TOOL_EXECUTORS,
}


# ---------------------------------------------------------------------------
# Human-readable tool call summaries
# ---------------------------------------------------------------------------

_LIFESTYLE_TOOL_NAMES = set(LIFESTYLE_TOOL_EXECUTORS.keys())
_FINANCE_TOOL_NAMES = set(FINANCE_TOOL_EXECUTORS.keys())


def _summarise_tool_call(name: str, args: dict) -> str:
    """Create a concise, human-readable summary of what a tool call is doing."""
    if name in _LIFESTYLE_TOOL_NAMES:
        return summarise_lifestyle_call(name, args)
    if name in _FINANCE_TOOL_NAMES:
        return summarise_finance_call(name, args)
    if name == "read_file":
        path = args.get("path", "")
        filename = Path(path).name if path else "unknown"
        return f"Reading {filename}"
    elif name == "write_file":
        path = args.get("path", "")
        filename = Path(path).name if path else "unknown"
        content = args.get("content", "")
        return f"Writing {filename} ({len(content)} chars)"
    elif name == "edit_file":
        path = args.get("path", "")
        filename = Path(path).name if path else "unknown"
        return f"Editing {filename}"
    elif name == "run_shell":
        cmd = args.get("command", "")
        return f"Running: {cmd[:80]}{'...' if len(cmd) > 80 else ''}"
    elif name == "search_files":
        pattern = args.get("pattern", "")
        directory = args.get("directory", "")
        dirname = Path(directory).name if directory else ""
        return f"Searching {dirname} for {pattern}"
    elif name == "list_dir":
        path = args.get("path", "")
        dirname = Path(path).name if path else ""
        return f"Listing {dirname}/"
    else:
        return f"Calling {name}"


def _summarise_tool_result(name: str, result: str) -> str:
    """Create a concise summary of a tool result for the UI."""
    if name in _LIFESTYLE_TOOL_NAMES:
        return summarise_lifestyle_result(name, result)
    if name in _FINANCE_TOOL_NAMES:
        return summarise_finance_result(name, result)
    if result.startswith("ERROR:"):
        return result[:120]
    elif name == "read_file":
        lines = result.count('\n') + 1
        return f"{lines} lines read"
    elif name in ("write_file", "edit_file"):
        return result[:100]
    elif name == "run_shell":
        # Show first meaningful line of output
        first_line = result.strip().split('\n')[0][:100] if result.strip() else "OK"
        return first_line
    elif name == "search_files":
        count = len(result.strip().split('\n')) if result.strip() else 0
        return f"{count} files found"
    elif name == "list_dir":
        count = len(result.strip().split('\n')) if result.strip() else 0
        return f"{count} entries"
    else:
        return result[:100]


# ---------------------------------------------------------------------------
# Streaming tool loop (v2.0)
# ---------------------------------------------------------------------------

async def stream_gemini_tool_loop(
    system_prompt: str,
    messages: List[dict],
    model_id: str = _DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 8192,
    content_parts: Optional[List] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream a Gemini conversation with native function calling.

    Yields structured events as the tool loop progresses so the frontend
    can render them in real-time.

    Event types:
        {"type": "tool_start", "tool": str, "summary": str}
        {"type": "tool_result", "tool": str, "summary": str}
        {"type": "thinking", "content": str}
        {"type": "final_text", "content": str}
        {"type": "error", "error": str}

    Args:
        system_prompt: The system prompt to use
        messages: Conversation history [{role, content}]
        model_id: Google model string
        temperature: Sampling temperature
        max_tokens: Max output tokens
        content_parts: Optional multimodal parts (video, images)
    """
    import google.generativeai as genai

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        yield {"type": "error", "error": "GOOGLE_API_KEY not set."}
        return

    genai.configure(api_key=api_key)

    # Build the model with tool declarations
    tools = [genai.protos.Tool(function_declarations=[
        genai.protos.FunctionDeclaration(
            name=td["name"],
            description=td["description"],
            parameters=genai.protos.Schema(**_convert_schema(td["parameters"])),
        )
        for td in TOOL_DECLARATIONS
    ])]

    model = genai.GenerativeModel(
        model_id,
        tools=tools,
        system_instruction=system_prompt,
    )

    # Build conversation history
    history = []
    for msg in messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [msg["content"]]})

    chat = model.start_chat(history=history)

    # Build user message (may include multimodal parts)
    user_text = messages[-1]["content"] if messages else ""
    if content_parts:
        user_message = [user_text] + list(content_parts)
        logger.info("[gemini_tool_loop] Sending multimodal message with %d extra parts", len(content_parts))
    else:
        user_message = user_text

    # Send initial message
    try:
        response = chat.send_message(user_message)
    except Exception as e:
        yield {"type": "error", "error": f"Gemini API call failed: {e}"}
        return

    # Tool loop — yield events as we go
    for _round in range(MAX_TOOL_ROUNDS):
        function_calls = _extract_function_calls(response)

        if not function_calls:
            # No tool calls — this is the final text response
            # Check for any intermediate text in this response too
            break

        # Check if there's thinking text alongside the tool calls
        thinking_text = _extract_text_parts(response)
        if thinking_text:
            yield {"type": "thinking", "content": thinking_text}

        # Execute each function call, yielding events as we go
        tool_responses = []
        for fc_name, fc_args in function_calls:
            # Yield tool_start so the UI shows what we're about to do
            yield {
                "type": "tool_start",
                "tool": fc_name,
                "summary": _summarise_tool_call(fc_name, fc_args),
            }

            # Execute the tool
            executor = _TOOL_EXECUTORS.get(fc_name)
            if executor:
                result = await executor(fc_args)
            else:
                result = f"ERROR: Unknown tool '{fc_name}'"

            # Yield tool_result so the UI shows what happened
            yield {
                "type": "tool_result",
                "tool": fc_name,
                "summary": _summarise_tool_result(fc_name, result),
            }

            tool_responses.append(
                genai.protos.Part(function_response=genai.protos.FunctionResponse(
                    name=fc_name,
                    response={"result": result},
                ))
            )

        # Send tool results back to the model
        try:
            response = chat.send_message(tool_responses)
        except Exception as e:
            yield {"type": "error", "error": f"Gemini API call failed after tool round {_round + 1}: {e}"}
            return
    else:
        # Hit MAX_TOOL_ROUNDS — nudge for a text summary
        logger.warning("[gemini_tool_loop] Hit max tool rounds (%d)", MAX_TOOL_ROUNDS)
        try:
            response = chat.send_message(
                "You have used all available tool calls. Stop using tools now and give "
                "your final answer as text based on what you have gathered so far."
            )
        except Exception as nudge_err:
            yield {"type": "error", "error": f"Failed to get final response: {nudge_err}"}
            return

    # Extract and stream the final text response
    final_text = _extract_full_text(response)
    if final_text:
        # Stream in chunks for a natural feel
        chunk_size = 60
        for i in range(0, len(final_text), chunk_size):
            yield {"type": "final_text", "content": final_text[i:i + chunk_size]}
    else:
        yield {
            "type": "final_text",
            "content": "(Model completed tool calls without producing a text response. Try a more specific question.)",
        }


# ---------------------------------------------------------------------------
# Legacy synchronous wrapper (for callers that haven't migrated yet)
# ---------------------------------------------------------------------------

async def run_gemini_tool_loop(
    system_prompt: str,
    messages: List[dict],
    model_id: str = _DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 8192,
    on_tool_call: Optional[callable] = None,
    content_parts: Optional[List] = None,
) -> str:
    """Legacy wrapper — collects all events and returns the final text.

    Preserved for backward compatibility with callers that expect a string.
    New code should use stream_gemini_tool_loop() instead.
    """
    final_parts = []
    async for event in stream_gemini_tool_loop(
        system_prompt=system_prompt,
        messages=messages,
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        content_parts=content_parts,
    ):
        if event["type"] == "final_text":
            final_parts.append(event["content"])
        elif event["type"] == "thinking":
            final_parts.append(event["content"])
        elif event["type"] in ("tool_start", "tool_result") and on_tool_call:
            try:
                on_tool_call(
                    event.get("tool", ""),
                    {},
                    event.get("summary", ""),
                )
            except Exception:
                pass
        elif event["type"] == "error":
            return f"ERROR: {event['error']}"
    return "".join(final_parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_function_calls(response) -> List[tuple]:
    """Extract (name, args) tuples from a Gemini response."""
    calls = []
    try:
        if not getattr(response, "candidates", None):
            return calls
        for part in response.candidates[0].content.parts:
            fc = getattr(part, "function_call", None)
            if fc:
                name = getattr(fc, "name", "")
                args = dict(getattr(fc, "args", {}) or {})
                if name:
                    calls.append((name, args))
    except Exception:
        pass
    return calls


def _extract_text_parts(response) -> str:
    """Extract any text parts from a response (may appear alongside tool calls)."""
    texts = []
    try:
        if not getattr(response, "candidates", None):
            return ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text and not hasattr(part, "function_call"):
                texts.append(part.text)
    except Exception:
        pass
    return "\n".join(texts)


def _extract_full_text(response) -> str:
    """Extract the full text from the final response."""
    try:
        return response.text
    except (ValueError, AttributeError):
        try:
            parts_text = []
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    parts_text.append(part.text)
            if parts_text:
                return "\n".join(parts_text)
        except Exception:
            pass
    return ""


def _convert_schema(schema: dict) -> dict:
    """Convert a JSON-Schema dict into kwargs for genai.protos.Schema.

    Recursive on purpose. Gemini's GenerateContentRequest REQUIRES that an
    ARRAY-typed schema carry its `items` element schema, and that nested
    OBJECT-typed schemas carry their own `properties`/`required`. The previous
    version copied only `type` + `description` one level deep, so every tool
    with an array parameter (log_nutrition.items, log_exercise.sets,
    save_recipe.ingredients, log_recipe.ingredient_names) reached Gemini with a
    type=ARRAY schema and no `items`, and the whole request was rejected with
    400 "...parameters.properties[x].items: missing field" — which surfaced to
    the user as a generic "Gemini API call failed" on every image turn.
    """
    import google.generativeai as genai

    type_map = {
        "string": 1,
        "number": 2,
        "integer": 3,
        "boolean": 4,
        "array": 5,
        "object": 6,
    }

    schema_type = schema.get("type", "object")
    result: dict = {"type": type_map.get(schema_type, 6)}

    if schema.get("description"):
        result["description"] = schema["description"]

    # Object: recurse into each property so nested objects keep their shape.
    if "properties" in schema:
        result["properties"] = {
            prop_name: genai.protos.Schema(**_convert_schema(prop_schema))
            for prop_name, prop_schema in schema["properties"].items()
        }

    if schema.get("required"):
        result["required"] = list(schema["required"])

    # Array: recurse into the element schema. Gemini rejects an array with no
    # `items`, so default to a string element if a declaration omitted it.
    if schema_type == "array":
        result["items"] = genai.protos.Schema(
            **_convert_schema(schema.get("items") or {"type": "string"})
        )

    return result
