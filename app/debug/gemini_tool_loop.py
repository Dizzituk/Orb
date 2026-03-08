# FILE: app/debug/gemini_tool_loop.py
"""
Gemini Tool Loop — native function-calling loop for debug lock mode.

Bypasses the generic provider registry and calls the Google Generative AI
SDK directly with proper function declarations. Gemini 3.1 Pro Custom Tools
is designed specifically for this use case — it prioritises registered
custom tools over bash commands.

Tools available:
  - read_file:     Read a file from the host codebase (D: drive, read-only)
  - write_file:    Write a file to the sandbox
  - run_shell:     Run a PowerShell command in the sandbox
  - search_files:  Search the host codebase with glob patterns
  - list_dir:      List directory contents on the host

v1.0 (2026-03-07): Initial implementation.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

logger = logging.getLogger(__name__)

SANDBOX_BASE = os.getenv("ASTRA_SANDBOX_URL", "http://192.168.250.2:8765")
MAX_TOOL_ROUNDS = 10  # prevent infinite loops


# ---------------------------------------------------------------------------
# Tool definitions (Google Generative AI function declaration format)
# ---------------------------------------------------------------------------

TOOL_DECLARATIONS = [
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file from the ASTRA codebase on the host. "
            "Use absolute paths starting with D:/Orb/ for backend or D:/orb-desktop/ for frontend. "
            "Returns the file contents as text. Host is READ ONLY."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute file path, e.g. D:/Orb/app/debug/debug_chat.py",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file in the SANDBOX. Only use for sandbox paths. "
            "The sandbox is at 192.168.250.2 and is a safe, isolated environment."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the sandbox project root",
                },
                "content": {
                    "type": "string",
                    "description": "The full file content to write",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_shell",
        "description": (
            "Run a PowerShell command in the SANDBOX. Use for syntax checks, "
            "installing packages, running tests, reading file contents via "
            "Get-Content, or any shell operation. Returns stdout, stderr, and exit code."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The PowerShell command to execute",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "search_files",
        "description": (
            "Search for files matching a pattern in the ASTRA host codebase. "
            "Returns a list of matching file paths. Use glob patterns like "
            "'*.py' or '**/*router*.py'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Base directory to search in, e.g. D:/Orb/app",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern, e.g. **/*debug*.py",
                },
            },
            "required": ["directory", "pattern"],
        },
    },
    {
        "name": "list_dir",
        "description": (
            "List files and directories at a path on the host codebase. "
            "Returns names with [FILE] or [DIR] prefixes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute directory path, e.g. D:/Orb/app/debug",
                },
            },
            "required": ["path"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool executors
# ---------------------------------------------------------------------------

async def _exec_read_file(args: dict) -> str:
    """Read a file from the host codebase (read-only)."""
    path = args.get("path", "")
    try:
        p = Path(path)
        if not p.exists():
            return f"ERROR: File not found: {path}"
        if not p.is_file():
            return f"ERROR: Not a file: {path}"
        content = p.read_text(encoding="utf-8", errors="replace")
        if len(content) > 15000:
            return content[:15000] + f"\n\n... (truncated, {len(content)} total chars)"
        return content
    except Exception as e:
        return f"ERROR: {e}"


async def _exec_write_file(args: dict) -> str:
    """Write a file to the sandbox."""
    path = args.get("path", "")
    content = args.get("content", "")
    try:
        from app.pipeline_v2.sandbox_tools import write_file
        ok = await write_file(path, content)
        if ok:
            return f"OK: Written {len(content)} chars to {path}"
        return f"ERROR: Failed to write {path} to sandbox"
    except Exception as e:
        return f"ERROR: {e}"


async def _exec_run_shell(args: dict) -> str:
    """Run a command in the sandbox."""
    cmd = args.get("command", "")
    try:
        from app.pipeline_v2.sandbox_tools import run_shell
        result = await run_shell(cmd, timeout_sec=30)
        parts = []
        if result.get("stdout"):
            parts.append(result["stdout"][:3000])
        if result.get("stderr"):
            parts.append(f"STDERR: {result['stderr'][:1000]}")
        parts.append(f"exit_code={result.get('exit_code', '?')}")
        return "\n".join(parts)
    except Exception as e:
        return f"ERROR: {e}"


async def _exec_search_files(args: dict) -> str:
    """Search for files matching a pattern."""
    directory = args.get("directory", "D:/Orb")
    pattern = args.get("pattern", "*.py")
    try:
        p = Path(directory)
        if not p.exists():
            return f"ERROR: Directory not found: {directory}"
        matches = sorted(str(m) for m in p.glob(pattern))[:30]
        if not matches:
            return f"No files matching '{pattern}' in {directory}"
        return "\n".join(matches)
    except Exception as e:
        return f"ERROR: {e}"


async def _exec_list_dir(args: dict) -> str:
    """List directory contents."""
    path = args.get("path", "")
    try:
        p = Path(path)
        if not p.exists():
            return f"ERROR: Not found: {path}"
        if not p.is_dir():
            return f"ERROR: Not a directory: {path}"
        entries = []
        for item in sorted(p.iterdir()):
            prefix = "[DIR]" if item.is_dir() else "[FILE]"
            entries.append(f"{prefix} {item.name}")
        return "\n".join(entries[:50]) if entries else "(empty directory)"
    except Exception as e:
        return f"ERROR: {e}"


_TOOL_EXECUTORS = {
    "read_file": _exec_read_file,
    "write_file": _exec_write_file,
    "run_shell": _exec_run_shell,
    "search_files": _exec_search_files,
    "list_dir": _exec_list_dir,
}


# ---------------------------------------------------------------------------
# Main tool loop
# ---------------------------------------------------------------------------

async def run_gemini_tool_loop(
    system_prompt: str,
    messages: List[dict],
    model_id: str = "gemini-3.1-pro-preview-customtools",
    temperature: float = 0.2,
    max_tokens: int = 8192,
    on_tool_call: Optional[callable] = None,
) -> str:
    """Run a Gemini conversation with native function calling.

    Handles the full tool loop: model responds → extracts function calls →
    executes tools → sends results back → model continues.

    Args:
        system_prompt: The system prompt to use
        messages: Conversation history [{role, content}]
        model_id: Google model string
        temperature: Sampling temperature
        max_tokens: Max output tokens
        on_tool_call: Optional callback(tool_name, args, result) for logging

    Returns:
        The final text response from the model.
    """
    import google.generativeai as genai

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        return "ERROR: GOOGLE_API_KEY not set."

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

    # Build conversation history for the SDK
    history = []
    for msg in messages[:-1]:  # All but the last (which we send as new input)
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [msg["content"]]})

    chat = model.start_chat(history=history)

    # The user's latest message
    user_message = messages[-1]["content"] if messages else ""

    # Tool loop
    response = chat.send_message(user_message)

    for _round in range(MAX_TOOL_ROUNDS):
        # Check for function calls
        function_calls = _extract_function_calls(response)
        if not function_calls:
            break  # Model returned text, we're done

        # Execute each function call
        tool_responses = []
        for fc_name, fc_args in function_calls:
            executor = _TOOL_EXECUTORS.get(fc_name)
            if executor:
                result = await executor(fc_args)
            else:
                result = f"ERROR: Unknown tool '{fc_name}'"

            if on_tool_call:
                try:
                    on_tool_call(fc_name, fc_args, result[:200])
                except Exception:
                    pass

            tool_responses.append(
                genai.protos.Part(function_response=genai.protos.FunctionResponse(
                    name=fc_name,
                    response={"result": result},
                ))
            )

        # Send tool results back to the model
        response = chat.send_message(tool_responses)

    # Extract final text
    try:
        return response.text
    except (ValueError, AttributeError):
        # Fallback: try to extract text from parts
        try:
            parts_text = []
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    parts_text.append(part.text)
            if parts_text:
                return "\n".join(parts_text)
        except Exception:
            pass
        return "(Model returned no text response after tool calls.)"


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


def _convert_schema(schema: dict) -> dict:
    """Convert a JSON schema dict to genai.protos.Schema kwargs."""
    type_map = {
        "string": 1,   # Type.STRING
        "number": 2,   # Type.NUMBER
        "integer": 3,  # Type.INTEGER
        "boolean": 4,  # Type.BOOLEAN
        "array": 5,    # Type.ARRAY
        "object": 6,   # Type.OBJECT
    }

    result = {}
    if "type" in schema:
        result["type"] = type_map.get(schema["type"], 6)

    if "properties" in schema:
        result["properties"] = {}
        for prop_name, prop_schema in schema["properties"].items():
            result["properties"][prop_name] = genai.protos.Schema(
                type=type_map.get(prop_schema.get("type", "string"), 1),
                description=prop_schema.get("description", ""),
            )

    if "required" in schema:
        result["required"] = schema["required"]

    return result
