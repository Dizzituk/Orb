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
MAX_TOOL_ROUNDS = 30  # generous allowance for complex debug investigations

# ---------------------------------------------------------------------------
# Filesystem safety — restrict write/edit/shell to known project directories
# Read and search are unrestricted (read-only is safe).
# ---------------------------------------------------------------------------

ALLOWED_WRITE_ROOTS = [
    Path("D:/Orb"),                              # ASTRA backend
    Path("D:/orb-desktop"),                       # ASTRA frontend
    Path("D:/Astra Android Folder"),              # All Android projects
    Path("C:/Users/dizzi/Documents"),             # User documents
    Path("C:/Users/dizzi/OneDrive/Documents"),    # OneDrive documents
    Path("C:/Users/dizzi/OneDrive/Pictures"),     # Pictures
]


def _is_path_allowed(path_str: str) -> bool:
    """Check if a path falls within allowed write roots."""
    try:
        target = Path(path_str).resolve()
        for root in ALLOWED_WRITE_ROOTS:
            try:
                if target == root.resolve() or root.resolve() in target.resolve().parents:
                    return True
            except (OSError, ValueError):
                continue
        return False
    except Exception:
        return False


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
            "Create or overwrite a file on the filesystem. Use absolute paths "
            "(e.g. D:/Orb/..., D:/Astra Android Folder/..., C:/Users/dizzi/...). "
            "Can write any file type: .py, .kt, .tsx, .txt, .html, .md, .json, etc. "
            "Parent directories are created automatically if they don't exist."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute file path, e.g. D:/Astra Android Folder/Astra-Bridge/ARCHITECTURE.txt",
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
        "name": "edit_file",
        "description": (
            "Apply targeted find-and-replace edits to a file. The old_text must "
            "match exactly and appear only once in the file. Use absolute paths."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute file path to edit",
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact text to find (must be unique in the file)",
                },
                "new_text": {
                    "type": "string",
                    "description": "Text to replace it with",
                },
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "run_shell",
        "description": (
            "Run a PowerShell command on the host machine. This is Windows with PowerShell. "
            "NEVER use Linux commands (grep, ls, cat, sed, awk, bash). "
            "Use PowerShell equivalents: Select-String (not grep), Get-ChildItem (not ls), "
            "Get-Content (not cat), python (not python3). "
            "Use for syntax checks, running builds, testing, installing packages, or any shell operation. "
            "Returns stdout, stderr, and exit code."
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
    """Write a file to the host filesystem (scoped to allowed directories)."""
    path = args.get("path", "")
    content = args.get("content", "")
    if not _is_path_allowed(path):
        return f"ERROR: Write blocked — {path} is outside allowed directories. Allowed roots: {', '.join(str(r) for r in ALLOWED_WRITE_ROOTS)}"
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        logger.info("[gemini_tool_loop] write_file: %s (%d chars)", path, len(content))
        return f"OK: Written {len(content)} chars to {path}"
    except Exception as e:
        return f"ERROR: {e}"


async def _exec_run_shell(args: dict) -> str:
    """Run a PowerShell command on the host (scoped to allowed directories)."""
    cmd = args.get("command", "")
    # Block dangerous commands that could affect system-wide state
    cmd_lower = cmd.lower()
    _BLOCKED_PATTERNS = [
        "remove-item c:\\", "remove-item 'c:", 'remove-item "c:',
        "del c:\\", "rd c:\\", "rmdir c:\\",
        "format-volume", "clear-disk",
        "stop-computer", "restart-computer",
        "set-executionpolicy",
        "new-service", "remove-service",
        "reg delete", "reg add",
        "net user", "net localgroup",
    ]
    for blocked in _BLOCKED_PATTERNS:
        if blocked in cmd_lower:
            return f"ERROR: Command blocked for safety — contains '{blocked}'"
    try:
        import asyncio
        proc = await asyncio.create_subprocess_shell(
            f'powershell.exe -NoProfile -Command "{cmd}"',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        parts = []
        if stdout:
            parts.append(stdout.decode("utf-8", errors="replace")[:3000])
        if stderr:
            parts.append(f"STDERR: {stderr.decode('utf-8', errors='replace')[:1000]}")
        parts.append(f"exit_code={proc.returncode}")
        return "\n".join(parts)
    except asyncio.TimeoutError:
        return "ERROR: Command timed out after 30 seconds"
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


async def _exec_edit_file(args: dict) -> str:
    """Apply targeted find-and-replace edits to a file (scoped to allowed directories)."""
    path = args.get("path", "")
    old_text = args.get("old_text", "")
    new_text = args.get("new_text", "")
    if not _is_path_allowed(path):
        return f"ERROR: Edit blocked — {path} is outside allowed directories."
    try:
        p = Path(path)
        if not p.exists():
            return f"ERROR: File not found: {path}"
        content = p.read_text(encoding="utf-8", errors="replace")
        if old_text not in content:
            return f"ERROR: old_text not found in {path}. Check exact match."
        if content.count(old_text) > 1:
            return f"ERROR: old_text appears {content.count(old_text)} times in {path}. Must be unique."
        new_content = content.replace(old_text, new_text, 1)
        p.write_text(new_content, encoding="utf-8")
        logger.info("[gemini_tool_loop] edit_file: %s (replaced %d chars with %d chars)", path, len(old_text), len(new_text))
        return f"OK: Edited {path} (replaced {len(old_text)} chars with {len(new_text)} chars)"
    except Exception as e:
        return f"ERROR: {e}"


_TOOL_EXECUTORS = {
    "read_file": _exec_read_file,
    "write_file": _exec_write_file,
    "edit_file": _exec_edit_file,
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
    content_parts: Optional[List] = None,
) -> str:
    """Run a Gemini conversation with native function calling.

    Handles the full tool loop: model responds -> extracts function calls ->
    executes tools -> sends results back -> model continues.

    Args:
        system_prompt: The system prompt to use
        messages: Conversation history [{role, content}]
        model_id: Google model string
        temperature: Sampling temperature
        max_tokens: Max output tokens
        on_tool_call: Optional callback(tool_name, args, result) for logging
        content_parts: Optional list of additional genai.protos.Part objects
                       (e.g. video file references) to include with the user message.

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

    # The user's latest message — may include multimodal parts (video, images)
    user_text = messages[-1]["content"] if messages else ""
    if content_parts:
        # Build multimodal message: text + video/image parts
        user_message = [user_text] + list(content_parts)
        logger.info("[gemini_tool_loop] Sending multimodal message with %d extra parts", len(content_parts))
    else:
        user_message = user_text

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
    else:
        # Hit MAX_TOOL_ROUNDS — nudge the model to produce a text summary
        logger.warning("[gemini_tool_loop] Hit max tool rounds (%d), forcing text response", MAX_TOOL_ROUNDS)
        try:
            response = chat.send_message(
                "You have used all available tool calls. Stop using tools now and give "
                "your final answer as text based on what you have gathered so far."
            )
        except Exception as nudge_err:
            logger.warning("[gemini_tool_loop] Nudge failed: %s", nudge_err)

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
        return "(Model exhausted tool calls without producing a response. Try asking a more specific question.)"


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
    import google.generativeai as genai

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
