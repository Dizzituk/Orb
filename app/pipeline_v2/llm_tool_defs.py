# FILE: app/pipeline_v2/llm_tool_defs.py
"""
Tool schema definitions for the agentic loops (OpenAI function format).

Split out of llm_tools.py on 2026-06-10 (file-size rule: llm_tools was at
the 30KB ceiling and Jobs 8-9 needed room there for provider routing and
the edit_file executor). Single responsibility: schemas only - no logic.

llm_tools_anthropic.to_anthropic_tools() converts these to Anthropic format
at call time, so this stays the single source of truth for both providers.

JOB 9 (2026-06-10): added edit_file - surgical exact-string replacement so
the verifier patches precisely instead of rewriting whole files.
"""
from __future__ import annotations

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the codebase. Returns the file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path (e.g. 'app/debug/models.py' or 'src/components/debug/DebugView.tsx')",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates or overwrites.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to write to",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete file content",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Surgically replace an EXACT string in a file with a new string. "
                "Use this for small fixes instead of rewriting whole files. "
                "old_str must appear EXACTLY ONCE in the file (match includes "
                "whitespace); if it matches zero or multiple times the edit is "
                "refused with the reason. A dated .bak of the original is saved "
                "alongside before the change."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative or absolute file path",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Exact existing text to replace (must be unique in the file)",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Replacement text (empty string deletes the matched text)",
                    },
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a PowerShell command. Use for syntax checks, booting app, npm commands.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "PowerShell command to execute",
                    },
                },
                "required": ["cmd"],
            },
        },
    },
]
