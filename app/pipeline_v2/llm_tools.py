# FILE: app/pipeline_v2/llm_tools.py
"""
LLM Tool-Calling Loop for the Agentic Builder.

Handles the OpenAI function-calling protocol:
  1. Send messages + tool definitions to the API
  2. If the model returns tool_calls, execute them
  3. Send tool results back as tool messages
  4. Repeat until the model returns a text response (no more tool calls)

Supports OpenAI (GPT-5.4) natively. Anthropic/Google would need
adapter logic (not implemented yet — GPT-5.4 is primary).

v1.0 (2026-03-07): Initial implementation for ASTRA v2.1.
v1.1 (2026-03-08): Added write integrity checking — pre-flight content
                    validation + post-write read-back verification.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main tool-calling loop
# ---------------------------------------------------------------------------

async def run_tool_loop(
    system_prompt: str,
    initial_user_message: str,
    provider: str,
    model: str,
    max_iterations: int = 50,
    max_tokens: int = 128000,
    on_tool_call: Optional[Callable[[str, Dict], None]] = None,
    on_text: Optional[Callable[[str], None]] = None,
    existing_messages: Optional[List[Dict]] = None,
) -> Tuple[List[Dict], int, int]:
    """Run an agentic tool-calling loop.

    Sends the initial prompt with tool definitions, then loops:
    model returns tool_calls -> we execute -> send results -> model continues.

    Args:
        system_prompt: System message.
        initial_user_message: First user message (ignored if existing_messages).
        provider: "openai" (primary).
        model: Model name (e.g. "gpt-5.4").
        max_iterations: Max tool-call rounds before stopping.
        max_tokens: Max output tokens per call.
        on_tool_call: Callback(tool_name, args) for progress tracking.
        on_text: Callback(text) for model's text responses.
        existing_messages: If provided, continues from this conversation history
            instead of starting fresh. The initial_user_message is appended as
            a new user message to this history. This keeps the builder in the
            same context window so it knows what it already built.

    Returns:
        (messages_history, total_input_tokens, total_output_tokens)
    """
    if existing_messages:
        messages = list(existing_messages)
        messages.append({"role": "user", "content": initial_user_message})
        logger.info(
            "[llm_tools] CONTINUATION MODE: %d existing messages + new instruction",
            len(existing_messages),
        )
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_user_message},
        ]

    total_input_tokens = 0
    total_output_tokens = 0

    for iteration in range(max_iterations):
        logger.info("[llm_tools] Iteration %d — calling %s/%s", iteration + 1, provider, model)

        response, in_tok, out_tok = await _call_api(
            provider=provider,
            model=model,
            messages=messages,
            tools=TOOLS,
            max_tokens=max_tokens,
        )
        total_input_tokens += in_tok
        total_output_tokens += out_tok

        if response is None:
            logger.error("[llm_tools] API call returned None at iteration %d", iteration + 1)
            break

        tool_calls = _extract_tool_calls(response, provider)

        if not tool_calls:
            text = _extract_text(response, provider)
            if on_text and text:
                on_text(text)
            messages.append(_make_assistant_message(response, provider))
            logger.info("[llm_tools] Text response at iteration %d (%d chars)", iteration + 1, len(text or ""))
            break

        messages.append(_make_assistant_message(response, provider))

        for tc in tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc["id"]

            if on_tool_call:
                on_tool_call(tool_name, tool_args)

            result = await _execute_tool(tool_name, tool_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": result,
            })

            logger.info("[llm_tools] Tool %s -> %d chars result", tool_name, len(result))

    else:
        logger.warning("[llm_tools] Hit max iterations (%d)", max_iterations)

    return messages, total_input_tokens, total_output_tokens


# ---------------------------------------------------------------------------
# API call (OpenAI)
# ---------------------------------------------------------------------------

async def _call_api(
    provider: str,
    model: str,
    messages: List[Dict],
    tools: List[Dict],
    max_tokens: int,
) -> Tuple[Optional[Any], int, int]:
    """Call the LLM API with tools. Returns (response, input_tokens, output_tokens)."""

    if provider == "openai":
        return await _call_openai(model, messages, tools, max_tokens)
    else:
        logger.warning("[llm_tools] Provider %s does not support tool calling — using text mode", provider)
        from app.pipeline_v2.llm_caller import call_llm
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        try:
            text = await call_llm(provider=provider, model=model,
                                  system_prompt=system, user_prompt=user,
                                  max_tokens=max_tokens)
            return {"text": text}, 0, 0
        except Exception as e:
            logger.error("[llm_tools] Fallback call failed: %s", e)
            return None, 0, 0


async def _call_openai(
    model: str,
    messages: List[Dict],
    tools: List[Dict],
    max_tokens: int,
) -> Tuple[Optional[Any], int, int]:
    """Call OpenAI API with function calling."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.error("[llm_tools] openai package not installed")
        return None, 0, 0

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.error("[llm_tools] OPENAI_API_KEY not set")
        return None, 0, 0

    client = AsyncOpenAI(api_key=api_key, timeout=180.0)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            max_completion_tokens=min(max_tokens, 16384),
            temperature=0,
        )

        usage = response.usage
        in_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0

        return response, in_tok, out_tok

    except Exception as e:
        logger.error("[llm_tools] OpenAI API error: %s", e)
        return None, 0, 0


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _extract_tool_calls(response: Any, provider: str) -> List[Dict]:
    """Extract tool calls from API response."""
    if provider == "openai" and hasattr(response, "choices"):
        msg = response.choices[0].message
        if msg.tool_calls:
            calls = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    args = {}
                calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": args,
                })
            return calls
    return []


def _extract_text(response: Any, provider: str) -> Optional[str]:
    """Extract text content from API response."""
    if provider == "openai" and hasattr(response, "choices"):
        return response.choices[0].message.content
    if isinstance(response, dict) and "text" in response:
        return response["text"]
    return None


def _make_assistant_message(response: Any, provider: str) -> Dict:
    """Convert API response to an assistant message for the conversation."""
    if provider == "openai" and hasattr(response, "choices"):
        msg = response.choices[0].message
        result = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return result
    text = _extract_text(response, provider) or ""
    return {"role": "assistant", "content": text}


# ---------------------------------------------------------------------------
# Tool execution (with write integrity checking)
# ---------------------------------------------------------------------------

# Module-level profile for tool execution routing (host vs sandbox).
_tool_profile: Optional["BuildTargetProfile"] = None


def set_tool_profile(profile: Optional["BuildTargetProfile"]) -> None:
    """Set the profile for tool execution routing."""
    global _tool_profile
    _tool_profile = profile


async def _execute_tool(name: str, args: Dict) -> str:
    """Execute a tool call and return the result as a string.

    Write operations include integrity checking:
      1. Pre-flight: scans content for patterns likely to be corrupted
      2. Write: sends content to sandbox
      3. Verify: reads file back and compares to original

    If corruption is detected, the error message tells the LLM exactly
    what went wrong so it can fix it (e.g. "backticks were stripped").
    """
    from app.pipeline_v2 import sandbox_tools

    try:
        if name == "read_file":
            path = args.get("path", "")
            content = await sandbox_tools.read_file(path, profile=_tool_profile)
            if content is None:
                return "ERROR: File not found: " + path
            if len(content) > 30000:
                return (
                    "[" + str(len(content)) + " chars total — showing first 15K + last 5K]\n"
                    + content[:15000] + "\n...\n" + content[-5000:]
                )
            return content

        elif name == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")

            # --- Write integrity: pre-flight check ---
            from app.pipeline_v2.write_integrity import (
                check_content_integrity,
                verify_write,
            )

            ext = ""
            if "." in path:
                ext = "." + path.rsplit(".", 1)[-1]

            pre_issues = check_content_integrity(content, ext)
            pre_errors = [w for w in pre_issues if w["severity"] == "error"]

            if pre_errors:
                logger.warning(
                    "[llm_tools] Pre-flight found %d issues for %s: %s",
                    len(pre_errors), path,
                    "; ".join(w["message"] for w in pre_errors[:3]),
                )

            # --- Write ---
            ok = await sandbox_tools.write_file(path, content, profile=_tool_profile)
            if not ok:
                return "ERROR: Write failed for " + path

            # --- Write integrity: post-write verify ---
            # v2.3: Skip integrity verification for host-mode writes.
            # Host writes use direct open() — no transport corruption possible.
            # The integrity checker causes false positives from \r\n vs \n diffs.
            _skip_verify = False
            if _tool_profile is not None:
                try:
                    from app.pipeline_v2.android_sandbox import is_android_build
                    _skip_verify = is_android_build(_tool_profile)
                except Exception:
                    pass

            if _skip_verify:
                verify_ok = True
                verify_issues = []
            else:
                verify_ok, verify_issues = await verify_write(path, content)

            if verify_ok:
                # Clean write
                result_msg = "OK: Written " + str(len(content)) + " chars to " + path
                if pre_errors:
                    result_msg += (
                        "\nWARNING: Pre-flight detected potential issues (but write verified OK): "
                        + "; ".join(w["message"] for w in pre_errors[:3])
                    )
                return result_msg
            else:
                # Corruption detected!
                logger.error(
                    "[llm_tools] WRITE INTEGRITY FAILURE for %s: %s",
                    path,
                    "; ".join(i.get("message", i.get("type", "unknown")) for i in verify_issues[:5]),
                )
                error_parts = [
                    "WRITE INTEGRITY ERROR for " + path + ":",
                    "The file was written but read-back verification FAILED.",
                    "The content on disk does not match what you sent.",
                ]
                for issue in verify_issues[:5]:
                    issue_type = issue.get("type", "unknown")
                    if issue_type == "backtick_stripped":
                        error_parts.append(
                            "CORRUPTION: Backticks were stripped during write. "
                            "This is a transport-layer bug. The content was correct "
                            "but the write path corrupted it."
                        )
                        if issue.get("fix_hint"):
                            error_parts.append("FIX: " + issue["fix_hint"])
                    elif issue_type == "dollar_interpolated":
                        error_parts.append(
                            "CORRUPTION: Dollar signs ($) were consumed during write. "
                            "Variable interpolation in the transport layer ate them."
                        )
                    elif issue_type == "line_diff":
                        error_parts.append(
                            "Line " + str(issue.get("line", "?")) + " differs:"
                        )
                        error_parts.append("  Expected: " + str(issue.get("expected", ""))[:80])
                        error_parts.append("  Got:      " + str(issue.get("actual", ""))[:80])
                    else:
                        error_parts.append(
                            issue_type + ": " + issue.get("message", str(issue))
                        )

                error_parts.append(
                    "ACTION: Please re-write the file. The content you generated was correct — "
                    "the transport layer corrupted it. Try writing again."
                )
                return "\n".join(error_parts)

        elif name == "run_shell":
            cmd = args.get("cmd", "")
            # v2.3: Longer timeout for Gradle builds (first run downloads deps)
            _shell_timeout = 600 if (_tool_profile and _tool_profile.build_system == "gradle" and "gradlew" in cmd) else 30
            result = await sandbox_tools.run_shell(cmd, timeout_sec=_shell_timeout, profile=_tool_profile)
            stdout = result.get("stdout", "")[:1000]
            stderr = result.get("stderr", "")[:500]
            rc = result.get("returncode", -1)
            parts = ["exit_code=" + str(rc)]
            if stdout:
                parts.append("STDOUT:\n" + stdout)
            if stderr:
                parts.append("STDERR:\n" + stderr)
            return "\n".join(parts)

        else:
            return "ERROR: Unknown tool: " + name

    except Exception as e:
        return "ERROR: " + name + " failed: " + str(e)
