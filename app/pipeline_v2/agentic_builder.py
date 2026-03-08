# FILE: app/pipeline_v2/agentic_builder.py
"""
ASTRA v2.1 Agentic Builder — one model, one loop, job done.

A single model (GPT-5.4) with full tool access works through
the build in a continuous context window:

  1. Read the spec and scaffold
  2. For each file: read existing content, write the logic, verify syntax
  3. Wire cross-file dependencies (imports, route registrations, props)
  4. Boot the application
  5. Read boot log for errors — fix if needed
  6. Hand off to Verification Model for visual check
  7. If FAIL: fix issues and loop back to step 4
  8. If PASS: done

The Builder has tools: read_file, write_file, run_shell, check_syntax.
It reasons about what to do, calls tools, reads results, continues.

Token budget gate: if approaching 80% of context, write a handover
summary and continue in a fresh context.

v1.0 (2026-03-07): Initial implementation for ASTRA v2.1.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from app.pipeline_v2.config import (
    BUILDER_PROVIDER, BUILDER_MODEL, BUILDER_MAX_OUTPUT,
    FALLBACK_BUILDER_PROVIDER, FALLBACK_BUILDER_MODEL,
    MAX_TOOL_CALLS, HANDOVER_THRESHOLD_PCT,
)
from app.pipeline_v2.models import (
    BuildResult, BuildSession, ScaffoldResult, ToolCall,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt for the Agentic Builder
# ---------------------------------------------------------------------------

BUILDER_SYSTEM = """You are ASTRA's Agentic Builder. You build features by reading, writing, and testing code.

You have tools to interact with the codebase in a Windows Sandbox:
- read_file(path): Read a file's contents
- write_file(path, content): Write or overwrite a file
- run_shell(cmd): Run a PowerShell command (syntax checks, boot app, etc.)

WORKFLOW:
1. Read the spec to understand what needs to be built
2. Read the scaffold files to see what structure exists
3. For each file, read it, write the logic, verify it compiles
4. Wire cross-file dependencies: imports, route registrations, component props
5. When you think you're done, boot the app and check for errors
6. Fix any errors and re-boot until clean

RULES:
- Read a file BEFORE modifying it. Never guess what's in a file.
- For MODIFY files (existing files), make TARGETED changes. Don't rewrite.
- For CREATE files (scaffold stubs), write the complete implementation.
- Check Python syntax after writing: run_shell("python -m py_compile <file>")
- After wiring everything, boot with: run_shell("cd D:\\Orb && .venv\\Scripts\\python.exe -c \\"from main import app; print('BOOT_OK')\\"")
- When you've fixed all errors and the app boots clean, say BUILDER_COMPLETE.

CRITICAL:
- Every import must reference a real file that exists
- Every function/class you import must actually be exported by that file
- Read the file you're importing from to verify the export exists
- Never add a route registration without reading main.py first
- Test compilation after every file write
"""


# ---------------------------------------------------------------------------
# Tool definitions for the LLM
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "Read a file from the sandbox codebase. Returns the file contents as text.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file (e.g. 'app/debug/models.py' or 'src/components/debug/DebugView.tsx')",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file in the sandbox. Creates the file if it doesn't exist, overwrites if it does.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file",
                },
                "content": {
                    "type": "string",
                    "description": "Complete file content to write",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_shell",
        "description": "Run a PowerShell command in the sandbox. Use for: syntax checks, booting the app, npm commands, reading logs.",
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
]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_agentic_builder(
    spec: Dict[str, Any],
    manifest: Dict[str, Any],
    scaffold: ScaffoldResult,
    job_dir: str,
    handover_context: Optional[str] = None,
    on_progress: Optional[Callable[[str], None]] = None,
    existing_messages: Optional[List[Dict]] = None,
) -> BuildResult:
    """Run the Agentic Builder.

    One model, one loop. Reads the spec and scaffold, writes code,
    tests, fixes, until the app boots clean.

    Args:
        spec: Verified spec from SpecGate.
        manifest: Segment manifest.
        scaffold: ScaffoldResult from the Scaffold Engine.
        job_dir: Job directory.
        handover_context: If continuing from a previous session.
        on_progress: Progress callback.

    Returns:
        BuildResult with all files written and session details.
    """
    t_start = time.time()
    emit = on_progress or (lambda msg: None)
    result = BuildResult()

    emit("🤖 Agentic Builder: Starting build loop...")
    emit(f"   Model: {BUILDER_PROVIDER}/{BUILDER_MODEL}")
    emit(f"   Scaffold: {scaffold.total_files} files")
    emit(f"   Max tool calls: {MAX_TOOL_CALLS}")

    # Build the initial prompt
    initial_prompt = _build_initial_prompt(spec, manifest, scaffold, handover_context)
    emit(f"   Initial prompt: {len(initial_prompt):,} chars")

    # Run the agentic loop using proper tool calling
    session = BuildSession(session_number=1)

    try:
        from app.pipeline_v2.llm_tools import run_tool_loop

        files_written = set()

        def on_tool(name, args):
            path = args.get("path", "")
            summary = path if name == "read_file" else (
                f"{path} ({len(args.get('content', ''))} chars)" if name == "write_file"
                else args.get("cmd", "")[:60]
            )
            emit(f"   🔧 {name}({summary})")
            session.tool_calls.append(ToolCall(tool=name, args=summary))
            if name == "write_file" and path:
                files_written.add(path)
                # Push a narrative for each file write
                try:
                    from app.pipeline_v2.orchestrator import _push_narrative
                    _push_narrative(
                        stage="critical_pipeline",
                        title=f"Write: {path.rsplit('/', 1)[-1] if '/' in path else path}",
                        output_summary=f"{len(args.get('content', ''))} chars written",
                        files_touched=[path],
                    )
                except Exception:
                    pass

        def on_text(text):
            if "BUILDER_COMPLETE" in text:
                emit("   ✅ Builder signalled COMPLETE")
                session.completed = True
            else:
                emit(f"   💬 {text[:150]}")

        messages, in_tok, out_tok = await run_tool_loop(
            system_prompt=BUILDER_SYSTEM,
            initial_user_message=initial_prompt,
            provider=BUILDER_PROVIDER,
            model=BUILDER_MODEL,
            max_iterations=MAX_TOOL_CALLS,
            max_tokens=BUILDER_MAX_OUTPUT,
            on_tool_call=on_tool,
            on_text=on_text,
            existing_messages=existing_messages,
        )

        session.files_created = list(files_written)
        session.total_input_tokens = in_tok
        session.total_output_tokens = out_tok
        if not session.completed:
            # Check last message for completion signal
            for msg in reversed(messages):
                if msg.get("role") == "assistant" and "BUILDER_COMPLETE" in (msg.get("content") or ""):
                    session.completed = True
                    break

    except Exception as e:
        logger.exception("[agentic_builder] Build loop failed")
        result.errors.append(f"Build loop error: {e}")

    # Collect results
    result.sessions.append(session)
    result.all_files_written = session.files_created + session.files_modified
    result.total_tool_calls = session.total_tool_calls
    result.total_llm_calls = len([m for m in messages if m.get("role") == "assistant"]) if 'messages' in dir() else 1
    result.total_input_tokens = session.total_input_tokens
    result.total_output_tokens = session.total_output_tokens
    result.total_duration_seconds = time.time() - t_start
    result.success = session.completed and len(result.all_files_written) > 0

    # v2.1.2: Store conversation history so verify→fix can continue in same context
    result.messages_history = messages if 'messages' in dir() else []

    emit(f"\n🤖 Builder complete: {len(result.all_files_written)} files, "
         f"{result.total_tool_calls} tool calls, "
         f"{result.total_duration_seconds:.1f}s")

    return result


# ---------------------------------------------------------------------------
# The agentic loop
# ---------------------------------------------------------------------------

async def _run_build_loop(
    initial_prompt: str,
    session: BuildSession,
    emit: Callable,
) -> BuildSession:
    """Run the agentic build loop with tool calling.

    The model sends tool calls, we execute them and return results,
    the model continues reasoning until it says BUILDER_COMPLETE.
    """
    from app.pipeline_v2.llm_caller import call_llm
    from app.pipeline_v2 import sandbox_tools

    # For now, use a simplified loop:
    # 1. Send the full prompt with tool definitions
    # 2. Parse tool calls from response
    # 3. Execute tools and accumulate results
    # 4. Send results back
    # 5. Repeat until BUILDER_COMPLETE or max tool calls

    messages = [initial_prompt]
    tool_results_context = ""

    for iteration in range(MAX_TOOL_CALLS):
        # Build the user prompt with accumulated tool results
        if tool_results_context:
            user_prompt = (
                f"PREVIOUS TOOL RESULTS:\n{tool_results_context}\n\n"
                f"Continue building. Use tools to read, write, and test. "
                f"Say BUILDER_COMPLETE when done."
            )
        else:
            user_prompt = initial_prompt

        emit(f"\n   --- Iteration {iteration + 1} ---")

        # Call the LLM
        try:
            response = await call_llm(
                provider=BUILDER_PROVIDER,
                model=BUILDER_MODEL,
                system_prompt=BUILDER_SYSTEM,
                user_prompt=user_prompt,
                max_tokens=BUILDER_MAX_OUTPUT,
            )
        except RuntimeError as e:
            emit(f"   ❌ LLM call failed: {e}")
            session.completed = False
            return session

        emit(f"   📝 Response: {len(response):,} chars")

        # Check for completion signal
        if "BUILDER_COMPLETE" in response:
            emit("   ✅ Builder signalled COMPLETE")
            session.completed = True
            break

        # Parse and execute tool calls from response
        tool_calls = _parse_tool_calls(response)

        if not tool_calls:
            emit("   ⚠️ No tool calls in response — asking to continue")
            tool_results_context += f"\n[No tools called. Continue building.]\n"
            continue

        # Execute each tool call
        batch_results = []
        for tc in tool_calls:
            emit(f"   🔧 {tc['name']}({tc.get('args_summary', '')})")

            result_text = await _execute_tool(tc, sandbox_tools)

            # Track the call
            tool_record = ToolCall(
                tool=tc["name"],
                args=tc.get("args_summary", ""),
                result_summary=result_text[:200],
                success="error" not in result_text.lower(),
            )
            session.tool_calls.append(tool_record)

            # Track files created/modified
            if tc["name"] == "write_file":
                path = tc.get("path", "")
                if path not in session.files_created and path not in session.files_modified:
                    session.files_created.append(path)

            batch_results.append(f"[{tc['name']}] {result_text[:500]}")
            emit(f"      → {result_text[:100]}")

        tool_results_context = "\n".join(batch_results)

    else:
        emit(f"   ⚠️ Reached max tool calls ({MAX_TOOL_CALLS})")
        session.completed = False

    return session


# ---------------------------------------------------------------------------
# Tool call parsing and execution
# ---------------------------------------------------------------------------

def _parse_tool_calls(response: str) -> List[Dict[str, Any]]:
    """Parse tool calls from the LLM response.

    The model is expected to output tool calls in a structured format.
    We support multiple formats to be robust:

    Format 1 (function-call style):
        <tool_call>read_file({"path": "app/debug/models.py"})</tool_call>

    Format 2 (JSON):
        {"tool": "read_file", "args": {"path": "app/debug/models.py"}}

    Format 3 (natural language with clear markers):
        TOOL: read_file
        PATH: app/debug/models.py
    """
    import re
    calls = []

    # Format 1: <tool_call> tags
    for m in re.finditer(r'<tool_call>(\w+)\((\{.*?\})\)</tool_call>', response, re.DOTALL):
        name = m.group(1)
        try:
            args = json.loads(m.group(2))
            calls.append({"name": name, **args, "args_summary": _summarise_args(name, args)})
        except json.JSONDecodeError:
            pass

    if calls:
        return calls

    # Format 2: JSON blocks
    for m in re.finditer(r'\{"tool":\s*"(\w+)",\s*"args":\s*(\{.*?\})\}', response, re.DOTALL):
        name = m.group(1)
        try:
            args = json.loads(m.group(2))
            calls.append({"name": name, **args, "args_summary": _summarise_args(name, args)})
        except json.JSONDecodeError:
            pass

    if calls:
        return calls

    # Format 3: TOOL:/PATH:/CMD: markers
    tool_blocks = re.split(r'\nTOOL:\s*', response)
    for block in tool_blocks[1:]:  # Skip first (before any TOOL: marker)
        lines = block.strip().split("\n")
        name = lines[0].strip()
        args = {}
        for line in lines[1:]:
            if line.startswith("PATH:"):
                args["path"] = line[5:].strip()
            elif line.startswith("CMD:"):
                args["cmd"] = line[4:].strip()
            elif line.startswith("CONTENT:"):
                args["content"] = "\n".join(lines[lines.index(line) + 1:])
                break
        if name in ("read_file", "write_file", "run_shell"):
            calls.append({"name": name, **args, "args_summary": _summarise_args(name, args)})

    return calls


async def _execute_tool(tc: Dict[str, Any], sandbox) -> str:
    """Execute a single tool call and return the result as text."""
    name = tc["name"]

    try:
        if name == "read_file":
            path = tc.get("path", "")
            content = await sandbox.read_file(path)
            if content is None:
                return f"ERROR: File not found: {path}"
            return f"OK ({len(content)} chars):\n{content}"

        elif name == "write_file":
            path = tc.get("path", "")
            content = tc.get("content", "")
            ok = await sandbox.write_file(path, content)
            return f"OK: Written {len(content)} chars to {path}" if ok else f"ERROR: Write failed for {path}"

        elif name == "run_shell":
            cmd = tc.get("cmd", "")
            result = await sandbox.run_shell(cmd, timeout_sec=30)
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            rc = result.get("returncode", -1)
            return f"exit={rc}\nSTDOUT: {stdout[:400]}\nSTDERR: {stderr[:200]}"

        else:
            return f"ERROR: Unknown tool: {name}"

    except Exception as e:
        return f"ERROR: {name} failed: {e}"


def _summarise_args(name: str, args: Dict) -> str:
    """Short summary of tool call args for logging."""
    if name == "read_file":
        return args.get("path", "?")
    elif name == "write_file":
        path = args.get("path", "?")
        content_len = len(args.get("content", ""))
        return f"{path} ({content_len} chars)"
    elif name == "run_shell":
        cmd = args.get("cmd", "?")
        return cmd[:60]
    return str(args)[:60]


# ---------------------------------------------------------------------------
# Initial prompt builder
# ---------------------------------------------------------------------------

def _build_initial_prompt(
    spec: Dict[str, Any],
    manifest: Dict[str, Any],
    scaffold: ScaffoldResult,
    handover_context: Optional[str] = None,
) -> str:
    """Build the initial prompt for the Agentic Builder."""
    parts = []

    # Handover from previous session?
    if handover_context:
        parts.append("=== HANDOVER FROM PREVIOUS SESSION ===")
        parts.append(handover_context)
        parts.append("")

    # Spec summary
    parts.append("=== SPECIFICATION ===")
    spec_text = json.dumps(spec, indent=2) if isinstance(spec, dict) else str(spec)
    parts.append(spec_text[:15000])  # Trim if very long
    parts.append("")

    # File scope from manifest
    parts.append("=== FILE SCOPE ===")
    segments = manifest.get("segments", [])
    for seg in segments:
        seg_id = seg.get("segment_id", "")
        files = seg.get("file_scope", [])
        reqs = seg.get("requirements", [])
        parts.append(f"\nSegment: {seg_id}")
        for f in files:
            parts.append(f"  - {f}")
        if reqs:
            parts.append(f"  Requirements:")
            for r in reqs:
                parts.append(f"    - {r}")
    parts.append("")

    # Scaffold summary
    parts.append("=== SCAFFOLD FILES ===")
    parts.append("The Scaffold Engine has written skeleton files. "
                 "CREATE files have stubs with TODO markers. "
                 "MODIFY files need targeted changes to existing code.")
    for sf in scaffold.files:
        tag = "[CREATE]" if sf.is_new else "[MODIFY]"
        parts.append(f"  {tag} {sf.path}")
    parts.append("")

    # Instructions
    parts.append("=== YOUR JOB ===")
    parts.append("1. Read each scaffold file and the files it depends on")
    parts.append("2. Write the business logic to replace TODO markers")
    parts.append("3. Wire imports, route registrations, and component props")
    parts.append("4. For MODIFY files, read the existing file first and make targeted changes")
    parts.append("5. Test syntax after each write")
    parts.append("6. Boot the app when all files are done")
    parts.append("7. Say BUILDER_COMPLETE when the app boots clean")
    parts.append("")
    parts.append("Start by reading the most foundational files (models, types) "
                 "then work outward to routes and UI components.")

    return "\n".join(parts)
