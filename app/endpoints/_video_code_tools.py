# FILE: app/endpoints/_video_code_tools.py
"""
Video + Code Tools pipeline.

v9.0 (2026-03-03): Single-model pipeline for compound video+codebase requests.

When the classifier detects VIDEO_CODE_TOOLS (video attachment + codebase intent),
this module:
1. Uploads the video to Gemini File API
2. Sends the video + user message to Gemini 3.1 Pro customtools
3. Gives the model read-only tools (read_file, list_files, search_files)
4. Runs the tool loop to completion
5. Returns the final response

This replaces the old 2-step approach (transcribe video → Sonnet) with a single
model that can both watch the video AND iteratively explore the codebase.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.memory import service as memory_service, schemas as memory_schemas

logger = logging.getLogger(__name__)

# Max tool rounds for video+code (lower than chat — video already costs tokens)
MAX_VIDEO_TOOL_ROUNDS = 15


def route_video_code_tools(
    video_attachments: List[dict],
    user_message: str,
    project_id: int,
    project: Any,
    attachments_summary: list,
    db: Session,
    model: str = "gemini-3.1-pro-preview-customtools",
) -> Any:
    """Route a video+codebase request through the single-model tool pipeline.

    This is a synchronous endpoint (called from chat_with_attachments).
    It runs the async tool loop via asyncio and returns the final response.
    """
    from app.endpoints.chat_attachments import ChatResponse

    print(f"[video-code-tools] Starting pipeline: {len(video_attachments)} video(s), model={model}")

    try:
        reply = asyncio.get_event_loop().run_until_complete(
            _run_video_code_pipeline(
                video_attachments=video_attachments,
                user_message=user_message,
                project_name=project.name,
                project_description=project.description or "",
                model=model,
            )
        )
    except RuntimeError:
        # No event loop running — create one
        reply = asyncio.run(
            _run_video_code_pipeline(
                video_attachments=video_attachments,
                user_message=user_message,
                project_name=project.name,
                project_description=project.description or "",
                model=model,
            )
        )

    provider_str = "google"
    model_str = model

    # Persist messages
    user_content = user_message or "[Video upload]"
    filenames = ", ".join(a.get("filename", "video") for a in video_attachments)
    user_content += f" [Uploaded: {filenames}]"

    memory_service.create_message(db, memory_schemas.MessageCreate(
        project_id=project_id,
        role="user",
        content=user_content,
        provider="local",
    ))
    memory_service.create_message(db, memory_schemas.MessageCreate(
        project_id=project_id,
        role="assistant",
        content=reply,
        provider=provider_str,
        model=model_str,
    ))

    print(f"[video-code-tools] Pipeline complete: {len(reply)} chars")

    return ChatResponse(
        project_id=project_id,
        provider=provider_str,
        model=model_str,
        reply=reply,
        was_reviewed=False,
        critic_review=None,
        attachments_summary=attachments_summary,
    )


def _gather_architecture_context(user_message: str) -> str:
    """Build a condensed architecture summary for the video+code pipeline.

    Combines:
    1. Architecture map summary (project structure, key files)
    2. RAG search results relevant to the user's message
    3. Frontend component tree for UI-related queries

    This gives the model a head start so it doesn't burn tool rounds
    on basic codebase discovery.
    """
    parts = []

    # 1. Load architecture map (condensed — first ~4000 chars covers
    #    executive summary + system architecture + entry points)
    try:
        arch_path = os.path.join("D:\\Orb", ".architecture", "ARCHITECTURE_MAP.md")
        if os.path.isfile(arch_path):
            with open(arch_path, "r", encoding="utf-8") as f:
                arch_content = f.read()
            # Take executive summary + first few sections (~4000 chars)
            # This covers tech stack, entry points, core infrastructure
            lines = arch_content.splitlines()
            summary_lines = []
            char_count = 0
            for line in lines:
                summary_lines.append(line)
                char_count += len(line)
                if char_count > 4000:
                    break
            parts.append("\n".join(summary_lines))
            logger.info("[video-code-tools] Injected architecture map: %d chars", char_count)
    except Exception as e:
        logger.debug("[video-code-tools] Could not load architecture map: %s", e)

    # 2. Frontend component tree from INDEX.json
    try:
        index_path = os.path.join("D:\\Orb", ".architecture", "INDEX.json")
        if os.path.isfile(index_path):
            import json
            with open(index_path, "r", encoding="utf-8") as f:
                idx = json.load(f)

            fe_files = [f for f in idx.get("files", []) if "orb-desktop" in f.get("path", "")]
            if fe_files:
                # Build a compact directory tree
                tree = {}
                for f in fe_files:
                    path = f["path"].replace("D:\\orb-desktop\\", "").replace("\\", "/")
                    dir_part = "/".join(path.split("/")[:-1]) or "."
                    if dir_part not in tree:
                        tree[dir_part] = []
                    tree[dir_part].append(f["name"])

                tree_lines = ["\n### Frontend File Tree (orb-desktop/)"]
                for dir_path in sorted(tree.keys()):
                    if dir_path.startswith("src/"):
                        files = tree[dir_path]
                        tree_lines.append(f"  {dir_path}/: {', '.join(sorted(files))}")
                parts.append("\n".join(tree_lines))
                logger.info("[video-code-tools] Injected frontend tree: %d dirs", len(tree))
    except Exception as e:
        logger.debug("[video-code-tools] Could not load frontend tree: %s", e)

    # 3. RAG search for message-relevant files
    if user_message:
        try:
            from app.db import get_db_sync
            from app.rag.retrieval.arch_search import search_architecture
            db = next(get_db_sync())
            try:
                results = search_architecture(db, query=user_message, top_k=10)
                if results:
                    rag_lines = ["\n### Relevant Files (RAG search)"]
                    for r in results:
                        path = r.get("path", r.get("file_path", ""))
                        score = r.get("score", 0)
                        desc = r.get("description", r.get("summary", ""))[:120]
                        rag_lines.append(f"  {path} (relevance: {score:.2f}) — {desc}")
                    parts.append("\n".join(rag_lines))
                    logger.info("[video-code-tools] Injected RAG results: %d files", len(results))
            finally:
                db.close()
        except Exception as e:
            logger.debug("[video-code-tools] RAG search failed: %s", e)

    if not parts:
        return "No architecture data available. Use tools to explore the codebase."

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Retry / timeout config
# ---------------------------------------------------------------------------
MAX_RETRIES = 3
RETRY_DELAYS = [5, 15, 30]  # seconds between retries
# Vision-only model for two-phase fallback (cheaper, no tools needed)
VISION_EXTRACT_MODEL = "gemini-2.5-flash"


async def _run_video_code_pipeline(
    video_attachments: List[dict],
    user_message: str,
    project_name: str,
    project_description: str,
    model: str,
) -> str:
    """Async pipeline with 3-tier resilience:

    Tier 1: Single-model (video + tools) with retry/backoff
    Tier 2: Two-phase — extract from video first, then tool-ground separately
    Tier 3: Graceful fallback — return raw extraction if grounding fails
    """
    import google.generativeai as genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY not set. Cannot run video+code pipeline."

    genai.configure(api_key=api_key)

    # ---- Upload videos (shared across all tiers) ----
    uploaded_files = await _upload_videos(genai, video_attachments)
    if not uploaded_files:
        return "Error: All video uploads failed."

    arch_context = _gather_architecture_context(user_message)

    # ---- TIER 1: Single-model with retry ----
    tier1_result = await _tier1_single_model(
        genai=genai,
        uploaded_files=uploaded_files,
        user_message=user_message,
        project_name=project_name,
        project_description=project_description,
        model=model,
        arch_context=arch_context,
    )
    if tier1_result:
        await _cleanup_videos(genai, uploaded_files)
        return tier1_result

    # ---- TIER 2: Two-phase (extract then ground) ----
    print("[video-code-tools] TIER 1 failed — falling back to two-phase extraction")
    tier2_result = await _tier2_two_phase(
        genai=genai,
        uploaded_files=uploaded_files,
        user_message=user_message,
        project_name=project_name,
        project_description=project_description,
        model=model,
        arch_context=arch_context,
    )
    await _cleanup_videos(genai, uploaded_files)
    if tier2_result:
        return tier2_result

    # ---- TIER 3: Should never reach here (tier 2 always returns something) ----
    return "Video analysis failed after all retry tiers. Please try with a shorter video or try again later."


async def _upload_videos(genai: Any, video_attachments: List[dict]) -> List[Any]:
    """Upload videos to Gemini File API. Returns list of ready file refs."""
    uploaded = []
    for video_att in video_attachments:
        video_path = video_att.get("path")
        filename = video_att.get("filename", "video.mp4")
        try:
            print(f"[video-code-tools] Uploading video: {filename}")
            video_file = genai.upload_file(path=str(video_path))
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)
            if video_file.state.name == "FAILED":
                print(f"[video-code-tools] Video processing failed: {filename}")
                continue
            uploaded.append(video_file)
            print(f"[video-code-tools] Video ready: {filename}")
        except Exception as e:
            logger.error("[video-code-tools] Video upload failed for %s: %s", filename, e)
    return uploaded


async def _cleanup_videos(genai: Any, uploaded_files: List[Any]) -> None:
    """Delete uploaded videos from Gemini File API."""
    for vf in uploaded_files:
        try:
            genai.delete_file(vf.name)
        except Exception as e:
            print(f"[video-code-tools] Cleanup warning: {e}")


def _build_system_prompt(
    project_name: str,
    project_description: str,
    arch_context: str,
    with_tools: bool = True,
) -> str:
    """Build the system prompt. Shared between tier 1 and tier 2 grounding."""
    tools_section = ""
    if with_tools:
        tools_section = """\nTOOLS AVAILABLE (read-only):
- read_file: Read file contents from the codebase
- list_files: List directory contents
- search_files: Search for files by pattern
- read_logs: Read recent log entries
- read_pipeline_state: Get current pipeline state

DO NOT:
- Create, write, or modify any files
- Execute commands
- Make changes to the codebase
- Guess at file contents — always read them first"""

    workflow = (
        """WORKFLOW:
1. First, watch and understand the video the user has provided
2. Use the architecture map above to identify which files are relevant — don't waste tool calls on discovery
3. Use your tools to read specific files and verify details
4. Ground your analysis in actual code — reference specific files and line numbers
5. Provide a concrete, actionable plan based on what you see in the video AND the code"""
        if with_tools
        else """WORKFLOW:
1. Watch the video carefully and extract EVERYTHING — every feature request, UI element, component mentioned, file referenced, and workflow described
2. Structure your extraction as a detailed specification
3. Reference the architecture map to identify likely file paths for each item mentioned
4. Be exhaustive — this extraction will be used to ground a codebase analysis"""
    )

    return f"""You are ASTRA's debug assistant analysing a video recording of the application.

Project: {project_name}. {project_description}

YOUR ROLE: You are a RESEARCHER with video context AND codebase access.

## PROJECT ROOTS (IMPORTANT)
This project has TWO separate directory roots:
- Backend (Python/FastAPI): D:/Orb/
- Frontend (React/TypeScript/Electron): D:/orb-desktop/

When using tools, ALWAYS use absolute paths:
- For backend files: D:/Orb/app/...
- For frontend files: D:/orb-desktop/src/...
NEVER search D:/Orb for frontend files — they are in D:/orb-desktop.

## CODEBASE ARCHITECTURE
{arch_context}

{workflow}
{tools_section}"""


async def _tier1_single_model(
    genai: Any,
    uploaded_files: List[Any],
    user_message: str,
    project_name: str,
    project_description: str,
    model: str,
    arch_context: str,
) -> Optional[str]:
    """Tier 1: Video + tools in one model call, with retry/backoff."""
    from app.llm.chat_tool_loop import get_chat_tools, TOOL_TIER_READ
    from app.llm._streaming_utils_3 import _convert_tools_to_gemini

    tools = get_chat_tools(TOOL_TIER_READ)
    gemini_tools = _convert_tools_to_gemini(tools)
    system_prompt = _build_system_prompt(project_name, project_description, arch_context, with_tools=True)

    model_kwargs = {"model_name": model}
    model_kwargs["system_instruction"] = system_prompt
    if gemini_tools:
        model_kwargs["tools"] = gemini_tools
    gemini_model = genai.GenerativeModel(**model_kwargs)

    prompt_parts = list(uploaded_files)
    prompt_parts.append(user_message or "Analyse this video and help me understand what's happening.")

    for attempt in range(MAX_RETRIES):
        try:
            print(f"[video-code-tools] TIER 1 attempt {attempt + 1}/{MAX_RETRIES}")
            result = await _run_tool_loop(
                gemini_model=gemini_model,
                initial_parts=prompt_parts,
                max_rounds=MAX_VIDEO_TOOL_ROUNDS,
            )
            if result and not result.startswith("Error"):
                print(f"[video-code-tools] TIER 1 succeeded on attempt {attempt + 1}")
                return result
        except Exception as e:
            is_timeout = "504" in str(e) or "Deadline" in str(e) or "timeout" in str(e).lower()
            logger.warning(
                "[video-code-tools] TIER 1 attempt %d failed%s: %s",
                attempt + 1, " (timeout)" if is_timeout else "", e,
            )
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                print(f"[video-code-tools] Retrying in {delay}s...")
                await asyncio.sleep(delay)

    print("[video-code-tools] TIER 1 exhausted all retries")
    return None


async def _tier2_two_phase(
    genai: Any,
    uploaded_files: List[Any],
    user_message: str,
    project_name: str,
    project_description: str,
    model: str,
    arch_context: str,
) -> str:
    """Tier 2: Extract from video first (no tools), then ground with tools.

    Phase A: Cheap/fast model watches video and extracts everything into text.
    Phase B: Tools model gets the extraction + architecture and grounds it in code.

    If Phase B fails, returns Phase A extraction as graceful fallback (Tier 3).
    """
    # ---- Phase A: Video extraction (no tools, simpler model) ----
    extraction = None
    extract_prompt = _build_system_prompt(
        project_name, project_description, arch_context, with_tools=False,
    )

    try:
        print(f"[video-code-tools] TIER 2 Phase A: extracting with {VISION_EXTRACT_MODEL}")
        extract_model = genai.GenerativeModel(
            model_name=VISION_EXTRACT_MODEL,
            system_instruction=extract_prompt,
        )
        prompt_parts = list(uploaded_files)
        prompt_parts.append(
            user_message
            + "\n\nExtract EVERYTHING from this video: every feature, UI element, "
            "file reference, component name, workflow step, and requirement. "
            "Be exhaustive — this will be used to build a grounded implementation plan."
        )
        response = extract_model.generate_content(prompt_parts)
        extraction = response.text
        print(f"[video-code-tools] TIER 2 Phase A complete: {len(extraction)} chars extracted")
    except Exception as e:
        logger.error("[video-code-tools] TIER 2 Phase A (extraction) failed: %s", e)
        return "Video analysis failed: could not extract content from video. Please try with a shorter video."

    if not extraction:
        return "Video analysis failed: extraction returned empty. Please try again."

    # ---- Phase B: Ground extraction with tools (no video, just text) ----
    try:
        from app.llm.chat_tool_loop import get_chat_tools, TOOL_TIER_READ
        from app.llm._streaming_utils_3 import _convert_tools_to_gemini

        print(f"[video-code-tools] TIER 2 Phase B: grounding with {model}")
        tools = get_chat_tools(TOOL_TIER_READ)
        gemini_tools = _convert_tools_to_gemini(tools)

        ground_prompt = f"""You are ASTRA's debug assistant grounding a video analysis in actual code.

Project: {project_name}. {project_description}

## CODEBASE ARCHITECTURE
{arch_context}

WORKFLOW:
1. Read the video extraction below carefully
2. Use your tools to read the specific files mentioned and verify they match the description
3. Ground every claim in actual code — reference specific files and line numbers
4. Produce a concrete, actionable plan with file paths and code references

TOOLS AVAILABLE (read-only):
- read_file, list_files, search_files, read_logs, read_pipeline_state

DO NOT guess at file contents — always read them first."""

        model_kwargs = {"model_name": model}
        model_kwargs["system_instruction"] = ground_prompt
        if gemini_tools:
            model_kwargs["tools"] = gemini_tools
        ground_model = genai.GenerativeModel(**model_kwargs)

        grounding_message = (
            f"Here is a detailed extraction from the user's video:\n\n"
            f"---\n{extraction}\n---\n\n"
            f"Original user request: {user_message}\n\n"
            f"Now ground this in the actual codebase. Read the relevant files "
            f"and produce a concrete implementation plan with real file paths and line numbers."
        )

        result = await _run_tool_loop(
            gemini_model=ground_model,
            initial_parts=[grounding_message],
            max_rounds=MAX_VIDEO_TOOL_ROUNDS,
        )
        if result and not result.startswith("Error"):
            print(f"[video-code-tools] TIER 2 Phase B complete: {len(result)} chars")
            return result

    except Exception as e:
        logger.warning("[video-code-tools] TIER 2 Phase B (grounding) failed: %s", e)

    # ---- Tier 3: Graceful fallback — return raw extraction ----
    print("[video-code-tools] TIER 3: Returning ungrounded extraction as fallback")
    return (
        f"**Note: Video extraction completed but codebase grounding failed. "
        f"The analysis below is based on video content only — file references are estimated "
        f"from the architecture map, not verified by reading the actual code.**\n\n"
        f"{extraction}"
    )


async def _run_tool_loop(
    gemini_model: Any,
    initial_parts: list,
    max_rounds: int = 15,
) -> str:
    """Run the Gemini function calling loop to completion.

    Sends the initial prompt, checks for function calls, executes them,
    sends results back, and repeats until the model produces a text response.
    """
    from app.llm.chat_tool_loop import execute_chat_tool
    from google.ai import generativelanguage as glm
    from google.protobuf import struct_pb2

    chat = gemini_model.start_chat()

    # First message includes the video
    response = chat.send_message(initial_parts)

    for round_num in range(max_rounds):
        # Check for function calls
        function_calls = []
        text_parts = []

        for candidate in response.candidates:
            if not hasattr(candidate, "content") or not candidate.content:
                continue
            for part in candidate.content.parts:
                fc = getattr(part, "function_call", None)
                if fc and fc.name:
                    args_dict = dict(fc.args) if fc.args else {}
                    function_calls.append({"name": fc.name, "args": args_dict})
                else:
                    text = getattr(part, "text", None)
                    if text:
                        text_parts.append(text)

        if not function_calls:
            # No more tool calls — return the text
            return "\n".join(text_parts) if text_parts else "Analysis complete (no text response)."

        # Execute tool calls
        print(f"[video-code-tools] Round {round_num + 1}: executing {len(function_calls)} tool call(s)")
        function_responses = []

        for fc in function_calls:
            print(f"[video-code-tools]   → {fc['name']}({list(fc['args'].keys())})")
            result_str = await execute_chat_tool(fc["name"], fc["args"])

            resp_struct = struct_pb2.Struct()
            resp_struct.update({"result": result_str})
            function_responses.append(
                glm.Part(
                    function_response=glm.FunctionResponse(
                        name=fc["name"],
                        response=resp_struct,
                    )
                )
            )

        # Send tool results back
        response = chat.send_message(function_responses)

    return "Analysis reached maximum tool rounds. Partial results may be available above."
