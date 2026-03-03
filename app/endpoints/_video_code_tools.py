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


async def _run_video_code_pipeline(
    video_attachments: List[dict],
    user_message: str,
    project_name: str,
    project_description: str,
    model: str,
) -> str:
    """Async pipeline: upload video, call model with tools, return text."""
    import google.generativeai as genai
    from app.llm.chat_tool_loop import get_chat_tools, execute_chat_tool, TOOL_TIER_READ

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY not set. Cannot run video+code pipeline."

    genai.configure(api_key=api_key)

    # Step 1: Upload videos to Gemini File API
    uploaded_files = []
    try:
        for video_att in video_attachments:
            video_path = video_att.get("path")
            filename = video_att.get("filename", "video.mp4")
            print(f"[video-code-tools] Uploading video: {filename}")

            video_file = genai.upload_file(path=str(video_path))

            # Wait for processing
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                print(f"[video-code-tools] Video processing failed: {filename}")
                continue

            uploaded_files.append(video_file)
            print(f"[video-code-tools] Video ready: {filename}")

    except Exception as e:
        logger.error("[video-code-tools] Video upload failed: %s", e)
        return f"Error uploading video: {e}"

    if not uploaded_files:
        return "Error: All video uploads failed."

    # Step 2: Build tools (read-only)
    tools = get_chat_tools(TOOL_TIER_READ)

    # Convert to Gemini format
    from app.llm._streaming_utils_3 import _convert_tools_to_gemini
    gemini_tools = _convert_tools_to_gemini(tools)

    # Step 3: Build system prompt
    system_prompt = f"""You are ASTRA's debug assistant analysing a video recording of the application.

Project: {project_name}. {project_description}

YOUR ROLE: You are a RESEARCHER with video context AND codebase access.

WORKFLOW:
1. First, watch and understand the video the user has provided
2. Then use your tools to explore the codebase and find relevant files
3. Ground your analysis in actual code — reference specific files and line numbers
4. Provide a concrete, actionable plan based on what you see in the video AND the code

TOOLS AVAILABLE (read-only):
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

    # Step 4: Build model with tools
    model_kwargs = {"model_name": model}
    if system_prompt:
        model_kwargs["system_instruction"] = system_prompt
    if gemini_tools:
        model_kwargs["tools"] = gemini_tools

    gemini_model = genai.GenerativeModel(**model_kwargs)

    # Step 5: Build initial prompt with video
    prompt_parts = list(uploaded_files)  # Video file references
    prompt_text = user_message or "Analyse this video and help me understand what's happening."
    prompt_parts.append(prompt_text)

    # Step 6: Run tool loop
    try:
        reply_text = await _run_tool_loop(
            gemini_model=gemini_model,
            initial_parts=prompt_parts,
            max_rounds=MAX_VIDEO_TOOL_ROUNDS,
        )
    except Exception as e:
        logger.error("[video-code-tools] Tool loop failed: %s", e)
        reply_text = f"Error during analysis: {e}"

    # Step 7: Clean up uploaded videos
    for vf in uploaded_files:
        try:
            genai.delete_file(vf.name)
        except Exception as cleanup_err:
            print(f"[video-code-tools] Cleanup warning: {cleanup_err}")

    return reply_text


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
