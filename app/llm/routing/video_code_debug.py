# FILE: app/llm/routing/video_code_debug.py
"""Video+Code debug pipeline.

Extracted from app.llm.routing.core to keep the core router smaller and easier to sanity-check.
Behavior is intended to be identical to the legacy implementation.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from app.llm.schemas import LLMTask, LLMResult
from app.jobs.schemas import JobEnvelope
from app.providers.registry import llm_call as registry_llm_call
from app.llm.job_classifier import compute_modality_flags
from app.llm.gemini_vision import transcribe_video_for_context

# Fallback handler (Spec §11)
try:
    from app.llm.fallbacks import (
        handle_video_failure,
        FallbackAction,
    )
    FALLBACKS_AVAILABLE = True
except ImportError:
    FALLBACKS_AVAILABLE = False

logger = logging.getLogger(__name__)


async def run_video_code_debug_pipeline(
    task: LLMTask,
    envelope: JobEnvelope,
    file_map: Optional[str] = None,
) -> LLMResult:
    """2-step pipeline for Video+Code debug jobs."""
    print("[video-code] Starting Video+Code debug pipeline")

    attachments = task.attachments or []
    flags = compute_modality_flags(attachments)

    video_attachments = flags.get("video_attachments", [])
    code_attachments = flags.get("code_attachments", [])

    print(f"[video-code] Found {len(video_attachments)} video(s), {len(code_attachments)} code file(s)")

    # Step 1: Transcribe all videos via Gemini 3 Pro
    video_transcripts = []

    for video_att in video_attachments:
        video_path = None
        if hasattr(video_att, "path") and video_att.path:
            video_path = video_att.path
        elif hasattr(video_att, "filename") and video_att.filename:
            # Attempt to build relative path from attachment filename
            # v0.15.0: Use project_id to locate file if provided
            project_id = getattr(task, 'project_id', 1) or 1
            video_path = f"data/files/{project_id}/{video_att.filename}"

        if video_path:
            print(f"[video-code] Step 1: Transcribing video: {video_att.filename}")
            try:
                transcript = await transcribe_video_for_context(video_path)
                video_transcripts.append({
                    "filename": video_att.filename,
                    "transcript": transcript,
                })
            except Exception as e:
                print(f"[video-code] Transcription failed for {video_att.filename}: {e}")
                # v0.15.0: Use fallback handler if available
                if FALLBACKS_AVAILABLE:
                    action, event = handle_video_failure(
                        str(e),
                        has_code=len(code_attachments) > 0,
                        task_id=envelope.job_id,
                    )
                    if action == FallbackAction.SKIP_STEP:
                        video_transcripts.append({
                            "filename": video_att.filename,
                            "transcript": f"[Video transcription failed: {e}]",
                        })
        else:
            print(f"[video-code] Could not determine path for video attachment: {video_att}")

    # Step 2: Enhance system prompt with transcripts
    transcript_text = ""
    for vt in video_transcripts:
        transcript_text += f"\n\nVIDEO: {vt['filename']}\nTRANSCRIPTS:\n{vt['transcript']}"

    system_content = f"""You are a code debugging assistant.

VIDEO CONTEXT (transcripts from videos provided by the user):
{transcript_text}

Use the video context above to understand what happened and help debug/fix the code.
Focus on:
- Any errors or issues visible in the video
- User actions that led to the problem
- Log output or console messages
- UI state changes"""

    if file_map:
        system_content += f"\n\n{file_map}\n\nIMPORTANT: When referring to files, use the [FILE_X] identifiers above."

    enhanced_messages = [{"role": "system", "content": system_content}]

    for msg in envelope.messages:
        if msg.get("role") != "system":
            enhanced_messages.append(msg)

    envelope.messages = enhanced_messages

    # Step 3: Call Sonnet
    sonnet_model = os.getenv("ANTHROPIC_SONNET_MODEL", "claude-sonnet-4-5-20250929")
    print(f"[video-code] Step 2: Calling Sonnet ({sonnet_model}) with code + transcripts")

    try:
        result = await registry_llm_call(
            provider_id="anthropic",
            model_id=sonnet_model,
            messages=envelope.messages,
            job_envelope=envelope,
        )

        if not result.is_success():
            print(f"[video-code] Sonnet call failed: {result.error_message}")
            return LLMResult(
                content=result.error_message or "Video+Code pipeline failed",
                provider="anthropic",
                model=sonnet_model,
                finish_reason="error",
                error_message=result.error_message,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                raw_response=None,
            )

        print(f"[video-code] Pipeline complete: {len(result.content)} chars")
        return result

    except Exception as exc:
        logger.exception("[video-code] Sonnet call failed: %s", exc)
        return LLMResult(
            content="",
            provider="anthropic",
            model=sonnet_model,
            finish_reason="error",
            error_message=str(exc),
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            raw_response=None,
        )
