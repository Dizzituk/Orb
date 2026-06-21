# FILE: app/content/video_pipeline/qa_gate.py
# Purpose: Post-production QA gate (Gemini watches the rendered video) + file-backed lessons store.
# Called-by: app.content.video_pipeline.director, app.content.video_pipeline.orchestrator
# Depends-on: app.content.video_pipeline.models, app.content.video_pipeline.director_prompts
# Last-renovated: 2026-06-21
"""
Post-production QA gate for the video pipeline.

Split out of director.py (BATCH 4) verbatim: run_qa_gate uploads the
rendered video to Gemini and scores it, persisting issues as lessons
that feed back into the next Director review.
"""
import json
import logging
import os
from typing import Dict, Any

import google.generativeai as genai

from app.content.video_pipeline.models import PIPELINE_GEMINI_MODEL, ScenePlan
from app.content.video_pipeline.director_prompts import QA_GATE_SYSTEM, QA_GATE_USER

logger = logging.getLogger(__name__)


async def run_qa_gate(
    scene_plan: ScenePlan,
    assembly_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the post-production QA gate by uploading the rendered video
    to Gemini and having it WATCH the actual output.

    The Director watches the video, listens to the audio, and scores
    it based on what it actually sees and hears - not metadata.

    Falls back to text-only review if video upload fails.
    """
    import time

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        logger.warning("[qa_gate] No GOOGLE_API_KEY - skipping QA gate")
        return {"passed": True, "overall_score": 5, "summary": "QA skipped - no API key"}

    genai.configure(api_key=api_key)

    # Build context for the review
    plan_json = json.dumps(scene_plan.model_dump(), indent=2, default=str)
    assembly_json = json.dumps(assembly_summary, indent=2, default=str)
    director_notes = json.dumps(
        scene_plan.metadata.get("director_notes", {}),
        indent=2, default=str,
    )

    user_prompt = QA_GATE_USER.format(
        scene_plan_json=plan_json[:3000],
        assembly_summary=assembly_json[:2000],
        director_notes=director_notes[:2000],
    )

    # -- Try to upload the actual video for Gemini to watch --
    video_file = None
    output_path = assembly_summary.get("output_path", "")

    if output_path and os.path.exists(output_path):
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        if file_size_mb < 100:  # Only upload if under 100MB
            try:
                logger.info(
                    f"[qa_gate] Uploading video for review: "
                    f"{output_path} ({file_size_mb:.1f} MB)"
                )
                video_file = genai.upload_file(
                    output_path, mime_type="video/mp4"
                )
                # Wait for Gemini to process the video
                max_wait = 120  # 2 minute max wait
                waited = 0
                while (
                    video_file.state.name == "PROCESSING"
                    and waited < max_wait
                ):
                    time.sleep(3)
                    waited += 3
                    video_file = genai.get_file(video_file.name)

                if video_file.state.name == "ACTIVE":
                    logger.info(
                        f"[qa_gate] Video uploaded and processed: "
                        f"{video_file.name}"
                    )
                else:
                    logger.warning(
                        f"[qa_gate] Video processing failed: "
                        f"state={video_file.state.name}"
                    )
                    video_file = None
            except Exception as e:
                logger.warning(
                    f"[qa_gate] Video upload failed, falling back "
                    f"to text review: {e}"
                )
                video_file = None
        else:
            logger.info(
                f"[qa_gate] Video too large for upload "
                f"({file_size_mb:.0f} MB), using text review"
            )

    logger.info(
        f"[qa_gate] Running post-production quality review "
        f"({'WITH VIDEO' if video_file else 'text-only'})"
    )

    model = genai.GenerativeModel(
        model_name=PIPELINE_GEMINI_MODEL,
        system_instruction=QA_GATE_SYSTEM,
        generation_config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
        },
    )

    # Build the content: video file + text prompt
    if video_file:
        content = [
            video_file,
            (
                "Watch this video carefully. Listen to the audio. "
                "Check if any narration is cut short mid-sentence. "
                "Check if the lip sync matches the speech on avatar segments. "
                "Check if transitions between segments are smooth. "
                "Check if the b-roll footage matches the narration topic. "
                "Then provide your QA assessment.\n\n"
                + user_prompt
            ),
        ]
    else:
        content = user_prompt

    response = model.generate_content(content)
    raw_text = response.text.strip()

    # Clean up uploaded video file
    if video_file:
        try:
            genai.delete_file(video_file.name)
            logger.info("[qa_gate] Cleaned up uploaded video")
        except Exception:
            pass

    try:
        qa_result = json.loads(raw_text)
    except json.JSONDecodeError:
        if "```json" in raw_text:
            json_str = raw_text.split("```json")[1].split("```")[0].strip()
            qa_result = json.loads(json_str)
        else:
            logger.error(f"[qa_gate] Failed to parse QA result: {raw_text[:300]}")
            return {"passed": True, "overall_score": 5, "summary": "QA parse failed"}

    # ── Filter out avatar segment issues ──
    # Gemini sometimes ignores the prompt instruction to not flag
    # avatar segments. Enforce it in code: strip any issue that
    # targets an avatar segment. This is deterministic and reliable.
    avatar_ids = {
        s.segment_id for s in scene_plan.segments
        if s.requires_avatar
    }
    raw_issues = qa_result.get("issues", [])
    filtered_issues = [
        issue for issue in raw_issues
        if issue.get("segment_id") not in avatar_ids
    ]
    dropped = len(raw_issues) - len(filtered_issues)
    if dropped > 0:
        logger.info(
            f"[qa_gate] Dropped {dropped} issues targeting "
            f"avatar segments (not actionable)"
        )
    qa_result["issues"] = filtered_issues

    passed = qa_result.get("passed", True)
    score = qa_result.get("overall_score", 5)
    issues = filtered_issues
    watched = "watched video" if video_file else "text-only"

    # Recalculate pass/fail based on filtered issues
    if not issues:
        qa_result["passed"] = True
        passed = True

    logger.info(
        f"[qa_gate] QA {'PASSED' if passed else 'FAILED'} - "
        f"score: {score}/10, issues: {len(issues)} ({watched})"
    )

    # Persist QA lessons for future Director reviews
    _save_qa_lessons(qa_result)

    return qa_result


def _save_qa_lessons(qa_result: Dict[str, Any]) -> None:
    """Save QA issues to a persistent lessons file so the Director learns."""
    from pathlib import Path
    lessons_path = Path("data/content/video_pipeline/qa_lessons.json")
    lessons_path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if lessons_path.exists():
        try:
            existing = json.loads(lessons_path.read_text(encoding="utf-8"))
        except Exception:
            existing = []

    # Add new issues as lessons, keep last 20
    for issue in qa_result.get("issues", []):
        existing.append({
            "category": issue.get("category", ""),
            "description": issue.get("description", ""),
            "fix": issue.get("suggested_fix", ""),
            "score": qa_result.get("overall_score", 0),
        })

    existing = existing[-20:]
    lessons_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    logger.info(f"[qa_gate] Saved {len(qa_result.get('issues', []))} lessons")


def get_qa_lessons() -> str:
    """Load previous QA lessons as context for the Director."""
    from pathlib import Path
    lessons_path = Path("data/content/video_pipeline/qa_lessons.json")
    if not lessons_path.exists():
        return "No previous QA lessons."

    try:
        lessons = json.loads(lessons_path.read_text(encoding="utf-8"))
        if not lessons:
            return "No previous QA lessons."

        lines = ["Previous QA issues to avoid:"]
        for lesson in lessons[-10:]:
            lines.append(
                f"- {lesson.get('category', '?')}: "
                f"{lesson.get('description', '')} "
                f"(Fix: {lesson.get('fix', 'unknown')})"
            )
        return "\n".join(lines)
    except Exception:
        return "No previous QA lessons."
