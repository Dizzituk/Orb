# FILE: app/content/video_pipeline/director.py
# Purpose: Director Layer — creative intelligence for the video pipeline.
# Called-by: app.content.video_pipeline.orchestrator
# Depends-on: app.content.video_pipeline.models, app.content.video_pipeline.director_prompts, app.content.video_pipeline.qa_gate
# Last-renovated: 2026-06-21
"""
Director Layer — creative intelligence for the video pipeline.

The Director is Gemini 3.1 Pro operating across production phases:
1. Pre-Production: Creative decisions about hooks, pacing, visual intent
2. Post-Production QA: Quality gate that watches the assembled video

Uses the same model as the script analyzer (PIPELINE_GEMINI_MODEL)
to maintain consistent creative context.

BATCH 4 split: the prompt-text constants moved to director_prompts.py and the QA gate
(run_qa_gate + lessons store) moved to qa_gate.py; both are re-exported below so
this module's public surface is unchanged.
"""
import json
import logging
import os
from typing import Optional, Dict, Any, List

import google.generativeai as genai

from app.content.video_pipeline.models import (
    PIPELINE_GEMINI_MODEL,
    ScenePlan,
    StyleProfile,
)
from app.content.video_pipeline.director_prompts import (
    DIRECTOR_SYSTEM, DIRECTOR_USER,
    QA_GATE_SYSTEM, QA_GATE_USER,
    SPLIT_SEGMENT_SYSTEM, SPLIT_SEGMENT_USER,
)
from app.content.video_pipeline.qa_gate import (
    run_qa_gate, _save_qa_lessons, get_qa_lessons,
)

logger = logging.getLogger(__name__)


async def run_director_review(
    scene_plan: ScenePlan,
    style_profile: Optional[StyleProfile] = None,
    target_platform: str = "youtube_longform",
) -> Dict[str, Any]:
    """
    Run the Director's pre-production review on a scene plan.

    Analyses the plan for hooks, pacing, visual intent, and transitions.
    Returns creative direction that the pipeline uses to improve the output.
    """
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        logger.warning("[director] No GOOGLE_API_KEY — skipping director review")
        return {}

    genai.configure(api_key=api_key)

    # Build style notes
    style_notes = "No style profile"
    if style_profile:
        style_notes = (
            f"Pacing: {style_profile.segment_rhythm}, "
            f"avg cut: {style_profile.avg_cut_duration_s}s, "
            f"tone: {style_profile.overall_tone}, "
            f"energy: {style_profile.energy_level}, "
            f"b-roll: {style_profile.b_roll_density}, "
            f"avatar: {style_profile.avatar_frequency}"
        )

    # Serialise scene plan for the prompt
    plan_data = scene_plan.model_dump()
    slim_segments = []
    for seg in plan_data["segments"]:
        slim_segments.append({
            "segment_id": seg["segment_id"],
            "segment_type": seg["segment_type"],
            "script_text": seg["script_text"][:200],
            "visual_description": seg["visual_description"],
            "search_keywords": seg["search_keywords"],
            "estimated_duration_s": seg["estimated_duration_s"],
            "requires_avatar": seg["requires_avatar"],
            "avatar_framing": seg["avatar_framing"],
        })

    scene_plan_json = json.dumps({
        "title": plan_data["title"],
        "total_segments": plan_data["total_segments"],
        "estimated_total_duration_s": plan_data["estimated_total_duration_s"],
        "segments": slim_segments,
    }, indent=2)

    # Include lessons from previous QA failures
    qa_lessons = get_qa_lessons()

    user_prompt = DIRECTOR_USER.format(
        title=scene_plan.title,
        target_platform=target_platform,
        style_notes=style_notes,
        scene_plan_json=scene_plan_json,
    )

    # Append QA lessons if any exist
    if qa_lessons and "No previous" not in qa_lessons:
        user_prompt += f"\n\n{qa_lessons}"

    logger.info(
        f"[director] Running pre-production review on "
        f"'{scene_plan.title}' ({scene_plan.total_segments} segments)"
    )

    model = genai.GenerativeModel(
        model_name=PIPELINE_GEMINI_MODEL,
        system_instruction=DIRECTOR_SYSTEM,
        generation_config={
            "temperature": 0.4,
            "response_mime_type": "application/json",
        },
    )

    response = model.generate_content(user_prompt)
    raw_text = response.text.strip()

    try:
        direction = json.loads(raw_text)
    except json.JSONDecodeError:
        if "```json" in raw_text:
            json_str = raw_text.split("```json")[1].split("```")[0].strip()
            direction = json.loads(json_str)
        else:
            logger.error(
                f"[director] Failed to parse direction: {raw_text[:300]}"
            )
            return {}

    logger.info(
        f"[director] Review complete — "
        f"hook: {direction.get('hook_assessment', '?')}, "
        f"score: {direction.get('overall_quality_score', '?')}/10, "
        f"AI budget: {len(direction.get('ai_budget_segments', []))} segments"
    )

    return direction


def apply_director_notes(
    scene_plan: ScenePlan,
    direction: Dict[str, Any],
) -> ScenePlan:
    """
    Apply the Director's creative direction to the scene plan.

    Updates search keywords, visual descriptions, and stores
    director notes and transition types in segment metadata.
    """
    if not direction or "segments" not in direction:
        return scene_plan

    dir_segments = {
        s["segment_id"]: s for s in direction.get("segments", [])
    }
    ai_budget = set(direction.get("ai_budget_segments", []))

    for segment in scene_plan.segments:
        dir_seg = dir_segments.get(segment.segment_id)
        if not dir_seg:
            continue

        # Update search keywords if the director revised them
        revised_kw = dir_seg.get("revised_search_keywords")
        if revised_kw:
            segment.search_keywords = revised_kw

        # Update visual description if revised
        revised_desc = dir_seg.get("revised_visual_description")
        if revised_desc:
            segment.visual_description = revised_desc

        # Update duration if the director adjusted it
        revised_dur = dir_seg.get("revised_duration_s")
        if revised_dur and isinstance(revised_dur, (int, float)):
            segment.estimated_duration_s = float(revised_dur)

        # Mark segments that need AI generation
        visual_intent = dir_seg.get("visual_intent", "stock")
        if (
            visual_intent in ("ai_generate", "diagram")
            and segment.segment_id in ai_budget
        ):
            from app.content.video_pipeline.models import AssetTier
            segment.priority_tier = AssetTier.AI_GENERATED

    # Store full director notes in scene plan metadata
    scene_plan.metadata["director_notes"] = direction
    scene_plan.metadata["hook_assessment"] = direction.get(
        "hook_assessment", "unknown"
    )
    scene_plan.metadata["overall_quality_score"] = direction.get(
        "overall_quality_score", 0
    )

    # Hook rewrite: inject a new b-roll hook segment before the avatar intro
    hook_rewrite = direction.get("hook_rewrite")
    if hook_rewrite and hook_rewrite.lower() not in ("null", "none", ""):
        # Clean stage directions that TTS would read literally
        import re as _re
        hook_rewrite = _re.sub(
            r'\[.*?\]|\(.*?\)|cut to host|cut to|fade in|fade out',
            '', hook_rewrite, flags=_re.IGNORECASE,
        ).strip()
        # Remove leading/trailing quotes if wrapped
        hook_rewrite = hook_rewrite.strip('"\'')
        hook_rewrite = hook_rewrite.strip()

    # Don't inject a hook if the script already opens with b-roll
    # (the author already wrote a hook). Only inject if seg_001 is avatar.
    first_seg = scene_plan.segments[0] if scene_plan.segments else None
    script_already_has_hook = (
        first_seg
        and not first_seg.requires_avatar
        and first_seg.segment_type.value != "avatar"
    )
    if script_already_has_hook:
        logger.info(
            "[director] Script already opens with b-roll — "
            "skipping hook injection"
        )
        hook_rewrite = None

    if hook_rewrite and len(hook_rewrite) > 10:
        from app.content.video_pipeline.models import (
            SceneSegment, SegmentType, AssetTier, AvatarFraming,
        )
        hook_seg = SceneSegment(
            segment_id="seg_000_hook",
            segment_type=SegmentType.INTRO,
            script_text=hook_rewrite,
            visual_description=(
                "Bold, attention-grabbing visual. "
                "Fast-paced, high-energy b-roll."
            ),
            search_keywords=["dramatic close up", "technology abstract"],
            mood_tags=["intense", "urgent"],
            estimated_duration_s=3.0,
            requires_avatar=False,
            avatar_framing=AvatarFraming.NONE,
            priority_tier=AssetTier.FREE_STOCK,
        )
        scene_plan.segments.insert(0, hook_seg)
        scene_plan.total_segments = len(scene_plan.segments)
        logger.info(
            f"[director] Injected hook rewrite: '{hook_rewrite[:60]}...'"
        )

    logger.info(
        f"[director] Applied notes to {len(dir_segments)} segments, "
        f"{len(ai_budget)} marked for AI generation"
    )

    return scene_plan


async def split_long_segment(
    segment_id: str,
    script_text: str,
    duration_s: float,
    keywords: list,
) -> List[Dict[str, Any]]:
    """
    Ask the Director to split a long b-roll segment into shorter
    sub-segments with different visual keywords for each.

    Returns a list of sub-segment dicts, or empty list on failure.
    """
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        return []

    genai.configure(api_key=api_key)

    user_prompt = SPLIT_SEGMENT_USER.format(
        segment_id=segment_id,
        duration_s=duration_s,
        script_text=script_text,
        keywords=keywords,
    )

    logger.info(
        f"[director] Splitting {segment_id} ({duration_s:.0f}s) "
        f"into sub-segments"
    )

    model = genai.GenerativeModel(
        model_name=PIPELINE_GEMINI_MODEL,
        system_instruction=SPLIT_SEGMENT_SYSTEM,
        generation_config={
            "temperature": 0.3,
            "response_mime_type": "application/json",
        },
    )

    try:
        response = model.generate_content(user_prompt)
        raw_text = response.text.strip()

        try:
            sub_segments = json.loads(raw_text)
        except json.JSONDecodeError:
            if "```json" in raw_text:
                json_str = raw_text.split("```json")[1].split("```")[0].strip()
                sub_segments = json.loads(json_str)
            else:
                logger.error(
                    f"[director] Failed to parse split: {raw_text[:200]}"
                )
                return []

        if not isinstance(sub_segments, list) or len(sub_segments) < 2:
            logger.warning(
                f"[director] Split returned {len(sub_segments) if isinstance(sub_segments, list) else 0} "
                f"sub-segments, expected 2-4"
            )
            return []

        logger.info(
            f"[director] Split {segment_id} into "
            f"{len(sub_segments)} sub-segments"
        )
        return sub_segments

    except Exception as e:
        logger.warning(f"[director] Segment split failed: {e}")
        return []
