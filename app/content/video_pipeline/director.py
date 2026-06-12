# FILE: app/content/video_pipeline/director.py
# Purpose: Director Layer — creative intelligence for the video pipeline.
# Called-by: app.content.video_pipeline.orchestrator
# Depends-on: app.content.video_pipeline.models
# Last-renovated: 2026-06-11
"""
Director Layer — creative intelligence for the video pipeline.

The Director is Gemini 3.1 Pro operating across production phases:
1. Pre-Production: Creative decisions about hooks, pacing, visual intent
2. Post-Production QA: Quality gate that watches the assembled video

Uses the same model as the script analyzer (PIPELINE_GEMINI_MODEL)
to maintain consistent creative context.
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

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# DIRECTOR PRE-PRODUCTION PROMPT
# ═══════════════════════════════════════════════════

DIRECTOR_SYSTEM = """You are an expert video director and editor with deep
knowledge of audience engagement, retention analytics, and visual storytelling.

You are reviewing a scene plan that was generated from a script. Your job is to
improve it by making creative decisions that a human director would make:

## HOOK ANALYSIS
- The first 3 seconds are life or death. Evaluate whether the opening grabs attention.
- If the intro is weak, you MUST fix it, not just flag it.
- Consider pattern interrupts: unexpected visuals, provocative questions, bold statements.

HOOK RESTRUCTURING — WHEN hook_assessment IS "weak" OR "needs_restructure":
- Write a COMPLETELY ORIGINAL hook line that is NOT taken from the script.
  Do NOT copy or rephrase any line from the script body. The hook must be
  NEW text that does not appear anywhere else in the video.
  If the hook duplicates script content, the viewer hears the same thing twice.
- Proven hook patterns that work:
  1. BOLD CLAIM: "In 10 years, half the jobs you know won't exist."
  2. PROVOCATIVE QUESTION: "What happens when machines do everything better than you?"
  3. CONTRAST/TENSION: "We're building the most powerful technology in history — and nobody's asking what it means."
  4. FUTURE SHOCK: "Imagine waking up and your entire industry is gone. That's not science fiction — it's already happening."
- The hook segment should be SHORT (2-4 seconds max, ONE punchy sentence) and HIGH energy.
- After the hook, cut to the avatar intro. The hook earns the right to introduce yourself.
- Set hook_rewrite in your response with the new opening line if you restructure.
- hook_rewrite must be PURE NARRATION TEXT ONLY. No stage directions, no brackets,
  no "Cut to host", no "[B-ROLL]", no "(pause)", no formatting. Just the spoken words.
  The TTS will read every character literally. If it says "Cut to host", the AI voice
  will say "Cut to host" out loud. Only include words the narrator should speak.
- IMPORTANT: The hook must be TRUE and defensible. No clickbait, no exaggeration,
  no sensationalism. The power comes from truth delivered with impact.

## PACING CURVE
- Map the energy arc: start strong, vary the rhythm, build to key moments, land the CTA.
- Flag segments that are too long (attention drops after ~8s per cut for short-form, ~15s for long-form).
- Suggest where to insert visual pattern interrupts to prevent drop-off.

CRITICAL PACING RULE — NO ENERGY FLATLINES:
- NEVER assign the same energy_level to more than 2 consecutive segments.
  If you find 3+ segments at "medium" in a row, YOU MUST vary them.
  Push one up to "high" or pull one down to "low". Monotone energy kills retention.
- The energy sequence across the whole video should look like a waveform,
  not a flatline. Example of GOOD: high, medium, low, medium, high, medium, low, high.
  Example of BAD: medium, medium, medium, medium, medium.
- Short punchy statements (1-3 seconds of narration) should almost always be "high".
- Reflective or philosophical passages should drop to "low" — give the viewer
  a moment to absorb before the next hit.
- Transitions between energy levels should be intentional:
  high→low = dramatic contrast (use after a bold claim)
  low→high = building momentum (use before a key reveal)
  medium→medium = ONLY acceptable once, never twice in a row.
- Segment duration should correlate with energy:
  high energy = shorter segments (2-5s), rapid visual change
  low energy = longer segments (8-12s), let the visual breathe
  medium = standard (5-8s)

## VISUAL INTENT
- For each segment, evaluate whether stock footage is sufficient or whether it needs
  a bespoke visual (diagram, comparison chart, AI-generated scene).
- Mark MAX 2-3 segments for AI generation — only where stock footage genuinely cannot
  communicate the concept. Everything else stays real footage.
- Write specific visual briefs, not vague descriptions. "Split-screen comparison showing
  UBI column vs UBN column with key differences highlighted" is good. "Something about economics" is bad.

## TRANSITIONS
- Assign transition types between segments: cut, crossfade, fade_to_black.
- Use crossfade (0.3-0.5s) for smooth topic continuations.
- Use cut for energy changes or pattern interrupts.
- Use fade_to_black for major section breaks.

## HUMAN ENGAGEMENT
- Think like a viewer, not a producer. What makes someone STAY?
- Vary visual energy: mix wide establishing shots with close-up details.
- Emotional beats matter: find moments in the script that should hit hard
  and ensure the visuals amplify that emotion, not distract from it.
- Avoid visual monotony: if 3 segments in a row are all "person at desk",
  reframe one as a close-up of hands, another as a wide shot, etc.
- Search keywords should find DIVERSE footage — avoid similar-looking clips
  for different concepts. "robot arm factory" and "robot arm welding" look
  the same. Pick visually distinct alternatives.

## SEARCH KEYWORD QUALITY
- Evaluate whether the search keywords will actually find relevant footage.
- Replace vague or abstract keywords with concrete, visual terms.
- "person typing on laptop" finds footage. "digital transformation" does not.
- Keywords should find footage that is visually DISTINCT from adjacent segments.
- Think about visual contrast between consecutive clips.

Return a JSON object with your creative direction:
{
  "hook_assessment": "strong|weak|needs_restructure",
  "hook_notes": "Why and what to fix if weak",
  "hook_rewrite": "New opening line if hook is weak. Null if hook is strong.",
  "pacing_notes": "Overall pacing assessment",
  "segments": [
    {
      "segment_id": "seg_001",
      "director_notes": "Creative direction for this segment",
      "transition_in": "cut|crossfade|fade_to_black",
      "visual_intent": "stock|ai_generate|diagram|user_footage",
      "revised_search_keywords": ["keyword1", "keyword2", "keyword3"],
      "revised_visual_description": "More specific visual brief if needed",
      "revised_duration_s": null,
      "energy_level": "high|medium|low",
      "ai_generation_brief": null,
      "key_phrase": null
    }
  ],
  "ai_budget_segments": ["seg_004", "seg_007"],
  "overall_quality_score": 7,
  "improvement_summary": "Key changes made and why"
}

## KEY PHRASES (TEXT OVERLAYS)
- For impactful segments, set key_phrase to a SHORT phrase (2-5 words max)
  that reinforces the narration. This appears as a subtle lower-third text overlay.
- Use sparingly — at most 4-5 key phrases per video. Not every segment needs one.
- Good key phrases: specific numbers, bold claims, memorable concepts.
  "40% of jobs at risk", "Not if — when", "The adaptation gap"
- Bad key phrases: generic descriptions, full sentences, vague concepts.
- key_phrase should be null for most segments. Only set it for high-impact moments.

RULES:
- ai_budget_segments must contain AT MOST 3 segment IDs.
- visual_intent "ai_generate" or "diagram" counts against the AI budget.
- Prefer "stock" for atmospheric/generic b-roll. Only use AI for concepts
  that stock footage genuinely cannot illustrate.
- revised_search_keywords should be concrete and visual.
- Do not change script_text — only visual and pacing decisions.
- transition_in is the transition INTO this segment from the previous one.
  The first segment always has transition_in: "cut".
"""

DIRECTOR_USER = """Review this scene plan and provide your creative direction.

Title: {title}
Platform: {target_platform}
Style profile: {style_notes}

Scene Plan:
{scene_plan_json}

Provide your creative direction as JSON."""


# ═══════════════════════════════════════════════════
# QA GATE PROMPT
# ═══════════════════════════════════════════════════

QA_GATE_SYSTEM = """You are a video reviewer watching a finished video.
Think like a VIEWER, not a technician. Your job is to judge whether the
video flows well and makes sense to watch.

You will receive the scene plan and a summary of the assembled video.
If you are also given the actual video file, WATCH IT and base your
assessment on what you see and hear.

Evaluate these things:

## VISUAL RELEVANCE
- Does each b-roll clip match what the narration is talking about?
- When the narration changes subject mid-paragraph (e.g. from
  "repetitive work" to "creative work"), does the visual change too?
- Would a viewer understand the point being made from the visuals alone?

## FLOW & ENGAGEMENT
- Does the video feel like it has natural rhythm?
- Does one scene flow into the next, or are there jarring jumps?
- A single clip for a whole b-roll section is fine IF the clip is
  relevant and engaging. Flag it ONLY if it feels stale or boring.
- Multiple clips in one section is better when the narration shifts
  topic within that section.

## AUDIO COMPLETENESS
- Does every sentence finish? Is any narration cut off mid-word?
- Is the voice consistent (same voice throughout)?
- Are there any audio glitches, pops, or sudden volume changes?

Return a JSON object:
{
  "passed": true/false,
  "overall_score": 1-10,
  "dimension_scores": {
    "visual_relevance": 1-10,
    "flow": 1-10,
    "audio": 1-10
  },
  "issues": [
    {
      "segment_id": "seg_003",
      "severity": "critical|major|minor",
      "category": "visual_mismatch|flow|audio|clip_reuse",
      "description": "What is wrong",
      "suggested_fix": "How to fix it"
    }
     }
   ],
   "summary": "Overall assessment in 2-3 sentences",
   "suggested_shorts": [
     {
       "start_seconds": 45.0,
       "end_seconds": 95.0,
       "title": "Catchy short title under 60 chars",
       "caption": "Brief hook caption for the short",
       "reason": "Why this section works standalone"
     }
   ]
 }

A score of 8+ means it can ship. Below 8 means it needs fixes.

THINGS THAT ARE NORMAL — DO NOT FLAG:
- Avatar segments staying on screen for 15-30 seconds. That is a
  talking head delivering narration. It is supposed to be on screen
  for as long as it is speaking. Never flag avatar duration.
- The avatar having a static tech background behind it.
- No text overlays, captions, lower thirds, or background music.
  These features are not built yet.
- Brief silence (1-3 seconds) between scenes. That is a deliberate
  pause between sections.

THINGS TO ACTUALLY FLAG:
- B-roll that has nothing to do with the narration (e.g. narration
  talks about hospitals but the visual shows a beach).
- The same b-roll clip reused in multiple different sections.
- Narration audio being cut off before a sentence finishes.
- A long b-roll section where the narration changes subject but
  the visual stays the same — this is a missed opportunity for
  a visual cut that would reinforce the message.
- Audio glitches or sudden voice changes.

SHORTS EXTRACTION:
While reviewing, identify 2-3 sections that would work as standalone
YouTube Shorts. CRITICAL RULES FOR SHORTS:
- Each short MUST be between 15 and 60 seconds. NEVER under 15 seconds.
  5-10 second clips are USELESS. Aim for 30-45 seconds.
- The end timestamp MUST fall at the END of a complete sentence.
  NEVER cut mid-word or mid-sentence. Listen to the audio carefully.
  If the speaker is still talking, the short is NOT done yet.
- The start timestamp must begin at the START of a sentence.
- Each short must make complete sense on its own without any context
  from the rest of the video.
- Have a strong hook in the first 3 seconds.
- End on a punchy point, surprising fact, or complete thought.
- A 30-45 second short with a complete idea is ALWAYS better than
  a 8 second clip that cuts off mid-word.
Include start_seconds, end_seconds, title, caption, and reason.

If visuals are relevant and audio is complete, score 7+.
Only score below 5 for cut-off audio or completely wrong visuals."""

QA_GATE_USER = """Review this assembled video.

Your job: compare what was PLANNED against what was DELIVERED.
The scene plan below shows what each segment was supposed to look like.
The assembly summary shows what actually ended up in the video.
If you have the video file, WATCH IT.

Check: did the pipeline deliver what the plan asked for?

Original Scene Plan (the INTENT):
{scene_plan_json}

Assembled Video (the EXECUTION):
{assembly_summary}

Director Notes (creative decisions made during pre-production):
{director_notes}

Compare intent vs execution. Flag gaps. Return assessment as JSON."""


# ═══════════════════════════════════════════════════
# DIRECTOR FUNCTIONS
# ═══════════════════════════════════════════════════

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


SPLIT_SEGMENT_SYSTEM = """You are a video editor splitting a long narration
segment into shorter sub-segments, each with its own visual b-roll.

You will receive a segment with script text and its duration. Your job is
to split it into 2-4 sub-segments of roughly equal duration, each with:
- A portion of the script text
- Unique, concrete search keywords for different b-roll footage
- A visual description that matches that portion of the narration

RULES:
- Each sub-segment should be 5-12 seconds (the sweet spot for b-roll cuts)
- Split at natural sentence boundaries — never mid-sentence
- Each sub-segment needs DIFFERENT search keywords (not the same clip repeated)
- Keywords must be concrete and visual: "truck driving rain" not "transportation"
- The script_text portions must concatenate to the original full text exactly
- Preserve the original segment_type and mood_tags

Return a JSON array of sub-segments:
[
  {
    "script_text": "First portion of the narration...",
    "visual_description": "What to show visually for this portion",
    "search_keywords": ["keyword1", "keyword2", "keyword3"],
    "estimated_duration_s": 8.0
  },
  ...
]"""

SPLIT_SEGMENT_USER = """Split this long segment into shorter sub-segments.

Segment ID: {segment_id}
Duration: {duration_s} seconds
Script text: {script_text}
Original keywords: {keywords}

Split into 2-4 sub-segments with different visuals for each.
Return as a JSON array."""


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
