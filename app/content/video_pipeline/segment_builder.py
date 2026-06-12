# FILE: app/content/video_pipeline/segment_builder.py
# Purpose: Segment Builder — Gemini-driven clip selection and arrangement.
# Called-by: app.content.video_pipeline.orchestrator
# Depends-on: app.content.video_pipeline.models
# Last-renovated: 2026-06-11
"""
Segment Builder — Gemini-driven clip selection and arrangement.

Instead of downloading clips blindly and verifying after the fact,
this module gives Gemini editorial control over each segment:

1. Receive all candidate clips for a segment (primary + clip B + fillers)
2. Extract thumbnail frames from each candidate
3. Send frames + narration text to Gemini in a single call
4. Gemini ranks clips by relevance and suggests which to use
5. Return the ranked selection for the bake step

Gemini acts as the video editor — it sees what is in each clip
and decides what fits the narration. The bake step then takes
those selections and assembles them mechanically.

Model: Gemini 3.1 Pro (multimodal — accepts images + text).
"""
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import google.generativeai as genai

from app.content.video_pipeline.models import PIPELINE_GEMINI_MODEL

logger = logging.getLogger(__name__)

_FRAME_DIR = Path("data/content/temp/builder_frames")


def _extract_frame(video_path, clip_id, seek_pct=0.3):
    """Extract a single representative frame from a clip."""
    if not os.path.exists(video_path):
        return None
    _FRAME_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(_FRAME_DIR / f"frame_{clip_id}.jpg")
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", os.path.abspath(video_path)],
            capture_output=True, text=True, timeout=10,
        )
        dur = float(result.stdout.strip())
        seek = dur * seek_pct
    except Exception:
        seek = 1.0
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(seek), "-i", os.path.abspath(video_path),
             "-frames:v", "1", "-q:v", "5", out_path],
            capture_output=True, timeout=15,
        )
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
            return out_path
    except Exception as e:
        logger.warning(f"[segment_builder] Frame extraction failed for {clip_id}: {e}")
    return None


def _probe_duration(file_path):
    """Get duration of a media file."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", os.path.abspath(file_path)],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _load_image_as_part(image_path):
    """Load an image file as a Gemini API content part."""
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        return {"mime_type": "image/jpeg", "data": data}
    except Exception:
        return None


async def select_clips_for_segment(
    segment_id,
    narration_text,
    candidate_clips,
    target_duration,
):
    """
    Have Gemini watch candidate clips and select the best ones.

    Args:
        segment_id: ID of the segment being built
        narration_text: The spoken narration for this segment
        candidate_clips: List of dicts with path, id, source
        target_duration: Seconds of footage needed

    Returns:
        Ranked list of clip dicts with relevance_score, use flag,
        and suggested order. Falls back to unranked if Gemini fails.
    """
    if not candidate_clips:
        return []

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        logger.warning("[segment_builder] No GOOGLE_API_KEY — returning unranked")
        return _fallback_unranked(candidate_clips, "No API key")

    genai.configure(api_key=api_key)

    # Extract frames from each candidate
    parts = []
    clip_descriptions = []

    for i, clip in enumerate(candidate_clips):
        frame_path = _extract_frame(clip["path"], f"{segment_id}_c{i}")
        if not frame_path:
            continue
        image_part = _load_image_as_part(frame_path)
        if not image_part:
            continue
        dur = _probe_duration(clip["path"])
        clip_descriptions.append(
            f"CLIP_{i}: source={clip.get('source', 'unknown')}, "
            f"duration={dur:.1f}s, file={Path(clip['path']).name}"
        )
        parts.append(f"[CLIP_{i}]")
        parts.append(image_part)

    if not parts:
        logger.warning(f"[segment_builder] No frames for {segment_id}")
        return _fallback_unranked(candidate_clips, "No frames")

    prompt = (
        "You are a video editor selecting b-roll clips for a segment.\n\n"
        f"NARRATION TEXT (what the viewer hears):\n\"{narration_text}\"\n\n"
        f"TARGET DURATION: {target_duration:.1f} seconds\n\n"
        f"CANDIDATE CLIPS:\n" + "\n".join(clip_descriptions) + "\n\n"
        "I have shown you a representative frame from each clip above.\n\n"
        "YOUR TASK:\n"
        "1. Score each clip 0-100 for visual match to the narration\n"
        "2. Select which clips to USE (enough to fill the target duration)\n"
        "3. Order them by suggested playback sequence\n\n"
        "A GOOD clip directly illustrates what is being said.\n"
        "A BAD clip shows something unrelated to the narration.\n\n"
        "Respond ONLY with valid JSON (no markdown fences):\n"
        '{"selections": [{"clip_index": 0, "relevance_score": 85, '
        '"reason": "...", "use": true, "order": 1}], '
        '"notes": "Brief editorial note"}\n\n'
        "Include ALL clips. Mark use:false for rejects. "
        "Order use:true clips by playback sequence."
    )

    parts.append(prompt)

    try:
        model = genai.GenerativeModel(PIPELINE_GEMINI_MODEL)
        response = model.generate_content(
            parts,
            generation_config=genai.GenerationConfig(
                temperature=0.3, max_output_tokens=2000,
            ),
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if "```" in text:
                text = text.split("```")[0]
        text = text.strip()

        result = json.loads(text)
        selections = result.get("selections", [])
        notes = result.get("notes", "")

        if notes:
            logger.info(f"[segment_builder] {segment_id}: {notes}")

        # Build ranked output
        used = sorted(
            [s for s in selections if s.get("use", False)],
            key=lambda s: s.get("order", 99),
        )
        unused = [s for s in selections if not s.get("use", False)]
        ranked = []

        for sel in used + unused:
            idx = sel.get("clip_index", 0)
            if idx < len(candidate_clips):
                clip = candidate_clips[idx]
                ranked.append({
                    "clip_id": clip["id"],
                    "path": clip["path"],
                    "relevance_score": sel.get("relevance_score", 50),
                    "reason": sel.get("reason", ""),
                    "use": sel.get("use", False),
                    "order": sel.get("order", 99),
                })

        used_count = sum(1 for r in ranked if r["use"])
        scores = [r["relevance_score"] for r in ranked if r["use"]]
        logger.info(
            f"[segment_builder] {segment_id}: "
            f"{used_count}/{len(ranked)} clips selected, scores: {scores}"
        )
        return ranked

    except json.JSONDecodeError as e:
        logger.warning(f"[segment_builder] JSON parse error for {segment_id}: {e}")
    except Exception as e:
        logger.warning(f"[segment_builder] Gemini failed for {segment_id}: {e}")

    return _fallback_unranked(candidate_clips, "Gemini unavailable")


def _fallback_unranked(candidate_clips, reason):
    """Return all candidates as-is when Gemini is unavailable."""
    return [
        {
            "clip_id": c["id"],
            "path": c["path"],
            "relevance_score": 50,
            "reason": reason,
            "use": True,
            "order": i,
        }
        for i, c in enumerate(candidate_clips)
    ]
