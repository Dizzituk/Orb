# FILE: app/content/video_pipeline/clip_analyzer.py
"""
Clip Analyzer — rich visual description of downloaded clips.

When a clip is downloaded (Pexels, fal.ai), this module extracts
frames and asks Gemini to describe exactly what is in the clip.
The rich description, quality score, and content tags get stored
in the asset library RAG index, making future semantic search
match against actual visual content rather than search queries.

Model: Gemini 3.1 Pro Preview (multimodal).
"""
import base64
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional

import google.generativeai as genai

from app.content.video_pipeline.models import PIPELINE_GEMINI_MODEL

logger = logging.getLogger(__name__)

_FRAME_DIR = Path("data/content/temp/analyze_frames")

ANALYZE_SYSTEM = """You are a stock footage librarian. You will receive 3-4 frames
from a short video clip. Your job is to describe exactly what is in the clip
so that future searches can find it accurately.

Provide:
1. description: A precise 2-3 sentence description of what happens in the clip.
   Be specific about: subjects, actions, camera angle, lighting, setting, mood.
   Write as if briefing a video editor who needs to find this exact clip later.

2. content_tags: 8-12 concrete tags describing the visual content.
   Tags should be specific and searchable. GOOD: "robotic arm", "factory floor",
   "overhead shot", "blue lighting". BAD: "technology", "future", "concept".

3. quality_score: Rate the clip quality 1-10 based on:
   - Resolution and sharpness (is it crisp or blurry?)
   - Stability (steady or shaky?)
   - Lighting (well-lit or dark/blown out?)
   - Composition (intentional framing or random?)
   - Usability (would a professional editor use this?)

4. dominant_colours: 2-3 dominant colour descriptions (e.g. "warm orange",
   "cool blue", "dark grey")

5. camera_motion: One of: static, pan_left, pan_right, tilt_up, tilt_down,
   zoom_in, zoom_out, tracking, handheld, aerial, orbiting

Return JSON:
{
  "description": "...",
  "content_tags": ["tag1", "tag2", ...],
  "quality_score": 7,
  "dominant_colours": ["warm orange", "cool blue"],
  "camera_motion": "static"
}

Be factual. Describe what you SEE, not what you think the clip is about."""


def _extract_analysis_frames(
    video_path: str,
    clip_id: str,
    num_frames: int = 4,
) -> list:
    """Extract evenly spaced frames from a clip for analysis.

    Returns list of JPEG file paths.
    """
    if not os.path.exists(video_path):
        return []

    _FRAME_DIR.mkdir(parents=True, exist_ok=True)

    # Get duration
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                os.path.abspath(video_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        dur = float(result.stdout.strip())
    except Exception:
        dur = 5.0

    if dur <= 0:
        dur = 5.0

    frames = []
    for i in range(num_frames):
        # Evenly space frames, avoiding the very start and end
        pct = (i + 1) / (num_frames + 1)
        seek = dur * pct

        out_path = str(_FRAME_DIR / f"analyze_{clip_id}_{i}.jpg")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", str(seek),
                    "-i", os.path.abspath(video_path),
                    "-frames:v", "1",
                    "-q:v", "5",
                    out_path,
                ],
                capture_output=True, timeout=15,
            )
            if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
                frames.append(out_path)
        except Exception:
            continue

    return frames


async def analyze_clip(video_path: str, clip_id: str = "") -> Optional[Dict]:
    """
    Analyze a video clip with Gemini to produce a rich description.

    Extracts frames, sends to Gemini, returns structured analysis.

    Args:
        video_path: Path to the video file
        clip_id: Identifier for the clip (used for temp files)

    Returns:
        Dict with description, content_tags, quality_score,
        dominant_colours, camera_motion. Or None on failure.
    """
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        logger.warning("[clip_analyzer] No GOOGLE_API_KEY — skipping analysis")
        return None

    if not clip_id:
        clip_id = Path(video_path).stem

    frames = _extract_analysis_frames(video_path, clip_id)
    if not frames:
        logger.warning(f"[clip_analyzer] No frames extracted for {clip_id}")
        return None

    genai.configure(api_key=api_key)

    # Build content: frames + instruction
    content_parts = []
    for frame_path in frames:
        try:
            with open(frame_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content_parts.append({
                "mime_type": "image/jpeg",
                "data": b64,
            })
        except Exception:
            continue

    if not content_parts:
        return None

    content_parts.append(
        f"These are {len(content_parts)} frames from a stock video clip "
        f"(filename: {os.path.basename(video_path)}). "
        f"Describe the clip and provide quality assessment as JSON."
    )

    try:
        model = genai.GenerativeModel(
            model_name=PIPELINE_GEMINI_MODEL,
            system_instruction=ANALYZE_SYSTEM,
            generation_config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
            },
        )

        response = model.generate_content(content_parts)
        raw_text = response.text.strip()

        try:
            analysis = json.loads(raw_text)
        except json.JSONDecodeError:
            if "```json" in raw_text:
                json_str = raw_text.split("```json")[1].split("```")[0].strip()
                analysis = json.loads(json_str)
            else:
                logger.error(f"[clip_analyzer] Parse failed: {raw_text[:200]}")
                return None

        # Gemini sometimes wraps the dict in a list — unwrap it
        if isinstance(analysis, list) and len(analysis) > 0:
            analysis = analysis[0]
        if not isinstance(analysis, dict):
            logger.error(f"[clip_analyzer] Unexpected type: {type(analysis)}")
            return None

        logger.info(
            f"[clip_analyzer] Analyzed {clip_id}: "
            f"quality={analysis.get('quality_score', '?')}/10, "
            f"tags={len(analysis.get('content_tags', []))}, "
            f"motion={analysis.get('camera_motion', '?')}"
        )

        # Cleanup frames
        for f in frames:
            try:
                os.remove(f)
            except OSError:
                pass

        return analysis

    except Exception as e:
        logger.error(f"[clip_analyzer] Analysis failed for {clip_id}: {e}")
        # Cleanup on error
        for f in frames:
            try:
                os.remove(f)
            except OSError:
                pass
        return None
