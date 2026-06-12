# FILE: app/content/video_pipeline/clip_verifier.py
# Purpose: Clip Verifier — pre-bake relevance scoring via Gemini.
# Called-by: app.content.video_pipeline.orchestrator
# Depends-on: app.content.video_pipeline.models
# Last-renovated: 2026-06-11
"""
Clip Verifier — pre-bake relevance scoring via Gemini.

Before clips get baked into segments, this module evaluates whether
each clip visually matches its narration text. Uses a single Gemini
call to batch-verify all b-roll segments at once.

Clips scoring below the relevance threshold (75/100) get flagged
for AI generation instead of being baked with irrelevant footage.

Model: Gemini 3.1 Pro Preview (multimodal — accepts images + text).
"""
import base64
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai

from app.content.video_pipeline.models import PIPELINE_GEMINI_MODEL

logger = logging.getLogger(__name__)

# Clips below this score get rejected and flagged for AI generation.
RELEVANCE_THRESHOLD = 75

# Temp directory for extracted thumbnail frames.
_FRAME_DIR = Path("data/content/temp/verify_frames")


def _extract_frame(video_path: str, segment_id: str, seek_pct: float = 0.3) -> Optional[str]:
    """Extract a single frame from a video clip as a JPEG.

    Seeks to seek_pct of the clip duration to avoid grabbing
    a black intro frame. Returns the path to the JPEG, or None.
    """
    if not os.path.exists(video_path):
        return None

    _FRAME_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(_FRAME_DIR / f"frame_{segment_id}.jpg")

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
        seek = dur * seek_pct
    except Exception:
        seek = 1.0

    # Extract frame
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
            return out_path
    except Exception as e:
        logger.warning(f"[clip_verifier] Frame extraction failed for {segment_id}: {e}")

    return None


def _load_frame_b64(frame_path: str) -> Optional[str]:
    """Load a JPEG frame as base64."""
    try:
        with open(frame_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


VERIFY_SYSTEM = """You are a video quality reviewer. You will receive a batch of
video segments. Each segment has:
- A narration script (what the viewer HEARS)
- A thumbnail frame from the assigned video clip (what the viewer SEES)
- The clip's description from the stock footage source

Your job: score how well the visual matches the narration on a scale of 0-100.

SCORING GUIDE:
- 90-100: Perfect match. The clip directly illustrates what is being said.
- 75-89: Good match. The clip is relevant and supports the narration.
- 50-74: Weak match. The clip is vaguely related but does not reinforce the message.
- 25-49: Poor match. The clip has nothing to do with the narration.
- 0-24: Completely wrong. The clip contradicts or undermines the narration.

EXAMPLES:
- Narration: "Factories now run with a handful of engineers"
  Clip: robotic assembly line → 95
  Clip: person typing at desk → 40
  Clip: beach sunset → 5

- Narration: "What does purpose look like when your job no longer defines you?"
  Clip: person standing alone on hilltop at sunset → 85
  Clip: busy office workers → 30

Return a JSON array with one entry per segment:
[
  {
    "segment_id": "seg_003",
    "relevance_score": 82,
    "reason": "Clip shows robotic arm in factory, matches narration about automation"
  },
  ...
]

Be honest and strict. A generic "technology" clip does NOT score 75+ for a specific
narration about healthcare, even if both are loosely about "the future".
Only segments with genuinely relevant visuals should pass."""


async def verify_clips(
    segments_to_verify: List[Dict],
    style_notes: str = "",
) -> Dict[str, Dict]:
    """
    Batch-verify clip relevance for all b-roll segments.

    Args:
        segments_to_verify: List of dicts, each with:
            - segment_id: str
            - script_text: str (narration)
            - clip_path: str (video file path)
            - clip_description: str (visual description from resolver)

    Returns:
        Dict mapping segment_id → {relevance_score, reason, passed}
    """
    if not segments_to_verify:
        return {}

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        logger.warning("[clip_verifier] No GOOGLE_API_KEY — skipping verification")
        return {
            s["segment_id"]: {"relevance_score": 75, "reason": "Skipped (no API key)", "passed": True}
            for s in segments_to_verify
        }

    genai.configure(api_key=api_key)

    # Extract frames and build the prompt content
    content_parts = []
    segment_map = []  # Track which segments we're sending

    for seg in segments_to_verify:
        sid = seg["segment_id"]
        frame_path = _extract_frame(seg["clip_path"], sid)
        if not frame_path:
            logger.warning(f"[clip_verifier] No frame for {sid}, auto-passing")
            continue

        frame_b64 = _load_frame_b64(frame_path)
        if not frame_b64:
            continue

        # Add the image
        content_parts.append({
            "mime_type": "image/jpeg",
            "data": frame_b64,
        })

        # Add the text context for this segment
        content_parts.append(
            f"SEGMENT: {sid}\n"
            f"NARRATION: {seg['script_text'][:300]}\n"
            f"CLIP DESCRIPTION: {seg.get('clip_description', 'No description')[:200]}\n"
            f"---"
        )

        segment_map.append(sid)

    if not segment_map:
        logger.info("[clip_verifier] No segments to verify")
        return {}

    # Add the final instruction
    content_parts.append(
        f"Score all {len(segment_map)} segments above. "
        f"Return a JSON array with relevance_score (0-100) and reason for each."
    )

    logger.info(
        f"[clip_verifier] Verifying {len(segment_map)} clips in one Gemini call"
    )

    try:
        model = genai.GenerativeModel(
            model_name=PIPELINE_GEMINI_MODEL,
            system_instruction=VERIFY_SYSTEM,
            generation_config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
            },
        )

        response = model.generate_content(content_parts)
        raw_text = response.text.strip()

        try:
            scores = json.loads(raw_text)
        except json.JSONDecodeError:
            if "```json" in raw_text:
                json_str = raw_text.split("```json")[1].split("```")[0].strip()
                scores = json.loads(json_str)
            else:
                logger.error(f"[clip_verifier] Failed to parse: {raw_text[:300]}")
                return {}

        # Build results dict
        results = {}
        for entry in scores:
            sid = entry.get("segment_id", "")
            score = entry.get("relevance_score", 50)
            reason = entry.get("reason", "")
            passed = score >= RELEVANCE_THRESHOLD

            results[sid] = {
                "relevance_score": score,
                "reason": reason,
                "passed": passed,
            }

            status = "PASS" if passed else "FAIL"
            logger.info(
                f"[clip_verifier] {sid}: {score}/100 {status} — {reason[:80]}"
            )

        # Any segments we sent but didn't get back — auto-pass
        for sid in segment_map:
            if sid not in results:
                results[sid] = {
                    "relevance_score": 75,
                    "reason": "Not scored by verifier",
                    "passed": True,
                }

        _cleanup_frames()

        passed_count = sum(1 for r in results.values() if r["passed"])
        failed_count = len(results) - passed_count
        logger.info(
            f"[clip_verifier] Verification complete: "
            f"{passed_count} passed, {failed_count} failed "
            f"(threshold={RELEVANCE_THRESHOLD})"
        )

        return results

    except Exception as e:
        logger.error(f"[clip_verifier] Verification failed: {e}")
        return {}


def _cleanup_frames():
    """Remove temporary frame files."""
    if _FRAME_DIR.exists():
        for f in _FRAME_DIR.glob("*.jpg"):
            try:
                os.remove(f)
            except OSError:
                pass
