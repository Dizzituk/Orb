# FILE: app/content/production/shorts_creator.py
"""
Shorts Creator — AI-powered short-form video extraction.

Takes a source video and a user prompt, uses Gemini to
analyse the video content and identify the best moments,
then uses the Edit Engine (FFmpeg) to cut them.

Flow:
1. User uploads video + prompt ("3 shorts, key insights, 30-60s")
2. Gemini analyses video and returns timestamped segments
3. Edit Engine cuts each segment
4. ContentOutput records created for review
5. User queues approved outputs for YouTube from Content Output
"""
import os
import json
import logging
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/content/output")


@dataclass
class ShortSegment:
    """A segment identified by AI for a short."""
    title: str
    start_seconds: float
    end_seconds: float
    description: str
    hook: str  # Opening hook text


async def analyse_video_for_shorts(
    video_path: str,
    prompt: str,
    max_shorts: int = 3,
    max_duration: int = 60,
) -> List[ShortSegment]:
    """
    Use Gemini to analyse a video and identify the best
    moments for short-form content.

    Gemini can process video natively via the File API.
    """
    import httpx

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("[shorts] No GOOGLE_API_KEY set")
        return []

    # Step 1: Get video duration with ffprobe
    duration = _get_video_duration(video_path)
    if not duration:
        logger.error("[shorts] Could not determine video duration")
        return []

    # Step 2: Upload video to Gemini File API
    file_uri = await _upload_to_gemini(video_path, api_key)
    if not file_uri:
        logger.error("[shorts] Failed to upload video to Gemini")
        return []

    # Step 3: Ask Gemini to identify the best moments
    analysis_prompt = _build_analysis_prompt(
        prompt, duration, max_shorts, max_duration
    )

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                "https://generativelanguage.googleapis.com/v1beta/"
                "models/gemini-2.0-flash:generateContent",
                params={"key": api_key},
                json={
                    "contents": [
                        {
                            "parts": [
                                {
                                    "fileData": {
                                        "mimeType": "video/mp4",
                                        "fileUri": file_uri,
                                    }
                                },
                                {"text": analysis_prompt},
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": 2048,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()

            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )

            return _parse_segments(text, duration)

    except Exception as e:
        logger.error("[shorts] Gemini analysis failed: %s", e)
        return []


async def _upload_to_gemini(
    video_path: str, api_key: str,
) -> Optional[str]:
    """Upload a video file to Gemini's File API."""
    import httpx

    file_size = os.path.getsize(video_path)
    filename = os.path.basename(video_path)

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            # Initiate resumable upload
            init_resp = await client.post(
                "https://generativelanguage.googleapis.com/upload/"
                "v1beta/files",
                params={"key": api_key},
                headers={
                    "X-Goog-Upload-Protocol": "resumable",
                    "X-Goog-Upload-Command": "start",
                    "X-Goog-Upload-Header-Content-Length": str(
                        file_size
                    ),
                    "X-Goog-Upload-Header-Content-Type": "video/mp4",
                    "Content-Type": "application/json",
                },
                json={
                    "file": {
                        "display_name": filename,
                    }
                },
            )
            init_resp.raise_for_status()

            upload_url = init_resp.headers.get(
                "X-Goog-Upload-URL"
            )
            if not upload_url:
                logger.error("[shorts] No upload URL returned")
                return None

            # Upload the actual file
            with open(video_path, "rb") as f:
                video_data = f.read()

            upload_resp = await client.put(
                upload_url,
                headers={
                    "X-Goog-Upload-Command": "upload, finalize",
                    "X-Goog-Upload-Offset": "0",
                    "Content-Length": str(file_size),
                },
                content=video_data,
            )
            upload_resp.raise_for_status()
            result = upload_resp.json()

            file_uri = result.get("file", {}).get("uri")
            state = result.get("file", {}).get("state")

            logger.info(
                "[shorts] Uploaded %s to Gemini: %s (state=%s)",
                filename, file_uri, state,
            )

            # Wait for processing if needed
            if state == "PROCESSING":
                file_uri = await _wait_for_processing(
                    result["file"]["name"], api_key
                )

            return file_uri

    except Exception as e:
        logger.error("[shorts] Upload to Gemini failed: %s", e)
        return None


async def _wait_for_processing(
    file_name: str, api_key: str, max_wait: int = 120,
) -> Optional[str]:
    """Poll until Gemini finishes processing the video."""
    import httpx
    import asyncio

    for _ in range(max_wait // 5):
        await asyncio.sleep(5)
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"https://generativelanguage.googleapis.com/"
                    f"v1beta/{file_name}",
                    params={"key": api_key},
                )
                resp.raise_for_status()
                data = resp.json()

                if data.get("state") == "ACTIVE":
                    return data.get("uri")
                elif data.get("state") == "FAILED":
                    logger.error(
                        "[shorts] Gemini file processing failed"
                    )
                    return None
        except Exception:
            continue

    logger.error("[shorts] Timeout waiting for Gemini processing")
    return None


def _get_video_duration(video_path: str) -> Optional[float]:
    """Get video duration in seconds using ffprobe."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                video_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception as e:
        logger.error("[shorts] ffprobe failed: %s", e)
        return None


def _build_analysis_prompt(
    user_prompt: str,
    duration: float,
    max_shorts: int,
    max_duration: int,
) -> str:
    """Build the Gemini prompt for video analysis."""
    return f"""You are a professional video editor. Watch this video carefully and identify the best moments for YouTube Shorts.

CRITICAL EDITING RULES:
- NEVER cut mid-sentence. Every short MUST start at the beginning of a sentence and end at the end of a sentence.
- Listen to the audio carefully. Find natural speech pauses and sentence boundaries for your cut points.
- The start_seconds should be the moment the speaker begins a complete thought.
- The end_seconds should be AFTER the speaker finishes their sentence — include the full final word.
- It is far better to include an extra second of silence than to cut off a word.
- Timestamps can be precise decimals (e.g. 14.5, 27.8) — do NOT round to whole numbers.

User request: {user_prompt}

Video duration: {duration:.0f} seconds
Maximum shorts to create: {max_shorts}
Target duration: 15 to {max_duration} seconds per short (flexible — a few seconds over is fine if it avoids cutting mid-sentence)

For each short, identify:
- A segment that works as a standalone, punchy piece of content
- Precise start and end timestamps aligned to sentence boundaries
- A catchy title
- Why this moment is compelling
- A hook — what grabs attention in the first 2 seconds

Respond in this exact JSON format (no markdown, no backticks):
[
    {{
        "title": "short title",
        "start_seconds": 14.5,
        "end_seconds": 42.3,
        "description": "why this moment is compelling",
        "hook": "opening hook text"
    }}
]

Rules:
- Timestamps within 0 to {duration:.0f}
- NEVER cut mid-word or mid-sentence — this is the most important rule
- Prefer segments that open with a strong statement or question
- Each short should make sense without any other context
- Shorts must not overlap
- Return ONLY valid JSON array"""


def _parse_segments(
    text: str, max_duration: float,
) -> List[ShortSegment]:
    """Parse Gemini's response into ShortSegments."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [
            l for l in lines if not l.strip().startswith("```")
        ]
        cleaned = "\n".join(lines)

    try:
        data = json.loads(cleaned)
        segments = []
        for item in data:
            start = float(item.get("start_seconds", 0))
            end = float(item.get("end_seconds", 0))

            # Validate timestamps
            if end <= start:
                continue
            if start < 0:
                start = 0
            if end > max_duration:
                # Allow up to 10% over to avoid cutting mid-sentence
                if end > max_duration * 1.1:
                    end = max_duration

            segments.append(ShortSegment(
                title=item.get("title", "Untitled"),
                start_seconds=start,
                end_seconds=end,
                description=item.get("description", ""),
                hook=item.get("hook", ""),
            ))

        return segments

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error("[shorts] Failed to parse segments: %s", e)
        return []


def cut_shorts(
    source_path: str,
    segments: List[ShortSegment],
    piece_id: str,
    output_format: str = "youtube_short",
    portrait: bool = True,
) -> List[Dict[str, Any]]:
    """
    Cut identified segments, convert to portrait 9:16.
    Hard cuts — no fades. Shorts should punch in immediately.

    Returns list of output file info dicts.
    """
    from app.content.production.edit_engine import (
        check_ffmpeg,
        _run_ffmpeg,
    )

    if not check_ffmpeg():
        logger.error("[shorts] FFmpeg not available")
        return []

    output_dir = OUTPUT_DIR / piece_id
    os.makedirs(output_dir, exist_ok=True)

    # Portrait (9:16) dimensions for shorts
    out_w = 1080
    out_h = 1920

    outputs = []
    for i, seg in enumerate(segments):
        safe_title = "".join(
            c if c.isalnum() or c in " -_" else ""
            for c in seg.title
        ).strip().replace(" ", "_")

        output_path = str(
            output_dir / f"short_{i+1}_{safe_title[:40]}.mp4"
        )

        duration = seg.end_seconds - seg.start_seconds

        # Build video filter chain — hard cut, no fades
        vf_parts = []

        if portrait:
            # Scale to fit width, then pad to 9:16 with black bars
            vf_parts.append(
                f"scale={out_w}:-2:force_original_aspect_ratio=decrease"
            )
            vf_parts.append(
                f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:black"
            )

        vf = ",".join(vf_parts) if vf_parts else None

        ffmpeg_args = [
            "-ss", str(seg.start_seconds),
            "-i", source_path,
            "-t", str(duration),
        ]
        if vf:
            ffmpeg_args.extend(["-vf", vf])

        ffmpeg_args.extend([
            "-c:v", "libx264", "-preset", "medium",
            "-b:v", "5M",
            "-force_key_frames", "expr:eq(n,0)",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", "128k",
            "-r", "30",
            output_path,
        ])

        success = _run_ffmpeg(ffmpeg_args)

        if success:
            outputs.append({
                "index": i + 1,
                "title": seg.title,
                "path": output_path,
                "start": seg.start_seconds,
                "end": seg.end_seconds,
                "duration": duration,
                "description": seg.description,
                "hook": seg.hook,
                "format": output_format,
                "resolution": f"{out_w}x{out_h}" if portrait else "original",
            })
            logger.info(
                "[shorts] Cut short %d: %s (%.1fs-%.1fs) %s",
                i + 1, seg.title,
                seg.start_seconds, seg.end_seconds,
                f"{out_w}x{out_h}" if portrait else "",
            )
        else:
            logger.error(
                "[shorts] Failed to cut short %d: %s",
                i + 1, seg.title,
            )

    return outputs





