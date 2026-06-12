# FILE: app/content/production/captions.py
# Purpose: Caption Generation and Burning (Spec Section 7.4).
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Caption Generation and Burning (Spec Section 7.4).

Deterministic caption system:
1. Generates SRT files from transcript segments
2. Burns captions into video via FFmpeg
3. Style driven by active style profile

85% of viewers watch without sound — captions are primary content delivery.
"""
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CaptionEntry:
    """A single caption entry with timing."""
    index: int
    start_seconds: float
    end_seconds: float
    text: str


# ─── CAPTION STYLE DEFAULTS ───
# From Spec Section 11.1

DEFAULT_STYLE = {
    "font": "Arial",
    "font_size": 22,
    "primary_colour": "&H00FFFFFF",  # White (ASS format: BBGGRR)
    "outline_colour": "&H00000000",  # Black outline
    "outline_width": 3,
    "shadow_depth": 1,
    "alignment": 2,  # Bottom centre
    "margin_v": 60,  # Vertical margin from bottom
    "bold": True,
}


# ═══════════════════════════════════════════════════
# SRT GENERATION
# ═══════════════════════════════════════════════════

def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def transcript_to_captions(
    transcript_segments: List[Dict[str, Any]],
    max_words_per_caption: int = 8,
    max_duration_seconds: float = 3.0,
) -> List[CaptionEntry]:
    """
    Convert transcript segments into caption entries.
    
    Breaks text into readable chunks that fit on screen.
    Each caption should be short enough to read in the display time.
    """
    captions = []
    index = 1

    for segment in transcript_segments:
        text = segment.get("text", "").strip()
        start = float(segment.get("timestamp", 0))
        end = float(segment.get("end_timestamp", start + 3))

        if not text:
            continue

        # Split into words
        words = text.split()
        if not words:
            continue

        # Calculate timing per word
        segment_duration = end - start
        if segment_duration <= 0:
            segment_duration = len(words) * 0.3  # ~300ms per word fallback

        time_per_word = segment_duration / len(words)

        # Chunk words into caption-sized groups
        chunks = []
        for i in range(0, len(words), max_words_per_caption):
            chunk_words = words[i:i + max_words_per_caption]
            chunks.append(" ".join(chunk_words))

        # Assign timing to each chunk
        current_time = start
        for chunk in chunks:
            word_count = len(chunk.split())
            chunk_duration = min(
                word_count * time_per_word,
                max_duration_seconds,
            )
            chunk_duration = max(chunk_duration, 0.5)  # Min 500ms

            captions.append(CaptionEntry(
                index=index,
                start_seconds=current_time,
                end_seconds=current_time + chunk_duration,
                text=chunk.upper() if DEFAULT_STYLE["bold"] else chunk,
            ))
            index += 1
            current_time += chunk_duration

    return captions


def captions_to_srt(captions: List[CaptionEntry]) -> str:
    """Convert caption entries to SRT format string."""
    lines = []
    for cap in captions:
        lines.append(str(cap.index))
        lines.append(
            f"{format_srt_timestamp(cap.start_seconds)} --> "
            f"{format_srt_timestamp(cap.end_seconds)}"
        )
        lines.append(cap.text)
        lines.append("")  # Blank line separator

    return "\n".join(lines)


def save_srt(captions: List[CaptionEntry], path: str) -> str:
    """Save captions as SRT file. Returns the file path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    srt_content = captions_to_srt(captions)
    with open(path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    logger.info(f"[captions] Saved SRT: {path} ({len(captions)} entries)")
    return path


# ═══════════════════════════════════════════════════
# ASS SUBTITLE GENERATION (for styled burn-in)
# ═══════════════════════════════════════════════════

def format_ass_timestamp(seconds: float) -> str:
    """Convert seconds to ASS timestamp (H:MM:SS.cc)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def captions_to_ass(
    captions: List[CaptionEntry],
    style: Optional[Dict[str, Any]] = None,
    video_width: int = 1080,
    video_height: int = 1920,
) -> str:
    """
    Generate ASS subtitle file with styled captions.
    ASS format allows precise control over font, colour, position.
    """
    s = {**DEFAULT_STYLE, **(style or {})}

    header = f"""[Script Info]
Title: ASTRA Content Captions
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{s['font']},{s['font_size']},{s['primary_colour']},&H000000FF,{s['outline_colour']},&H80000000,-1,0,0,0,100,100,0,0,1,{s['outline_width']},{s['shadow_depth']},{s['alignment']},40,40,{s['margin_v']},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []
    for cap in captions:
        start = format_ass_timestamp(cap.start_seconds)
        end = format_ass_timestamp(cap.end_seconds)
        # Escape special ASS characters
        text = cap.text.replace("\\", "\\\\").replace("{", "\\{")
        events.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"
        )

    return header + "\n".join(events) + "\n"


def save_ass(
    captions: List[CaptionEntry],
    path: str,
    style: Optional[Dict[str, Any]] = None,
    video_width: int = 1080,
    video_height: int = 1920,
) -> str:
    """Save captions as ASS subtitle file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ass_content = captions_to_ass(captions, style, video_width, video_height)
    with open(path, "w", encoding="utf-8") as f:
        f.write(ass_content)
    logger.info(f"[captions] Saved ASS: {path} ({len(captions)} entries)")
    return path


# ═══════════════════════════════════════════════════
# BURN CAPTIONS INTO VIDEO
# ═══════════════════════════════════════════════════

def burn_captions(
    video_path: str,
    subtitle_path: str,
    output_path: str,
) -> bool:
    """
    Burn subtitles (SRT or ASS) into video using FFmpeg.
    ASS subtitles preserve styling; SRT uses FFmpeg defaults.
    """
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        logger.error("[captions] FFmpeg not available")
        return False

    # Determine subtitle filter based on file type
    ext = Path(subtitle_path).suffix.lower()
    if ext == ".ass":
        # ASS: use ass filter for styled rendering
        sub_filter = f"ass='{subtitle_path}'"
    else:
        # SRT: use subtitles filter with force_style
        sub_filter = (
            f"subtitles='{subtitle_path}'"
            f":force_style='FontSize=22,Bold=1,"
            f"PrimaryColour=&H00FFFFFF,"
            f"OutlineColour=&H00000000,"
            f"Outline=3,Shadow=1,"
            f"Alignment=2,MarginV=60'"
        )

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", sub_filter,
        "-c:v", "libx264", "-preset", "medium",
        "-c:a", "copy",
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            logger.error(f"[captions] Burn failed: {result.stderr[-300:]}")
            return False
        logger.info(f"[captions] Burned captions into {output_path}")
        return True
    except Exception as e:
        logger.error(f"[captions] Burn error: {e}")
        return False
