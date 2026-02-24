# FILE: app/content/production/edit_engine.py
"""
Edit Engine — Deterministic video assembly via FFmpeg (Spec Section 7.4).

Takes structured inputs (video analysis, cutaway assets, style profile)
and produces finished video content. No AI involved — pure deterministic
FFmpeg operations.

Requires: FFmpeg installed and on PATH.
"""
import json
import logging
import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Output directories
OUTPUT_BASE = Path("data/content/output")
TEMP_DIR = Path("data/content/temp")


@dataclass
class EditSegment:
    """A single segment in the edit decision list."""
    source_path: str
    start_seconds: float
    end_seconds: float
    segment_type: str = "anchor"  # anchor | cutaway | intro | outro
    # Optional overlay text for captions
    caption_text: Optional[str] = None
    caption_start: Optional[float] = None
    caption_end: Optional[float] = None


@dataclass
class EditDecisionList:
    """
    Complete edit decision list for a video production.
    The deterministic blueprint that FFmpeg executes.
    """
    piece_id: str
    output_format: str  # instagram_reel, youtube_short, etc.
    segments: List[EditSegment] = field(default_factory=list)
    # Output specs
    width: int = 1080
    height: int = 1920  # 9:16 default
    fps: int = 30
    # Audio
    background_music_path: Optional[str] = None
    music_volume: float = 0.15
    voice_volume: float = 1.0
    # Colour grading LUT
    lut_path: Optional[str] = None

    @property
    def total_duration(self) -> float:
        return sum(s.end_seconds - s.start_seconds for s in self.segments)


# ─── FORMAT PRESETS ───

FORMAT_PRESETS: Dict[str, Dict[str, Any]] = {
    "instagram_reel": {
        "width": 1080, "height": 1920, "fps": 30,
        "max_duration": 90, "codec": "libx264",
        "audio_codec": "aac", "audio_bitrate": "128k",
        "video_bitrate": "4M",
    },
    "youtube_short": {
        "width": 1080, "height": 1920, "fps": 30,
        "max_duration": 180, "codec": "libx264",
        "audio_codec": "aac", "audio_bitrate": "128k",
        "video_bitrate": "5M",
    },
    "youtube_longform": {
        "width": 1920, "height": 1080, "fps": 30,
        "max_duration": None, "codec": "libx264",
        "audio_codec": "aac", "audio_bitrate": "192k",
        "video_bitrate": "8M",
    },
    "tiktok": {
        "width": 1080, "height": 1920, "fps": 30,
        "max_duration": 180, "codec": "libx264",
        "audio_codec": "aac", "audio_bitrate": "128k",
        "video_bitrate": "4M",
    },
    "facebook_video": {
        "width": 1920, "height": 1080, "fps": 30,
        "max_duration": 300, "codec": "libx264",
        "audio_codec": "aac", "audio_bitrate": "128k",
        "video_bitrate": "6M",
    },
}


# ═══════════════════════════════════════════════════
# FFMPEG AVAILABILITY
# ═══════════════════════════════════════════════════

def check_ffmpeg() -> bool:
    """Check if FFmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


def _run_ffmpeg(args: List[str], timeout: int = 300) -> bool:
    """Run an FFmpeg command. Returns True on success."""
    cmd = ["ffmpeg", "-y"] + args  # -y = overwrite output
    logger.debug(f"[edit_engine] Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            logger.error(f"[edit_engine] FFmpeg error: {result.stderr[-500:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("[edit_engine] FFmpeg timed out")
        return False
    except FileNotFoundError:
        logger.error("[edit_engine] FFmpeg not found on PATH")
        return False


# ═══════════════════════════════════════════════════
# CORE OPERATIONS
# ═══════════════════════════════════════════════════

def trim_clip(
    input_path: str,
    output_path: str,
    start: float,
    end: float,
) -> bool:
    """Extract a clip from a video file."""
    duration = end - start
    return _run_ffmpeg([
        "-ss", str(start),
        "-i", input_path,
        "-t", str(duration),
        "-c", "copy",
        output_path,
    ])


def concat_clips(
    clip_paths: List[str],
    output_path: str,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
) -> bool:
    """
    Concatenate multiple clips into a single video.
    Re-encodes to ensure consistent format.
    """
    # Create concat file list
    list_path = str(TEMP_DIR / "concat_list.txt")
    os.makedirs(TEMP_DIR, exist_ok=True)

    with open(list_path, "w") as f:
        for clip in clip_paths:
            # FFmpeg concat requires escaped paths
            escaped = clip.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    return _run_ffmpeg([
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps}",
        "-c:v", "libx264", "-preset", "medium",
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ])


def add_audio_track(
    video_path: str,
    audio_path: str,
    output_path: str,
    audio_volume: float = 0.15,
) -> bool:
    """Mix background audio into a video at specified volume."""
    return _run_ffmpeg([
        "-i", video_path,
        "-i", audio_path,
        "-filter_complex",
        f"[1:a]volume={audio_volume}[bg];"
        f"[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[out]",
        "-map", "0:v", "-map", "[out]",
        "-c:v", "copy", "-c:a", "aac",
        "-shortest",
        output_path,
    ])


def apply_colour_lut(
    input_path: str,
    output_path: str,
    lut_path: str,
) -> bool:
    """Apply a colour grading LUT to a video."""
    return _run_ffmpeg([
        "-i", input_path,
        "-vf", f"lut3d='{lut_path}'",
        "-c:v", "libx264", "-preset", "medium",
        "-c:a", "copy",
        output_path,
    ])


def convert_aspect_ratio(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
) -> bool:
    """Convert video to a specific aspect ratio with padding."""
    return _run_ffmpeg([
        "-i", input_path,
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264", "-preset", "medium",
        "-c:a", "copy",
        output_path,
    ])


# ═══════════════════════════════════════════════════
# EDL EXECUTION
# ═══════════════════════════════════════════════════

def execute_edl(edl: EditDecisionList) -> Optional[str]:
    """
    Execute a complete Edit Decision List.
    Returns path to the final output video, or None on failure.
    """
    if not check_ffmpeg():
        logger.error("[edit_engine] FFmpeg not available — cannot execute EDL")
        return None

    if not edl.segments:
        logger.error("[edit_engine] EDL has no segments")
        return None

    # Setup directories
    piece_dir = OUTPUT_BASE / edl.piece_id
    temp_clips_dir = TEMP_DIR / edl.piece_id
    os.makedirs(piece_dir, exist_ok=True)
    os.makedirs(temp_clips_dir, exist_ok=True)

    # Get format preset
    preset = FORMAT_PRESETS.get(edl.output_format, FORMAT_PRESETS["youtube_short"])

    # Step 1: Extract individual segments
    clip_paths = []
    for i, segment in enumerate(edl.segments):
        clip_path = str(temp_clips_dir / f"seg_{i:03d}.mp4")

        if not os.path.exists(segment.source_path):
            logger.warning(
                f"[edit_engine] Source not found: {segment.source_path}, "
                f"skipping segment {i}"
            )
            continue

        success = trim_clip(
            segment.source_path, clip_path,
            segment.start_seconds, segment.end_seconds,
        )
        if success:
            clip_paths.append(clip_path)
        else:
            logger.warning(f"[edit_engine] Failed to extract segment {i}")

    if not clip_paths:
        logger.error("[edit_engine] No clips extracted — aborting")
        return None

    # Step 2: Concatenate all segments
    concat_path = str(temp_clips_dir / "concat.mp4")
    success = concat_clips(
        clip_paths, concat_path,
        width=preset["width"],
        height=preset["height"],
        fps=preset["fps"],
    )
    if not success:
        logger.error("[edit_engine] Concatenation failed")
        return None

    current_path = concat_path

    # Step 3: Apply colour grading if LUT specified
    if edl.lut_path and os.path.exists(edl.lut_path):
        graded_path = str(temp_clips_dir / "graded.mp4")
        if apply_colour_lut(current_path, graded_path, edl.lut_path):
            current_path = graded_path

    # Step 4: Mix background music if specified
    if edl.background_music_path and os.path.exists(edl.background_music_path):
        music_path = str(temp_clips_dir / "with_music.mp4")
        if add_audio_track(
            current_path, edl.background_music_path,
            music_path, edl.music_volume
        ):
            current_path = music_path

    # Step 5: Copy to final output
    output_filename = f"{edl.output_format}.mp4"
    final_path = str(piece_dir / output_filename)
    shutil.copy2(current_path, final_path)

    # Cleanup temp files
    try:
        shutil.rmtree(temp_clips_dir)
    except Exception as e:
        logger.warning(f"[edit_engine] Temp cleanup failed: {e}")

    logger.info(
        f"[edit_engine] EDL executed: {final_path} "
        f"({len(clip_paths)} segments, {edl.output_format})"
    )
    return final_path


def save_edl_to_json(edl: EditDecisionList, path: str) -> None:
    """Persist an EDL to disk for debugging/replay."""
    data = {
        "piece_id": edl.piece_id,
        "output_format": edl.output_format,
        "width": edl.width,
        "height": edl.height,
        "fps": edl.fps,
        "background_music_path": edl.background_music_path,
        "music_volume": edl.music_volume,
        "lut_path": edl.lut_path,
        "total_duration": edl.total_duration,
        "segments": [
            {
                "source_path": s.source_path,
                "start_seconds": s.start_seconds,
                "end_seconds": s.end_seconds,
                "segment_type": s.segment_type,
                "caption_text": s.caption_text,
            }
            for s in edl.segments
        ],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
