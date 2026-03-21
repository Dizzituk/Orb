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
            cmd, capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
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

    The bake step already produces perfectly normalised clips
    (1920x1080, libx264, aac 44100Hz stereo). We just stitch
    them with -c copy. NO re-encoding — re-encoding causes
    cumulative audio timing drift that breaks lip sync.

    Only falls back to re-encode normalisation if the direct
    concat fails (e.g. mixed codecs from non-baked sources).
    """
    os.makedirs(TEMP_DIR, exist_ok=True)

    # FAST PATH: direct concat with -c copy (no re-encoding).
    # The bake step guarantees consistent format, so this should
    # always work for pipeline-generated clips.
    list_path = str(TEMP_DIR / "concat_list.txt")
    with open(list_path, "w") as f:
        for clip in clip_paths:
            abs_clip = os.path.abspath(clip)
            escaped = abs_clip.replace("'", "'\\''") 
            f.write(f"file '{escaped}'\n")
    result = _run_ffmpeg([
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        output_path,
    ])

    if result and os.path.exists(output_path):
        return True

    # SLOW PATH: concat failed (mixed formats). Normalise first.
    logger.warning(
        "[edit_engine] Direct concat failed — normalising clips"
    )
    normalized_paths = []
    for i, clip in enumerate(clip_paths):
        norm_path = str(TEMP_DIR / f"norm_{i:03d}.mp4")
        success = _run_ffmpeg([
            "-i", os.path.abspath(clip),
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                   f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps}",
            "-c:v", "libx264", "-preset", "fast",
            "-crf", "23",
            "-ar", "44100", "-ac", "2",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            "-vsync", "cfr",
            "-async", "1",
            norm_path,
        ])
        if success and os.path.exists(norm_path):
            normalized_paths.append(norm_path)
        else:
            logger.warning(
                f"[edit_engine] Failed to normalize clip {i}: {clip}"
            )

    if not normalized_paths:
        logger.error("[edit_engine] No clips survived normalization")
        return False

    list_path2 = str(TEMP_DIR / "concat_list_norm.txt")
    with open(list_path2, "w") as f:
        for clip in normalized_paths:
            abs_clip = os.path.abspath(clip)
            escaped = abs_clip.replace("'", "'\\''") 
            f.write(f"file '{escaped}'\n")
    result = _run_ffmpeg([
        "-f", "concat", "-safe", "0",
        "-i", list_path2,
        "-c", "copy",
        output_path,
    ])

    # Cleanup normalized temp files
    for p in normalized_paths:
        try:
            os.remove(p)
        except OSError:
            pass

    return result


def composite_avatar_on_background(
    avatar_path: str,
    output_path: str,
    background_path: Optional[str] = None,
) -> bool:
    """
    Composite a HeyGen avatar clip onto a background image.

    Uses FFmpeg colorkey to remove the dark HeyGen background (#080812)
    and overlay the avatar onto the specified background image.
    Falls back to a generated dark tech gradient if no background is set.

    Args:
        avatar_path: Path to the HeyGen avatar .mp4
        output_path: Where to save the composited result
        background_path: Path to a 1920x1080 background image (PNG/JPG)
    """
    # Find background image
    if not background_path or not os.path.exists(background_path):
        # Check for default backgrounds
        bg_dir = Path("data/content/video_pipeline/backgrounds")
        if bg_dir.exists():
            candidates = list(bg_dir.glob("*.png")) + list(bg_dir.glob("*.jpg"))
            if candidates:
                background_path = str(candidates[0])

    if not background_path or not os.path.exists(background_path):
        logger.info(
            "[edit_engine] No background image available, "
            "skipping avatar compositing"
        )
        return False

    abs_avatar = os.path.abspath(avatar_path)
    abs_bg = os.path.abspath(background_path)
    abs_out = os.path.abspath(output_path)

    # Composite avatar onto background.
    #
    # Strategy: colorkey removes the dark HeyGen background,
    # then a box blur on the alpha channel softens the edges
    # to prevent hard rectangular cutout boundaries.
    #
    # Filter chain:
    # 1. Scale background to 1920x1080
    # 2. Colorkey the avatar (remove #080812 with soft blend)
    # 3. Split alpha, blur it, merge back (feathers edges)
    # 4. Overlay onto background
    success = _run_ffmpeg(
        [
            "-loop", "1", "-i", abs_bg,
            "-i", abs_avatar,
            "-filter_complex",
            "[0:v]scale=1920:1080,format=rgba[bg];"
            "[1:v]colorkey=0x080812:0.12:0.15,format=rgba,"
            # Split into RGB + Alpha, blur the alpha, recombine.
            # This feathers all edges of the keyed region.
            "split[rgb][a];"
            "[a]alphaextract,boxblur=2:2[soft_alpha];"
            "[rgb][soft_alpha]alphamerge[keyed];"
            "[bg][keyed]overlay=0:0:shortest=1,format=yuv420p[out]",
            "-map", "[out]", "-map", "1:a",
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            abs_out,
        ],
        timeout=300,
    )

    if success:
        logger.info(
            f"[edit_engine] Avatar composited onto background: {output_path}"
        )
    else:
        logger.warning(
            f"[edit_engine] Avatar compositing failed for {avatar_path}"
        )

    return success


def apply_crossfades(
    clip_paths: List[str],
    output_path: str,
    crossfade_duration: float = 0.4,
) -> bool:
    """
    Concatenate clips with crossfade transitions between each pair.

    Uses FFmpeg's xfade filter. All clips must be pre-normalised to
    identical format (resolution, fps, audio codec/sample rate).

    If only 1 clip, just copies it. Falls back to simple concat on failure.
    """
    if len(clip_paths) < 2:
        if clip_paths:
            shutil.copy2(clip_paths[0], output_path)
            return True
        return False

    # Get durations of each clip via ffprobe
    durations = []
    for clip in clip_paths:
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0", os.path.abspath(clip),
                ],
                capture_output=True, text=True, timeout=10,
            )
            dur = float(result.stdout.strip())
            durations.append(dur)
        except Exception:
            durations.append(5.0)  # fallback

    # Build the xfade filter chain
    # For N clips, we need N-1 xfade operations chained together
    n = len(clip_paths)
    inputs = []
    for clip in clip_paths:
        inputs.extend(["-i", os.path.abspath(clip)])

    # Build video xfade chain
    xf_dur = min(crossfade_duration, 0.5)  # cap at 0.5s
    video_filters = []
    audio_filters = []

    # Calculate offsets — each xfade starts at (cumulative duration - xfade_dur)
    offsets = []
    cumulative = 0.0
    for i in range(n - 1):
        cumulative += durations[i]
        offset = cumulative - xf_dur * (i + 1)
        offsets.append(max(0, offset))

    if n == 2:
        # Simple case: 2 clips, 1 xfade
        video_filters.append(
            f"[0:v][1:v]xfade=transition=fade:duration={xf_dur}:"
            f"offset={offsets[0]:.2f}[vout]"
        )
        audio_filters.append(
            f"[0:a][1:a]acrossfade=d={xf_dur}:c1=tri:c2=tri[aout]"
        )
        filter_complex = ";".join(video_filters + audio_filters)
        maps = ["-map", "[vout]", "-map", "[aout]"]
    else:
        # Multi-clip: chain xfades
        # Video chain
        prev_label = "0:v"
        for i in range(1, n):
            out_label = f"v{i}" if i < n - 1 else "vout"
            video_filters.append(
                f"[{prev_label}][{i}:v]xfade=transition=fade:"
                f"duration={xf_dur}:offset={offsets[i-1]:.2f}[{out_label}]"
            )
            prev_label = out_label

        # Audio chain
        prev_label = "0:a"
        for i in range(1, n):
            out_label = f"a{i}" if i < n - 1 else "aout"
            audio_filters.append(
                f"[{prev_label}][{i}:a]acrossfade=d={xf_dur}:"
                f"c1=tri:c2=tri[{out_label}]"
            )
            prev_label = out_label

        filter_complex = ";".join(video_filters + audio_filters)
        maps = ["-map", "[vout]", "-map", "[aout]"]

    success = _run_ffmpeg(
        inputs + [
            "-filter_complex", filter_complex,
        ] + maps + [
            "-c:v", "libx264", "-preset", "medium",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            output_path,
        ],
        timeout=600,
    )

    if not success:
        logger.warning(
            "[edit_engine] Crossfade failed, falling back to simple concat"
        )
        # Fallback: simple concat without transitions
        list_path = str(TEMP_DIR / "xfade_fallback.txt")
        with open(list_path, "w") as f:
            for clip in clip_paths:
                escaped = os.path.abspath(clip).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")
        return _run_ffmpeg([
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            output_path,
        ])

    return success


def add_audio_track(
    video_path: str,
    audio_path: str,
    output_path: str,
    audio_volume: float = 0.12,
    use_ducking: bool = True,
) -> bool:
    """
    Mix background music into a video with voice-activated ducking.

    When use_ducking=True, the video's voice track acts as a sidechain
    trigger: music dips when voice is present and gently rises during
    pauses. This creates a professional broadcast feel.

    Ducking parameters (tuned for subtle, smooth behaviour):
    - threshold=0.03: only trigger on actual speech, not micro-pauses
    - ratio=10: strong ducking (music drops significantly)
    - attack=500ms: smooth 500ms fade down (half a second)
    - release=3000ms: gentle 3s rise back up (only swells during real pauses)
    """
    if use_ducking:
        return _run_ffmpeg([
            "-i", video_path,
            "-i", audio_path,
            "-filter_complex",
            # Set music base volume, then duck it under voice
            f"[1:a]volume={audio_volume}[music];"
            f"[0:a]asplit[voice][sc];"
            f"[music][sc]sidechaincompress="
            f"threshold=0.03:ratio=10:"
            f"attack=500:release=3000"
            f"[ducked];"
            f"[voice][ducked]amix=inputs=2:"
            f"duration=first:dropout_transition=2[out]",
            "-map", "0:v", "-map", "[out]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output_path,
        ])
    else:
        # Flat mix without ducking (fallback)
        return _run_ffmpeg([
            "-i", video_path,
            "-i", audio_path,
            "-filter_complex",
            f"[1:a]volume={audio_volume}[bg];"
            f"[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[out]",
            "-map", "0:v", "-map", "[out]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output_path,
        ])


def generate_thumbnail(
    video_path: str,
    output_path: str,
    title: str = "",
    astra_head_path: Optional[str] = None,
    seek_percent: float = 0.25,
) -> bool:
    """
    Generate a YouTube thumbnail from the video.

    Takes a frame at seek_percent into the video, blurs it,
    darkens slightly, places the Astra head bottom-right,
    and adds bold title text centre-left.

    Output is 1280x720 PNG (YouTube standard).
    """
    # Find Astra head branding
    if not astra_head_path or not os.path.exists(astra_head_path):
        brand_dir = Path("data/content/video_pipeline/branding")
        if brand_dir.exists():
            candidates = list(brand_dir.glob("astra_head*"))
            if candidates:
                astra_head_path = str(candidates[0])

    # Get video duration for seek point
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
        duration = float(result.stdout.strip())
    except Exception:
        duration = 60.0

    seek_to = duration * seek_percent

    # Clean title for FFmpeg drawtext (escape special chars)
    safe_title = title.upper().replace("'", "").replace("\\", "").replace(":", "").replace(";", "")
    if len(safe_title) > 20:
        # Always wrap to two lines for readability
        words = safe_title.split()
        mid = len(words) // 2
        safe_title = " ".join(words[:mid]) + "\\n" + " ".join(words[mid:])

    font_path = "C\\:\\Windows\\Fonts\\impact.ttf"

    if astra_head_path and os.path.exists(astra_head_path):
        # Full thumbnail: blurred bg + Astra head + title
        abs_head = os.path.abspath(astra_head_path)
        success = _run_ffmpeg([
            "-ss", str(seek_to),
            "-i", os.path.abspath(video_path),
            "-i", abs_head,
            "-frames:v", "1",
            "-filter_complex",
            f"[0:v]scale=1280:720,gblur=sigma=12,"
            f"eq=brightness=-0.15:saturation=1.2[bg];"
            f"[1:v]scale=400:-1[head];"
            f"[bg][head]overlay=W-w-40:H-h-20[comp];"
            f"[comp]drawtext=fontfile='{font_path}':"
            f"text='{safe_title}':fontsize=80:"
            f"fontcolor=white:borderw=5:bordercolor=black:"
            f"x=50:y=80",
            output_path,
        ])
    else:
        # No Astra head: just blurred bg + title
        success = _run_ffmpeg([
            "-ss", str(seek_to),
            "-i", os.path.abspath(video_path),
            "-frames:v", "1",
            "-vf",
            f"scale=1280:720,gblur=sigma=8,"
            f"eq=brightness=-0.1:saturation=1.2,"
            f"drawtext=fontfile='{font_path}':"
            f"text='{safe_title}':fontsize=80:"
            f"fontcolor=white:borderw=5:bordercolor=black:"
            f"x=(w-text_w)/2:y=(h-text_h)/2",
            output_path,
        ])

    if success:
        logger.info(f"[edit_engine] Thumbnail generated: {output_path}")
    else:
        logger.warning(f"[edit_engine] Thumbnail generation failed")

    return success


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

        # If start is 0, the source is already a complete clip
        # (baked with audio + gap). Just copy it — don't trim,
        # because trim can cut off audio at the end.
        if segment.start_seconds == 0.0:
            import shutil as _sh
            _sh.copy2(
                os.path.abspath(segment.source_path),
                clip_path,
            )
            success = os.path.exists(clip_path)
        else:
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

    # Step 5: Force keyframes every 2s for smooth seeking.
    # The -c copy concat produces sparse keyframes (sometimes
    # 250+ second gaps) which makes VLC/WMP freeze on seek.
    # This fast re-encode fixes it with negligible quality loss.
    seekable_path = str(temp_clips_dir / "seekable.mp4")
    kf_ok = _run_ffmpeg([
        "-i", current_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-g", "60",  # keyframe every 60 frames = 2s at 30fps
        "-keyint_min", "30",
        "-c:a", "copy",
        "-movflags", "+faststart",
        seekable_path,
    ])
    if kf_ok and os.path.exists(seekable_path):
        current_path = seekable_path
        logger.info("[edit_engine] Keyframe pass: forced KF every 2s")
    else:
        logger.warning(
            "[edit_engine] Keyframe pass failed, using concat output"
        )

    # Step 6: Copy to final output
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
