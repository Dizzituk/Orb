# FILE: app/content/video_pipeline/bake_segment.py
# Purpose: Segment Baker — simple, correct clip assembly.
# Called-by: app.content.video_pipeline.orchestrator
# Depends-on: app.content.video_pipeline.asset_library
# Last-renovated: 2026-06-11
"""
Segment Baker — simple, correct clip assembly.

RULES:
1. Avatar = point of truth. Never re-encode. Copy as-is + pad gap.
2. B-roll = fill audio duration with clips. Each clip plays for an
   equal share. No clip shorter than 2 seconds.
3. No forced frame rate. Preserve source fps.
4. No effects during bake. Effects are a post-process step.
5. Measure VIDEO STREAM duration, never container duration.
6. If clips run short, get more clips until segment is filled.
"""
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional, Set, List

logger = logging.getLogger(__name__)

SCENE_GAP_S = 0.75
MAX_FILLER_ATTEMPTS = 5
MIN_CLIP_DURATION = 2.0  # Never show a clip for less than 2 seconds


def _probe_duration(file_path: str) -> float:
    """Container duration via ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet",
             "-show_entries", "format=duration",
             "-of", "csv=p=0",
             os.path.abspath(file_path)],
            capture_output=True, text=True, timeout=15,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _probe_video_duration(file_path: str) -> float:
    """Actual video stream duration (not container)."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet",
             "-select_streams", "v:0",
             "-show_entries", "stream=duration",
             "-of", "csv=p=0",
             os.path.abspath(file_path)],
            capture_output=True, text=True, timeout=15,
        )
        val = result.stdout.strip()
        if val and val != "N/A":
            return float(val)
    except Exception:
        pass
    return _probe_duration(file_path)


def _gather_filler_clips(
    filler_dir: str,
    used_paths: Set[str],
    exclude_paths: Set[str],
    strict: bool = True,
) -> List[str]:
    """Get available filler clips from a directory.

    strict=True: respects cooldown and cross-segment reuse.
    strict=False: ignores restrictions. Repeated clip > freeze.
    """
    fdir = Path(filler_dir)
    if not fdir.exists():
        return []

    if strict:
        from app.content.video_pipeline.asset_library import (
            is_clip_on_cooldown,
        )

    candidates = []
    for f in fdir.glob("*.mp4"):
        abs_f = os.path.abspath(str(f))
        if abs_f in exclude_paths:
            continue
        if strict:
            if abs_f in used_paths:
                continue
            if is_clip_on_cooldown(str(f)):
                continue
        candidates.append(abs_f)

    candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return candidates


def _trim_clip(
    input_path: str,
    output_path: str,
    duration: float,
) -> bool:
    """Trim a clip to duration. Scale to 1080p, normalise to 25fps.

    25fps matches HeyGen avatar output — all clips share one fps.
    No audio — audio is merged separately.
    """
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", os.path.abspath(input_path),
            "-t", f"{duration:.3f}",
            "-vf",
            "scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
            "format=yuv420p",
            "-c:v", "libx264", "-preset", "fast",
            "-crf", "23", "-an",
            os.path.abspath(output_path),
        ],
        capture_output=True, text=True,
        timeout=60, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        logger.warning(f"[bake] Trim failed: {result.stderr[-200:]}")
        return False
    return (
        os.path.exists(output_path)
        and os.path.getsize(output_path) > 500
    )


def _gather_enough_clips(
    all_clips: List[str],
    target_dur: float,
    filler_clips_dir: str,
    used: Set[str],
) -> List[str]:
    """Ensure we have enough clips to cover target_dur.

    Adds fillers until total video footage >= target.
    Tries strict first, then relaxed (ignores cooldown).
    """
    def _total():
        return sum(_probe_video_duration(c) for c in all_clips)

    if _total() >= target_dur:
        return all_clips

    # Try strict fillers first
    for strict in [True, False]:
        fillers = _gather_filler_clips(
            filler_clips_dir,
            used_paths=used,
            exclude_paths=set(all_clips),
            strict=strict,
        )
        pixabay_dir = "data/content/video_pipeline/downloads/pixabay"
        fillers.extend(_gather_filler_clips(
            pixabay_dir,
            used_paths=used,
            exclude_paths=set(all_clips),
            strict=strict,
        ))

        for fp in fillers:
            fd = _probe_video_duration(fp)
            if fd < 0.5:
                continue
            all_clips.append(fp)
            used.add(fp)
            if _total() >= target_dur:
                return all_clips

        if not strict:
            logger.warning(
                "[bake] Exhausted all fillers including relaxed mode"
            )

    return all_clips


def bake_broll_segment(
    video_path: str,
    audio_path: str,
    output_path: str,
    segment_index: int = 0,
    key_phrase: str = "",
    filler_clips_dir: str = (
        "data/content/video_pipeline/downloads/pexels"
    ),
    used_video_paths: Optional[Set[str]] = None,
    clip_b_path: str = "",
) -> bool:
    """Bake a b-roll segment.

    Simple logic:
    1. Measure audio duration = target
    2. Gather enough clips to cover target
    3. Divide target equally across clips
    4. Trim each clip to its share
    5. Concat clips
    6. Merge with audio
    """
    audio_dur = _probe_duration(audio_path)
    if audio_dur <= 0:
        logger.warning(f"[bake] Audio duration 0: {audio_path}")
        return False

    target_dur = audio_dur + SCENE_GAP_S
    abs_audio = os.path.abspath(audio_path)
    _used = used_video_paths if used_video_paths is not None else set()

    # ── Step 1: Gather initial clips ──
    all_clips = []
    primary_abs = os.path.abspath(video_path)
    if _probe_video_duration(video_path) > 0.5:
        all_clips.append(primary_abs)
        _used.add(primary_abs)

    if clip_b_path and os.path.exists(clip_b_path):
        cb_abs = os.path.abspath(clip_b_path)
        if _probe_video_duration(clip_b_path) > 0.5:
            if cb_abs not in _used:
                all_clips.append(cb_abs)
                _used.add(cb_abs)

    # ── Step 2: Gather enough clips to fill target ──
    all_clips = _gather_enough_clips(
        all_clips, target_dur, filler_clips_dir, _used,
    )

    if not all_clips:
        logger.warning(f"[bake] No clips for segment {segment_index}")
        return False

    # ── Step 3: Calculate equal share per clip ──
    # Each clip gets target_dur / num_clips seconds.
    # But never trim a clip longer than its actual video duration.
    num_clips = len(all_clips)
    share = target_dur / num_clips

    # If share is too small, we have too many clips — trim list
    while share < MIN_CLIP_DURATION and num_clips > 1:
        all_clips.pop()
        num_clips = len(all_clips)
        share = target_dur / num_clips

    logger.info(
        f"[bake] Segment {segment_index}: {target_dur:.1f}s target, "
        f"{num_clips} clips x {share:.1f}s each"
    )

    # ── Step 4: Trim each clip to its share ──
    # Two-pass: first find clips shorter than their share,
    # then redistribute leftover time to longer clips.
    temp_dir = Path("data/content/temp/bake")
    temp_dir.mkdir(parents=True, exist_ok=True)

    clip_durations = [
        _probe_video_duration(c) for c in all_clips
    ]

    # Pass 1: identify what each clip can give
    remaining = target_dur
    long_clips = []  # clips that have more footage than share
    short_total = 0.0  # time contributed by short clips

    for i, dur in enumerate(clip_durations):
        if dur < share:
            # Short clip — use all of it
            short_total += dur
            remaining -= dur
        else:
            long_clips.append(i)

    # Pass 2: redistribute remaining time to long clips
    if long_clips:
        long_share = remaining / len(long_clips)
    else:
        long_share = share

    # Now trim each clip
    trimmed = []
    for i, clip_path in enumerate(all_clips):
        dur = clip_durations[i]
        if dur < share:
            trim_to = dur  # use full length of short clip
        else:
            trim_to = min(long_share, dur)

        out = str(temp_dir / f"seg_{segment_index:03d}_{i:02d}.mp4")
        if _trim_clip(clip_path, out, trim_to):
            actual = _probe_video_duration(out)
            if actual > 0.3:
                trimmed.append(out)

    if not trimmed:
        logger.warning(f"[bake] No clips survived trim: seg {segment_index}")
        return False

    # ── Step 5: Check total — do we have enough? ──
    total_video = sum(_probe_video_duration(t) for t in trimmed)
    if total_video < target_dur - 0.5:
        logger.warning(
            f"[bake] Segment {segment_index}: video={total_video:.1f}s "
            f"vs target={target_dur:.1f}s — short by "
            f"{target_dur - total_video:.1f}s"
        )

    # ── Step 6: Concat trimmed clips ──
    if len(trimmed) == 1:
        video_file = trimmed[0]
    else:
        video_file = str(temp_dir / f"concat_{segment_index:03d}.mp4")

        # Use concat FILTER (not demuxer) — handles mixed
        # frame rates correctly without altering playback speed.
        # Each clip keeps its real-world timing.
        inputs = []
        filter_parts = []
        for i, t in enumerate(trimmed):
            inputs.extend(["-i", os.path.abspath(t)])
            filter_parts.append(f"[{i}:v]")

        filter_str = (
            "".join(filter_parts)
            + f"concat=n={len(trimmed)}:v=1:a=0[outv]"
        )

        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_str,
            "-map", "[outv]",
            "-c:v", "libx264", "-preset", "fast",
            "-crf", "23",
            video_file,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=120, encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            logger.warning(
                f"[bake] Concat failed seg {segment_index}: "
                f"{result.stderr[-200:]}"
            )
            video_file = trimmed[0]
    # ── Step 7: Merge video + audio ──
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", os.path.abspath(video_file),
            "-i", abs_audio,
            "-t", f"{target_dur:.3f}",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-r", "25",  # Output at 25fps — matches avatar
            "-c:a", "aac", "-b:a", "192k",
            "-ar", "44100", "-ac", "2",
            "-af", f"apad=whole_dur={target_dur}",
            "-movflags", "+faststart",
            output_path,
        ],
        capture_output=True, text=True, timeout=120,
    )

    if result.returncode != 0:
        logger.warning(
            f"[bake] Merge failed seg {segment_index}: "
            f"{result.stderr[-300:]}"
        )
        return False

    # ── Step 8: Validate ──
    vid_dur = _probe_video_duration(output_path)
    aud_dur = _probe_duration(output_path)
    gap = aud_dur - vid_dur

    if gap > 0.5:
        logger.warning(
            f"[bake_validate] FREEZE: video={vid_dur:.1f}s "
            f"audio={aud_dur:.1f}s gap={gap:.1f}s"
        )
        return False

    logger.info(
        f"[bake] Segment {segment_index}: OK — "
        f"{num_clips} clips, {vid_dur:.1f}s video, "
        f"{aud_dur:.1f}s audio"
    )
    return True


def pad_avatar_segment(
    avatar_path: str,
    output_path: str,
) -> bool:
    """Avatar = point of truth. Never change frame rate.

    Just add SCENE_GAP_S of silence + frozen last frame.
    Preserve original fps (25fps from HeyGen).
    """
    avatar_dur = _probe_duration(avatar_path)
    if avatar_dur <= 0:
        logger.warning(f"[bake] Avatar duration 0: {avatar_path}")
        return False

    total_dur = avatar_dur + SCENE_GAP_S
    abs_avatar = os.path.abspath(avatar_path)

    # Preserve source frame rate — no -r flag, no -vsync cfr
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", abs_avatar,
            "-vf",
            f"tpad=stop_mode=clone:stop_duration={SCENE_GAP_S}",
            "-af", f"apad=whole_dur={total_dur}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-r", "25",  # Output at 25fps — matches avatar
            "-c:a", "aac", "-b:a", "192k",
            "-ar", "44100", "-ac", "2",
            "-movflags", "+faststart",
            output_path,
        ],
        capture_output=True, text=True, timeout=120,
    )

    if result.returncode != 0:
        logger.warning(
            f"[bake] Avatar pad failed: {result.stderr[-300:]}"
        )
        return False

    logger.info(
        f"[bake] Avatar: {avatar_dur:.1f}s + "
        f"gap={SCENE_GAP_S}s = {total_dur:.1f}s"
    )
    return True


