# FILE: app/content/video_pipeline/edl_builder.py
"""
EDL Builder — deterministic Edit Decision List construction.

Takes resolved assets, narration audio, avatar clips, and style profile
and constructs an EditDecisionList for the existing FFmpeg Edit Engine.

No AI involved — pure deterministic assembly.
"""
import os
import logging
import subprocess
from typing import List, Dict, Optional
from pathlib import Path

from app.content.video_pipeline.models import (
    ScenePlan, ResolvedPlan, ResolvedAsset, StyleProfile,
    AssetSource, AvatarFraming,
)

logger = logging.getLogger(__name__)


def _sanitise_filename(name: str) -> str:
    """Remove characters illegal in Windows filenames."""
    import re
    # Replace : < > " / \ | ? * with underscore
    sanitised = re.sub(r'[:<>"/\\|?*]', '_', name)
    # Strip trailing dots/spaces (Windows issue)
    return sanitised.strip('. ')


def get_audio_duration(audio_path: str) -> float:
    """Get duration of an audio file in seconds via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0", audio_path,
            ],
            capture_output=True, text=True, timeout=15,
        )
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"[edl_builder] Cannot get duration for {audio_path}: {e}")
        return 0.0


def build_edl(
    scene_plan: ScenePlan,
    resolved_plan: ResolvedPlan,
    narration_audio_paths: Dict[str, str],
    style: StyleProfile,
    output_format: str = "youtube_longform",
) -> Dict:
    """
    Build an EditDecisionList from all resolved inputs.

    This bridges to the existing edit_engine.py EditDecisionList dataclass.

    Args:
        scene_plan: The structured scene plan
        resolved_plan: Resolved assets per segment
        narration_audio_paths: Dict of segment_id -> WAV file path
        style: Style profile for transitions, audio, captions
        output_format: Target format preset

    Returns:
        Dict matching EditDecisionList structure for execute_edl()
    """
    from app.content.production.edit_engine import (
        EditDecisionList, EditSegment, FORMAT_PRESETS,
    )

    # Get format preset
    preset = FORMAT_PRESETS.get(output_format, FORMAT_PRESETS["youtube_longform"])

    # Build asset lookup: segment_id -> ResolvedAsset
    asset_map = {a.segment_id: a for a in resolved_plan.assets}

    segments = []
    current_time = 0.0

    for scene_seg in scene_plan.segments:
        asset = asset_map.get(scene_seg.segment_id)
        if not asset or not asset.file_path:
            logger.warning(
                f"[edl_builder] No asset for {scene_seg.segment_id}, skipping"
            )
            continue

        # Determine segment duration from the ground truth source:
        # - B-roll segments: use TTS narration audio duration
        # - Avatar segments: probe the actual video file duration
        #   (HeyGen renders the full text, duration may differ
        #    from Gemini's estimate)
        audio_path = narration_audio_paths.get(scene_seg.segment_id)
        if audio_path and os.path.exists(audio_path):
            # B-roll: duration = narration audio length
            duration = get_audio_duration(audio_path)
        elif (
            scene_seg.requires_avatar
            and asset.file_path
            and os.path.exists(asset.file_path)
        ):
            # Avatar: probe the actual video clip duration
            duration = get_audio_duration(asset.file_path)
            if duration > 0:
                logger.info(
                    f"[edl_builder] {scene_seg.segment_id}: "
                    f"avatar clip actual duration = {duration:.1f}s "
                    f"(estimate was {scene_seg.estimated_duration_s:.1f}s)"
                )
        else:
            duration = scene_seg.estimated_duration_s

        if duration <= 0:
            duration = scene_seg.estimated_duration_s

        # Determine segment type for the EDL
        if scene_seg.segment_type.value == "cutaway":
            seg_type = "cutaway"
        elif scene_seg.segment_type.value == "intro":
            seg_type = "intro"
        elif scene_seg.segment_type.value == "outro":
            seg_type = "outro"
        else:
            seg_type = "anchor"

        # Handle avatar framing modes
        framing = scene_seg.avatar_framing

        if framing == AvatarFraming.PIP and asset.source == AssetSource.HEYGEN:
            # PiP mode: avatar is a small overlay in bottom-right.
            # We need the b-roll as the main source and the avatar
            # clip path stored in metadata for FFmpeg overlay.
            # The b-roll asset should have been resolved separately
            # by the resolver (avatar + b-roll for the same segment).
            edl_seg = EditSegment(
                source_path=asset.file_path,
                start_seconds=0.0,
                end_seconds=duration,
                segment_type=seg_type,
                caption_text=scene_seg.script_text if style.caption_style != "none" else None,
                caption_start=0.0 if scene_seg.script_text else None,
                caption_end=duration if scene_seg.script_text else None,
            )
            # Store avatar overlay info for FFmpeg compositing
            # The assembly step will use chromakey + overlay filter
            edl_seg._avatar_overlay = {
                "mode": "pip",
                "avatar_path": asset.file_path,
                "position": "bottom_right",
                "scale": 0.25,  # 25% of frame width
                "chromakey_color": "0x00FF00",
            }
        elif framing == AvatarFraming.FULL_FRAME and asset.source == AssetSource.HEYGEN:
            # Full frame: avatar is the main visual.
            # Green screen will be keyed out and replaced with
            # a generated digital environment background.
            edl_seg = EditSegment(
                source_path=asset.file_path,
                start_seconds=0.0,
                end_seconds=duration,
                segment_type=seg_type,
                caption_text=scene_seg.script_text if style.caption_style != "none" else None,
                caption_start=0.0 if scene_seg.script_text else None,
                caption_end=duration if scene_seg.script_text else None,
            )
            edl_seg._avatar_overlay = {
                "mode": "full_frame",
                "avatar_path": asset.file_path,
                "chromakey_color": "0x00FF00",
            }
        else:
            # Standard segment — no avatar compositing
            edl_seg = EditSegment(
                source_path=asset.file_path,
                start_seconds=0.0,
                end_seconds=duration,
                segment_type=seg_type,
                caption_text=scene_seg.script_text if style.caption_style != "none" else None,
                caption_start=0.0 if scene_seg.script_text else None,
                caption_end=duration if scene_seg.script_text else None,
            )

        segments.append(edl_seg)
        current_time += duration

    # Build the EDL
    edl = EditDecisionList(
        piece_id=f"video_pipeline_{_sanitise_filename(scene_plan.title[:30])}",
        output_format=output_format,
        segments=segments,
        width=preset.get("width", 1920),
        height=preset.get("height", 1080),
        fps=preset.get("fps", 30),
        background_music_path=None,  # TODO: music selection
        music_volume=style.music_volume_ratio,
        voice_volume=style.voice_volume,
        lut_path=style.lut_reference,
    )

    logger.info(
        f"[edl_builder] EDL built: {len(segments)} segments, "
        f"~{edl.total_duration:.1f}s total, format={output_format}"
    )
    return edl


def build_narration_map(
    scene_plan: ScenePlan,
    tts_output_dir: str,
) -> Dict[str, str]:
    """
    Map segment IDs to narration audio file paths.
    Assumes TTS has generated files named {segment_id}.wav
    in the output directory.
    """
    output_dir = Path(tts_output_dir)
    narration_map = {}

    for segment in scene_plan.segments:
        # Check for WAV file
        wav_path = output_dir / f"{segment.segment_id}.wav"
        if wav_path.exists():
            narration_map[segment.segment_id] = str(wav_path)
        else:
            # Also check mp3
            mp3_path = output_dir / f"{segment.segment_id}.mp3"
            if mp3_path.exists():
                narration_map[segment.segment_id] = str(mp3_path)

    logger.info(
        f"[edl_builder] Narration map: {len(narration_map)}/{len(scene_plan.segments)} "
        f"segments have audio"
    )
    return narration_map
