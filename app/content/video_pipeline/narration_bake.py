# FILE: app/content/video_pipeline/narration_bake.py
# Purpose: Audio-first bake — produce one self-contained clip per segment for the video pipeline.
# Called-by: app.content.video_pipeline.orchestrator
# Depends-on: app.content.video_pipeline.bake_segment, app.content.video_pipeline.segment_builder
# Last-renovated: 2026-06-21
"""
Audio-first bake helper.

Split out of orchestrator.py (BATCH 4) verbatim. Given an EDL and a
narration map, produces one self-contained clip per segment (equal-part
clip filling via the bake_segment module) and reports which source clips
were actually baked into the final output (for cooldown marking).
"""
import os
import logging
from typing import Callable, Optional
from pathlib import Path

from app.content.video_pipeline.models import ScenePlan, ResolvedPlan

logger = logging.getLogger(__name__)


# ── Audio-First Bake (uses bake_segment module) ──

async def _bake_narration_into_clips(
    edl,
    narration_map: dict,
    scene_plan: ScenePlan,
    job_dir: str,
    emit: Callable,
    resolved_plan: Optional["ResolvedPlan"] = None,
) -> tuple:
    """
    Audio-first bake: produce one self-contained clip per segment.

    Uses the bake_segment module for equal-part clip filling.
    Modifies edl.segments[].source_path in-place.
    Returns (baked_count, used_source_clips) — the count and the set
    of all original source clip paths that were actually baked into
    the final output. Only these should be marked for cooldown.
    """
    from app.content.video_pipeline.bake_segment import (
        bake_broll_segment, pad_avatar_segment, _probe_duration,
    )

    job_path = Path(job_dir)
    baked_dir = job_path / "baked_clips"
    baked_dir.mkdir(exist_ok=True)

    baked_count = 0
    edl_idx = 0
    used_video_paths: set = set()

    for segment in scene_plan.segments:
        if edl_idx >= len(edl.segments):
            break

        edl_seg = edl.segments[edl_idx]
        audio_path = narration_map.get(segment.segment_id)
        baked_path = str(baked_dir / f"baked_{edl_idx:03d}.mp4")

        if segment.requires_avatar:
            src = os.path.abspath(edl_seg.source_path)
            success = pad_avatar_segment(src, baked_path)
            if success:
                edl_seg.source_path = baked_path
                new_dur = _probe_duration(baked_path)
                if new_dur > 0:
                    edl_seg.start_seconds = 0.0
                    edl_seg.end_seconds = new_dur
                baked_count += 1
            else:
                logger.warning(
                    f"[bake_audio] Avatar pad failed for "
                    f"{segment.segment_id}, using original"
                )

        elif audio_path and os.path.exists(audio_path):
            src = os.path.abspath(edl_seg.source_path)
            used_video_paths.add(src)

            clip_b_path = ""
            if resolved_plan:
                for asset in resolved_plan.assets:
                    if asset.segment_id == segment.segment_id:
                        clip_b_path = asset.metadata.get("clip_b_path", "")
                        break

            # ── Segment Builder: Gemini selects clips ──
            # Gather candidates, let Gemini rank them,
            # then pass the best to the bake step.
            try:
                from app.content.video_pipeline.segment_builder import (
                    select_clips_for_segment, _probe_duration as _sb_probe,
                )

                candidates = [{"id": "primary", "path": src, "source": "pexels"}]
                if clip_b_path and os.path.exists(clip_b_path):
                    candidates.append({
                        "id": "clip_b", "path": os.path.abspath(clip_b_path),
                        "source": "pexels",
                    })

                audio_dur = _probe_duration(audio_path)
                ranked = await select_clips_for_segment(
                    segment_id=segment.segment_id,
                    narration_text=segment.script_text,
                    candidate_clips=candidates,
                    target_duration=audio_dur + 0.75,
                )

                # Use top-ranked clip as primary, second as clip_b
                if ranked and ranked[0].get("use"):
                    src = ranked[0]["path"]
                    used_video_paths.add(src)
                if len(ranked) > 1 and ranked[1].get("use"):
                    clip_b_path = ranked[1]["path"]
                else:
                    clip_b_path = ""

            except Exception as sb_err:
                logger.warning(
                    f"[bake_audio] Segment builder failed for "
                    f"{segment.segment_id}: {sb_err} — using defaults"
                )

            success = bake_broll_segment(
                video_path=src,
                audio_path=audio_path,
                output_path=baked_path,
                segment_index=edl_idx,
                used_video_paths=used_video_paths,
                clip_b_path=clip_b_path,
            )
            if success:
                edl_seg.source_path = baked_path
                new_dur = _probe_duration(baked_path)
                if new_dur > 0:
                    edl_seg.start_seconds = 0.0
                    edl_seg.end_seconds = new_dur
                baked_count += 1
            else:
                logger.warning(
                    f"[bake_audio] B-roll bake failed for "
                    f"{segment.segment_id}, using original"
                )

        else:
            logger.info(
                f"[bake_audio] No audio for "
                f"{segment.segment_id}, keeping original"
            )

        edl_idx += 1

    logger.info(
        f"[bake_audio] Complete: {baked_count}/{edl_idx} clips baked, "
        f"{len(used_video_paths)} source clips used"
    )
    return baked_count, used_video_paths
