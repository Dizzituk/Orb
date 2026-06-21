# FILE: app/content/video_pipeline/_resolver_indexing.py
# Purpose: Post-download Gemini clip analysis + asset-library indexing (shared by the fetchers).
# Called-by: app.content.video_pipeline._resolver_fetchers, app.content.video_pipeline.asset_resolver
# Depends-on: app.content.video_pipeline.models, app.content.video_pipeline.clip_analyzer (lazy), app.content.video_pipeline.asset_library (lazy)
# Last-renovated: 2026-06-20
"""
_analyze_and_index - analyze a downloaded clip with Gemini (clip_analyzer)
and index it in the asset library. Non-blocking: failures are logged.

Extracted verbatim from asset_resolver.py on 2026-06-20 (split campaign,
batch 2). Logic byte-identical to the pre-split module.
"""
import logging
from typing import Optional

from app.content.video_pipeline.models import SceneSegment

logger = logging.getLogger(__name__)


async def _analyze_and_index(
    file_path: str,
    source: str,
    segment: SceneSegment,
    extra_keywords: Optional[list] = None,
    duration_s: float = 0.0,
    cost_usd: float = 0.0,
) -> None:
    """Analyze a downloaded clip with Gemini and index it in the library.

    Runs clip_analyzer to get a rich visual description, then
    passes it to asset_library.index_asset for embedding + storage.
    Non-blocking: failures are logged but do not break the pipeline.
    """
    try:
        from app.content.video_pipeline.clip_analyzer import analyze_clip
        from app.content.video_pipeline.asset_library import index_asset

        # Analyze the clip (Gemini multimodal — extracts frames)
        analysis = await analyze_clip(file_path, clip_id=segment.segment_id)

        # Build keyword list
        all_kw = list(segment.search_keywords)
        if extra_keywords:
            all_kw.extend(extra_keywords)
        for clip in segment.clips:
            if clip.search_query:
                all_kw.append(clip.search_query)

        # Visual description: prefer analysis, fallback to segment
        vis_desc = segment.visual_description
        if segment.clips and segment.clips[0].shot_description:
            vis_desc = segment.clips[0].shot_description

        index_asset(
            file_path=file_path,
            source=source,
            segment_id=segment.segment_id,
            search_keywords=all_kw,
            visual_description=vis_desc,
            duration_s=duration_s,
            cost_usd=cost_usd,
            clip_analysis=analysis,
        )
    except Exception as e:
        logger.debug(f"[asset_resolver] Analyze+index failed: {e}")
