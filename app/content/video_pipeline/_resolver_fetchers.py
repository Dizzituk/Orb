# FILE: app/content/video_pipeline/_resolver_fetchers.py
# Purpose: Per-tier footage fetchers (local RAG / Pexels / Pixabay / fal.ai / HeyGen) for each segment.
# Called-by: app.content.video_pipeline.asset_resolver
# Depends-on: app.content.video_pipeline.models, app.content.video_pipeline._resolver_budget, app.content.video_pipeline._resolver_indexing (+ lazy provider clients)
# Last-renovated: 2026-06-20
"""
Per-tier footage fetchers - each searches/generates and returns a ResolvedAsset.

Extracted verbatim from asset_resolver.py on 2026-06-20 (split campaign,
batch 2). Logic byte-identical to the pre-split module. The provider clients
(pexels / pixabay / fal / heygen / asset_library) stay lazily imported inside
each function exactly as before.
"""
import os
import logging
from typing import Optional

from app.content.video_pipeline.models import (
    SceneSegment, ResolvedAsset, AssetSource, AssetTier,
)
from app.content.video_pipeline._resolver_budget import BudgetTracker
from app.content.video_pipeline._resolver_indexing import _analyze_and_index

logger = logging.getLogger(__name__)


async def _try_local(
    segment: SceneSegment,
    used_clips: Optional[set] = None,
) -> Optional[ResolvedAsset]:
    """Search the video asset library (RAG-backed semantic search)."""
    try:
        from app.content.video_pipeline.asset_library import search_library

        query = (
            f"{' '.join(segment.search_keywords)}. "
            f"{segment.visual_description}"
        )
        # Request extra results so we can skip already-used clips.
        # Similarity threshold 0.80 — only return genuinely relevant
        # clips. Below 0.80, fall through to Pexels for a fresh search.
        results = search_library(query, max_results=5, min_similarity=0.80)

        # Filter out clips already used in this video
        for candidate in (results or []):
            fpath = candidate.get("file_path", "")
            if not fpath or not os.path.exists(fpath):
                continue
            if used_clips and os.path.abspath(fpath) in used_clips:
                logger.info(
                    f"[asset_resolver] Skipping reused clip: "
                    f"{os.path.basename(fpath)} for {segment.segment_id}"
                )
                continue
            # Found a valid, non-reused clip
            best = candidate
            break
        else:
            # All candidates were used or missing
            best = None

        if best:
            fpath = best.get("file_path", "")

            if fpath and os.path.exists(fpath):
                # Pull the rich description if available
                analysis = best.get("analysis", {})
                desc = analysis.get("description", "")
                quality = analysis.get("quality_score", 0)
                quality_str = f", quality={quality}/10" if quality else ""

                logger.info(
                    f"[asset_resolver] LIBRARY HIT for "
                    f"{segment.segment_id}: "
                    f"{os.path.basename(fpath)} "
                    f"(similarity={best.get('similarity', 0):.2f}"
                    f"{quality_str})"
                )
                return ResolvedAsset(
                    segment_id=segment.segment_id,
                    source=AssetSource.LOCAL_INDEX,
                    tier=AssetTier.LOCAL,
                    file_path=fpath,
                    duration_s=best.get("duration_s", 0),
                    cost_usd=0.0,
                    confidence_score=best.get("similarity", 0.7),
                    metadata={
                        "library_match": True,
                        "original_source": best.get("source", ""),
                        "similarity": best.get("similarity", 0),
                        "clip_description": desc or best.get("visual_description", ""),
                    },
                )
            else:
                logger.debug(
                    f"[asset_resolver] Library match file missing: {fpath}"
                )
    except Exception as e:
        logger.debug(f"[asset_resolver] Library search failed: {e}")

    return None


async def _try_pexels(
    segment: SceneSegment,
    used_clips: Optional[set] = None,
    style_hint: str = "",
) -> Optional[ResolvedAsset]:
    """Search Pexels for free stock footage."""
    try:
        from app.content.video_pipeline.pexels_client import (
            search_and_download,
        )

        # Use clips[0].search_query as the primary search if available.
        search_queries = []
        if segment.clips:
            for clip in segment.clips:
                if clip.search_query:
                    search_queries.append(clip.search_query)
        if not search_queries:
            search_queries = segment.search_keywords[:3]

        results = None
        for query in search_queries:
            short_q = " ".join(query.split()[:5])
            # Append style hint to search if available
            if style_hint:
                short_q = f"{short_q} {style_hint}"
                short_q = " ".join(short_q.split()[:6])  # Cap at 6 words
            results = await search_and_download(short_q, max_results=3)
            if results:
                # Find first result not already used
                for r in results:
                    fp = r.get("file_path", "")
                    if not fp:
                        continue
                    if used_clips and os.path.abspath(fp) in used_clips:
                        continue
                    results = [r]
                    break
                else:
                    results = None
                    continue
                break
        if not results:
            results = []

        if results and results[0].get("file_path"):
            best = results[0]
            _fp = os.path.abspath(best["file_path"])
            if used_clips and _fp in used_clips:
                logger.info(
                    f"[asset_resolver] Pexels clip already used, "
                    f"skipping for {segment.segment_id}"
                )
                return None

            logger.info(
                f"[asset_resolver] Pexels match for {segment.segment_id}: "
                f"id={best['id']}"
            )
            meta = {"pexels_id": best["id"]}

            # Analyze and index the downloaded clip (non-blocking)
            await _analyze_and_index(
                file_path=best["file_path"],
                source="pexels",
                segment=segment,
                duration_s=best.get("duration", 0),
            )

            asset = ResolvedAsset(
                segment_id=segment.segment_id,
                source=AssetSource.PEXELS,
                tier=AssetTier.FREE_STOCK,
                file_path=best["file_path"],
                url=best.get("url", ""),
                duration_s=best.get("duration", 0),
                cost_usd=0.0,
                confidence_score=0.6,
                metadata=meta,
            )

            # Download clip B if the director specified two clips
            if len(segment.clips) >= 2 and segment.clips[1].search_query:
                try:
                    clip_b_q = " ".join(
                        segment.clips[1].search_query.split()[:5]
                    )
                    b_results = await search_and_download(
                        clip_b_q, max_results=3,
                    )
                    for b_res in (b_results or []):
                        b_fp = b_res.get("file_path", "")
                        if not b_fp:
                            continue
                        b_abs = os.path.abspath(b_fp)
                        if used_clips and b_abs in used_clips:
                            continue
                        asset.metadata["clip_b_path"] = b_fp
                        if used_clips is not None:
                            used_clips.add(b_abs)

                        # Analyze clip B too
                        await _analyze_and_index(
                            file_path=b_fp,
                            source="pexels",
                            segment=segment,
                            duration_s=b_res.get("duration", 0),
                        )

                        logger.info(
                            f"[asset_resolver] Clip B for "
                            f"{segment.segment_id}: '{clip_b_q}'"
                        )
                        break
                except Exception:
                    pass  # Non-critical, filler will be used

            return asset
    except Exception as e:
        logger.warning(f"[asset_resolver] Pexels search failed: {e}")

    return None


async def _try_pixabay(
    segment: SceneSegment,
    used_clips: Optional[set] = None,
) -> Optional[ResolvedAsset]:
    """Search Pixabay for free stock footage (second free tier)."""
    try:
        from app.content.video_pipeline.pixabay_client import (
            search_and_download,
        )

        search_queries = []
        if segment.clips:
            for clip in segment.clips:
                if clip.search_query:
                    search_queries.append(clip.search_query)
        if not search_queries:
            search_queries = segment.search_keywords[:3]

        results = None
        for query in search_queries:
            short_q = " ".join(query.split()[:5])
            results = await search_and_download(short_q, max_results=3)
            if results:
                for r in results:
                    fp = r.get("file_path", "")
                    if not fp:
                        continue
                    if used_clips and os.path.abspath(fp) in used_clips:
                        continue
                    results = [r]
                    break
                else:
                    results = None
                    continue
                break
        if not results:
            results = []

        if results and results[0].get("file_path"):
            best = results[0]
            _fp = os.path.abspath(best["file_path"])
            if used_clips and _fp in used_clips:
                return None

            logger.info(
                f"[asset_resolver] Pixabay match for {segment.segment_id}: "
                f"id={best['id']}"
            )
            meta = {"pixabay_id": best["id"]}

            await _analyze_and_index(
                file_path=best["file_path"],
                source="pixabay",
                segment=segment,
                duration_s=best.get("duration", 0),
            )

            asset = ResolvedAsset(
                segment_id=segment.segment_id,
                source=AssetSource.PIXABAY,
                tier=AssetTier.FREE_STOCK,
                file_path=best["file_path"],
                url=best.get("url", ""),
                duration_s=best.get("duration", 0),
                cost_usd=0.0,
                confidence_score=0.6,
                metadata=meta,
            )

            if len(segment.clips) >= 2 and segment.clips[1].search_query:
                try:
                    clip_b_q = " ".join(
                        segment.clips[1].search_query.split()[:5]
                    )
                    b_results = await search_and_download(
                        clip_b_q, max_results=3,
                    )
                    for b_res in (b_results or []):
                        b_fp = b_res.get("file_path", "")
                        if not b_fp:
                            continue
                        b_abs = os.path.abspath(b_fp)
                        if used_clips and b_abs in used_clips:
                            continue
                        asset.metadata["clip_b_path"] = b_fp
                        if used_clips is not None:
                            used_clips.add(b_abs)
                        await _analyze_and_index(
                            file_path=b_fp,
                            source="pixabay",
                            segment=segment,
                            duration_s=b_res.get("duration", 0),
                        )
                        break
                except Exception:
                    pass

            return asset
    except Exception as e:
        logger.warning(f"[asset_resolver] Pixabay search failed: {e}")

    return None


async def _try_fal_ai(
    segment: SceneSegment,
    budget: BudgetTracker,
) -> Optional[ResolvedAsset]:
    """Generate footage via fal.ai."""
    try:
        from app.content.video_pipeline.fal_client import (
            generate_video, download_generated_video,
        )

        # Build generation prompt from the director's shot description.
        if segment.clips and segment.clips[0].shot_description:
            prompt = segment.clips[0].shot_description
        else:
            prompt = segment.visual_description
        if segment.mood_tags:
            prompt += f". Mood: {', '.join(segment.mood_tags)}"

        duration_s = min(int(segment.estimated_duration_s), 8)
        duration_str = f"{duration_s}s"

        # Check budget
        from app.content.video_pipeline.fal_client import _get_model_config
        config = _get_model_config()
        estimated_cost = duration_s * config["cost_per_second"]

        if not budget.can_spend(estimated_cost):
            logger.info(
                f"[asset_resolver] fal.ai skipped for "
                f"{segment.segment_id}: budget insufficient"
            )
            return None

        result = await generate_video(
            prompt=prompt,
            duration=duration_str,
        )

        video_url = result.get("video_url", "")
        if not video_url:
            return None

        file_path = await download_generated_video(
            video_url, segment.segment_id,
        )

        actual_cost = result.get("cost_usd", estimated_cost)
        budget.record("fal_ai", segment.segment_id, actual_cost)

        asset = ResolvedAsset(
            segment_id=segment.segment_id,
            source=AssetSource.FAL_AI,
            tier=AssetTier.AI_GENERATED,
            file_path=file_path or "",
            url=video_url,
            duration_s=result.get("duration_s", duration_s),
            cost_usd=actual_cost,
            confidence_score=0.7,
            metadata={
                "model": result.get("model", ""),
                "prompt": prompt[:200],
            },
        )

        # Analyze and index AI-generated content
        if file_path:
            await _analyze_and_index(
                file_path=file_path,
                source="fal_ai",
                segment=segment,
                duration_s=result.get("duration_s", duration_s),
                cost_usd=actual_cost,
            )

        return asset
    except Exception as e:
        logger.warning(f"[asset_resolver] fal.ai generation failed: {e}")

    return None


async def _try_heygen(
    segment: SceneSegment,
    budget: BudgetTracker,
) -> Optional[ResolvedAsset]:
    """Generate avatar clip via HeyGen, with disk caching."""
    import hashlib
    from pathlib import Path

    text_hash = hashlib.md5(
        segment.script_text.strip().encode()
    ).hexdigest()[:10]

    cache_dir = Path("data/content/video_pipeline/downloads/heygen")
    pattern = f"heygen_{segment.segment_id}_{text_hash}_*.mp4"
    cached = sorted(cache_dir.glob(pattern)) if cache_dir.exists() else []

    if cached:
        cached_path = str(cached[-1])
        logger.info(
            f"[asset_resolver] HeyGen CACHED for {segment.segment_id}: "
            f"{cached_path} (text hash match, skipping API call)"
        )
        return ResolvedAsset(
            segment_id=segment.segment_id,
            source=AssetSource.HEYGEN,
            tier=AssetTier.AVATAR,
            file_path=cached_path,
            url="",
            duration_s=0,
            cost_usd=0.0,
            confidence_score=0.95,
            metadata={"cached": True, "text_hash": text_hash},
        )

    # No cache hit — generate fresh
    try:
        from app.content.video_pipeline.heygen_client import (
            generate_and_download,
        )

        result = await generate_and_download(
            text=segment.script_text,
            segment_id=segment.segment_id,
        )

        cost = result.get("cost_usd", 0)
        budget.record("heygen", segment.segment_id, cost)

        return ResolvedAsset(
            segment_id=segment.segment_id,
            source=AssetSource.HEYGEN,
            tier=AssetTier.AVATAR,
            file_path=result.get("file_path", ""),
            url=result.get("video_url", ""),
            duration_s=result.get("duration_s", 0),
            cost_usd=cost,
            confidence_score=0.95,
            metadata={"heygen_video_id": result.get("video_id", "")},
        )
    except Exception as e:
        logger.warning(f"[asset_resolver] HeyGen generation failed: {e}")

    return None
