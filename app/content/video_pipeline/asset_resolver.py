# FILE: app/content/video_pipeline/asset_resolver.py
# Purpose: Tiered Asset Resolver — cascading footage sourcing for each segment.
# Called-by: app.content.video_pipeline.orchestrator
# Depends-on: app.content.video_pipeline.asset_library, app.content.video_pipeline.clip_analyzer, app.content.video_pipeline.fal_client, app.content.video_pipeline.heygen_client (+3 more)
# Last-renovated: 2026-06-11
"""
Tiered Asset Resolver — cascading footage sourcing for each segment.

For each segment in the scene plan, cascades through tiers:
  Tier 0: Local stock (indexed by scanner)
  Tier 1: Free stock (Pexels API)
  Tier 2: Paid stock (Shutterstock API)
  Tier 3: AI generated (fal.ai)
  Tier A: Avatar (HeyGen) for requires_avatar segments

Enhanced: downloads now get analyzed by Gemini (clip_analyzer) for
rich visual descriptions before being indexed in the asset library.
"""
import os
import logging
from typing import Optional, Callable, Awaitable

from app.content.video_pipeline.models import (
    ScenePlan, SceneSegment, ResolvedAsset, ResolvedPlan,
    AssetSource, AssetTier,
)

logger = logging.getLogger(__name__)

# Asset clusters extracted to leaf modules (split 2026-06-20). Imported back
# so resolve_assets / _resolve_segment below + the orchestrator imports are
# unchanged.
from app.content.video_pipeline._resolver_budget import BudgetTracker
from app.content.video_pipeline._resolver_indexing import _analyze_and_index  # noqa: F401
from app.content.video_pipeline._resolver_fetchers import (
    _try_local,
    _try_pexels,
    _try_pixabay,
    _try_fal_ai,
    _try_heygen,
)


async def resolve_assets(
    scene_plan: ScenePlan,
    max_budget_usd: float = 10.0,
    progress_callback: Optional[Callable] = None,
    force_ai_segments: Optional[set] = None,
    style_profile: Optional["StyleProfile"] = None,
) -> ResolvedPlan:
    """
    Resolve assets for every segment in the scene plan.

    Cascades through tiers for each segment until footage is found.
    Respects budget ceiling for paid tiers.

    force_ai_segments: set of segment IDs that should skip library/Pexels
    and go straight to fal.ai generation.
    style_profile: Visual style guide from reference videos.
    """
    budget = BudgetTracker(max_budget_usd)
    assets = []
    unresolved = []
    used_clips: set = set()
    force_ai = force_ai_segments or set()

    # Build style hint for search queries
    style_hint = ""
    if style_profile and style_profile.visual_mood_keywords:
        style_hint = " ".join(style_profile.visual_mood_keywords[:3])
        logger.info(f"[asset_resolver] Style hint: {style_hint}")
    total = len(scene_plan.segments)
    for idx, segment in enumerate(scene_plan.segments):
        if progress_callback:
            await progress_callback(
                f"Resolving segment {idx + 1}/{total}: {segment.segment_id}",
                (idx / total) * 100,
            )

        # If this segment is marked for forced AI generation,
        # skip library and Pexels entirely.
        if segment.segment_id in force_ai and not segment.requires_avatar:
            logger.info(
                f"[asset_resolver] FORCE AI for {segment.segment_id}: "
                f"verification/QA flagged, stock footage inadequate"
            )
            asset = await _try_fal_ai(segment, budget)
            if asset:
                if asset.file_path:
                    used_clips.add(os.path.abspath(asset.file_path))
                assets.append(asset)
                continue
            # If fal.ai fails (budget etc), fall through to normal cascade
            logger.warning(
                f"[asset_resolver] fal.ai failed for {segment.segment_id}, "
                f"falling back to normal cascade"
            )

        asset = await _resolve_segment(segment, budget, used_clips, style_hint=style_hint)
        if asset and asset.file_path:
            used_clips.add(os.path.abspath(asset.file_path))

        if asset:
            assets.append(asset)
        else:
            unresolved.append(segment.segment_id)
            logger.warning(
                f"[asset_resolver] UNRESOLVED: {segment.segment_id} "
                f"({segment.segment_type})"
            )

    plan = ResolvedPlan(
        assets=assets,
        total_cost_usd=budget.spent,
        unresolved_segments=unresolved,
    )

    logger.info(
        f"[asset_resolver] Resolved {len(assets)}/{total} segments. "
        f"Total cost: ${budget.spent:.2f}. "
        f"Unresolved: {len(unresolved)}"
    )
    return plan


async def _resolve_segment(
    segment: SceneSegment,
    budget: BudgetTracker,
    used_clips: Optional[set] = None,
    style_hint: str = "",
) -> Optional[ResolvedAsset]:
    """Resolve a single segment through the tier cascade."""

    # Avatar segments: generate the avatar clip via HeyGen.
    if segment.requires_avatar:
        return await _try_heygen(segment, budget)

    # Tier 0: Local stock (skip clips already used in this video)
    result = await _try_local(segment, used_clips)
    if result:
        return result

    # Tier 1: Free stock — Pexels
    result = await _try_pexels(segment, used_clips, style_hint=style_hint)
    if result:
        return result

    # Tier 1b: Free stock — Pixabay (different library, more variety)
    result = await _try_pixabay(segment, used_clips)
    if result:
        return result

    # Tier 2: AI generated (fal.ai) — budget gated
    if budget.remaining > 0:
        result = await _try_fal_ai(segment, budget)
        if result:
            return result

    return None
