# FILE: app/content/distribution/youtube_router.py
"""
YouTube-specific API endpoints.

Handles OAuth flow and channel-level analytics that
go beyond the generic distribution_router endpoints.

Endpoints:
- /content/youtube/auth/status  — Check auth status
- /content/youtube/auth/start   — Start OAuth flow
- /content/youtube/auth/revoke  — Revoke access
- /content/youtube/channel      — Channel info + stats
- /content/youtube/videos       — Recent videos with stats
- /content/youtube/analytics    — Channel analytics (watch time, etc.)
- /content/youtube/top-videos   — Top videos by performance
"""
import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/content/youtube",
    tags=["YouTube"],
    dependencies=[Depends(require_auth)],
)


# ═══════════════════════════════════════════════════
# OAUTH FLOW
# ═══════════════════════════════════════════════════

@router.get("/auth/status", response_model=dict)
def youtube_auth_status():
    """Check YouTube OAuth authentication status."""
    from app.content.distribution.youtube_auth import check_auth_status
    return check_auth_status()


@router.post("/auth/start", response_model=dict)
def youtube_auth_start():
    """Start YouTube OAuth flow.

    Opens a browser window for the user to sign in to Google
    and grant YouTube access. Frontend should poll /auth/status.
    """
    from app.content.distribution.youtube_auth import start_auth_flow
    return start_auth_flow()


@router.post("/auth/revoke", response_model=dict)
def youtube_auth_revoke():
    """Revoke YouTube OAuth access."""
    from app.content.distribution.youtube_auth import revoke_auth
    return revoke_auth()


# ═══════════════════════════════════════════════════
# CHANNEL DATA
# ═══════════════════════════════════════════════════

@router.get("/channel", response_model=dict)
async def youtube_channel_info(
    channel_id: Optional[str] = Query(
        None,
        description="Channel ID. Omit to get your own channel.",
    ),
):
    """Get channel info: name, subscribers, total views, etc."""
    from app.content.distribution.youtube_channel import (
        get_channel_info,
    )

    info = await get_channel_info(channel_id)
    if not info:
        return {
            "error": "not_available",
            "message": (
                "Could not fetch channel info. "
                "Check OAuth status or provide a channel ID."
            ),
        }
    return info


@router.get("/videos", response_model=dict)
async def youtube_recent_videos(
    channel_id: Optional[str] = Query(None),
    max_results: int = Query(10, ge=1, le=50),
):
    """Get recent videos with statistics."""
    from app.content.distribution.youtube_channel import (
        get_recent_videos,
    )

    videos = await get_recent_videos(channel_id, max_results)
    return {"count": len(videos), "videos": videos}


@router.get("/analytics", response_model=dict)
async def youtube_channel_analytics(
    days: int = Query(30, ge=1, le=365),
):
    """Get channel-wide analytics (views, watch time, subscribers).

    Requires OAuth with yt-analytics.readonly scope.
    """
    from app.content.distribution.youtube_channel import (
        get_channel_analytics,
    )

    data = await get_channel_analytics(days)
    if not data:
        return {
            "error": "not_available",
            "message": (
                "YouTube Analytics not available. "
                "Complete OAuth first."
            ),
        }
    return data


@router.get("/top-videos", response_model=dict)
async def youtube_top_videos(
    days: int = Query(30, ge=1, le=365),
    max_results: int = Query(10, ge=1, le=50),
):
    """Get top performing videos by views."""
    from app.content.distribution.youtube_channel import (
        get_top_videos_analytics,
    )

    data = await get_top_videos_analytics(days, max_results)
    if not data:
        return {
            "error": "not_available",
            "message": "YouTube Analytics not available.",
        }
    return {"count": len(data), "videos": data}


# ═══════════════════════════════════════════════════
# SYNC
# ═══════════════════════════════════════════════════

@router.post("/sync", response_model=dict)
async def youtube_sync_channel(
    max_results: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Sync existing YouTube videos into ASTRA's content database.

    Pulls videos from the authenticated channel and creates
    ContentPiece + ContentOutput records for tracking.
    """
    from app.content.distribution.youtube_sync import (
        sync_channel_videos,
    )

    return await sync_channel_videos(db, max_results)


# ═══════════════════════════════════════════════════
# MANUAL UPLOAD & QUEUE
# ═══════════════════════════════════════════════════

class ManualUploadRequest(BaseModel):
    """Request body for manual video upload/queue."""
    video_path: str
    title: str
    description: str = ""
    output_format: str = "youtube_longform"  # or "youtube_short"
    tags: Optional[List[str]] = None
    category_id: str = "28"
    privacy: str = "private"
    scheduled_time: Optional[str] = None  # ISO format, or "optimal"
    thumbnail_path: Optional[str] = None
    auto_optimise: bool = True


@router.post("/upload", response_model=dict)
async def youtube_manual_upload(
    body: ManualUploadRequest,
    db: Session = Depends(get_db),
):
    """Manually upload and queue a video to YouTube.

    Creates content records, optionally optimises metadata
    via LLM, schedules at the best time (or a specific time),
    and publishes.

    Set scheduled_time to:
    - null/omit: publish immediately
    - "optimal": let ASTRA pick the best time
    - ISO datetime: schedule for a specific time
    """
    import os
    from datetime import datetime as dt, timezone as tz

    from app.content.models import ContentPiece, ContentOutput
    from app.content.distribution.scheduler import find_next_slot

    # Validate video exists
    if not os.path.exists(body.video_path):
        return {
            "error": "file_not_found",
            "message": f"Video not found: {body.video_path}",
        }

    # Step 1: Optimise metadata if requested
    tags = body.tags or []
    title = body.title
    description = body.description
    category_id = body.category_id

    if body.auto_optimise:
        from app.content.distribution.youtube_optimiser import (
            optimise_metadata,
        )

        optimised = await optimise_metadata(
            title=title,
            description=description,
            existing_tags=tags,
        )
        tags = optimised.get("tags", tags)
        title = optimised.get("optimised_title", title)
        description = optimised.get(
            "optimised_description", description
        )
        category_id = optimised.get(
            "suggested_category_id", category_id
        )

    # Step 2: Determine schedule time
    scheduled_at = None
    if body.scheduled_time == "optimal":
        scheduled_at = find_next_slot(db, "youtube")
    elif body.scheduled_time:
        try:
            scheduled_at = dt.fromisoformat(body.scheduled_time)
            if scheduled_at.tzinfo is None:
                scheduled_at = scheduled_at.replace(tzinfo=tz.utc)
        except ValueError:
            return {
                "error": "invalid_datetime",
                "message": "scheduled_time must be ISO format or 'optimal'",
            }

    # Step 3: Detect if this is a Short
    is_short = body.output_format == "youtube_short"

    # Auto-detect from duration if not explicitly set
    if body.output_format == "youtube_longform":
        try:
            import subprocess, json as _json
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format=duration", "-of", "json", body.video_path],
                capture_output=True, text=True, timeout=10,
            )
            dur = float(_json.loads(probe.stdout)["format"]["duration"])
            if dur <= 60:
                is_short = True
        except Exception:
            pass

    # Add #Shorts to description for YouTube Shorts
    if is_short and "#Shorts" not in description:
        description = description.rstrip() + "\n\n#Shorts"

    # Create content records
    piece = ContentPiece(
        title=title,
        description=description,
        content_category="educational",
        status="approved",
        source_conversation_ids=[],
        source_tag_ids=[],
    )
    db.add(piece)
    db.flush()

    privacy = body.privacy
    if scheduled_at and privacy == "public":
        privacy = "private"  # Must be private for scheduled publish

    output = ContentOutput(
        piece_id=piece.id,
        output_format="youtube_short" if is_short else "youtube_longform",
        platform="youtube",
        primary_asset_path=body.video_path,
        thumbnail_path=body.thumbnail_path,
        caption_text=description,
        platform_metadata={
            "title": title,
            "tags": tags,
            "category_id": category_id,
            "privacy": privacy,
            "manual_upload": True,
        },
        scheduled_at=scheduled_at,
        publish_device="desktop",
    )
    db.add(output)
    db.commit()
    db.refresh(output)

    # Step 4: If no schedule time, publish immediately
    result_data = {
        "piece_id": piece.id,
        "output_id": output.id,
        "title": title,
        "tags": tags,
        "category_id": category_id,
        "auto_optimised": body.auto_optimise,
    }

    if scheduled_at:
        result_data["status"] = "scheduled"
        result_data["scheduled_at"] = scheduled_at.isoformat()
        result_data["message"] = (
            f"Video queued for {scheduled_at.strftime('%Y-%m-%d %H:%M UTC')}"
        )
    else:
        # Publish now
        from app.content.distribution.publisher import publish_output

        pub_result = await publish_output(db, output.id)
        result_data["status"] = pub_result.get("status", "unknown")
        result_data["publish_result"] = pub_result

    return result_data


# ═══════════════════════════════════════════════════
# TAG OPTIMISATION (standalone)
# ═══════════════════════════════════════════════════

class OptimiseRequest(BaseModel):
    """Request body for metadata optimisation."""
    title: str
    description: str = ""
    existing_tags: Optional[List[str]] = None
    category: str = "science_tech"


@router.post("/optimise", response_model=dict)
async def youtube_optimise_metadata(
    body: OptimiseRequest,
):
    """Run LLM-powered metadata optimisation without uploading.

    Useful for previewing tag suggestions before committing.
    """
    from app.content.distribution.youtube_optimiser import (
        optimise_metadata,
    )

    return await optimise_metadata(
        title=body.title,
        description=body.description,
        existing_tags=body.existing_tags,
        category=body.category,
    )


# ═══════════════════════════════════════════════════
# OPTIMISE EXISTING VIDEO
# ═══════════════════════════════════════════════════

@router.post("/videos/{video_id}/optimise", response_model=dict)
async def youtube_optimise_existing(
    video_id: str,
):
    """Optimise tags/metadata for an already-published video.

    Fetches current metadata, runs through the optimiser,
    then updates the video on YouTube.
    """
    from app.content.distribution.youtube_channel import (
        get_recent_videos,
    )
    from app.content.distribution.youtube_optimiser import (
        optimise_metadata,
    )

    # Fetch current video info
    headers = None
    try:
        from app.content.distribution.youtube_auth import (
            get_youtube_credentials,
        )
        import httpx

        creds = get_youtube_credentials()
        if not creds:
            return {"error": "not_authenticated"}

        async with httpx.AsyncClient(timeout=15) as client:
            # Get current metadata
            resp = await client.get(
                "https://www.googleapis.com/youtube/v3/videos",
                headers={"Authorization": f"Bearer {creds.token}"},
                params={
                    "part": "snippet,statistics",
                    "id": video_id,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", [])

            if not items:
                return {"error": "video_not_found"}

            snippet = items[0]["snippet"]
            current_title = snippet.get("title", "")
            current_desc = snippet.get("description", "")
            current_tags = snippet.get("tags", [])

            # Run optimisation
            optimised = await optimise_metadata(
                title=current_title,
                description=current_desc,
                existing_tags=current_tags,
            )

            # Update video on YouTube
            update_resp = await client.put(
                "https://www.googleapis.com/youtube/v3/videos",
                headers={
                    "Authorization": f"Bearer {creds.token}",
                    "Content-Type": "application/json",
                },
                params={"part": "snippet"},
                json={
                    "id": video_id,
                    "snippet": {
                        "title": optimised["optimised_title"],
                        "description": optimised[
                            "optimised_description"
                        ],
                        "tags": optimised["tags"],
                        "categoryId": optimised[
                            "suggested_category_id"
                        ],
                    },
                },
            )
            update_resp.raise_for_status()

            return {
                "video_id": video_id,
                "status": "optimised",
                "previous_tags": current_tags,
                "new_tags": optimised["tags"],
                "title_changed": (
                    current_title != optimised["optimised_title"]
                ),
                "optimised": optimised,
            }

    except Exception as e:
        logger.error(
            "[youtube] Optimise existing failed: %s", e
        )
        return {"error": str(e)}



# ═══════════════════════════════════════════════════
# ALGORITHM STRATEGY
# ═══════════════════════════════════════════════════

@router.get("/strategy", response_model=dict)
def youtube_strategy_summary():
    """Get the current YouTube algorithm optimisation strategy.

    Returns ranking signals, optimal posting times, and rules
    that ASTRA uses when optimising content for YouTube.
    """
    from app.content.distribution.algorithm_strategy import (
        get_strategy_summary,
    )
    return get_strategy_summary()


@router.post("/strategy/score-title", response_model=dict)
async def youtube_score_title(
    body: OptimiseRequest,
):
    """Score a title against algorithm best practices."""
    from app.content.distribution.algorithm_strategy import (
        score_title,
    )
    result = score_title(body.title)
    return result


@router.get("/strategy/next-slot", response_model=dict)
def youtube_next_posting_slot(
    content_type: str = Query("youtube_short"),
):
    """Get the next algorithmically optimal posting time."""
    from app.content.distribution.algorithm_strategy import (
        get_optimal_posting_time,
    )
    result = get_optimal_posting_time(content_type)
    result["scheduled_time"] = result["scheduled_time"].isoformat()
    return result

