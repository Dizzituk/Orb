# FILE: app/social/meta_post.py
# Purpose: High-level Meta posting orchestration.
# Called-by: app.debug.executors.social_actions
# Depends-on: app.social.meta_client
# Last-renovated: 2026-06-11
"""
High-level Meta posting orchestration.

Wraps meta_client with input validation, cross-channel verification, and
structured error handling. This is the single entry point chat tools and
other ASTRA components should use for "post this to Meta" actions.

Cross-channel verification rationale:
    Upload     uses POST multipart  -> success means "Meta accepted bytes"
    Verify     uses GET on post obj -> success means "post is in the graph"
  These are independent paths through Meta's infrastructure. A
  single-channel "did it succeed?" can lie (cached responses, partial
  failures, async indexing). Two-channel agreement is the stronger signal.

Currently supports Facebook Page only. Instagram support requires a public
image URL (Meta won't accept multipart for IG container creation), which
needs a separate hosting solution (OneDrive share / S3 / tunnel) — that's
a deliberate Phase-2 task, not an oversight.

Never raises: failures are returned in the result dict so the chat tool
layer doesn't need defensive try/except wrapping.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from app.social.meta_client import (
    MetaApiError,
    get_page_id,
    get_token,
    post_photo_to_page,
    verify_post_exists,
)

logger = logging.getLogger(__name__)

VALID_TARGETS = ("facebook",)  # Instagram pending — needs URL hosting decision


async def post_to_meta(
    image_path: str,
    caption: str,
    *,
    target: str = "facebook",
    scheduled_at: Optional[int] = None,
    verify: bool = True,
) -> Dict[str, Any]:
    """
    Post an image with caption to Meta.

    Args:
        image_path   : Absolute path to image file on disk.
        caption      : Post caption text. Empty string allowed.
        target       : 'facebook' (default). Instagram pending hosting.
        scheduled_at : Unix timestamp for scheduled publish, or None for now.
        verify       : If True (default), performs a GET-channel verification
                       after upload. Skipped automatically for scheduled posts
                       (object isn't queryable until publish time).

    Returns dict with shape:
        ok            : bool — overall success
        target        : str
        post_id       : str | None
        permalink_url : str | None
        scheduled     : bool
        scheduled_at  : int | None
        verified      : bool — cross-channel verification result
        error         : str | None — populated when ok=False
        error_kind    : str | None — 'config' | 'input' | 'api' | 'verify'
    """
    target = (target or "facebook").lower().strip()
    if target not in VALID_TARGETS:
        return _failure(
            target,
            f"Target '{target}' not supported yet. Valid: {VALID_TARGETS}. "
            f"Instagram support pending public image hosting.",
            kind="input",
        )

    # Pre-flight: credentials
    if not get_token():
        return _failure(
            target,
            "Meta access token not configured. Add 'meta_access_token' in "
            "Settings -> API Keys (long-lived User or Page Access Token "
            "with pages_manage_posts scope).",
            kind="config",
        )
    if not get_page_id():
        return _failure(
            target,
            "Facebook Page ID not configured. Add 'facebook_page_id' in "
            "Settings -> API Keys (find it in your Page -> About).",
            kind="config",
        )

    # Pre-flight: image
    if not Path(image_path).is_file():
        return _failure(
            target,
            f"Image not found at: {image_path}",
            kind="input",
        )

    # Upload
    try:
        result = await post_photo_to_page(
            image_path=image_path,
            caption=caption,
            scheduled_at=scheduled_at,
        )
    except FileNotFoundError as exc:
        return _failure(target, str(exc), kind="input")
    except ValueError as exc:
        return _failure(target, str(exc), kind="input")
    except MetaApiError as exc:
        return _failure(
            target,
            f"Graph API error: {exc} "
            f"(code={exc.graph_code}, subcode={exc.graph_subcode})",
            kind="api",
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception("[meta_post] Unexpected error during upload")
        return _failure(target, f"Unexpected error: {exc}", kind="api")

    post_id = result.get("post_id")
    scheduled = bool(result.get("scheduled"))

    # Cross-channel verify (skipped for scheduled posts — verify would
    # always return False since the post isn't yet in the graph)
    verified = False
    if verify and post_id and not scheduled:
        try:
            verified = await verify_post_exists(post_id)
        except Exception as exc:  # pragma: no cover
            logger.warning("[meta_post] Verify call failed: %s", exc)
            verified = False

    return {
        "ok": True,
        "target": target,
        "post_id": post_id,
        "permalink_url": result.get("permalink_url"),
        "scheduled": scheduled,
        "scheduled_at": result.get("scheduled_at"),
        "verified": verified,
        "error": None,
        "error_kind": None,
    }


def _failure(target: str, message: str, *, kind: str) -> Dict[str, Any]:
    """Uniform failure shape so callers can introspect without parsing strings."""
    return {
        "ok": False,
        "target": target,
        "post_id": None,
        "permalink_url": None,
        "scheduled": False,
        "scheduled_at": None,
        "verified": False,
        "error": message,
        "error_kind": kind,
    }
