# FILE: app/debug/executors/social_actions.py
# Purpose: Chat tool executor for social media API actions.
# Called-by: app.debug.executors
# Depends-on: app.social.meta_post
# Last-renovated: 2026-06-11
"""
Chat tool executor for social media API actions.

Currently exposes:
    meta_post — post an image + caption to a Facebook Page via Graph API.

This replaces the brittle browser-driven Meta upload flow with a
deterministic single-API-call path. The browser flow stays in place for
agentic browse-the-web work (engagement reading, comment drafting,
content discovery) — this executor is just the deterministic publish
path that should be preferred whenever an API equivalent exists.

Returns user-facing strings (chat LLM reads these directly), so error
messages are written as helpful diagnostics, not stack traces.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from app.social.meta_post import post_to_meta

logger = logging.getLogger(__name__)


async def execute_meta_post(params: Dict[str, Any]) -> str:
    """
    Tool handler for `meta_post`.

    Params:
        image_path   (required) : absolute path to image on disk
        caption      (required) : post caption (empty string allowed)
        target       (optional) : 'facebook' (default)
        scheduled_at (optional) : Unix timestamp for scheduled publish
        verify       (optional) : bool, default true
    """
    p = params or {}
    image_path = p.get("image_path")
    caption = p.get("caption")
    target = p.get("target", "facebook")
    scheduled_at = p.get("scheduled_at")
    verify = p.get("verify", True)

    if not image_path:
        return "Error: 'image_path' is required."
    if caption is None:
        return "Error: 'caption' is required (empty string is allowed)."

    if scheduled_at is not None:
        try:
            scheduled_at = int(scheduled_at)
        except (TypeError, ValueError):
            return (
                f"Error: 'scheduled_at' must be a Unix timestamp integer, "
                f"got {scheduled_at!r}."
            )

    result = await post_to_meta(
        image_path=image_path,
        caption=caption,
        target=target,
        scheduled_at=scheduled_at,
        verify=bool(verify),
    )

    return _format_result(result)


def _format_result(result: Dict[str, Any]) -> str:
    """Render the post_to_meta result as a string for the chat LLM."""
    if not result.get("ok"):
        kind = result.get("error_kind") or "error"
        return (
            f"meta_post FAILED ({kind}): {result.get('error')}\n"
            f"target={result.get('target')}"
        )

    lines = [
        f"meta_post OK target={result['target']} post_id={result['post_id']}",
    ]
    if result.get("scheduled"):
        lines.append(f"scheduled_at={result.get('scheduled_at')} (Unix)")
        lines.append("Permalink will be available after the publish time.")
    else:
        permalink = result.get("permalink_url")
        if permalink:
            lines.append(f"permalink={permalink}")
        verified = result.get("verified")
        lines.append(
            "cross-channel verify: "
            + ("CONFIRMED" if verified else "unverified (post may still be propagating)")
        )
    return "\n".join(lines)
