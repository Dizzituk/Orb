# FILE: app/content/video_pipeline/fal_client.py
"""
fal.ai unified API client for AI video generation.

Supports multiple models through fal.ai's unified endpoint:
- Veo 3.1 (latest, highest quality, 4K support)
- Veo 3.1 Fast (budget, speed-optimised)
- Kling 3 Pro (motion quality, character consistency)
- Wan 2.2 (cheapest option)

Model selection via .env: FAL_VIDEO_MODEL (default: veo31_fast)
API key from encrypted settings: FAL_API_KEY

Model cascade on failure: veo31_fast → veo3_fast → wan22
"""
import os
import time
import logging
import httpx
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

FAL_QUEUE_URL = "https://queue.fal.run"
DOWNLOAD_DIR = Path("data/content/video_pipeline/downloads/fal_ai")

# Model presets with cost per second (no audio, 720p)
MODEL_PRESETS = {
    "veo31_fast": {
        "endpoint": "fal-ai/veo3.1/fast",
        "cost_per_second": 0.10,
        "max_duration": "8s",
        "description": "Google Veo 3.1 Fast — best value",
    },
    "veo31": {
        "endpoint": "fal-ai/veo3.1",
        "cost_per_second": 0.20,
        "max_duration": "8s",
        "description": "Google Veo 3.1 Standard — highest quality",
    },
    "veo3_fast": {
        "endpoint": "fal-ai/veo3/fast",
        "cost_per_second": 0.10,
        "max_duration": "8s",
        "description": "Google Veo 3 Fast — legacy fast",
    },
    "veo3": {
        "endpoint": "fal-ai/veo3",
        "cost_per_second": 0.20,
        "max_duration": "8s",
        "description": "Google Veo 3 Standard — legacy",
    },
    "kling3_pro": {
        "endpoint": "fal-ai/kling-video/v3/pro/text-to-video",
        "cost_per_second": 0.224,
        "max_duration": "10s",
        "description": "Kling 3 Pro — best motion quality",
    },
    "wan22": {
        "endpoint": "fal-ai/wan/v2.2/text-to-video",
        "cost_per_second": 0.10,
        "max_duration": "5s",
        "description": "Wan 2.2 — cheapest option",
    },
}

# Fallback order if the primary model 404s
FALLBACK_ORDER = ["veo31_fast", "veo3_fast", "wan22"]

DEFAULT_MODEL = "veo31_fast"


def _get_api_key() -> str:
    """Get fal.ai API key from environment."""
    key = os.getenv("FAL_API_KEY", "")
    if not key:
        raise ValueError(
            "FAL_API_KEY not set. Add it in Settings > API Keys."
        )
    return key


def _get_model_config() -> Dict[str, Any]:
    """Get configured model or default."""
    model_name = os.getenv("FAL_VIDEO_MODEL", DEFAULT_MODEL)
    if model_name in MODEL_PRESETS:
        return MODEL_PRESETS[model_name]
    # Treat as raw endpoint
    return {
        "endpoint": model_name,
        "cost_per_second": 0.25,
        "max_duration": "8s",
        "description": f"Custom: {model_name}",
    }


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Key {_get_api_key()}",
        "Content-Type": "application/json",
    }


async def _submit_to_model(
    endpoint: str,
    payload: dict,
) -> Optional[Dict[str, Any]]:
    """Try submitting to a specific fal.ai model endpoint.

    Uses queue.fal.run/{endpoint} with payload wrapped in "input".
    Returns the queue response dict, or None if the endpoint 404s.
    Raises on other errors (auth, server, etc).
    """
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{FAL_QUEUE_URL}/{endpoint}",
            headers=_headers(),
            json={"input": payload},
        )
        if resp.status_code == 404:
            logger.warning(
                f"[fal.ai] Endpoint not found: {endpoint} (404)"
            )
            return None
        resp.raise_for_status()
        return resp.json()


async def generate_video(
    prompt: str,
    duration: str = "5s",
    aspect_ratio: str = "16:9",
    resolution: str = "720p",
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a video via fal.ai with automatic model fallback.

    Tries the configured model first. If it 404s, cascades through
    fallback models until one works.

    Returns dict with: video_url, duration_s, cost_usd, model
    """
    config = MODEL_PRESETS.get(model_override, _get_model_config())
    endpoint = config["endpoint"]

    payload = {
        "prompt": prompt,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
    }

    logger.info(
        f"[fal.ai] Generating video: model={endpoint}, "
        f"duration={duration}, prompt='{prompt[:60]}...'"
    )

    # Try primary model, then fallbacks on 404
    queue_data = await _submit_to_model(endpoint, payload)

    if queue_data is None:
        # Primary model 404d — try fallbacks
        for fallback_name in FALLBACK_ORDER:
            fb_config = MODEL_PRESETS.get(fallback_name)
            if not fb_config or fb_config["endpoint"] == endpoint:
                continue

            logger.info(
                f"[fal.ai] Trying fallback: {fb_config['endpoint']}"
            )
            queue_data = await _submit_to_model(
                fb_config["endpoint"], payload,
            )
            if queue_data is not None:
                endpoint = fb_config["endpoint"]
                config = fb_config
                logger.info(
                    f"[fal.ai] Fallback succeeded: {endpoint}"
                )
                break

    if queue_data is None:
        raise RuntimeError(
            "[fal.ai] All model endpoints returned 404. "
            "Check fal.ai status or update model presets."
        )

    request_id = queue_data.get("request_id", "")
    if not request_id:
        raise RuntimeError("[fal.ai] No request_id in queue response")

    # Use response_url from the submit response if available,
    # otherwise construct it. The response_url is the canonical
    # way to fetch the result from fal.ai's queue API.
    response_url = queue_data.get(
        "response_url",
        f"{FAL_QUEUE_URL}/{endpoint}/requests/{request_id}",
    )
    # Status URL is the response_url + /status
    status_url = response_url + "/status"

    logger.info(
        f"[fal.ai] Queued: request_id={request_id}, "
        f"status_url={status_url}"
    )

    # Poll for completion
    max_wait = 1800
    poll_interval = 8
    elapsed = 0

    async with httpx.AsyncClient(timeout=15) as client:
        while elapsed < max_wait:
            resp = await client.get(
                status_url,
                headers=_headers(),
            )
            resp.raise_for_status()
            status_data = resp.json()

            status = status_data.get("status", "")
            if status == "COMPLETED":
                break
            elif status in ("FAILED", "CANCELLED"):
                error = status_data.get("error", "Unknown error")
                raise RuntimeError(
                    f"[fal.ai] Generation failed: {error}"
                )

            time.sleep(poll_interval)
            elapsed += poll_interval

    if elapsed >= max_wait:
        raise TimeoutError(
            f"[fal.ai] Generation timed out after {max_wait}s"
        )

    # Fetch result from response_url
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            response_url,
            headers=_headers(),
        )
        resp.raise_for_status()
        result = resp.json()

    video_info = result.get("video", {})
    video_url = video_info.get("url", "")

    dur_seconds = int(duration.replace("s", ""))
    cost = dur_seconds * config["cost_per_second"]

    logger.info(
        f"[fal.ai] Generated: {video_url[:80]}... "
        f"(model={endpoint}, est. cost: ${cost:.2f})"
    )

    return {
        "video_url": video_url,
        "duration_s": dur_seconds,
        "cost_usd": cost,
        "model": endpoint,
        "request_id": request_id,
    }


async def download_generated_video(
    video_url: str,
    segment_id: str,
) -> Optional[str]:
    """Download a generated video to local cache."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"fal_{segment_id}_{int(time.time())}.mp4"
    filepath = DOWNLOAD_DIR / filename

    async with httpx.AsyncClient(
        timeout=120, follow_redirects=True,
    ) as client:
        resp = await client.get(video_url)
        resp.raise_for_status()
        filepath.write_bytes(resp.content)

    logger.info(
        f"[fal.ai] Downloaded: {filepath} "
        f"({len(resp.content) / 1024 / 1024:.1f} MB)"
    )
    return str(filepath)
