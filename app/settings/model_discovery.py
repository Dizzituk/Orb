# FILE: app/settings/model_discovery.py
# Purpose: Provider model discovery — GET /settings/models/available pulls the
#          live model lists from Anthropic/OpenAI/Gemini + local (vLLM/ollama),
#          cached 24h in-memory with an on-disk fallback. LANE D Task 3.
# Called-by: app.settings.router (include_router; inherits require_auth),
#            orb-desktop ModelSwitcher + Models settings panel
# Depends-on: httpx, app.llm.frontier_models (audit CLI stays separate)
# Last-renovated: 2026-07-02
"""
Discovery is the ONLY way the model lists stay current: dateless Anthropic IDs
from the 4.6 generation are PINNED snapshots, not evergreen — new releases
arrive as NEW ids. API keys come from the encrypted settings_api_keys store,
which main.py syncs into os.environ at startup (and env_model_store re-syncs
after every reload), so os.getenv here IS the store's runtime surface.

Cache policy: 24h TTL in-memory; every successful fetch also lands in
data/model_discovery_cache.json. Provider outage -> serve the newest cached
list with stale=true (UI still renders). POST /models/available/refresh
forces a refetch. The `python -m app.llm.frontier_models` audit CLI is
untouched and keeps working.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter

logger = logging.getLogger(__name__)

discovery_router = APIRouter(tags=["Settings — Models"])

CACHE_TTL_S = 24 * 3600
_CACHE_FILE = Path(__file__).resolve().parents[2] / "data" / "model_discovery_cache.json"
_HTTP_TIMEOUT = 15.0
_LOCAL_TIMEOUT = 3.0

# provider -> {"models": [...], "fetched_at": epoch, "error": str|None}
_cache: Dict[str, Dict[str, Any]] = {}

_OPENAI_CHAT_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")
_OPENAI_EXCLUDE = ("embedding", "audio", "tts", "whisper", "realtime", "image",
                   "dall-e", "moderation", "transcribe", "search", "instruct")


async def _fetch_anthropic() -> List[str]:
    key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not configured")
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        r = await client.get(
            "https://api.anthropic.com/v1/models",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            params={"limit": 100},
        )
        r.raise_for_status()
        return sorted(m["id"] for m in r.json().get("data", []) if m.get("id"))


async def _fetch_openai() -> List[str]:
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not configured")
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        r = await client.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        r.raise_for_status()
        ids = [m["id"] for m in r.json().get("data", []) if m.get("id")]
    chat = [
        mid for mid in ids
        if mid.lower().startswith(_OPENAI_CHAT_PREFIXES)
        and not any(x in mid.lower() for x in _OPENAI_EXCLUDE)
    ]
    return sorted(chat)


async def _fetch_google() -> List[str]:
    key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("GOOGLE_API_KEY / GEMINI_API_KEY not configured")
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        r = await client.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": key, "pageSize": 200},
        )
        r.raise_for_status()
        models = r.json().get("models", [])
    out = [
        m.get("name", "").removeprefix("models/")
        for m in models
        if "generateContent" in (m.get("supportedGenerationMethods") or [])
    ]
    return sorted(x for x in out if x)


async def _fetch_local() -> List[str]:
    """vLLM (Nat, port 8003) /v1/models + ollama :11434 tags — merged."""
    found: List[str] = []
    vllm_base = (os.getenv("LOCAL_LLM_BASE_URL") or
                 os.getenv("VLLM_BASE_URL") or "http://localhost:8003/v1").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=_LOCAL_TIMEOUT) as client:
            r = await client.get(f"{vllm_base}/models")
            r.raise_for_status()
            found.extend(m["id"] for m in r.json().get("data", []) if m.get("id"))
    except Exception as exc:
        logger.debug("[model_discovery] vLLM not reachable: %s", exc)
    ollama_base = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=_LOCAL_TIMEOUT) as client:
            r = await client.get(f"{ollama_base}/api/tags")
            r.raise_for_status()
            found.extend(m["name"] for m in r.json().get("models", []) if m.get("name"))
    except Exception as exc:
        logger.debug("[model_discovery] ollama not reachable: %s", exc)
    if not found:
        raise RuntimeError("no local model server reachable (vLLM :8003 / ollama :11434)")
    return sorted(set(found))


_FETCHERS = {
    "anthropic": _fetch_anthropic,
    "openai": _fetch_openai,
    "google": _fetch_google,
    "local": _fetch_local,
}


def _load_disk_cache() -> None:
    if _cache or not _CACHE_FILE.exists():
        return
    try:
        data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        for provider, entry in data.items():
            if isinstance(entry, dict) and isinstance(entry.get("models"), list):
                _cache[provider] = entry
        logger.info("[model_discovery] disk cache loaded (%d providers)", len(_cache))
    except Exception as exc:
        logger.warning("[model_discovery] disk cache unreadable: %s", exc)


def _save_disk_cache() -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(_cache, indent=1), encoding="utf-8")
        os.replace(tmp, _CACHE_FILE)
    except Exception as exc:
        logger.warning("[model_discovery] disk cache write failed: %s", exc)


async def _get_provider_models(provider: str, force: bool) -> Dict[str, Any]:
    _load_disk_cache()
    now = time.time()
    cached = _cache.get(provider)
    if cached and not force and (now - cached.get("fetched_at", 0)) < CACHE_TTL_S:
        return {**cached, "stale": False, "from_cache": True}
    try:
        models = await _FETCHERS[provider]()
        entry = {"models": models, "fetched_at": now, "error": None}
        _cache[provider] = entry
        _save_disk_cache()
        return {**entry, "stale": False, "from_cache": False}
    except Exception as exc:
        logger.warning("[model_discovery] %s fetch failed: %s", provider, exc)
        if cached and cached.get("models"):
            return {**cached, "stale": True, "from_cache": True,
                    "error": f"fetch failed, serving cache: {exc}"}
        return {"models": [], "fetched_at": None, "stale": True,
                "from_cache": False, "error": str(exc)}


async def _gather(force: bool) -> Dict[str, Any]:
    providers = list(_FETCHERS.keys())
    results = await asyncio.gather(
        *(_get_provider_models(p, force) for p in providers),
        return_exceptions=False,
    )
    payload: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}
    for provider, res in zip(providers, results):
        payload[provider] = res["models"]
        fetched = res.get("fetched_at")
        meta[provider] = {
            "stale": res.get("stale", False),
            "error": res.get("error"),
            "from_cache": res.get("from_cache", False),
            "fetched_at": (
                datetime.fromtimestamp(fetched, tz=timezone.utc).isoformat()
                if fetched else None
            ),
        }
    payload["_meta"] = meta
    return payload


@discovery_router.get("/models/available", response_model=dict)
async def get_available_models():
    """{provider: [model_ids...]} from live provider APIs, 24h-cached.

    Provider outage -> newest cached list with _meta[provider].stale = true.
    """
    return await _gather(force=False)


@discovery_router.post("/models/available/refresh", response_model=dict)
async def refresh_available_models():
    """Force a refetch of every provider list (bypasses the 24h cache)."""
    return await _gather(force=True)


__all__ = ["discovery_router"]
