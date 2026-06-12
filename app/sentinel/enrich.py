# FILE: app/sentinel/enrich.py
# Purpose: Reverse-DNS + GeoIP enrichment for public IPs via ip-api.com batch endpoint —
#          cache-first (data\sentinel\geo_cache.json), rate-limit aware, private IPs never sent.
# Called-by: app.sentinel.collector, app.sentinel.router
# Depends-on: httpx; ip-api.com free tier (15 batch req/min, 100 IPs/batch, http only)
# Last-renovated: 2026-06-12
from __future__ import annotations

import ipaddress
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_CACHE_FILE = Path(__file__).resolve().parents[2] / "data" / "sentinel" / "geo_cache.json"
_BATCH_URL = "http://ip-api.com/batch?fields=status,message,country,countryCode,reverse,org,query"
_BATCH_MAX = 100
_BACKOFF_SECONDS = 90.0  # after an error or exhausted rate-limit window

_cache: Optional[Dict[str, dict]] = None
_backoff_until: float = 0.0


def is_public_ip(ip: str) -> bool:
    """True for routable public IPs; private/loopback/link-local etc. short-circuit."""
    try:
        parsed = ipaddress.ip_address(str(ip).strip())
    except ValueError:
        return False
    return not (
        parsed.is_private
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_multicast
        or parsed.is_reserved
        or parsed.is_unspecified
    )


def _load_cache() -> Dict[str, dict]:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(_CACHE_FILE.read_text(encoding="utf-8")) if _CACHE_FILE.exists() else {}
        except Exception:
            logger.warning("[sentinel] geo cache unreadable — starting fresh")
            _cache = {}
    return _cache


def _save_cache() -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(_load_cache(), indent=1), encoding="utf-8")
    except Exception as exc:
        logger.debug("[sentinel] geo cache save failed: %s", exc)


def cached_info(ip: str) -> Optional[dict]:
    return _load_cache().get(ip)


async def enrich_ips(ips: List[str]) -> Dict[str, dict]:
    """Enrich a set of IPs. Returns {ip: {rdns, country, country_code, org, private}}.

    Cache-first; only public cache-misses are sent to ip-api (batched, one
    request per call, max 100). During backoff, misses are simply returned
    empty and retried on a later cycle — enrichment is best-effort.
    """
    global _backoff_until
    cache = _load_cache()
    out: Dict[str, dict] = {}
    misses: List[str] = []

    for ip in dict.fromkeys(ips):  # de-dupe, keep order
        if not ip:
            continue
        if not is_public_ip(ip):
            out[ip] = {"rdns": "", "country": "", "country_code": "", "org": "", "private": True}
            continue
        hit = cache.get(ip)
        if hit:
            out[ip] = hit
            continue
        misses.append(ip)

    if not misses or time.monotonic() < _backoff_until:
        return out

    batch = misses[:_BATCH_MAX]
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(_BATCH_URL, json=batch)
        remaining = resp.headers.get("X-Rl")
        ttl = resp.headers.get("X-Ttl")
        if resp.status_code == 429 or remaining == "0":
            _backoff_until = time.monotonic() + (float(ttl) if ttl else _BACKOFF_SECONDS)
            logger.info("[sentinel] ip-api rate limited — backing off %ss", ttl or _BACKOFF_SECONDS)
            return out
        resp.raise_for_status()
        rows = resp.json()
    except Exception as exc:
        _backoff_until = time.monotonic() + _BACKOFF_SECONDS
        logger.warning("[sentinel] ip-api batch lookup failed (%s) — backoff %ss", exc, _BACKOFF_SECONDS)
        return out

    stamped = datetime.now(timezone.utc).isoformat()
    for row in rows if isinstance(rows, list) else []:
        ip = str(row.get("query") or "")
        if not ip:
            continue
        if row.get("status") != "success":
            info = {"rdns": "", "country": "", "country_code": "", "org": "",
                    "private": False, "cached_at": stamped}
        else:
            info = {
                "rdns": str(row.get("reverse") or ""),
                "country": str(row.get("country") or ""),
                "country_code": str(row.get("countryCode") or ""),
                "org": str(row.get("org") or ""),
                "private": False,
                "cached_at": stamped,
            }
        cache[ip] = info
        out[ip] = info
    _save_cache()
    return out
