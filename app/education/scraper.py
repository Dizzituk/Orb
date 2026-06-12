# Purpose: scraper
# Called-by: app.education.llm_scraper, app.education.service
# Depends-on: app.education.schemas
# Last-renovated: 2026-06-11
from __future__ import annotations

import json
import logging
import re
from html import unescape
from typing import List
from urllib.parse import urlparse

import httpx

from app.education.schemas import ScrapedModule

logger = logging.getLogger(__name__)


class EducationScrapeError(Exception):
    pass


def scrape_coursera_course(url: str) -> List[ScrapedModule]:
    parsed = urlparse(url)
    if "coursera.org" not in parsed.netloc.lower():
        raise EducationScrapeError("Only Coursera URLs are supported right now.")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        response = httpx.get(url, headers=headers, timeout=20.0, follow_redirects=True)
        response.raise_for_status()
    except Exception as exc:
        raise EducationScrapeError(f"Failed to fetch course page: {exc}") from exc

    html = response.text
    modules = _extract_from_json_ld(html)
    if not modules:
        modules = _extract_from_next_data(html)
    if not modules:
        modules = _extract_from_headings(html)
    if not modules:
        raise EducationScrapeError("Could not extract curriculum from this Coursera page.")

    return [
        ScrapedModule(
            title=module.title.strip(),
            description=(module.description or "").strip() or None,
            order_index=index + 1,
        )
        for index, module in enumerate(modules)
    ]


def _extract_from_json_ld(html: str) -> List[ScrapedModule]:
    modules: List[ScrapedModule] = []
    for raw in re.findall(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', html, re.S | re.I):
        try:
            payload = json.loads(unescape(raw.strip()))
        except Exception:
            continue
        for item in _walk_json(payload):
            if not isinstance(item, dict):
                continue
            title = item.get("name") or item.get("title")
            desc = item.get("description")
            if isinstance(title, list):
                title = title[0] if title else None
            if isinstance(desc, list):
                desc = desc[0] if desc else None
            if title and isinstance(title, str) and _looks_like_module(title):
                modules.append(ScrapedModule(title=title, description=str(desc) if desc else None, order_index=len(modules) + 1))
    return _dedupe_modules(modules)


def _extract_from_next_data(html: str) -> List[ScrapedModule]:
    match = re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S | re.I)
    if not match:
        return []
    try:
        payload = json.loads(unescape(match.group(1)))
    except Exception:
        return []

    modules: List[ScrapedModule] = []
    for item in _walk_json(payload):
        if not isinstance(item, dict):
            continue
        title = item.get("name") or item.get("title") or item.get("moduleName")
        desc = item.get("description") or item.get("summary") or item.get("subtitle")
        if title and _looks_like_module(title):
            modules.append(ScrapedModule(title=title, description=desc, order_index=len(modules) + 1))
    return _dedupe_modules(modules)


def _extract_from_headings(html: str) -> List[ScrapedModule]:
    pattern = re.compile(r'<h[1-4][^>]*>(.*?)</h[1-4]>', re.S | re.I)
    modules: List[ScrapedModule] = []
    for match in pattern.finditer(html):
        title = _strip_tags(match.group(1))
        if not _looks_like_module(title):
            continue
        window = html[match.end(): match.end() + 600]
        desc_match = re.search(r'<p[^>]*>(.*?)</p>', window, re.S | re.I)
        desc = _strip_tags(desc_match.group(1)) if desc_match else None
        modules.append(ScrapedModule(title=title, description=desc, order_index=len(modules) + 1))
    return _dedupe_modules(modules)


def _strip_tags(value: str) -> str:
    return re.sub(r'<[^>]+>', ' ', unescape(value or '')).strip()


def _walk_json(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _looks_like_module(title) -> bool:
    if isinstance(title, list):
        title = title[0] if title else ""
    if not isinstance(title, str):
        return False
    text = title.strip()
    if len(text) < 4 or len(text) > 180:
        return False
    lowered = text.lower()
    keywords = ["module", "week", "lesson", "introduction", "project", "capstone", "basics", "foundations"]
    return any(keyword in lowered for keyword in keywords) or bool(re.match(r"^(module|week|lesson)\s*\d+", lowered))


def _dedupe_modules(modules: List[ScrapedModule]) -> List[ScrapedModule]:
    seen = set()
    result: List[ScrapedModule] = []
    for module in modules:
        key = module.title.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(module)
    return result[:50]
