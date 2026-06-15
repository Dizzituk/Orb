# FILE: app/scene_director/research/researcher.py
# Purpose: Turn an intent into a structured FACT PACK by REUSING ASTRA's existing web-search
#          plumbing (app.llm.web_search.search_and_answer) — bounded, distilled facts + sources
#          (title+url) for scene composition + provenance. Does NOT build a new search client.
# Called-by: app.scene_director.research (maybe_research)
# Depends-on: app.llm.web_search (search_and_answer, WebSearchRequest)
# Last-renovated: 2026-06-13
"""research(intent, era) -> fact_pack dict | None.

Reuses search_and_answer: ONE bounded async round-trip (search → fetch → LLM-distil → cited
answer + sources with titles/URLs). We keep distilled facts in OUR words + source links
(copyright-respectful — no large verbatim text). fact_pack shape:
  {"summary": str, "facts": [str, ...], "sources": [{"title","url"}], "query": str}

Synthesis model + provider are the web-search stage's own env knobs (WEB_SEARCH_PROVIDER_ID /
WEB_SEARCH_MODEL_ID); search provider key is BRAVE_SEARCH_API_KEY (falls back to DuckDuckGo).
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

_SENT = re.compile(r"(?<=[.!?])\s+")


def _max_results() -> int:
    try:
        return max(1, min(10, int(os.getenv("SCENE_RESEARCH_MAX_RESULTS", "5"))))
    except ValueError:
        return 5


def _build_query(intent: str, era: Optional[str]) -> str:
    """A scene-oriented query: what a builder needs (what it looked like, layout, scale)."""
    base = intent.strip().rstrip(".")
    bits = [base]
    if era and era.strip().lower() not in ("", "modern", "any", "none"):
        bits.append(f"{era} period")
    bits.append("what it looked like layout buildings scale visual details")
    return " ".join(bits)[:400]


def _facts_from_answer(answer: str, limit: int = 8) -> list[str]:
    """Split the distilled answer into concise fact bullets (strip [1][2] citation marks)."""
    clean = re.sub(r"\[\d+\]", "", answer or "").strip()
    sentences = [s.strip() for s in _SENT.split(clean) if len(s.strip()) > 20]
    return sentences[:limit]


async def research(intent: str, era: Optional[str] = None) -> Optional[dict]:
    """Run one bounded web-search round and distil a fact pack. Returns None on failure
    (caller composes without research). Never raises out of here for a search problem."""
    try:
        from app.llm.web_search import search_and_answer, WebSearchRequest
    except Exception as exc:  # plumbing missing — degrade gracefully
        logger.warning("[scene.research] web-search plumbing unavailable: %s", exc)
        return None

    query = _build_query(intent, era)
    try:
        resp = await search_and_answer(WebSearchRequest(query=query, max_results=_max_results()))
    except Exception as exc:
        logger.warning("[scene.research] search failed (%s) — composing without research", exc)
        return None

    if resp is None or not getattr(resp, "ok", False) or not getattr(resp, "answer", "").strip():
        logger.info("[scene.research] no usable research result for '%s'", query[:80])
        return None

    sources = []
    for s in (getattr(resp, "sources", None) or []):
        title = getattr(s, "title", "") or ""
        url = getattr(s, "url", "") or ""
        if url:
            sources.append({"title": title, "url": url})

    fact_pack = {
        "summary": (resp.answer or "").strip()[:1500],
        "facts": _facts_from_answer(resp.answer),
        "sources": sources[:8],
        "query": query,
    }
    logger.info("[scene.research] fact pack: %d facts, %d sources (query='%s')",
                len(fact_pack["facts"]), len(fact_pack["sources"]), query[:60])
    return fact_pack
