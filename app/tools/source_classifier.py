# FILE: app/tools/source_classifier.py
# Purpose: Source diversity & credibility tagging for web search results.
# Called-by: app.llm.deep_research_engine, app.llm.web_search
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Source diversity & credibility tagging for web search results.

Maps domains to credibility tiers and source types so the LLM
synthesiser can weigh sources appropriately and present a balanced
view across source types.

Tiers:
    T1 — Primary / Authoritative: Official docs, gov sites, peer-reviewed
    T2 — Established: Major news, well-known tech blogs, Wikipedia
    T3 — Professional: Industry blogs, company engineering blogs
    T4 — Community: Forums, Reddit, Stack Overflow, personal blogs
    T5 — Unknown: Unclassified domains (default)

Source types:
    academic, government, official_docs, mainstream_news, tech_news,
    industry_blog, company_blog, encyclopedia, forum, social_media,
    independent_researcher, aggregator, ecommerce, unknown

v2.1 (2026-02): Initial implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse


@dataclass(frozen=True)
class SourceTag:
    """Credibility tag for a search result source."""
    domain: str
    tier: int                   # 1-5, lower = more authoritative
    tier_label: str             # "authoritative", "established", etc.
    source_type: str            # "academic", "forum", etc.
    bias_note: Optional[str]    # Optional note about known bias


# =========================================================================
# Domain registry — known domains mapped to tier + type
# =========================================================================
# Format: domain_pattern -> (tier, source_type, bias_note)
# Patterns are checked with endswith() for subdomains.

_DOMAIN_REGISTRY: dict[str, tuple[int, str, Optional[str]]] = {
    # --- T1: Primary / Authoritative ---
    # Government
    "gov.uk": (1, "government", None),
    ".gov": (1, "government", None),
    "europa.eu": (1, "government", None),
    "who.int": (1, "government", None),
    "ons.gov.uk": (1, "government", None),
    # Academic / Research
    ".edu": (1, "academic", None),
    ".ac.uk": (1, "academic", None),
    "arxiv.org": (1, "academic", None),
    "scholar.google.com": (1, "academic", None),
    "pubmed.ncbi.nlm.nih.gov": (1, "academic", None),
    "nature.com": (1, "academic", None),
    "sciencedirect.com": (1, "academic", None),
    "ieee.org": (1, "academic", None),
    "acm.org": (1, "academic", None),
    "researchgate.net": (1, "academic", None),
    # Official documentation
    "docs.python.org": (1, "official_docs", None),
    "developer.mozilla.org": (1, "official_docs", None),
    "docs.microsoft.com": (1, "official_docs", None),
    "learn.microsoft.com": (1, "official_docs", None),
    "docs.google.com": (1, "official_docs", None),
    "docs.github.com": (1, "official_docs", None),
    "docs.docker.com": (1, "official_docs", None),
    "docs.aws.amazon.com": (1, "official_docs", None),
    "cloud.google.com": (1, "official_docs", None),
    "docs.anthropic.com": (1, "official_docs", None),
    "platform.openai.com": (1, "official_docs", None),
    "fastapi.tiangolo.com": (1, "official_docs", None),
    "reactjs.org": (1, "official_docs", None),
    "react.dev": (1, "official_docs", None),
    "nodejs.org": (1, "official_docs", None),
    "kubernetes.io": (1, "official_docs", None),
    "www.postgresql.org": (1, "official_docs", None),
    "redis.io": (1, "official_docs", None),

    # --- T2: Established ---
    # Major news
    "bbc.co.uk": (2, "mainstream_news", None),
    "bbc.com": (2, "mainstream_news", None),
    "reuters.com": (2, "mainstream_news", None),
    "apnews.com": (2, "mainstream_news", None),
    "theguardian.com": (2, "mainstream_news", None),
    "nytimes.com": (2, "mainstream_news", None),
    "washingtonpost.com": (2, "mainstream_news", None),
    "ft.com": (2, "mainstream_news", None),
    "economist.com": (2, "mainstream_news", None),
    "bloomberg.com": (2, "mainstream_news", None),
    "cnbc.com": (2, "mainstream_news", None),
    "sky.com": (2, "mainstream_news", None),
    "independent.co.uk": (2, "mainstream_news", None),
    "telegraph.co.uk": (2, "mainstream_news", None),
    # Tech news
    "techcrunch.com": (2, "tech_news", None),
    "theverge.com": (2, "tech_news", None),
    "arstechnica.com": (2, "tech_news", None),
    "wired.com": (2, "tech_news", None),
    "zdnet.com": (2, "tech_news", None),
    "theregister.com": (2, "tech_news", None),
    # Encyclopedia
    "wikipedia.org": (2, "encyclopedia", "Community-edited, verify claims"),
    "britannica.com": (2, "encyclopedia", None),

    # --- T3: Professional / Industry ---
    # Company engineering blogs
    "engineering.fb.com": (3, "company_blog", None),
    "netflixtechblog.com": (3, "company_blog", None),
    "blog.google": (3, "company_blog", None),
    "aws.amazon.com/blogs": (3, "company_blog", None),
    "openai.com/blog": (3, "company_blog", None),
    "anthropic.com": (3, "company_blog", None),
    "github.blog": (3, "company_blog", None),
    "stripe.com/blog": (3, "company_blog", None),
    # Industry / tech blogs
    "martinfowler.com": (3, "industry_blog", None),
    "kentcdodds.com": (3, "industry_blog", None),
    "simonwillison.net": (3, "industry_blog", None),
    "tldrsec.com": (3, "industry_blog", None),
    "css-tricks.com": (3, "industry_blog", None),
    "smashingmagazine.com": (3, "industry_blog", None),
    "dev.to": (3, "industry_blog", None),
    "hashnode.dev": (3, "industry_blog", None),
    "freecodecamp.org": (3, "industry_blog", None),
    # Independent researchers
    "substack.com": (3, "independent_researcher", "Quality varies by author"),

    # --- T4: Community ---
    # Forums
    "stackoverflow.com": (4, "forum", "Community answers, check accepted/votes"),
    "superuser.com": (4, "forum", None),
    "serverfault.com": (4, "forum", None),
    "askubuntu.com": (4, "forum", None),
    "stackexchange.com": (4, "forum", None),
    "github.com": (4, "forum", "Issues/discussions are community content"),
    # Social / discussion
    "reddit.com": (4, "social_media", "Anecdotal, verify claims independently"),
    "news.ycombinator.com": (4, "social_media", "Tech-focused community discussion"),
    "twitter.com": (4, "social_media", "Unverified individual claims"),
    "x.com": (4, "social_media", "Unverified individual claims"),
    "quora.com": (4, "forum", "Quality varies widely"),
    # Aggregators
    "medium.com": (4, "aggregator", "Quality varies widely by author"),
    "towardsdatascience.com": (4, "aggregator", "Community-written, verify"),

    # --- T4: E-commerce (not authoritative for info) ---
    "amazon.co.uk": (4, "ecommerce", "Product listings, not editorial"),
    "amazon.com": (4, "ecommerce", "Product listings, not editorial"),
    "ebay.co.uk": (4, "ecommerce", None),
    "ebay.com": (4, "ecommerce", None),
}


# Tier labels
_TIER_LABELS = {
    1: "authoritative",
    2: "established",
    3: "professional",
    4: "community",
    5: "unknown",
}


def _extract_domain(url: str) -> str:
    """Extract the domain from a URL."""
    try:
        parsed = urlparse(url)
        return (parsed.hostname or "").lower()
    except Exception:
        return ""


def classify_source(url: str) -> SourceTag:
    """
    Classify a URL into a credibility tier and source type.

    Checks the domain against the registry using suffix matching
    so subdomains are caught (e.g. "blog.example.com" matches "example.com").
    """
    domain = _extract_domain(url)
    if not domain:
        return SourceTag(
            domain="",
            tier=5,
            tier_label="unknown",
            source_type="unknown",
            bias_note=None,
        )

    # Check registry — longest match wins (most specific)
    best_match = None
    best_length = 0

    for pattern, (tier, source_type, bias_note) in _DOMAIN_REGISTRY.items():
        if domain == pattern or domain.endswith("." + pattern) or domain.endswith(pattern):
            if len(pattern) > best_length:
                best_match = (tier, source_type, bias_note)
                best_length = len(pattern)

    if best_match:
        tier, source_type, bias_note = best_match
        return SourceTag(
            domain=domain,
            tier=tier,
            tier_label=_TIER_LABELS[tier],
            source_type=source_type,
            bias_note=bias_note,
        )

    return SourceTag(
        domain=domain,
        tier=5,
        tier_label="unknown",
        source_type="unknown",
        bias_note=None,
    )


def classify_sources(urls: list[str]) -> list[SourceTag]:
    """Classify multiple URLs."""
    return [classify_source(url) for url in urls]


def get_diversity_summary(tags: list[SourceTag]) -> dict:
    """
    Analyse source diversity across a result set.

    Returns:
        tier_distribution: Count per tier
        type_distribution: Count per source type
        diversity_score: 0.0-1.0 (higher = more diverse)
        bias_notes: Any bias warnings
        missing_perspectives: Source types not represented
    """
    if not tags:
        return {
            "tier_distribution": {},
            "type_distribution": {},
            "diversity_score": 0.0,
            "bias_notes": [],
            "missing_perspectives": [],
        }

    tier_dist: dict[int, int] = {}
    type_dist: dict[str, int] = {}
    bias_notes: list[str] = []

    for tag in tags:
        tier_dist[tag.tier] = tier_dist.get(tag.tier, 0) + 1
        type_dist[tag.source_type] = type_dist.get(tag.source_type, 0) + 1
        if tag.bias_note:
            bias_notes.append(f"[{tag.domain}] {tag.bias_note}")

    # Diversity score: how many unique source types / total possible
    # More types = more diverse perspective
    unique_types = len([t for t in type_dist if t != "unknown"])
    max_useful_types = min(len(tags), 6)  # Cap at 6 types for scoring
    diversity_score = unique_types / max_useful_types if max_useful_types > 0 else 0.0
    diversity_score = min(1.0, diversity_score)

    # Identify missing perspectives
    ideal_types = {"official_docs", "mainstream_news", "industry_blog", "forum"}
    present_types = set(type_dist.keys())
    missing = ideal_types - present_types

    return {
        "tier_distribution": tier_dist,
        "type_distribution": type_dist,
        "diversity_score": round(diversity_score, 2),
        "bias_notes": bias_notes,
        "missing_perspectives": sorted(missing),
    }


def format_source_tag_for_evidence(tag: SourceTag) -> str:
    """Format a source tag for inclusion in LLM evidence prompt."""
    parts = [f"[{tag.tier_label.upper()}]", f"[{tag.source_type}]"]
    if tag.bias_note:
        parts.append(f"(Note: {tag.bias_note})")
    return " ".join(parts)
