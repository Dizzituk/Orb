# FILE: app/astra_memory/topic_tagger.py
# Purpose: Topic + entity tagger for HotIndex content.
# Called-by: app.astra_memory.indexer, app.llm.routing.memory_injection
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Topic + entity tagger for HotIndex content.

Replaces the legacy six-tag keyword extractor in indexer.py which only
recognised code-related tags (`code`, `python`, `testing`, `architecture`,
`documentation`, `debugging`). That meant every Astra conversation \u2014 even
investment discussions, political analysis, legal-claim context, content
strategy \u2014 was either tagged with a code label or untagged. The arbiter
and the memory_injection layer query by tag, so untagged-by-topic records
were effectively invisible to topic-driven retrieval.

This module produces:
  - tags:     coarse topic domains (investments, politics, ai_society, work,
              legal, portugal, content, finance, lifestyle, astra_dev,
              philosophy, plus the legacy code/architecture/testing/etc.)
  - entities: specific named items (TAO, Bittensor, Yodel, Leigh Day,
              Reform UK, Zack Polanski, ASTRA, etc.)

Both are extracted by curated keyword matching \u2014 fast, deterministic, no
LLM dependency. The keyword lists are intentionally focused on the
topics Taz actually discusses (per his ASTRA usage patterns). When a
record matches none of the topic domains it falls back to a coarse
`general` tag rather than the empty list, so retrieval can still find it
by recency / priority.

Authoritative since Phase 9 retrieval-quality work (2026-05-13).
"""
from __future__ import annotations

import re
from typing import List, Set


# =============================================================================
# Topic-domain keyword sets
# =============================================================================
#
# Order matters: a record gets every domain whose keywords match. So a chat
# about "TAO valuation under different AGI displacement scenarios" picks up
# {investments, ai_society} \u2014 both correctly.
#
# Keep keyword sets tight: a single false-positive keyword can over-tag a
# whole content type. Test additions against /scripts/audit_retrieval_*.py
# before shipping.

_TOPIC_KEYWORDS = {
    # Investing / markets / specific instruments Taz holds.
    # Keep keywords unambiguous: avoid common English words that overload
    # (share/stake/position/hold/yield mean different things outside
    # markets). Multi-word phrases and ticker-shaped terms only.
    "investments": {
        "tao", "bittensor", "subnet",
        "rebalance", "reallocate",
        "my portfolio", "the portfolio", "portfolio allocation",
        "investment", "investments", "investing",
        "my investment", "my investments",
        "stocks and shares", "stock market", "equity", "equities",
        "etf", "etfs", "index fund",
        "dividend yield", "dividend payout",
        "ticker", "price target", "valuation",
        "bull case", "bear case", "moonbull", "moon bull",
        "quantum computing", "ionq", "rigetti",
        "nvidia", "tsmc", "asml",
        "ftse", "nasdaq", "s&p 500",
        "gold price", "silver price", "bullion",
        "crypto", "bitcoin", "ethereum",
        "market cap", "market commentary",
        "add to my position", "build a position",
        "my holdings", "my position in",
    },
    # UK + global politics, ideology, electoral.
    # Avoid bare common words (labour, policy, conservative, party, green)
    # which match ordinary English. Multi-word phrases or proper-noun
    # combinations only.
    "politics": {
        "reform uk", "reform party", "reform's",
        "green party", "polanski", "zack polanski",
        "labour party", "keir starmer", "starmer",
        "tory party", "the tories",
        "conservative party",
        "lib dem", "liberal democrat",
        "by-election", "local election", "general election",
        "ge2024", "ge 2024",
        "immigration", "asylum seeker", "small boats", "channel crossings",
        "brexit",
        "political", "manifesto",
        "wealth inequality", "wealth tax",
    },
    # AI as a societal/economic force, displacement, AGI economics.
    # Note: 'agi' is a strong signal — keep it. 'displacement' on its own
    # is ambiguous (cursor/water displacement), so require it next to AI
    # markers via the multi-word form.
    "ai_society": {
        "agi", "asi", "superintelligence",
        "automation dividend", "ubi", "universal basic income",
        "sovereign wealth fund",
        "ai displacement", "job displacement", "labour displacement",
        "agi economics", "agi timeline",
        "ai literacy", "digital sovereignty",
        "alex imas", "deepmind", "deepmind hire",
        "recursive self-improvement",
        "wealth redistribution",
    },
    # Delivery driving / route / DSP / Yodel / InPost.
    # Avoid 'ming' (Flash Gordon villain) and 'parcel' alone (too generic).
    "work": {
        "yodel", "inpost",
        "dsp subcontractor",
        "route to09", "to09 route", " to09 ",
        "parcels", "parcel count",
        "delivery driver", "delivery work", "delivery route",
        "depot", "stem time", "drop rate",
        "redruth depot", "cornwall route",
    },
    # The worker-status claim / GDPR claim.
    # Avoid bare 'claim' (overloads with 'I claim that X').
    "legal": {
        "leigh day", "barings law", "abbleys",
        "worker status", "limb b",
        "era 1996", "employment rights act",
        "s.230(3)", "section 230(3)",
        "legal claim", "the claim", "my claim against",
        "solicitor", "solicitors",
        "gb domestic", "drivers hours", "driver's hours",
        "11-hour duty", "11 hour duty", "duty limit",
        "gdpr claim", "data breach",
        "amazon dsp",
        "tribunal", "settled with payout",
    },
    # Portugal D7 visa plans.
    # Avoid bare 'portuguese' (matches 'Portuguese man-of-war') and bare
    # 'algarve' on its own; require place-context.
    "portugal": {
        "portugal", "move to portugal", "live in portugal",
        "d7 visa", "d-7 visa", "passive income visa",
        "lagos algarve", "lagos, algarve", "lagos portugal",
        "algarve coast", "in the algarve",
        "alentejo",
        "portuguese residency", "portuguese passport",
        "expat", "relocation to portugal",
        "nhr regime", "non-habitual resident",
    },
    # Social media output, Man in the Van persona.
    # Avoid 'thumbnail' (README thumbnails), 'audience' (any reader),
    # 'caption' (any caption). Require platform names or persona refs.
    "content": {
        "facebook post", "facebook page", "facebook content",
        "instagram post", "instagram reel", "instagram content",
        "tiktok", "tiktok video",
        "youtube short", "youtube channel", "youtube video",
        "man in the van", "man-in-the-van",
        "social media post", "social media content",
        "engagement rate", "engagement metric",
        "d-id avatar", "avatar video",
        "meta business suite",
    },
    # Personal finance, tax, accounting.
    # Avoid 'budget' alone (budget the time / budget your tokens).
    "finance": {
        "self-assessment", "self assessment", "tax return",
        "mtd", "making tax digital",
        "hmrc",
        "mileage log", "mileage claim",
        "business expense", "deductible expense",
        "credit card statement", "savings goal",
        "weekly earnings", "monthly earnings",
        "personal budget", "household budget",
        "sole trader", "self-employed",
    },
    # Lifestyle, health, fitness, ADHD, surfing.
    # Avoid 'pt' (too short and overloads), 'fitness' as a singleton
    # (matches 'fitness function'), 'sugar craving' (matches 'craving for').
    "lifestyle": {
        "surf", "surfing", "surfboard", "longboard surfboard",
        "low-carb", "low carb diet", "keto diet", "ketogenic diet",
        "adhd", "dyslexia",
        "personal trainer", "personal training",
        "food craving", "a sugar craving",
        "sleep quality", "sleep schedule",
        "workout session", "exercise routine",
        "porthtowan", "fistral", "watergate bay",
    },
    # ASTRA-specific dev work (distinct from generic 'code').
    # Avoid 'implementer' alone (GoF Implementer pattern). Use full proper
    # nouns or multi-word ASTRA terms.
    "astra_dev": {
        "astra", "astrabridge", "astra bridge",
        "scaffold engine", "agentic builder", "visual verifier",
        "windows sandbox", "specgate", "spec gate",
        "overwatcher", "critical pipeline",
        "pattern library db",
        "hot index", "hotindex", "preference record",
        "decay job", "decay scheduler",
        "astra implementer",
        "chatterbox tts", "tts pipeline",
        "memory tier", "tier 1", "tier 2", "tier 3",
        "arbiter mode", "shadow mode", "enforce mode",
        "proposed facts",
    },
    # Worldview, philosophy, frameworks.
    "philosophy": {
        "stoic", "stoicism", "buddhist", "buddhism",
        "first principles", "systems thinking",
        "epistemology",
        "moral philosophy", "ethical framework",
    },
}

# Legacy code-related tags (kept for backward-compat with existing records
# and for retrieval queries that legitimately want technical content).
# The match conditions here are looser than _TOPIC_KEYWORDS because code
# signals are syntactic (e.g. 'def ' as a marker of Python code).

_CODE_PATTERNS = (
    (r"\bdef\s+\w+\(", "code"),
    (r"\bclass\s+\w+", "code"),
    (r"\bimport\s+\w+", "code"),
    (r"\basync\s+def", "code"),
    (r"\bfunction\s+\w+", "code"),
    (r"\bpython\b", "python"),
    (r"\b(pytest|unittest|test_)", "testing"),
    (r"\b(architecture|architect|design pattern|design decision)", "architecture"),
    (r"\b(readme|documentation|docstring|tutorial)", "documentation"),
    (r"\b(traceback|stacktrace|error:|exception:|bug|debug)", "debugging"),
)


# =============================================================================
# Entity patterns
# =============================================================================
#
# Entities are SPECIFIC named items that warrant exact retrieval matches.
# A user asking about "TAO" should hit records that mention TAO explicitly,
# not generic "investments" records. So entities are the high-precision
# index whereas tags are the high-recall index.

# Curated entity vocabulary \u2014 case-insensitive match, returned in canonical
# form. Add new entries here when a topic gets discussed often enough to
# warrant precise retrieval rather than fuzzy keyword fallback.

_ENTITY_VOCAB = {
    # Investments / instruments
    "TAO": ("tao", "bittensor", "$tao"),
    "Bittensor": ("bittensor",),
    "IonQ": ("ionq", "ion q"),
    "Rigetti": ("rigetti",),
    "NVIDIA": ("nvidia", "nvda"),
    "AMD": (),                # tricky \u2014 too short, can be a false positive
    "TSMC": ("tsmc",),
    # Work / legal
    "Yodel": ("yodel",),
    "InPost": ("inpost", "in post"),
    "Leigh Day": ("leigh day",),
    "Barings Law": ("barings law", "barings"),
    "Abbleys": ("abbleys",),
    "ERA 1996": ("era 1996", "employment rights act"),
    # Politics
    "Reform UK": ("reform uk", "reform party"),
    "Zack Polanski": ("zack polanski", "polanski"),
    "Green Party": ("green party", "the greens"),
    "Keir Starmer": ("starmer", "keir starmer"),
    # ASTRA / Anthropic
    "ASTRA": ("astra", "astrabridge"),
    "Anthropic": ("anthropic", "claude"),
    "DeepMind": ("deepmind",),
    "OpenAI": ("openai", "open ai", "chatgpt"),
    # Places
    "Cornwall": ("cornwall", "redruth"),
    "Portugal": ("portugal", "portuguese"),
    "Lagos": ("lagos algarve", "lagos, algarve"),
    "Barnstaple": ("barnstaple",),
    # People (frequently-mentioned)
    "Alex Imas": ("alex imas",),
    # Concepts
    "AGI": ("agi", "artificial general intelligence"),
    "ASI": ("asi", "artificial superintelligence"),
    "UBI": ("universal basic income", "ubi"),
    "D7 Visa": ("d7 visa", "d-7 visa"),
}


# =============================================================================
# Public API
# =============================================================================

def extract_tags(text: str, max_tags: int = 8) -> List[str]:
    """
    Extract topic-domain tags from `text`. Returns up to `max_tags` items
    in deterministic order (sorted alphabetically).

    Always returns at least one tag: if no domain matches, returns
    ['general'] so retrieval can still rank the record by recency.
    """
    if not text:
        return ["general"]

    lower = text.lower()
    found: Set[str] = set()

    # Topic-domain matching
    for domain, keywords in _TOPIC_KEYWORDS.items():
        for kw in keywords:
            # Word-boundary match for short keywords; substring for long.
            if " " in kw or "-" in kw or len(kw) >= 6:
                if kw in lower:
                    found.add(domain)
                    break
            else:
                if re.search(rf"\b{re.escape(kw)}\b", lower):
                    found.add(domain)
                    break

    # Code-pattern matching (legacy compatibility)
    for pattern, tag in _CODE_PATTERNS:
        if re.search(pattern, lower):
            found.add(tag)

    if not found:
        return ["general"]

    return sorted(found)[:max_tags]


def extract_entities(text: str, max_entities: int = 12) -> List[str]:
    """
    Extract specific named entities from `text`. Returns canonical names.

    Returns [] (empty list) if nothing matches \u2014 this is the correct
    JSON-serialisable empty value, NOT None / 'null'.
    """
    if not text:
        return []

    lower = text.lower()
    found: List[str] = []
    seen: Set[str] = set()

    for canonical, aliases in _ENTITY_VOCAB.items():
        # Check canonical form first (case-insensitive substring), then aliases
        canonical_lower = canonical.lower()
        hit = False
        if canonical_lower in lower and len(canonical_lower) >= 3:
            hit = True
        else:
            for alias in aliases:
                if alias in lower:
                    hit = True
                    break
        if hit and canonical not in seen:
            found.append(canonical)
            seen.add(canonical)

    return found[:max_entities]


def extract_tags_and_entities(text: str) -> dict:
    """Convenience: return both as a single dict."""
    return {
        "tags": extract_tags(text),
        "entities": extract_entities(text),
    }
