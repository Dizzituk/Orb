# FILE: app/content/engagement/classifier.py
# Purpose: Comment Sentiment Classifier.
# Called-by: app.content.engagement.scanner
# Depends-on: app.llm
# Last-renovated: 2026-06-11
"""
Comment Sentiment Classifier.

Tiered classification approach:
1. Keyword matching (deterministic, free, instant)
2. LLM fallback (only for ambiguous cases)

Classification tiers:
- positive  → auto-respond candidate
- neutral   → ignore / monitor
- question  → flag for review
- negative  → flag for review
- toxic     → flag urgently
- spam      → ignore / hide
"""
import re
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# KEYWORD DICTIONARIES
# ═══════════════════════════════════════════════════

# Patterns that strongly indicate positive sentiment
POSITIVE_PATTERNS = [
    r"\b(love|loved|loving)\s+(this|it|your|the)\b",
    r"\b(great|amazing|awesome|fantastic|brilliant|incredible)\b",
    r"\b(thank|thanks|thankyou|thank\s*you)\b",
    r"\b(helpful|useful|insightful|informative)\b",
    r"\b(well\s+said|spot\s+on|nailed\s+it|so\s+true)\b",
    r"\b(subscribed|followed|shared)\b",
    r"\b(keep\s+(it\s+)?up|more\s+(of\s+)?this|keep\s+going)\b",
    r"\b(inspired|motivation|motivating|inspiring)\b",
    r"\b(respect|appreciated|valuable)\b",
    r"\b(legend|goat|king|queen|boss)\b",
]

# Patterns that indicate a question
QUESTION_PATTERNS = [
    r"\?\s*$",
    r"^(what|how|why|when|where|who|which|can|could|would|do|does|is|are)\b",
    r"\b(please\s+explain|can\s+you\s+(tell|explain|show))\b",
    r"\b(question|wondering|curious)\b",
]

# Patterns that indicate negative sentiment
NEGATIVE_PATTERNS = [
    r"\b(disagree|wrong|incorrect|misleading|misinformation)\b",
    r"\b(bad|terrible|awful|worst|waste)\b",
    r"\b(disappointed|disappointing|let\s+down)\b",
    r"\b(clickbait|click\s*bait|misleading\s+title)\b",
    r"\b(unsubscribed|unfollowed)\b",
    r"\b(boring|pointless|useless)\b",
]

# Patterns that indicate toxic content — flag urgently
TOXIC_PATTERNS = [
    r"\b(kill\s+yourself|kys|die|death\s+threat)\b",
    r"\b(retard|retarded)\b",
    r"\b(racist|sexist|homophobic)\b",
    r"\b(shut\s+up|stfu|fuck\s+(off|you)|fck)\b",
    r"\b(hate\s+you|hate\s+this\s+guy)\b",
    r"\b(loser|idiot|moron|stupid)\b",
]

# Patterns that indicate spam
SPAM_PATTERNS = [
    r"https?://\S+.*https?://\S+",  # Multiple URLs
    r"\b(check\s+my|visit\s+my|sub\s+(to\s+)?my)\b",
    r"\b(free\s+(money|gift|followers|subs))\b",
    r"\b(dm\s+me|whatsapp|telegram)\b.*\d{5,}",
    r"\b(promo(ting|tion)?|discount|sale)\b.*https?://",
    r"(.)\1{5,}",  # Repeated characters (e.g., "aaaaaa")
]

# Emoji-only positive indicators
POSITIVE_EMOJI_PATTERN = re.compile(
    r"^[\s🔥❤️💯👏👍🙏💪✨🎯💡🫡😍🥰😊👌💖"
    r"\U0001f44d\U0001f44f\U0001f525\U0001f4af\U0001f64f"
    r"\U0001f4aa\U00002764\U0001f60d\U0001f970]+$"
)


def _compile_patterns(patterns: list[str]) -> list[re.Pattern]:
    """Pre-compile regex patterns for performance."""
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_POSITIVE = _compile_patterns(POSITIVE_PATTERNS)
_QUESTION = _compile_patterns(QUESTION_PATTERNS)
_NEGATIVE = _compile_patterns(NEGATIVE_PATTERNS)
_TOXIC = _compile_patterns(TOXIC_PATTERNS)
_SPAM = _compile_patterns(SPAM_PATTERNS)


# ═══════════════════════════════════════════════════
# KEYWORD CLASSIFIER (Tier 1)
# ═══════════════════════════════════════════════════

def classify_by_keywords(text: str) -> Tuple[Optional[str], float]:
    """
    Classify comment sentiment using keyword matching.

    Returns:
        (sentiment, confidence) — or (None, 0.0) if ambiguous.
        Confidence 0.7+ means keyword match is strong enough.
    """
    if not text or not text.strip():
        return ("spam", 0.9)

    cleaned = text.strip()

    # Check for emoji-only positive
    if POSITIVE_EMOJI_PATTERN.match(cleaned):
        return ("positive", 0.95)

    # Check toxic first (highest priority)
    toxic_hits = sum(1 for p in _TOXIC if p.search(cleaned))
    if toxic_hits >= 1:
        return ("toxic", min(0.7 + toxic_hits * 0.1, 0.99))

    # Check spam
    spam_hits = sum(1 for p in _SPAM if p.search(cleaned))
    if spam_hits >= 1:
        return ("spam", min(0.7 + spam_hits * 0.1, 0.99))

    # Count hits per category
    pos_hits = sum(1 for p in _POSITIVE if p.search(cleaned))
    neg_hits = sum(1 for p in _NEGATIVE if p.search(cleaned))
    q_hits = sum(1 for p in _QUESTION if p.search(cleaned))

    # Clear positive (no negative signals)
    if pos_hits >= 1 and neg_hits == 0:
        return ("positive", min(0.6 + pos_hits * 0.1, 0.95))

    # Clear negative (no positive signals)
    if neg_hits >= 1 and pos_hits == 0:
        return ("negative", min(0.6 + neg_hits * 0.1, 0.95))

    # Question detected
    if q_hits >= 1 and neg_hits == 0 and pos_hits == 0:
        return ("question", min(0.6 + q_hits * 0.1, 0.95))

    # Mixed signals or no clear match → ambiguous
    if pos_hits > 0 and neg_hits > 0:
        return (None, 0.0)  # Send to LLM

    # Short comments with no matches
    if len(cleaned.split()) <= 3:
        return ("neutral", 0.5)

    # No clear match → ambiguous
    return (None, 0.0)


# ═══════════════════════════════════════════════════
# LLM CLASSIFIER (Tier 2 — only for ambiguous)
# ═══════════════════════════════════════════════════

LLM_CLASSIFY_PROMPT = """Classify this social media comment into exactly one category.

Categories:
- positive: Supportive, encouraging, complimentary
- neutral: Neither positive nor negative, simple observations
- question: Asking something, seeking information
- negative: Critical, disappointed, disagreeing (but civil)
- toxic: Abusive, threatening, hateful
- spam: Self-promotion, scams, irrelevant

Comment: "{comment}"

Respond with ONLY the category name, nothing else."""


async def classify_by_llm(text: str) -> Tuple[str, float]:
    """
    Classify comment using LLM. Only called for ambiguous cases.
    Returns (sentiment, confidence).
    """
    try:
        from app.llm.gateway import quick_complete

        prompt = LLM_CLASSIFY_PROMPT.format(comment=text[:500])
        result = await quick_complete(
            prompt=prompt,
            max_tokens=10,
            temperature=0.0,
        )

        category = result.strip().lower()
        valid = {"positive", "neutral", "question", "negative", "toxic", "spam"}

        if category in valid:
            return (category, 0.75)

        logger.warning(f"[classifier] LLM returned invalid category: {category}")
        return ("neutral", 0.3)

    except Exception as e:
        logger.error(f"[classifier] LLM classification failed: {e}")
        return ("neutral", 0.3)


# ═══════════════════════════════════════════════════
# UNIFIED CLASSIFIER
# ═══════════════════════════════════════════════════

async def classify_comment(
    text: str,
    use_llm_fallback: bool = True,
) -> Tuple[str, float, str]:
    """
    Classify a comment using tiered approach.

    Returns:
        (sentiment, confidence, method)
        method is 'keyword' or 'llm'
    """
    # Tier 1: keyword matching
    sentiment, confidence = classify_by_keywords(text)

    if sentiment is not None and confidence >= 0.5:
        return (sentiment, confidence, "keyword")

    # Tier 2: LLM fallback (if enabled)
    if use_llm_fallback:
        sentiment, confidence = await classify_by_llm(text)
        return (sentiment, confidence, "llm")

    # Default to neutral if LLM disabled
    return ("neutral", 0.3, "keyword")
