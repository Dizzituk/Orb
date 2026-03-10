# FILE: app/content/distribution/algorithm_strategy.py
"""
YouTube Algorithm Strategy — platform-aware content optimisation.

Encodes known YouTube ranking signals and best practices
into actionable rules that ASTRA applies automatically
when optimising titles, descriptions, tags, thumbnails,
and posting schedules.

This is the intelligence layer that sits between raw content
and what actually gets published. Every publish action should
consult this module.

Sources: YouTube Creator Academy, published research on
recommendation algorithms, creator best practices.
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# CORE RANKING SIGNALS (what YouTube's algorithm cares about)
# ═══════════════════════════════════════════════════

RANKING_SIGNALS = {
    "click_through_rate": {
        "weight": "critical",
        "description": "% of impressions that become views",
        "target": "> 5% (good), > 10% (excellent)",
        "levers": ["thumbnail", "title", "topic_relevance"],
    },
    "average_view_duration": {
        "weight": "critical",
        "description": "How long viewers watch before leaving",
        "target": "> 50% of video length",
        "levers": ["content_quality", "pacing", "hook_strength"],
    },
    "average_percentage_viewed": {
        "weight": "high",
        "description": "What % of the video people watch",
        "target": "> 60% for shorts, > 40% for longform",
        "levers": ["video_length", "pacing", "content_density"],
    },
    "engagement_rate": {
        "weight": "high",
        "description": "Likes + comments + shares / views",
        "target": "> 4% (good), > 8% (viral)",
        "levers": ["call_to_action", "controversial_angle", "question_hooks"],
    },
    "session_time": {
        "weight": "high",
        "description": "Does watching this lead to more YouTube watching",
        "target": "Longer sessions = more promotion",
        "levers": ["end_screens", "playlists", "series_content"],
    },
    "upload_frequency": {
        "weight": "medium",
        "description": "Consistent uploads get algorithmic preference",
        "target": "Minimum 1/week, ideally 2-3/week for shorts",
        "levers": ["content_calendar", "batch_production"],
    },
}


# ═══════════════════════════════════════════════════
# TITLE OPTIMISATION RULES
# ═══════════════════════════════════════════════════

TITLE_RULES = {
    "max_length": 60,
    "ideal_length": (40, 55),
    "rules": [
        "Front-load the most compelling keyword in first 5 words",
        "Use numbers where possible (3 Ways, 5 Reasons, etc.)",
        "Create curiosity gap — promise value without revealing everything",
        "Use power words: secret, shocking, incredible, truth, revealed",
        "Include the primary search keyword naturally",
        "Question format performs well for educational content",
        "Never use ALL CAPS for entire title — one word max",
        "Avoid clickbait that doesn't deliver — kills retention",
    ],
    "shorts_specific": [
        "Keep under 40 characters — gets truncated in shorts feed",
        "Hook word first — viewer decides in 0.5 seconds",
        "Emoji at start can boost CTR in shorts feed",
    ],
}


# ═══════════════════════════════════════════════════
# DESCRIPTION OPTIMISATION RULES
# ═══════════════════════════════════════════════════

DESCRIPTION_RULES = {
    "first_two_lines": (
        "Most important — this is what shows above 'Show More'. "
        "Include primary keyword and a compelling reason to watch."
    ),
    "structure": [
        "Line 1-2: Hook + primary keyword (visible without expanding)",
        "Line 3-5: Brief summary of what viewer will learn/see",
        "Relevant hashtags (3-5, mix of broad and niche)",
        "Timestamps/chapters for longform (boosts retention metrics)",
        "Links to related content (boosts session time)",
        "Call to action (subscribe, comment a specific thing)",
    ],
    "keyword_density": (
        "Include primary keyword 2-3 times naturally. "
        "Include 3-5 related keywords once each. "
        "Never keyword-stuff — YouTube penalises this."
    ),
    "shorts_specific": [
        "Keep very short — 1-2 lines + hashtags",
        "Use #Shorts hashtag (still helps discovery)",
        "3-5 niche hashtags that match search intent",
    ],
}


# ═══════════════════════════════════════════════════
# TAG STRATEGY
# ═══════════════════════════════════════════════════

TAG_STRATEGY = {
    "total_tags": (15, 30),
    "structure": {
        "primary": "1-2 exact-match keywords (what the video IS about)",
        "secondary": "3-5 closely related topics",
        "broad": "3-5 broader category tags",
        "long_tail": "5-10 specific phrases people search for",
        "channel": "1-2 channel/brand tags (builds your tag footprint)",
    },
    "rules": [
        "Most important tags first — YouTube weights order",
        "Include common misspellings of key terms",
        "Mix single words and multi-word phrases",
        "Check competitor tags for ideas (vidIQ/TubeBuddy approach)",
        "Never use irrelevant tags — YouTube penalises this",
    ],
}


# ═══════════════════════════════════════════════════
# POSTING TIME OPTIMISATION
# ═══════════════════════════════════════════════════

# Based on general YouTube data for UK audience
# These get refined by actual analytics over time
OPTIMAL_POSTING_TIMES = {
    "youtube_short": {
        "best_days": ["Tuesday", "Thursday", "Saturday"],
        "best_hours_utc": [7, 12, 17, 19],
        "rationale": (
            "Shorts peak during commute times and lunch breaks. "
            "Weekend mornings also strong for casual browsing."
        ),
    },
    "youtube_longform": {
        "best_days": ["Tuesday", "Thursday", "Saturday"],
        "best_hours_utc": [14, 17, 19],
        "rationale": (
            "Longform performs best when people have time to watch. "
            "Afternoon/evening on weekdays, morning on weekends."
        ),
    },
}


# ═══════════════════════════════════════════════════
# SHORTS-SPECIFIC STRATEGY
# ═══════════════════════════════════════════════════

SHORTS_STRATEGY = {
    "ideal_length": (15, 45),
    "max_length": 60,
    "rules": [
        "Hook in first 1-2 seconds — viewer decides instantly",
        "No intro/outro — jump straight into the content",
        "Fast pacing — new visual or info every 3-5 seconds",
        "End with a loop or callback to the start (boosts replays)",
        "Vertical 9:16 is mandatory",
        "Text overlays help accessibility and silent viewers",
        "Trending audio can boost discovery (if relevant)",
    ],
    "content_types_ranked": [
        "Hot take / controversial opinion (highest engagement)",
        "Quick tutorial / how-to (highest search value)",
        "Surprising fact / stat (highest share rate)",
        "Behind the scenes (highest authenticity signal)",
        "Reaction / commentary (moderate all metrics)",
    ],
}


# ═══════════════════════════════════════════════════
# OPTIMISATION FUNCTIONS
# ═══════════════════════════════════════════════════

def get_optimisation_prompt(
    content_type: str = "youtube_short",
    title: str = "",
    description: str = "",
    topic: str = "",
) -> str:
    """
    Build an optimisation prompt that encodes algorithm knowledge.

    Used by the youtube_optimiser to guide the LLM when
    generating tags, titles, and descriptions.
    """
    is_short = "short" in content_type.lower()

    title_rules = TITLE_RULES["shorts_specific"] if is_short else TITLE_RULES["rules"]
    desc_rules = DESCRIPTION_RULES["shorts_specific"] if is_short else DESCRIPTION_RULES["structure"]

    strategy = SHORTS_STRATEGY if is_short else {}

    return f"""You are optimising a YouTube {'Short' if is_short else 'video'} for maximum algorithmic performance.

YOUTUBE ALGORITHM PRIORITIES (in order):
1. Click-Through Rate — the thumbnail and title must create curiosity
2. Average View Duration — the content must hook and hold attention
3. Engagement — likes, comments, shares signal quality to the algorithm
4. Session Time — content that keeps people on YouTube gets promoted

TITLE RULES:
{chr(10).join('- ' + r for r in title_rules)}

DESCRIPTION RULES:
{chr(10).join('- ' + r for r in desc_rules)}

TAG STRATEGY:
- {TAG_STRATEGY['total_tags'][0]}-{TAG_STRATEGY['total_tags'][1]} tags total
- Primary (exact match): {TAG_STRATEGY['structure']['primary']}
- Secondary (related): {TAG_STRATEGY['structure']['secondary']}
- Long tail (search phrases): {TAG_STRATEGY['structure']['long_tail']}
- Most important tags FIRST

{'SHORTS RULES:' + chr(10) + chr(10).join('- ' + r for r in SHORTS_STRATEGY['rules']) if is_short else ''}

Current title: {title}
Current description: {description}
Topic: {topic}

Optimise for MAXIMUM algorithmic performance while staying authentic and delivering on the promise."""


def get_optimal_posting_time(
    content_type: str = "youtube_short",
    current_utc: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Calculate the next optimal posting time based on
    algorithm strategy and content type.
    """
    if current_utc is None:
        current_utc = datetime.now(timezone.utc)

    config = OPTIMAL_POSTING_TIMES.get(
        content_type,
        OPTIMAL_POSTING_TIMES["youtube_short"],
    )

    day_names = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]

    best_days = config["best_days"]
    best_hours = config["best_hours_utc"]

    # Find the next best day+hour combination
    for days_ahead in range(7):
        candidate = current_utc + timedelta(days=days_ahead)
        day_name = day_names[candidate.weekday()]

        if day_name in best_days or days_ahead > 3:
            for hour in sorted(best_hours):
                slot = candidate.replace(
                    hour=hour, minute=0, second=0, microsecond=0,
                )
                if slot > current_utc + timedelta(hours=2):
                    return {
                        "scheduled_time": slot,
                        "day": day_name,
                        "hour_utc": hour,
                        "rationale": config["rationale"],
                    }

    # Fallback: tomorrow at best hour
    tomorrow = current_utc + timedelta(days=1)
    return {
        "scheduled_time": tomorrow.replace(
            hour=best_hours[0], minute=0, second=0, microsecond=0,
        ),
        "day": day_names[tomorrow.weekday()],
        "hour_utc": best_hours[0],
        "rationale": "Fallback — next available slot",
    }


def score_title(title: str) -> Dict[str, Any]:
    """Score a title against algorithm best practices."""
    issues = []
    score = 10.0

    if len(title) > TITLE_RULES["max_length"]:
        issues.append(f"Too long ({len(title)} chars, max {TITLE_RULES['max_length']})")
        score -= 2

    if len(title) < 20:
        issues.append("Too short — not enough keywords for discovery")
        score -= 1

    if title == title.upper():
        issues.append("All caps — looks spammy, reduces trust")
        score -= 2

    if not any(c in title for c in "?!:"):
        issues.append("No punctuation hook — questions and exclamations boost CTR")
        score -= 0.5

    # Check for power words
    power_words = [
        "secret", "shocking", "truth", "revealed", "ultimate",
        "complete", "essential", "proven", "fastest", "best",
        "worst", "never", "always", "mistake", "hack",
    ]
    has_power = any(w in title.lower() for w in power_words)
    if not has_power:
        issues.append("No power words — consider adding one for CTR")
        score -= 0.5

    return {
        "score": max(0, min(10, score)),
        "issues": issues,
        "length": len(title),
    }


def get_strategy_summary() -> Dict[str, Any]:
    """Get a summary of the current algorithm strategy.
    
    Useful for displaying in the UI or feeding to LLMs as context.
    """
    return {
        "ranking_signals": {
            k: {"weight": v["weight"], "target": v["target"]}
            for k, v in RANKING_SIGNALS.items()
        },
        "shorts_ideal_length": SHORTS_STRATEGY["ideal_length"],
        "title_max_length": TITLE_RULES["max_length"],
        "tag_count": TAG_STRATEGY["total_tags"],
        "posting_times": OPTIMAL_POSTING_TIMES,
    }
