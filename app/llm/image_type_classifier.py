# FILE: app/llm/image_type_classifier.py
# Purpose: Image request classifier — determines which rendering pipeline to use.
# Called-by: app.llm.image_router
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Image request classifier — determines which rendering pipeline to use.

Categories:
  DATA_CHART     — user provided data inline (percentages, numbers, labels)
                   → Plotly deterministic renderer (no AI, exact data)
  DATA_RESEARCH  — user wants a chart but needs data gathered first
                   → Web search → extract → Plotly
  CREATIVE       — scenic, portrait, avatar, illustration, thumbnail, etc.
                   → GPT Image 1.5 / Nano Banana (AI generation)

The key insight: if the user already gave you the numbers, NEVER send them
to an AI image model. AI models hallucinate text and can't render exact
percentages. Use Plotly for data, AI for creative.

v1.0 (2026-04-02): Initial implementation.
"""
from __future__ import annotations

import re
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class ImageType(str, Enum):
    DATA_CHART = "data_chart"          # User provided inline data
    DATA_RESEARCH = "data_research"    # Needs web search for data
    CREATIVE = "creative"              # AI-generated creative image


@dataclass
class ImageClassification:
    image_type: ImageType
    confidence: float
    reason: str
    inline_data_detected: bool = False


# Patterns that indicate inline numerical data
_INLINE_DATA_PATTERNS = [
    # Percentage with number: "White: 88.6%", "45%", "12.4%"
    re.compile(r'\b\d+\.?\d*\s*%', re.IGNORECASE),
    # Labelled values: "White: 88.6", "Asian: 5.6"
    re.compile(r'\b[A-Z][a-z]+\s*:\s*\d+\.?\d*', re.IGNORECASE),
    # Parenthetical counts: "(7,353)", "(464)"
    re.compile(r'\(\d[\d,]+\)'),
    # Explicit data markers
    re.compile(r'\b(?:data|values?|numbers?)\s*(?:\(|:)', re.IGNORECASE),
]

# Minimum matches to be confident data is inline
_MIN_DATA_MATCHES = 3

# Keywords indicating a chart/graph/data viz request
_CHART_KEYWORDS = re.compile(
    r'\b(?:chart|graph|bar\s*chart|pie\s*chart|line\s*chart|infographic|'
    r'histogram|scatter\s*plot|data\s*vis|plot|horizontal\s*bar|'
    r'stacked\s*bar|grouped\s*bar|comparison\s*chart)\b',
    re.IGNORECASE,
)

# Keywords that suggest research is needed (no inline data)
_RESEARCH_KEYWORDS = re.compile(
    r'\b(?:latest|recent|current|compare|comparison|benchmark|'
    r'ranking|trend|over\s+time|growth|decline|statistics|'
    r'how\s+(?:many|much)|what\s+(?:are|is)\s+the)\b',
    re.IGNORECASE,
)

# Strong signals of creative/artistic image requests
_CREATIVE_KEYWORDS = re.compile(
    r'\b(?:photo(?:graph)?|portrait|landscape|scenic|artistic|'
    r'illustration|avatar|icon|logo|thumbnail|banner|cover|'
    r'painting|sketch|anime|cartoon|realistic|fantasy|abstract|'
    r'wallpaper|poster|album\s*cover|profile\s*(?:pic|picture)|'
    r'cinematic|dramatic|beautiful|aesthetic|style\s+of)\b',
    re.IGNORECASE,
)

# Layout/structure keywords that suggest a designed data slide
_LAYOUT_KEYWORDS = re.compile(
    r'\b(?:layout|footer|header|labelled|stacked|side\s*by\s*side|'
    r'source\s*:|font|colour|color|accent|background|dark\s*(?:background|theme)|'
    r'slide|1080x1080|carousel|swipe)\b',
    re.IGNORECASE,
)


def classify_image_request(message: str) -> ImageClassification:
    """Classify an image generation request into DATA_CHART, DATA_RESEARCH, or CREATIVE.

    Priority:
    1. If inline numerical data is detected (3+ data patterns) → DATA_CHART
    2. If chart keywords + research keywords but no inline data → DATA_RESEARCH
    3. If creative keywords dominate → CREATIVE
    4. If chart keywords but ambiguous → DATA_RESEARCH (safer to search)
    5. Default → CREATIVE
    """
    inline_data_count = sum(
        len(pat.findall(message)) for pat in _INLINE_DATA_PATTERNS
    )
    has_chart_kw = bool(_CHART_KEYWORDS.search(message))
    has_layout_kw = bool(_LAYOUT_KEYWORDS.search(message))
    has_research_kw = bool(_RESEARCH_KEYWORDS.search(message))
    has_creative_kw = bool(_CREATIVE_KEYWORDS.search(message))

    logger.info(
        "[image_classifier] inline_data=%d, chart=%s, layout=%s, research=%s, creative=%s",
        inline_data_count, has_chart_kw, has_layout_kw, has_research_kw, has_creative_kw,
    )

    # 1. Strong inline data → DATA_CHART (deterministic Plotly)
    if inline_data_count >= _MIN_DATA_MATCHES:
        return ImageClassification(
            image_type=ImageType.DATA_CHART,
            confidence=min(1.0, 0.7 + inline_data_count * 0.05),
            reason=f"Inline data detected: {inline_data_count} numeric patterns",
            inline_data_detected=True,
        )

    # 2. Chart + layout keywords with moderate data → DATA_CHART
    if (has_chart_kw or has_layout_kw) and inline_data_count >= 2:
        return ImageClassification(
            image_type=ImageType.DATA_CHART,
            confidence=0.8,
            reason=f"Chart/layout keywords + {inline_data_count} data patterns",
            inline_data_detected=True,
        )

    # 3. Chart keywords + research keywords, no inline data → DATA_RESEARCH
    if has_chart_kw and has_research_kw and not has_creative_kw:
        return ImageClassification(
            image_type=ImageType.DATA_RESEARCH,
            confidence=0.75,
            reason="Chart + research keywords, no inline data",
        )

    # 4. Creative keywords dominate → CREATIVE
    if has_creative_kw and not has_chart_kw and inline_data_count < 2:
        return ImageClassification(
            image_type=ImageType.CREATIVE,
            confidence=0.85,
            reason="Creative/artistic keywords detected",
        )

    # 5. Chart keywords but no data, no research signal → DATA_RESEARCH (safer)
    if has_chart_kw and not has_creative_kw:
        return ImageClassification(
            image_type=ImageType.DATA_RESEARCH,
            confidence=0.6,
            reason="Chart keywords present, defaulting to research path",
        )

    # 6. Default: CREATIVE
    return ImageClassification(
        image_type=ImageType.CREATIVE,
        confidence=0.5,
        reason="No strong chart/data signals, defaulting to creative",
    )


__all__ = ["classify_image_request", "ImageType", "ImageClassification"]
