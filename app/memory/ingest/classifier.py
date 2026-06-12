# FILE: app/memory/ingest/classifier.py
# Purpose: Item classifier (Spec Section 9.2, Stage 3).
# Called-by: app.memory.ingest, app.memory.ingest.pipeline
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Item classifier (Spec Section 9.2, Stage 3).

Takes extracted knowledge items and classifies each one:
    - domain: development, content, fitness, catering, finance, general
    - memory_layer: preference, knowledge, decision, ephemeral
    - confidence: How certain the classification is (0.0–1.0)
    - project_id: Which project this belongs to

Items below the confidence threshold go to the review queue
instead of being stored directly.

Classification uses keyword matching and role-based heuristics.
Can be upgraded to LLM-based classification later.

Usage:
    from app.memory.ingest.classifier import classify_item

    result = classify_item(
        text="I prefer 4-space indentation in Python",
        source="gpt_export",
        role="user",
    )
    # result.domain = "development"
    # result.memory_layer = "preference"
    # result.confidence = 0.85
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# =========================================================================
# Classification result
# =========================================================================

@dataclass
class ClassifiedItem:
    """Result of classifying a parsed chunk."""
    text: str
    domain: str                 # development, content, fitness, etc.
    memory_layer: str           # preference, knowledge, decision, ephemeral
    confidence: float           # 0.0–1.0
    project_id: str             # Usually astra-core
    source: str                 # user_stated, researched, inferred
    source_file: str
    metadata: dict

    @property
    def needs_review(self) -> bool:
        return self.confidence < REVIEW_THRESHOLD


# Confidence below this goes to review queue
REVIEW_THRESHOLD = 0.5


# =========================================================================
# Domain keyword sets
# =========================================================================

DOMAIN_KEYWORDS = {
    "development": {
        "code", "coding", "python", "javascript", "function", "class",
        "refactor", "debug", "api", "database", "git", "deploy",
        "file size", "modularity", "architecture", "pipeline",
        "indent", "formatting", "linting", "test", "build",
        "variable", "import", "module", "package", "framework",
    },
    "content": {
        "blog", "article", "post", "writing", "tone", "voice",
        "headline", "paragraph", "social media", "caption",
        "thumbnail", "seo", "keyword", "publish", "draft",
        "editing", "proofread", "copywrite", "content calendar",
    },
    "video": {
        "video", "edit", "footage", "clip", "timeline", "render",
        "resolution", "export", "thumbnail", "transition",
        "colour grade", "color grade", "frame rate", "codec",
    },
    "fitness": {
        "training", "exercise", "workout", "programme", "program",
        "sets", "reps", "weight", "cardio", "strength",
        "client", "pt", "personal training", "split",
        "muscle", "recovery", "nutrition", "protein",
    },
    "catering": {
        "recipe", "menu", "portion", "ingredients", "cooking",
        "kitchen", "supplier", "catering", "food cost",
        "allergen", "prep", "service", "cover", "banquet",
    },
    "finance": {
        "invoice", "expense", "revenue", "profit", "tax",
        "vat", "account", "budget", "cost", "price",
        "payment", "billing", "receipt", "financial",
    },
}

# =========================================================================
# Memory layer patterns
# =========================================================================

PREFERENCE_PATTERNS = [
    r"i (?:prefer|like|want|always|never)",
    r"(?:always|never) (?:use|do|make)",
    r"my (?:preferred|favourite|favorite|default)",
    r"(?:don't|do not|dont) (?:ever|use|like)",
    r"make sure (?:to|you)",
    r"from now on",
    r"going forward",
]

DECISION_PATTERNS = [
    r"(?:decided|decision|chose|chosen) to",
    r"we (?:went|go) with",
    r"the (?:approach|strategy|plan) is",
    r"(?:rationale|reason|because we)",
    r"(?:trade-?off|alternative|instead of)",
    r"(?:rejected|ruled out|dismissed)",
]

EPHEMERAL_PATTERNS = [
    r"(?:right now|currently|at the moment)",
    r"(?:today|this session|this task)",
    r"(?:working on|in progress|next step)",
    r"(?:todo|to-?do|action item)",
]


# =========================================================================
# Classifier
# =========================================================================

def classify_item(
    text: str,
    source: str = "inferred",
    role: Optional[str] = None,
    source_file: str = "",
    metadata: Optional[dict] = None,
    project_id: str = "astra-core",
) -> ClassifiedItem:
    """
    Classify a text item into domain and memory layer.

    Args:
        text: The content to classify.
        source: Origin (user_stated, researched, inferred, gpt_export).
        role: Speaker role (user, assistant) for GPT exports.
        source_file: Original file path.
        metadata: Additional context from parser.
        project_id: Project scope.

    Returns:
        ClassifiedItem with domain, memory_layer, and confidence.
    """
    meta = metadata or {}
    text_lower = text.lower()

    # Classify domain
    domain, domain_conf = _classify_domain(text_lower)

    # Classify memory layer
    layer, layer_conf = _classify_layer(text_lower, role)

    # Determine source reliability
    if source == "user_stated" or role == "user":
        source_type = "user_stated"
    elif source == "gpt_export" and role == "assistant":
        source_type = "researched"
    else:
        source_type = "inferred"

    # Combined confidence
    confidence = (domain_conf * 0.4) + (layer_conf * 0.6)

    # Boost confidence for user-stated items
    if source_type == "user_stated":
        confidence = min(1.0, confidence + 0.15)

    return ClassifiedItem(
        text=text,
        domain=domain,
        memory_layer=layer,
        confidence=round(confidence, 3),
        project_id=project_id,
        source=source_type,
        source_file=source_file,
        metadata=meta,
    )


def _classify_domain(text: str) -> tuple[str, float]:
    """Classify into a domain by keyword scoring."""
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text)
        if hits > 0:
            scores[domain] = hits

    if not scores:
        return ("general", 0.3)

    best = max(scores, key=scores.get)
    # Normalise: more hits = higher confidence, cap at 0.95
    conf = min(0.95, 0.5 + (scores[best] * 0.1))
    return (best, conf)


def _classify_layer(
    text: str,
    role: Optional[str],
) -> tuple[str, float]:
    """Classify into a memory layer."""
    # Check preference patterns
    pref_hits = sum(
        1 for p in PREFERENCE_PATTERNS
        if re.search(p, text, re.IGNORECASE)
    )
    if pref_hits >= 1 and (role == "user" or role is None):
        return ("preference", min(0.9, 0.6 + pref_hits * 0.15))

    # Check decision patterns
    dec_hits = sum(
        1 for p in DECISION_PATTERNS
        if re.search(p, text, re.IGNORECASE)
    )
    if dec_hits >= 1:
        return ("decision", min(0.85, 0.5 + dec_hits * 0.15))

    # Check ephemeral patterns
    eph_hits = sum(
        1 for p in EPHEMERAL_PATTERNS
        if re.search(p, text, re.IGNORECASE)
    )
    if eph_hits >= 1:
        return ("ephemeral", min(0.8, 0.5 + eph_hits * 0.1))

    # Default: knowledge
    return ("knowledge", 0.5)
