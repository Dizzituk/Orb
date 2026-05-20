# FILE: app/memory/complexity.py
"""
Query Complexity Classifier (Spec Section 10, Job 6A).

Classifies incoming queries into complexity tiers to drive
model selection. Works alongside the existing 8-route job
classifier (which routes by content type) — this classifies
by DIFFICULTY.

Tiers:
    ping_pong   — Trivial: greetings, yes/no, acknowledgements
    lookup      — Factual: single-fact retrieval, definitions
    reasoning   — Analytical: comparisons, explanations, planning
    deep        — Complex: multi-step architecture, system design
    multimodal  — Specialist: vision, video, audio processing

Signals used:
    - Query length (word count)
    - Intent type (from translation layer)
    - Domain keywords (architecture, security, etc.)
    - Conversation context depth
    - Confidence score (from confidence store)
    - Attachment presence (images, video, PDFs)

Returns:
    ComplexityResult with tier, confidence, model_target,
    and escalation_hint.

Usage:
    from app.memory.complexity import classify_complexity

    result = classify_complexity(
        query="redesign the pipeline to support multi-project",
        intent="ARCHITECTURE",
    )
    # result.tier = "deep"
    # result.model_target = "opus"
    # result.escalation_hint = None
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# =========================================================================
# Result dataclass
# =========================================================================

@dataclass
class ComplexityResult:
    """Result of complexity classification."""
    tier: str               # ping_pong, lookup, reasoning, deep, multimodal
    confidence: float       # How sure we are about the classification
    model_target: str       # Suggested model tier: local, local_rag, sonnet, opus, specialist
    escalation_hint: Optional[str]  # If set, suggests escalation reason
    signals: dict           # Debug: which signals contributed

    @property
    def is_local_capable(self) -> bool:
        return self.tier in ("ping_pong", "lookup")

    @property
    def needs_rag(self) -> bool:
        return self.tier in ("lookup", "reasoning", "deep")


# =========================================================================
# Tier definitions
# =========================================================================

TIER_MODEL_MAP = {
    "ping_pong": "local",
    "lookup": "local_rag",
    "reasoning": "sonnet",
    "deep": "opus",
    "multimodal": "specialist",
    # Explicit user model requests
    "explicit_gpt54": "gpt54",
    "explicit_claude": "claude",
    "explicit_gemini": "gemini",
}


# =========================================================================
# Keyword sets
# =========================================================================

# Deep complexity indicators
# v9.2: Deep keywords split into architecture (needs big model) and exploration (needs tools only)
DEEP_ARCHITECTURE_KEYWORDS = {
    "architecture", "architectural", "redesign", "refactor",
    "system design", "multi-file", "multi-project",
    "migration", "security review", "threat model",
    "dependency graph", "blast radius", "breaking change",
    "trade-off", "tradeoff", "design decision",
    "scalability", "performance audit", "infrastructure",
    "pipeline", "roadmap", "integrate",
    "integration", "end-to-end", "production", "deploy",
    "come up with a plan", "full plan", "full spec", "architectural plan", "design plan",
}

DEEP_EXPLORATION_KEYWORDS = {
    # Codebase exploration — needs tool access but not a big model
    "code base", "codebase", "go through the code",
    "look at the code", "look through the code",
    "check the code", "explore the code",
    "go through the project", "look at the project",
    "check the backend", "check the frontend",
    # Filesystem/app exploration
    "find the app", "find the android", "find the bridge",
    "look at the app", "look at the folder", "look at the android",
    "check the app", "check the folder", "check the android",
    "tell me about the app", "tell me about the project",
    "overview of the app", "overview of the project",
    "explore the app", "explore the folder",
    "inspect the", "browse the", "scan the",
    "what's in the", "what is in the",
    "astra android", "astra-bridge", "astra bridge", "bridge app",
    "android folder", "android project", "android app",
    "driver copilot", "fitness app", "bar stock",  # ASTRA flagship apps
    # Domain awareness — broad terms that indicate the user wants ASTRA to act
    "domain", "capability", "capabilities",
    # Cloud storage — needs tool access for cloud_upload/cloud_list
    "upload to drive", "upload to google", "upload to cloud",
    "put on drive", "put on google drive", "save to drive",
    "google drive", "cloud storage", "upload the file",
    "send to drive", "share on drive", "make accessible",
    "upload it", "upload this", "put it on drive",
    "download from drive", "get from drive", "list drive",
}

# Combined set for backward compat — anything that needs deep processing or tools
DEEP_KEYWORDS = DEEP_ARCHITECTURE_KEYWORDS | DEEP_EXPLORATION_KEYWORDS

# Reasoning indicators
REASONING_KEYWORDS = {
    "compare", "explain", "analyse", "analyze", "plan",
    "strategy", "approach", "recommend", "evaluate",
    "debug", "diagnose", "troubleshoot", "optimise",
    "optimize", "review", "critique", "assess",
    "implement", "build", "create", "write code",
    "how should", "what if", "pros and cons",
}

# Lookup indicators
LOOKUP_KEYWORDS = {
    "what is", "what are", "define", "definition",
    "show me", "list", "status", "where is",
    "find", "search", "look up", "how many",
    "who is", "when was", "which file",
}

# Ping-pong indicators
PING_PONG_PATTERNS = [
    r"^(hi|hello|hey|yo|sup|thanks|thank you|ok|okay|yes|no|sure|cool|great|cheers|ta|right|yep|nah|nope)[\s!.?]*$",
    r"^(good\s+(morning|afternoon|evening|night))[\s!.?]*$",
    r"^(?:hi|hello|hey)?\s*(how are you|what'?s up|how'?s it going)[\s!.?]*$",
]

# Multimodal indicators (attachment-based, not keyword)
MULTIMODAL_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".mp3", ".wav", ".ogg", ".m4a",
}

# Intent-to-tier overrides
INTENT_TIER_OVERRIDES = {
    "CHAT_ONLY": "ping_pong",
    "SHOW_STATUS": "lookup",
    "LIST_FILES": "lookup",
    "SHOW_ARCHITECTURE": "lookup",
    "SHOW_LOGS": "lookup",
    "SHOW_PREFERENCES": "lookup",
    "DESCRIBE_FILE": "lookup",
    "DELETE_FILE": "reasoning",
    "PURGE_QUARANTINE": "reasoning",
    "RUN_PIPELINE": "reasoning",
    "REFACTOR": "deep",
    "SCAN_CODEBASE": "reasoning",
    "CREATE_ARCHITECTURE_MAP": "deep",
    "BUILD_SPEC": "deep",
    "EXECUTE_SANDBOX": "reasoning",
}


# =========================================================================
# Explicit model request patterns
# =========================================================================

# When the user explicitly asks for a specific model, skip all other
# classification and route directly. These are intentional requests,
# not complexity-based upgrades.
EXPLICIT_MODEL_PATTERNS = {
    "gpt_54": {
        "patterns": [
            r"(?:use|switch (?:that |it |this )?to|escalate (?:that |it |this )?to|send (?:it |that )?to|route (?:it |that |this )?to|put (?:it |that )?through)\s*(?:gpt[\s\-]?5[\.\s]?4|gpt5\.4|gpt 5\.4|openai)",
            r"(?:use|switch (?:that |it |this )?to|escalate (?:that |it |this )?to)\s*(?:the\s+)?(?:reasoning|creative|openai)\s+model",
            r"can you (?:escalate|send|route|switch)\s+(?:it |that |this )?to\s+(?:gpt[\s\-]?5[\.\s]?4|openai)",
        ],
        "tier": "explicit_gpt54",
        "provider": "openai",
        "model": "gpt-5.4",
        "target": "gpt54",
    },
    "claude": {
        "patterns": [
            r"(?:use|switch (?:that |it |this )?to|escalate (?:that |it |this )?to|send (?:it |that )?to|route (?:it |that |this )?to|put (?:it |that )?through)\s*(?:claude|anthropic|opus)",
            r"(?:use|switch (?:that |it |this )?to|escalate (?:that |it |this )?to)\s*(?:the\s+)?(?:deep|architecture)\s+model",
            r"can you (?:escalate|send|route|switch)\s+(?:it |that |this )?to\s+(?:claude|anthropic|opus)",
        ],
        "tier": "explicit_claude",
        "provider": "anthropic",
        "model": "claude-opus-4-7",
        "target": "claude",
    },
    "gemini": {
        "patterns": [
            r"(?:use|switch (?:that |it |this )?to|escalate (?:that |it |this )?to|send (?:it |that )?to|route (?:it |that |this )?to|put (?:it |that )?through)\s*(?:gemini|google)",
            r"(?:use|switch (?:that |it |this )?to|escalate (?:that |it |this )?to)\s*(?:the\s+)?(?:vision|multimodal|google)\s+model",
            r"can you (?:escalate|send|route|switch)\s+(?:it |that |this )?to\s+(?:gemini|google)",
        ],
        "tier": "explicit_gemini",
        "provider": "google",
        "model": "gemini-3.1-pro-preview",
        "target": "gemini",
    },
}


def _detect_explicit_model_request(query_lower: str) -> Optional[dict]:
    """Detect if the user is explicitly requesting a specific model.

    Returns the model config dict if detected, None otherwise.
    """
    for model_key, config in EXPLICIT_MODEL_PATTERNS.items():
        for pattern in config["patterns"]:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return config
    return None


# =========================================================================
# Main classifier
# =========================================================================

def classify_complexity(
    query: str,
    intent: Optional[str] = None,
    attachments: Optional[list] = None,
    context_depth: int = 0,
    confidence_score: Optional[float] = None,
) -> ComplexityResult:
    """
    Classify query complexity into a tier.

    Args:
        query: The user's input text.
        intent: Resolved intent from translation layer (if available).
        attachments: List of attachment file paths or dicts.
        context_depth: Number of active context entries (conversation depth).
        confidence_score: Confidence score from the confidence store.

    Returns:
        ComplexityResult with tier, model_target, and debug signals.
    """
    signals = {}
    query_lower = query.lower().strip()
    word_count = len(query.split())

    # ── Signal 0: Explicit model request (HIGHEST priority) ──────
    # When the user says "use GPT-5.4" or "send to Claude",
    # skip all other classification and route directly.
    # Check BOTH the full message AND just the first line —
    # users often put "escalate to X" on the first line
    # followed by a big block of content.
    first_line = query_lower.split("\n")[0].strip()
    explicit = (
        _detect_explicit_model_request(first_line)
        or _detect_explicit_model_request(query_lower)
    )
    if explicit:
        signals["explicit_model"] = explicit["model"]
        _prov = explicit["provider"]
        _mod = explicit["model"]
        logger.info(
            f"[complexity] Explicit model request: {_prov}/{_mod}"
        )
        return ComplexityResult(
            tier=explicit["tier"],
            confidence=1.0,
            model_target=explicit["target"],
            escalation_hint=f"User requested {_mod}",
            signals=signals,
        )

    # ── Signal 1: Multimodal check (highest priority) ────────────
    if attachments:
        has_multimodal = _check_multimodal(attachments)
        signals["multimodal"] = has_multimodal
        if has_multimodal:
            return ComplexityResult(
                tier="multimodal",
                confidence=0.95,
                model_target="specialist",
                escalation_hint=None,
                signals=signals,
            )

    # ── Signal 2: Intent override ────────────────────────────────
    if intent and intent.upper() in INTENT_TIER_OVERRIDES:
        tier = INTENT_TIER_OVERRIDES[intent.upper()]
        signals["intent_override"] = intent
        return ComplexityResult(
            tier=tier,
            confidence=0.85,
            model_target=TIER_MODEL_MAP[tier],
            escalation_hint=None,
            signals=signals,
        )

    # ── Signal 3: Ping-pong detection ────────────────────────────
    if _is_ping_pong(query_lower):
        signals["ping_pong_match"] = True
        return ComplexityResult(
            tier="ping_pong",
            confidence=0.95,
            model_target="local",
            escalation_hint=None,
            signals=signals,
        )

    # ── Signal 4: Keyword scoring ────────────────────────────────
    deep_hits = _count_keyword_hits(query_lower, DEEP_KEYWORDS)
    arch_hits = _count_keyword_hits(query_lower, DEEP_ARCHITECTURE_KEYWORDS)
    explore_hits = _count_keyword_hits(query_lower, DEEP_EXPLORATION_KEYWORDS)
    reasoning_hits = _count_keyword_hits(query_lower, REASONING_KEYWORDS)
    lookup_hits = _count_keyword_hits(query_lower, LOOKUP_KEYWORDS)

    signals["deep_hits"] = deep_hits
    signals["arch_hits"] = arch_hits
    signals["explore_hits"] = explore_hits
    signals["reasoning_hits"] = reasoning_hits
    signals["lookup_hits"] = lookup_hits
    signals["word_count"] = word_count

    # ── Signal 5: Word count heuristic ───────────────────────────
    length_tier = _length_heuristic(word_count)
    signals["length_tier"] = length_tier

    # ── Signal 6: Context depth ──────────────────────────────────
    signals["context_depth"] = context_depth

    # ── Combine signals ──────────────────────────────────────────
    tier, conf = _resolve_tier(
        deep_hits, reasoning_hits, lookup_hits,
        word_count, length_tier, context_depth,
    )

    return ComplexityResult(
        tier=tier,
        confidence=round(conf, 3),
        model_target=TIER_MODEL_MAP[tier],
        escalation_hint=None,
        signals=signals,
    )


# =========================================================================
# Signal helpers
# =========================================================================

def _is_ping_pong(query_lower: str) -> bool:
    """Check if query matches trivial ping-pong patterns."""
    for pattern in PING_PONG_PATTERNS:
        if re.match(pattern, query_lower, re.IGNORECASE):
            return True
    # Also catch very short messages with no substance
    if len(query_lower.split()) <= 2 and not any(
        kw in query_lower for kw in ("refactor", "build", "scan", "show", "create")
    ):
        return True
    return False


def _count_keyword_hits(text: str, keywords: set) -> int:
    """Count how many keywords from the set appear in text.

    Uses word-boundary matching for single words to prevent
    'architect' matching inside 'architecture'. Multi-word
    phrases use plain substring matching.
    """
    hits = 0
    for kw in keywords:
        if " " in kw:
            # Multi-word phrase: substring match is fine
            if kw in text:
                hits += 1
        else:
            # Single word: require word boundaries
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                hits += 1
    return hits


def _length_heuristic(word_count: int) -> str:
    """Classify by message length."""
    if word_count <= 5:
        return "ping_pong"
    if word_count <= 15:
        return "lookup"
    if word_count <= 50:
        return "reasoning"
    return "deep"


def _check_multimodal(attachments: list) -> bool:
    """Check if any attachments are multimodal (image/video/audio)."""
    for att in attachments:
        # Support both string paths and dict attachments
        if isinstance(att, str):
            path = att
        else:
            # v14.0: Also check mime type directly (synthetic attachments
            # from file_upload_* fields may have a mime key)
            mime = att.get("mime", "")
            if mime.startswith(("image/", "video/", "audio/")):
                return True
            path = att.get("path", "")
        ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if ext in MULTIMODAL_EXTENSIONS:
            return True
    return False


def _resolve_tier(
    deep_hits: int,
    reasoning_hits: int,
    lookup_hits: int,
    word_count: int,
    length_tier: str,
    context_depth: int,
) -> tuple[str, float]:
    """
    Combine all signals into a final tier + confidence.

    Priority: deep > reasoning > lookup > ping_pong
    Context depth pushes toward higher tiers.
    """
    # Deep: 2+ deep keywords OR 1 deep + long message OR 1 deep + reasoning
    # v2: Very long messages (100+ words) with reasoning keywords are also deep
    # v9.0: 1 deep + 1 reasoning = deep (catches "go through codebase" + "plan")
    if (deep_hits >= 2
            or (deep_hits >= 1 and word_count > 30)
            or (deep_hits >= 1 and reasoning_hits >= 1)):
        conf = min(0.95, 0.7 + (deep_hits * 0.1))
        return ("deep", conf)

    # Reasoning: reasoning keywords dominate, or moderate complexity
    if reasoning_hits >= 2 or (reasoning_hits >= 1 and word_count > 15):
        conf = min(0.9, 0.6 + (reasoning_hits * 0.1))
        # Context depth can push reasoning → deep
        if context_depth > 5 and reasoning_hits >= 2:
            return ("deep", conf * 0.9)
        # Very long messages with multiple reasoning signals → deep
        # (multi-domain planning, brainstorming, strategic requests)
        if word_count > 100 and reasoning_hits >= 3:
            return ("deep", min(0.85, conf))
        return ("reasoning", conf)

    # Lookup: lookup keywords or short factual query
    if lookup_hits >= 1 or (word_count <= 10 and deep_hits == 0 and reasoning_hits == 0):
        conf = min(0.85, 0.6 + (lookup_hits * 0.1))
        return ("lookup", conf)

    # Length-based fallback
    if length_tier == "ping_pong":
        return ("ping_pong", 0.7)

    if length_tier == "deep":
        return ("deep", 0.6)

    # v2.3: No keyword signals at all → default to lookup, not reasoning.
    # A 30-word message with zero reasoning/deep/lookup hits is just a
    # conversational request (e.g. "write me a poem and save it as a txt").
    # Only escalate to reasoning when there is actual keyword evidence.
    if deep_hits == 0 and reasoning_hits == 0 and lookup_hits == 0:
        return ("lookup", 0.5)

    # Default to reasoning for ambiguous medium-length queries with some signal
    return ("reasoning", 0.5)
