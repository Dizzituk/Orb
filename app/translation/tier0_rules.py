# FILE: app/translation/tier0_rules.py
"""
Tier 0: Pure rule-based intent classification (no LLM calls).

Routes wake phrases, pipeline triggers, RAG queries, and obvious chat patterns.
Trigger check functions are in _tier0_trigger_checks.py.
Filesystem, multi-file, refactor, and chat pattern rules are in _tier0_filesystem.py.
"""
from __future__ import annotations
import re
from typing import Optional, Tuple, List
from .schemas import CanonicalIntent, LatencyTier
from .intents import INTENT_DEFINITIONS, get_intent_by_trigger_phrase
from .gates import is_obvious_chat


# =============================================================================
# TIER 0 RULE ENGINE
# =============================================================================

class Tier0RuleResult:
    """Result from Tier 0 rule matching."""

    def __init__(
        self,
        matched: bool,
        intent: Optional[CanonicalIntent] = None,
        confidence: float = 1.0,
        rule_name: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        self.matched = matched
        self.intent = intent
        self.confidence = confidence
        self.rule_name = rule_name
        self.reason = reason


# --- Trigger check functions (extracted to _tier0_trigger_checks.py) ---------
from app.translation._tier0_trigger_checks import (  # noqa: E402
    check_architecture_map_variant,
    check_zombie_start,
    check_update_architecture,
    check_scan_sandbox_trigger,
    check_weaver_trigger,
    check_spec_gate_trigger,
    check_critical_pipeline_trigger,
    check_overwatcher_trigger,
    check_segment_loop_trigger,
    check_codebase_report_trigger,
    check_latest_report_trigger,
    check_embedding_commands,
    check_project_scope_trigger,
)

# --- Filesystem, multi-file, refactor, chat patterns --------------------------
from app.translation._tier0_filesystem import (  # noqa: E402
    MULTI_FILE_SEARCH_PATTERNS,
    MULTI_FILE_REFACTOR_PATTERNS,
    MULTI_FILE_SCOPE_KEYWORDS,
    check_multi_file_trigger,
    check_refactor_codebase_trigger,
    check_filesystem_query_trigger,
    USER_CHAT_PATTERNS,
    is_user_chat_pattern,
    TAZISH_CHAT_PATTERNS,
    is_tazish_chat,
)

# --- Codebase questions (v10.0 — Job 10A) ------------------------------------
from app.translation._tier0_codebase_questions import (  # noqa: E402
    check_rag_codebase_query,
    check_natural_codebase_question,
    CODEBASE_GUARD_KEYWORDS,
)

# --- Web search (v2.1) -------------------------------------------------------
from app.translation._tier0_web_search import (  # noqa: E402
    check_web_search_trigger,
    check_deep_research_trigger,
)

# --- Project registry (v11.0) ------------------------------------------------
try:
    from app.project_registry.tier0_checks import (  # noqa: E402
        check_project_scan_trigger,
        check_register_project_trigger,
        check_list_projects_trigger,
    )
    _PROJECT_REGISTRY_CHECKS = True
except ImportError:
    _PROJECT_REGISTRY_CHECKS = False

# --- Memory ingest (v2.1) ----------------------------------------------------
from app.translation._tier0_memory_ingest import (  # noqa: E402
    check_memory_trigger,
)


# =============================================================================
# MAIN CLASSIFIER
# =============================================================================

def tier0_classify(text: str) -> Tier0RuleResult:
    """
    Attempt to classify intent using pure rules.
    Returns Tier0RuleResult with matched=True if successful.

    Order of checks:
    1. Exact trigger phrase match
    2. Pattern match against intent patterns
    3. Spec Gate flow special handlers
    4. Obvious chat short-circuit
    5. No match (needs Tier 1)
    """
    text_stripped = text.strip()

    # Strip wake phrase prefix
    text_stripped = re.sub(
        r'^(?:(?:astra|orb),?\s*)?(?:command:?\s*)?(?:feedback:?\s*)?',
        '', text_stripped, flags=re.IGNORECASE
    ).strip()

    # 1. Exact trigger phrase matches
    result = _check_exact_trigger_phrases(text_stripped)
    if result.matched:
        return result

    # 2. Pattern matches
    result = _check_trigger_patterns(text_stripped)
    if result.matched:
        return result

    # 2b. Domain intent patterns (v3.0 — unified cross-domain routing)
    # Run EARLY — before special-case handlers that might classify
    # domain queries as CHAT_ONLY.
    result = _check_domain_patterns(text_stripped)
    if result.matched:
        return result

    # 3. Special case handlers
    for check_fn in (
        check_update_architecture,
        check_scan_sandbox_trigger,
        check_architecture_map_variant,
        check_weaver_trigger,
        check_spec_gate_trigger,
        check_critical_pipeline_trigger,
        check_overwatcher_trigger,
        check_segment_loop_trigger,
    ):
        result = check_fn(text_stripped)
        if result.matched:
            return result

    # 3b. Project scoping
    result = check_project_scope_trigger(text_stripped)
    if result.matched:
        return result

    # 4. Data & query handlers
    result = check_rag_codebase_query(text_stripped)
    if result.matched:
        return result

    result = check_natural_codebase_question(text_stripped)
    if result.matched:
        return result

    result = check_embedding_commands(text_stripped)
    if result.matched:
        return result

    result = check_codebase_report_trigger(text_stripped)
    if result.matched:
        return result

    # 4e. Multi-file operations (BEFORE filesystem query!)
    result = check_multi_file_trigger(text_stripped)
    if result.matched:
        return result

    result = check_refactor_codebase_trigger(text_stripped)
    if result.matched:
        return result

    # 4f. Filesystem query (MUST be before is_obvious_chat!)
    result = check_filesystem_query_trigger(text)
    if result.matched:
        return result

    result = check_latest_report_trigger(text_stripped)
    if result.matched:
        return result

    # 4h. Web search & memory (v2.1)
    result = check_deep_research_trigger(text_stripped)
    if result.matched:
        return result

    result = check_web_search_trigger(text_stripped)
    if result.matched:
        return result

    result = check_memory_trigger(text_stripped)
    if result.matched:
        return result

    # 4j. Project registry commands (v11.0)
    if _PROJECT_REGISTRY_CHECKS:
        for check_fn in (
            check_project_scan_trigger,
            check_register_project_trigger,
            check_list_projects_trigger,
        ):
            result = check_fn(text_stripped)
            if result.matched:
                return result

    # 4i. Graduated confidence rules (v2.2)
    result = _check_graduated_rules(text_stripped)
    if result.matched:
        return result

    # 5. Obvious chat patterns
    is_chat, chat_reason = is_obvious_chat(text_stripped)
    if is_chat:
        return Tier0RuleResult(
            matched=True,
            intent=CanonicalIntent.CHAT_ONLY,
            confidence=0.95,
            rule_name="obvious_chat",
            reason=f"Obvious chat pattern: {chat_reason}",
        )

    # 6. No match — needs Tier 1 classifier
    return Tier0RuleResult(
        matched=False,
        intent=None,
        confidence=0.0,
        rule_name=None,
        reason="No rule match - requires Tier 1 classification",
    )


# =============================================================================
# GENERIC MATCHERS (used by tier0_classify)
# =============================================================================

def _check_exact_trigger_phrases(text: str) -> Tier0RuleResult:
    """Check for exact trigger phrase matches."""
    for intent, defn in INTENT_DEFINITIONS.items():
        for phrase in defn.trigger_phrases:
            if phrase.isupper():
                if text == phrase or text.startswith(phrase + " "):
                    return Tier0RuleResult(
                        matched=True, intent=intent, confidence=1.0,
                        rule_name="exact_trigger_phrase",
                        reason=f"Exact match: '{phrase}'",
                    )
            else:
                if text.lower() == phrase.lower() or text.lower().startswith(phrase.lower() + " "):
                    return Tier0RuleResult(
                        matched=True, intent=intent, confidence=1.0,
                        rule_name="exact_trigger_phrase",
                        reason=f"Exact match (case-insensitive): '{phrase}'",
                    )
    return Tier0RuleResult(matched=False)


# =============================================================================
# DOMAIN PATTERN MATCHING (v3.0 — unified cross-domain routing)
# =============================================================================

# Domain intents that should be checked via regex search (not match)
_DOMAIN_INTENT_IDS = {
    CanonicalIntent.DOMAIN_FINANCE,
    CanonicalIntent.DOMAIN_INVESTMENTS,
    CanonicalIntent.DOMAIN_CONTENT,
    CanonicalIntent.DOMAIN_SOCIAL,
    CanonicalIntent.DOMAIN_LIFESTYLE,
    CanonicalIntent.DOMAIN_DEBUG,
    CanonicalIntent.DOMAIN_EDUCATION,
    CanonicalIntent.DOMAIN_BUILDS,
}


def _check_domain_patterns(text: str) -> Tier0RuleResult:
    """Check for domain intent patterns using regex search.

    Domain patterns use word boundaries and need search() not match().
    Scores by number of pattern matches to pick the best domain
    when multiple could apply.
    """
    text_lower = text.lower()
    best_intent = None
    best_score = 0
    best_pattern = ""

    for intent_id in _DOMAIN_INTENT_IDS:
        defn = INTENT_DEFINITIONS.get(intent_id)
        if not defn:
            continue

        score = 0
        matched_pattern = ""

        # Check trigger phrases (substring match)
        for phrase in defn.trigger_phrases:
            if phrase.lower() in text_lower:
                score += 2
                matched_pattern = phrase

        # Check trigger patterns (regex search)
        for pattern_str in defn.trigger_patterns:
            try:
                if re.search(pattern_str, text, re.IGNORECASE):
                    score += 1
                    if not matched_pattern:
                        matched_pattern = pattern_str
            except re.error:
                continue

        if score > best_score:
            best_score = score
            best_intent = intent_id
            best_pattern = matched_pattern

    # Also check for generic navigation patterns: "go to X", "open X", "take me to X"
    # These should match the domain that X corresponds to
    nav_match = re.search(
        r'\b(?:go\s+to|open|show|move\s+to|switch\s+to|take\s+me\s+to|navigate\s+to)\s+'
        r'(investment|finance|account|content|social|health|fitness|debug|education|build|project\s+build)',
        text_lower,
    )
    if nav_match:
        nav_target = nav_match.group(1).strip()
        nav_map = {
            "investment": CanonicalIntent.DOMAIN_INVESTMENTS,
            "finance": CanonicalIntent.DOMAIN_FINANCE,
            "account": CanonicalIntent.DOMAIN_FINANCE,
            "content": CanonicalIntent.DOMAIN_CONTENT,
            "social": CanonicalIntent.DOMAIN_SOCIAL,
            "health": CanonicalIntent.DOMAIN_LIFESTYLE,
            "fitness": CanonicalIntent.DOMAIN_LIFESTYLE,
            "debug": CanonicalIntent.DOMAIN_DEBUG,
            "education": CanonicalIntent.DOMAIN_EDUCATION,
            "build": CanonicalIntent.DOMAIN_BUILDS,
            "project build": CanonicalIntent.DOMAIN_BUILDS,
        }
        matched_intent = nav_map.get(nav_target)
        if matched_intent:
            return Tier0RuleResult(
                matched=True,
                intent=matched_intent,
                confidence=0.95,
                rule_name="domain_navigation",
                reason=f"Navigation command: '{nav_match.group(0)}' -> {matched_intent.value}",
            )

    # v3.1: Require at least a score of 2 to trigger domain routing.
    # A single regex pattern match (score=1) is too easily triggered by
    # common English words like "share" (investments), "market" (investments),
    # "content" (content) in unrelated contexts.  Score of 2 means either
    # a trigger phrase matched (+2) or two separate regex patterns matched.
    if best_intent and best_score >= 2:
        return Tier0RuleResult(
            matched=True,
            intent=best_intent,
            confidence=min(0.95, 0.7 + best_score * 0.05),
            rule_name="domain_pattern",
            reason=f"Domain match: {best_intent.value} (score={best_score}, pattern='{best_pattern}')",
        )

    return Tier0RuleResult(matched=False)


def _check_trigger_patterns(text: str) -> Tier0RuleResult:
    """Check for regex pattern matches against intent patterns."""
    for intent, defn in INTENT_DEFINITIONS.items():
        for pattern_str in defn.trigger_patterns:
            try:
                pattern = re.compile(
                    pattern_str,
                    re.IGNORECASE if not _is_case_sensitive_pattern(pattern_str) else 0,
                )
                if pattern.match(text):
                    return Tier0RuleResult(
                        matched=True, intent=intent, confidence=0.98,
                        rule_name="trigger_pattern",
                        reason=f"Pattern match: '{pattern_str}'",
                    )
            except re.error:
                continue
    return Tier0RuleResult(matched=False)


def _is_case_sensitive_pattern(pattern: str) -> bool:
    """Determine if a pattern should be case-sensitive (has uppercase literals)."""
    in_class = False
    for char in pattern:
        if char == '[':
            in_class = True
        elif char == ']':
            in_class = False
        elif not in_class and char.isupper():
            return True
    return False


# =============================================================================
# GRADUATED CONFIDENCE RULES (v2.2)
# =============================================================================

def _check_graduated_rules(text: str) -> Tier0RuleResult:
    """Check input against graduated confidence rules promoted from Tier 1."""
    try:
        from app.translation.tier0_learned_rules import check_graduated_rules
        result = check_graduated_rules(text)
        if result.matched:
            intent_enum = None
            try:
                intent_enum = CanonicalIntent(result.intent)
            except ValueError:
                pass
            if intent_enum is not None:
                return Tier0RuleResult(
                    matched=True,
                    intent=intent_enum,
                    confidence=result.confidence,
                    rule_name=result.rule_name,
                    reason=result.reason,
                )
    except ImportError:
        pass
    except Exception:
        pass
    return Tier0RuleResult(matched=False)
