# FILE: app/translation/tier0_rules.py
"""
Tier 0: Pure rule-based intent classification.
No LLM calls. Fastest tier.

Handles:
- Wake phrases ("Astra, command:", "Astra, feedback:")
- Obvious capitalised commands ("CREATE ARCHITECTURE MAP")
- Obvious natural commands ("Start your zombie", "update architecture")
- Scan sandbox commands ("scan sandbox", "scan filesystem")
- Spec Gate flow commands ("How does that look all together", "Send to Spec Gate")
- Critical Pipeline commands ("run critical pipeline")
- Overwatcher commands ("send to overwatcher")
- Obvious chat patterns (questions, past tense, etc.)
- Filesystem READ queries ("what's written in", "read file", "show contents of")
- Filesystem command mode ("Astra, command: list/read/head/lines/find/append/overwrite/delete_area/delete_lines")

v5.7 (2026-01): Fixed corrupted regex in check_filesystem_query_trigger
  - Restored correct command mode detection regex
  - Removed duplicated code blocks
v5.5 (2026-01): Stage 1 deterministic file editing
  - Routes "Astra, command: append <path> \"content\"" to FILESYSTEM_QUERY
  - Routes "Astra, command: overwrite <path> \"content\"" to FILESYSTEM_QUERY
  - Routes "Astra, command: delete_area <path>" to FILESYSTEM_QUERY (ASTRA_BLOCK markers)
  - Routes "Astra, command: delete_lines <path> <start> <end>" to FILESYSTEM_QUERY
v5.0 (2026-01): Added explicit command mode for filesystem operations
  - Routes "Astra, command: list <path>" to FILESYSTEM_QUERY
  - Routes "Astra, command: read <path>" to FILESYSTEM_QUERY (supports quoted paths)
  - Routes "Astra, command: head <path> <n>" to FILESYSTEM_QUERY
  - Routes "Astra, command: lines <path> <start> <end>" to FILESYSTEM_QUERY
  - Routes "Astra, command: find <term> [under <path>]" to FILESYSTEM_QUERY
  - Added line range natural language detection ("show me line 45-65 of...")
v1.6 (2026-01): Added READ file patterns to check_filesystem_query_trigger
  - Routes "what's written in <path>" to FILESYSTEM_QUERY
  - Routes "read file <path>" to FILESYSTEM_QUERY
  - Routes "show contents of <path>" to FILESYSTEM_QUERY
  - Routes "view/display/cat <path>" to FILESYSTEM_QUERY
  - Handles apostrophe variants (straight ' and curly ')
v1.5 (2026-01): Added check_filesystem_query_trigger for list/find
v1.4 (2026-01): Added check_scan_sandbox_trigger, wired check_update_architecture
v1.3 (2026-01): Added check_overwatcher_trigger, wired check_critical_pipeline_trigger
v1.2 (2026-01): Fixed "critical architecture" -> SEND_TO_SPEC_GATE (was incorrectly in Weaver)
v1.1 (2026-01): Added Spec Gate flow handlers (WEAVER_BUILD_SPEC, SEND_TO_SPEC_GATE)
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
    
    # Strip wake phrase prefix if present (e.g., "Astra, command: X" -> "X")
    text_stripped = re.sub(
        r'^(?:astra,?\s*)?(?:command:?\s*)?(?:feedback:?\s*)?',
        '', text_stripped, flags=re.IGNORECASE
    ).strip()
    
    # 1. Check exact trigger phrase matches
    result = _check_exact_trigger_phrases(text_stripped)
    if result.matched:
        return result
    
    # 2. Check pattern matches
    result = _check_trigger_patterns(text_stripped)
    if result.matched:
        return result
    
    # 3. Check architecture update trigger
    result = check_update_architecture(text_stripped)
    if result.matched:
        return result
    
    # 3b. Check scan sandbox trigger
    result = check_scan_sandbox_trigger(text_stripped)
    if result.matched:
        return result
    
    # 3c. Check architecture map variants (ALL CAPS vs lowercase)
    result = check_architecture_map_variant(text_stripped)
    if result.matched:
        return result
    
    # 4. Check Spec Gate flow special handlers
    result = check_weaver_trigger(text_stripped)
    if result.matched:
        return result
    
    result = check_spec_gate_trigger(text_stripped)
    if result.matched:
        return result
    
    result = check_critical_pipeline_trigger(text_stripped)
    if result.matched:
        return result
    
    result = check_overwatcher_trigger(text_stripped)
    if result.matched:
        return result
    
    # 4b. Check RAG codebase query (v1.3)
    result = check_rag_codebase_query(text_stripped)
    if result.matched:
        return result
    
    # 4c. Check embedding commands (v1.3)
    result = check_embedding_commands(text_stripped)
    if result.matched:
        return result
    
    # 4d. Check codebase report (v1.5)
    result = check_codebase_report_trigger(text_stripped)
    if result.matched:
        return result
    
    # 4e. Check filesystem query (v1.5) - MUST be before is_obvious_chat!
    result = check_filesystem_query_trigger(text)
    if result.matched:
        return result
    
    # 5. Check obvious chat patterns (short-circuit)
    is_chat, chat_reason = is_obvious_chat(text_stripped)
    if is_chat:
        return Tier0RuleResult(
            matched=True,
            intent=CanonicalIntent.CHAT_ONLY,
            confidence=0.95,  # High but not 1.0 - classifier could override
            rule_name="obvious_chat",
            reason=f"Obvious chat pattern: {chat_reason}",
        )
    
    # 6. No match - needs Tier 1 classifier
    return Tier0RuleResult(
        matched=False,
        intent=None,
        confidence=0.0,
        rule_name=None,
        reason="No rule match - requires Tier 1 classification",
    )


def _check_exact_trigger_phrases(text: str) -> Tier0RuleResult:
    """
    Check for exact trigger phrase matches.
    These are high-confidence matches.
    """
    # Check each intent's trigger phrases
    for intent, defn in INTENT_DEFINITIONS.items():
        for phrase in defn.trigger_phrases:
            # Case-sensitive match for ALL CAPS phrases
            if phrase.isupper():
                if text == phrase or text.startswith(phrase + " "):
                    return Tier0RuleResult(
                        matched=True,
                        intent=intent,
                        confidence=1.0,
                        rule_name="exact_trigger_phrase",
                        reason=f"Exact match: '{phrase}'",
                    )
            else:
                # Case-insensitive for normal phrases
                if text.lower() == phrase.lower() or text.lower().startswith(phrase.lower() + " "):
                    return Tier0RuleResult(
                        matched=True,
                        intent=intent,
                        confidence=1.0,
                        rule_name="exact_trigger_phrase",
                        reason=f"Exact match (case-insensitive): '{phrase}'",
                    )
    
    return Tier0RuleResult(matched=False)


def _check_trigger_patterns(text: str) -> Tier0RuleResult:
    """
    Check for regex pattern matches against intent patterns.
    """
    for intent, defn in INTENT_DEFINITIONS.items():
        for pattern_str in defn.trigger_patterns:
            try:
                pattern = re.compile(pattern_str, re.IGNORECASE if not _is_case_sensitive_pattern(pattern_str) else 0)
                if pattern.match(text):
                    return Tier0RuleResult(
                        matched=True,
                        intent=intent,
                        confidence=0.98,  # Slightly less than exact match
                        rule_name="trigger_pattern",
                        reason=f"Pattern match: '{pattern_str}'",
                    )
            except re.error:
                continue  # Skip invalid patterns
    
    return Tier0RuleResult(matched=False)


def _is_case_sensitive_pattern(pattern: str) -> bool:
    """
    Determine if a pattern should be case-sensitive.
    Patterns with ALL CAPS literals are case-sensitive.
    """
    # Check if pattern has uppercase literals (not character classes)
    # Simple heuristic: if pattern contains uppercase letters outside []
    in_class = False
    has_upper = False
    for char in pattern:
        if char == '[':
            in_class = True
        elif char == ']':
            in_class = False
        elif not in_class and char.isupper():
            has_upper = True
            break
    return has_upper


# =============================================================================
# SPECIAL CASE HANDLERS
# =============================================================================

def check_architecture_map_variant(text: str) -> Tier0RuleResult:
    """
    Special handler for architecture map variants.
    - ALL CAPS "CREATE ARCHITECTURE MAP" -> with files
    - Normal case "Create architecture map" -> structure only
    """
    text_stripped = text.strip()
    
    # Check ALL CAPS variant
    if text_stripped.startswith("CREATE ARCHITECTURE MAP"):
        return Tier0RuleResult(
            matched=True,
            intent=CanonicalIntent.ARCHITECTURE_MAP_WITH_FILES,
            confidence=1.0,
            rule_name="archmap_all_caps",
            reason="ALL CAPS trigger for full architecture map with files",
        )
    
    # Check normal case variant
    if re.match(r"^[Cc]reate [Aa]rchitecture [Mm]ap", text_stripped):
        # Make sure it's NOT all caps
        if not text_stripped.startswith("CREATE ARCHITECTURE MAP"):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.ARCHITECTURE_MAP_STRUCTURE_ONLY,
                confidence=1.0,
                rule_name="archmap_normal_case",
                reason="Normal case trigger for structure-only architecture map",
            )
    
    return Tier0RuleResult(matched=False)


def check_zombie_start(text: str) -> Tier0RuleResult:
    """
    Special handler for zombie/sandbox start commands.
    """
    text_lower = text.strip().lower()
    
    patterns = [
        r"^start your zombie$",
        r"^start the zombie$",
        r"^launch zombie$",
        r"^spin up zombie$",
    ]
    
    for pattern in patterns:
        if re.match(pattern, text_lower):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.START_SANDBOX_ZOMBIE_SELF,
                confidence=1.0,
                rule_name="zombie_start",
                reason="Zombie start command detected",
            )
    
    return Tier0RuleResult(matched=False)


def check_update_architecture(text: str) -> Tier0RuleResult:
    """
    Special handler for architecture update commands.
    """
    text_lower = text.strip().lower()
    
    patterns = [
        r"^update architecture$",
        r"^update your architecture$",
        r"^refresh architecture$",
        r"^update code atlas$",
        r"^refresh code atlas$",
    ]
    
    for pattern in patterns:
        if re.match(pattern, text_lower):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.ARCHITECTURE_UPDATE_ATLAS_ONLY,
                confidence=1.0,
                rule_name="update_architecture",
                reason="Architecture update command detected",
            )
    
    return Tier0RuleResult(matched=False)


def check_scan_sandbox_trigger(text: str) -> Tier0RuleResult:
    """
    Special handler for sandbox scanning commands.
    
    Scans the sandbox filesystem structure via sandbox controller.
    Does NOT scan host PC - only sandbox environment.
    """
    text_lower = text.strip().lower()
    
    patterns = [
        r"^scan sandbox$",
        r"^scan the sandbox$",
        r"^sandbox scan$",
        r"^scan sandbox structure$",
        r"^scan sandbox filesystem$",
        r"^scan filesystem$",
        r"^filesystem scan$",
    ]
    
    for pattern in patterns:
        if re.match(pattern, text_lower):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.SCAN_SANDBOX_STRUCTURE,
                confidence=1.0,
                rule_name="scan_sandbox",
                reason="Sandbox scan command detected",
            )
    
    return Tier0RuleResult(matched=False)


# =============================================================================
# SPEC GATE FLOW HANDLERS (v1.1)
# =============================================================================

def check_weaver_trigger(text: str) -> Tier0RuleResult:
    """
    Special handler for Weaver (spec building) triggers.
    
    Triggers Weaver to consolidate ramble/conversation into candidate spec.
    Natural language triggers that indicate "put it all together".
    """
    text_lower = text.strip().lower()
    text_stripped = text.strip()
    
    # Exact phrase matches (case-insensitive)
    # NOTE: "critical architecture" moved to check_spec_gate_trigger (v1.2)
    exact_phrases = [
        "how does that look all together",
        "how does that look all together?",
        "weave this into a spec",
        "weave that into a spec",
        "build spec from ramble",
        "compile the spec",
        "put that all together",
        "put this all together",
        "put it all together",
        "consolidate that into a spec",
        "consolidate this into a spec",
        "summarize the ramble into a spec",
        "summarize my ramble into a spec",
        "turn this into a spec",
        "turn that into a spec",
    ]
    
    for phrase in exact_phrases:
        if text_lower == phrase or text_lower.rstrip("?") == phrase.rstrip("?"):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.WEAVER_BUILD_SPEC,
                confidence=1.0,
                rule_name="weaver_exact_phrase",
                reason=f"Weaver trigger: '{phrase}'",
            )
    
    # Pattern matches
    weaver_patterns = [
        r"^how does (?:that|this|it) (?:all )?look(?: all together)?\??$",
        r"^weave (?:this|that|it) into a spec$",
        r"^build (?:a )?spec from (?:the )?ramble$",
        r"^compile (?:the )?spec$",
        r"^put (?:that|this|it) all together$",
        r"^consolidate (?:that|this|it) into a spec$",
        r"^summarize (?:the|my) ramble into a spec$",
        r"^turn (?:this|that|it) into a spec$",
        r"^okay,? (?:now )?(?:weave|build|compile|consolidate)",
        r"^ok,? (?:now )?(?:weave|build|compile|consolidate)",
    ]
    
    for pattern in weaver_patterns:
        if re.match(pattern, text_lower):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.WEAVER_BUILD_SPEC,
                confidence=0.98,
                rule_name="weaver_pattern",
                reason=f"Weaver trigger pattern: '{pattern}'",
            )
    
    return Tier0RuleResult(matched=False)


def check_spec_gate_trigger(text: str) -> Tier0RuleResult:
    """
    Special handler for Spec Gate (validation) triggers.
    
    Sends refined candidate spec to Spec Gate for validation.
    Explicit "send to spec gate" type commands.
    Also handles simple "Yes" confirmations after Weaver prompt.
    """
    text_lower = text.strip().lower()
    
    # Simple affirmative responses (after Weaver asks "Shall I send to Spec Gate?")
    # These are short, so we match exactly
    simple_affirmatives = [
        "yes",
        "yes please",
        "yes, please",
        "yep",
        "yeah",
        "sure",
        "go ahead",
        "do it",
        "proceed",
        "send it",
        "ok",
        "okay",
        "affirmative",
        "confirmed",
        "confirm",
        "y",
    ]
    
    if text_lower in simple_affirmatives:
        return Tier0RuleResult(
            matched=True,
            intent=CanonicalIntent.SEND_TO_SPEC_GATE,
            confidence=0.95,  # Slightly lower - relies on context
            rule_name="specgate_affirmative",
            reason=f"Spec Gate affirmative confirmation: '{text_lower}'",
        )
    
    # Exact phrase matches (case-insensitive)
    # "critical architecture" triggers Spec Gate validation (v1.2)
    exact_phrases = [
        "critical architecture",
        "send to spec gate",
        "send that to spec gate",
        "send this to spec gate",
        "send it to spec gate",
        "okay, send that to spec gate",
        "okay, send to spec gate",
        "ok, send that to spec gate",
        "ok, send to spec gate",
        "validate the spec",
        "validate spec",
        "run spec gate",
        "submit spec for validation",
        "submit the spec",
        "spec gate validate",
        "specgate validate",
    ]
    
    for phrase in exact_phrases:
        if text_lower == phrase or text_lower.startswith(phrase):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.SEND_TO_SPEC_GATE,
                confidence=1.0,
                rule_name="specgate_exact_phrase",
                reason=f"Spec Gate trigger: '{phrase}'",
            )
    
    # Pattern matches
    specgate_patterns = [
        r"^(?:ok(?:ay)?,?\s*)?send (?:that|this|it) to spec ?gate$",
        r"^send to spec ?gate$",
        r"^validate (?:the )?spec$",
        r"^run spec ?gate$",
        r"^submit (?:the )?spec(?: for validation)?$",
        r"^spec ?gate[,:]?\s*validate$",
        r"^(?:now )?send (?:it )?to spec ?gate$",
        r"^(?:go ahead and )?send (?:it )?to spec ?gate$",
        r"^critical architecture$",
    ]
    
    for pattern in specgate_patterns:
        if re.match(pattern, text_lower):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.SEND_TO_SPEC_GATE,
                confidence=0.98,
                rule_name="specgate_pattern",
                reason=f"Spec Gate trigger pattern: '{pattern}'",
            )
    
    return Tier0RuleResult(matched=False)


def check_critical_pipeline_trigger(text: str) -> Tier0RuleResult:
    """
    Special handler for critical pipeline triggers.
    
    Requires validated spec from Spec Gate before execution.
    High-stakes - requires confirmation.
    """
    text_lower = text.strip().lower()
    
    patterns = [
        r"^run (?:the )?critical pipeline$",
        r"^execute (?:the )?critical pipeline$",
        r"^start the pipeline$",
        r"^run (?:the )?critical pipeline for job\s+",
        r"^execute (?:the )?pipeline$",
        r"^critical pipeline$",
    ]
    
    for pattern in patterns:
        if re.match(pattern, text_lower):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB,
                confidence=1.0,
                rule_name="critical_pipeline",
                reason="Critical pipeline command detected",
            )
    
    return Tier0RuleResult(matched=False)


def check_overwatcher_trigger(text: str) -> Tier0RuleResult:
    """
    Special handler for Overwatcher execution triggers.
    
    Final stage: Overwatcher implements the approved changes.
    High-stakes - requires confirmation.
    """
    text_lower = text.strip().lower()
    
    patterns = [
        r"^send to overwatcher$",
        r"^send (?:it |that )?to overwatcher$",
        r"^overwatcher execute$",
        r"^run overwatcher$",
        r"^execute overwatcher$",
        r"^overwatcher$",
        r"^(?:ok(?:ay)?,?\s*)?send (?:it |that )?to overwatcher$",
    ]
    
    for pattern in patterns:
        if re.match(pattern, text_lower):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES,
                confidence=1.0,
                rule_name="overwatcher_execute",
                reason="Overwatcher execute command detected",
            )
    
    return Tier0RuleResult(matched=False)


def check_rag_codebase_query(text: str) -> Tier0RuleResult:
    """
    Special handler for RAG codebase queries.
    
    Searches indexed codebase and answers questions.
    v1.5: Fixed patterns to match trailing punctuation (?, !)
    v1.4: Expanded patterns to catch architecture/codebase questions automatically.
    v1.3: Added RAG query support.
    """
    text_stripped = text.strip()
    text_lower = text_stripped.lower()
    
    # Remove trailing punctuation for pattern matching
    text_clean = text_lower.rstrip('?.!')
    
    # Index commands
    index_patterns = [
        r"^index\s+(?:the\s+)?(?:architecture|codebase|rag)$",
        r"^run\s+rag\s+index$",
    ]
    
    for pattern in index_patterns:
        if re.match(pattern, text_clean):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.RAG_CODEBASE_QUERY,
                confidence=1.0,
                rule_name="rag_index",
                reason="RAG index command detected",
            )
    
    # Explicit search queries (highest confidence)
    explicit_search_patterns = [
        r"^search\s+(?:the\s+)?codebase:\s*.+",
        r"^ask\s+about\s+(?:the\s+)?codebase:\s*.+",
        r"^codebase\s+(?:search|query):\s*.+",
        r"^in\s+(?:the|this)\s+codebase,?\s+.+",
    ]
    
    for pattern in explicit_search_patterns:
        if re.match(pattern, text_lower):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.RAG_CODEBASE_QUERY,
                confidence=1.0,
                rule_name="rag_explicit_search",
                reason="RAG codebase search detected",
            )
    
    # ==========================================================================
    # Architecture-related questions (auto-route to RAG)
    # These patterns catch questions about the codebase structure/architecture
    # Patterns match against text_clean (no trailing punctuation)
    # ==========================================================================
    
    architecture_question_patterns = [
        # Entry points / main flow
        r"^what\s+(?:are|is)\s+(?:the\s+)?(?:main\s+)?entry\s*points?$",
        r"^where\s+(?:are|is)\s+(?:the\s+)?(?:main\s+)?entry\s*points?$",
        r"^show\s+(?:me\s+)?(?:the\s+)?entry\s*points?$",
        r"^(?:list|find)\s+(?:the\s+)?entry\s*points?$",
        
        # Bottlenecks / performance
        r"^what\s+(?:are|is)\s+(?:the\s+)?(?:potential\s+)?bottlenecks?$",
        r"^where\s+(?:are|is)\s+(?:the\s+)?bottlenecks?$",
        r"^(?:find|identify|show)\s+(?:the\s+)?bottlenecks?$",
        r"^bottlenecks$",  # Single word
        
        # Coupling / dependencies
        r"^(?:which|what)\s+modules?\s+(?:are|is)\s+(?:tightly\s+)?coupled",
        r"^where\s+(?:is|are)\s+(?:the\s+)?tight\s+coupling",
        r"^show\s+(?:me\s+)?(?:the\s+)?dependencies",
        
        # Where should X live / feature placement
        r"^where\s+should\s+(?:a\s+)?(?:new\s+)?(?:feature|functionality|code|module|component)\s+.+\s+(?:live|go)",
        r"^where\s+should\s+I\s+(?:put|add|place|implement)\s+.+",
        r"^where\s+does\s+.+\s+(?:belong|go|fit)",
        
        # What functions/classes handle X
        r"^what\s+(?:function|class|method|module|file)s?\s+(?:handle|process|manage|control)s?\s+.+",
        r"^(?:which|what)\s+(?:function|class|method|module|file)s?\s+(?:are\s+)?responsible\s+for\s+.+",
        r"^(?:which|what)\s+(?:function|class|method|module|file)s?\s+(?:do|does)\s+.+",
        
        # Where is X implemented/defined/located
        r"^where\s+is\s+.+\s+(?:implemented|defined|located|declared|found)",
        r"^find\s+(?:the\s+)?(?:implementation|definition|location)\s+of\s+.+",
        r"^(?:locate|find)\s+.+\s+(?:in\s+the\s+codebase|code)",
        
        # How does X work / flow
        r"^how\s+does\s+(?:the\s+)?(?:routing|streaming|pipeline|job|auth|memory|rag)\s+(?:work|function|flow)",
        r"^explain\s+(?:the\s+)?(?:routing|streaming|pipeline|job|auth|memory|rag)\s+(?:flow|system|architecture)",
        
        # Optimization / improvement
        r"^how\s+(?:would|should|could)\s+(?:you|I|we)\s+optimize\s+.+",
        r"^(?:suggest|recommend)\s+(?:optimizations?|improvements?)\s+(?:for|to)\s+.+",
        
        # Code structure questions
        r"^what\s+is\s+the\s+(?:purpose|role|responsibility)\s+of\s+.+",
        r"^what\s+does\s+(?:the\s+)?(?:file|module|class|function)\s+.+\s+do",
        r"^describe\s+(?:the\s+)?(?:file|module|class|function|component)\s+.+",
        
        # List/show components
        r"^(?:list|show|display)\s+(?:all\s+)?(?:the\s+)?(?:modules?|components?|services?|handlers?|routers?)$",
        r"^what\s+(?:modules?|components?|services?)\s+(?:exist|are\s+there)",
    ]
    
    for pattern in architecture_question_patterns:
        if re.match(pattern, text_clean):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.RAG_CODEBASE_QUERY,
                confidence=0.95,  # High but slightly below explicit commands
                rule_name="rag_architecture_question",
                reason=f"Architecture question detected: '{text_clean}'",
            )
    
    return Tier0RuleResult(matched=False)


def check_codebase_report_trigger(text: str) -> Tier0RuleResult:
    """
    Special handler for codebase report commands.
    
    Commands:
    - "codebase report fast" → Quick metadata scan
    - "codebase report full" → Deep content scan
    
    v1.5: Added codebase report support.
    """
    text_lower = text.strip().lower()
    
    # Codebase report patterns
    codebase_report_patterns = [
        r"^codebase\s+report\s+fast$",
        r"^codebase\s+report\s+full$",
        r"^generate\s+codebase\s+report$",
        r"^run\s+codebase\s+report$",
    ]
    
    for pattern in codebase_report_patterns:
        if re.match(pattern, text_lower):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.CODEBASE_REPORT,
                confidence=1.0,
                rule_name="codebase_report",
                reason="Codebase report command detected",
            )
    
    return Tier0RuleResult(matched=False)


def check_embedding_commands(text: str) -> Tier0RuleResult:
    """
    Special handler for embedding management commands.
    
    Commands:
    - "embedding status" → Check status
    - "generate embeddings" → Trigger job
    
    v1.3: Added embedding command support.
    """
    text_lower = text.strip().lower()
    
    # Status commands
    status_patterns = [
        r"^embedding[s]?\s+status$",
        r"^check\s+embedding[s]?$",
        r"^embedding[s]?\s+progress$",
        r"^how\s+are\s+embeddings\s+doing$",
    ]
    
    for pattern in status_patterns:
        if re.match(pattern, text_lower):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.EMBEDDING_STATUS,
                confidence=1.0,
                rule_name="embedding_status",
                reason="Embedding status command detected",
            )
    
    # Generate commands
    generate_patterns = [
        r"^generate\s+embedding[s]?$",
        r"^run\s+embedding[s]?$",
        r"^start\s+embedding[s]?$",
        r"^embed\s+(?:the\s+)?(?:code\s+)?chunks$",
    ]
    
    for pattern in generate_patterns:
        if re.match(pattern, text_lower):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.GENERATE_EMBEDDINGS,
                confidence=1.0,
                rule_name="generate_embeddings",
                reason="Generate embeddings command detected",
            )
    
    return Tier0RuleResult(matched=False)


# =============================================================================
# FILESYSTEM QUERY HANDLER (v5.7 - FIXED)
# =============================================================================

# Known user folder keywords (for queries that don't have explicit paths)
_KNOWN_FOLDER_KEYWORDS = {
    "desktop", "onedrive", "documents", "downloads", 
    "pictures", "videos", "music", "appdata",
}

# Allowed scan roots - reject queries outside these
_ALLOWED_FS_ROOTS = [
    r"D:\\",
    r"C:\\Users\\dizzi",
]


def _has_windows_path(text: str) -> bool:
    """Check if text contains a Windows path like C:\\ or D:\\."""
    return bool(re.search(r'[A-Za-z]:[/\\]', text))


def _has_known_folder_keyword(text: str) -> bool:
    """Check if text contains a known user folder keyword."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in _KNOWN_FOLDER_KEYWORDS)


def _is_within_allowed_roots(text: str) -> bool:
    """
    Check if query references a path within allowed roots.
    
    Allowed: D:\\ and C:\\Users\\dizzi
    Reject: C:\\Windows, C:\\Program Files, etc.
    """
    text_lower = text.lower()
    
    # If contains C:\ but NOT C:\Users\dizzi, reject
    if re.search(r'c:[/\\]', text_lower):
        if not re.search(r'c:[/\\]users[/\\]dizzi', text_lower):
            return False
    
    # D:\ is always allowed
    # Known folder keywords (Desktop, OneDrive) are under C:\Users\dizzi
    return True


def check_filesystem_query_trigger(text: str) -> Tier0RuleResult:
    """
    Special handler for filesystem listing/search queries (v5.7 - FIXED).
    
    v5.7 (2026-01): Fixed corrupted regex, removed duplicated code blocks
    
    v5.6 (2026-01): Fixed to handle text with command prefix already stripped
      - Also matches raw verbs without "command:" prefix
      - Handles case where translator.py strips prefix before calling
    
    v5.0: Added explicit command mode support:
    - "Astra, command: list <path>"
    - "Astra, command: read <path>" or read "<path with spaces>"
    - "Astra, command: head <path> <n>"
    - "Astra, command: lines <path> <start> <end>"
    - "Astra, command: find <term> [under <path>]"
    
    Natural language queries:
    - "List everything on C:\\Users\\dizzi\\Desktop"
    - "What's in C:\\Users\\dizzi\\OneDrive"
    - "Find folder named Jobs under C:\\Users\\dizzi"
    - "Show me line 45-65 of stream_router.py"
    - "First 10 lines of main.py"
    
    Requirements:
    - Must have Windows path OR known folder keyword
    - Must be within allowed roots
    """
    text_stripped = text.strip()
    
    # FS command verbs - used in multiple checks below
    fs_command_verbs = [
        r'^list\s+',           # list <path>
        r'^read\s+',           # read <path> or read "<path>"
        r'^head\s+',           # head <path> [n]
        r'^lines\s+',          # lines <path> <start> <end>
        r'^find\s+',           # find <term> [under <path>]
        # Stage 1 write commands (v5.5)
        r'^append\s+',         # append <path> "content" or append <path> + fenced block
        r'^overwrite\s+',      # overwrite <path> "content" or overwrite <path> + fenced block
        r'^delete_area\s+',    # delete_area <path> (uses ASTRA_BLOCK markers)
        r'^delete_lines\s+',   # delete_lines <path> <start> <end>
    ]
    
    # ==========================================================================
    # v5.6: DIRECT COMMAND VERB DETECTION (highest priority)
    # When text is already stripped of "Astra, command:" prefix, it starts
    # directly with the verb like "append path content" or "read path"
    # ==========================================================================
    text_lower = text_stripped.lower()
    for pattern in fs_command_verbs:
        if re.match(pattern, text_lower):
            verb = text_lower.split()[0]
            print(f"[FILESYSTEM_QUERY] Direct verb detected: {text_stripped[:80]}")
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.FILESYSTEM_QUERY,
                confidence=1.0,
                rule_name="filesystem_direct_verb",
                reason=f"Filesystem command (direct verb): {verb}",
            )
    
    # ==========================================================================
    # v5.7: EXPLICIT COMMAND MODE DETECTION (with prefix) - FIXED REGEX
    # Patterns: "Astra, command: <cmd> <args>" or "command: <cmd> <args>"
    # ==========================================================================
    
    # Check for command mode prefix
    command_match = re.match(
        r'^(?:astra,?\s*)?command:?\s*(.+)$',
        text_stripped, re.IGNORECASE
    )
    
    if command_match:
        cmd_text = command_match.group(1).strip()
        cmd_lower = cmd_text.lower()
        
        # Check for FS commands
        for pattern in fs_command_verbs:
            if re.match(pattern, cmd_lower):
                print(f"[FILESYSTEM_QUERY] Command mode detected: {cmd_text[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=1.0,
                    rule_name="filesystem_command_mode",
                    reason=f"Filesystem command: {cmd_text.split()[0]}",
                )
    
    # ==========================================================================
    # v5.0: LINE RANGE QUERIES (natural language)
    # "Show me line 45-65 of...", "What's on lines 10-20..."
    # ==========================================================================
    
    line_range_patterns = [
        r'lines?\s+\d+\s*[-\u2013\u2014to]+\s*\d+\s+(?:of|in|from)\s+',
        r'(?:show|what\'?s)\s+(?:me\s+)?(?:on\s+)?lines?\s+\d+',
        r'first\s+\d+\s+lines?\s+(?:of|in|from)\s+',
        r'head\s+\d+\s+(?:of|in|from)\s+',
    ]
    
    text_lower_raw = text_stripped.lower()
    for pattern in line_range_patterns:
        if re.search(pattern, text_lower_raw):
            # Must have a path reference
            if _has_windows_path(text_stripped) or _has_known_folder_keyword(text_lower_raw):
                print(f"[FILESYSTEM_QUERY] Line range query: {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=1.0,
                    rule_name="filesystem_line_range",
                    reason="Filesystem line range query",
                )
    
    # Strip optional "After scan sandbox, " prefix
    text_for_match = re.sub(
        r'^[Aa]fter\s+scan\s+sandbox,?\s*',
        '', text_stripped
    ).strip()
    
    text_lower = text_for_match.lower()
    
    # ==========================================================================
    # TIGHT PATTERNS - each requires Windows path context
    # ==========================================================================
    
    # Pattern 1: "List everything/all/contents/files/folders on/in/at <path>"
    # Requires Windows path
    list_with_path_patterns = [
        r"^list\s+(?:everything|all|contents?|files?(?:\s+and\s+folders?)?|folders?(?:\s+and\s+files?)?|top[- ]?level\s+folders?)\s+(?:on|in|at|under|inside)\s+[A-Za-z]:[/\\]",
    ]
    
    for pattern in list_with_path_patterns:
        if re.match(pattern, text_lower):
            if _is_within_allowed_roots(text_lower):
                print(f"[FILESYSTEM_QUERY] Detected list with path: {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=1.0,
                    rule_name="filesystem_list_with_path",
                    reason=f"Filesystem list query with Windows path",
                )
    
    # Pattern 2: "What's in / What is in <path>"
    # Requires Windows path
    whats_in_patterns = [
        r"^what(?:'s|\s+is)\s+(?:in|on|at|inside)\s+[A-Za-z]:[/\\]",
    ]
    
    for pattern in whats_in_patterns:
        if re.match(pattern, text_lower):
            if _is_within_allowed_roots(text_lower):
                print(f"[FILESYSTEM_QUERY] Detected what's in: {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=1.0,
                    rule_name="filesystem_whats_in",
                    reason=f"Filesystem 'what's in' query with Windows path",
                )
    
    # Pattern 3: "Show me everything/contents in <path>"
    # Requires Windows path
    show_patterns = [
        r"^show\s+(?:me\s+)?(?:everything|all|contents?|files?|folders?)\s+(?:in|on|at|under|inside)\s+[A-Za-z]:[/\\]",
    ]
    
    for pattern in show_patterns:
        if re.match(pattern, text_lower):
            if _is_within_allowed_roots(text_lower):
                print(f"[FILESYSTEM_QUERY] Detected show: {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=1.0,
                    rule_name="filesystem_show",
                    reason=f"Filesystem show query with Windows path",
                )
    
    # Pattern 4: "Contents of <path>"
    # Requires Windows path
    contents_of_patterns = [
        r"^contents?\s+of\s+[A-Za-z]:[/\\]",
    ]
    
    for pattern in contents_of_patterns:
        if re.match(pattern, text_lower):
            if _is_within_allowed_roots(text_lower):
                print(f"[FILESYSTEM_QUERY] Detected contents of: {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=1.0,
                    rule_name="filesystem_contents_of",
                    reason=f"Filesystem 'contents of' query with Windows path",
                )
    
    # Pattern 5: "Find folder/file/directory named X under/in <path>"
    # Requires Windows path
    find_named_with_path_patterns = [
        r"^find\s+(?:folder|file|directory)\s+(?:named?\s+)?[\w\s]+\s+(?:under|in|on|inside)\s+[A-Za-z]:[/\\]",
    ]
    
    for pattern in find_named_with_path_patterns:
        if re.match(pattern, text_lower):
            if _is_within_allowed_roots(text_lower):
                print(f"[FILESYSTEM_QUERY] Detected find named with path: {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=1.0,
                    rule_name="filesystem_find_named_path",
                    reason=f"Filesystem find query with Windows path",
                )
    
    # Pattern 6: "Find X under/in <known folder>"
    # Requires known folder keyword (Desktop, OneDrive, etc.)
    find_under_folder_patterns = [
        r"^find\s+[\w\s]+\s+(?:under|in|inside|on)\s+(?:my\s+)?(?:desktop|onedrive|documents|downloads)",
    ]
    
    for pattern in find_under_folder_patterns:
        if re.match(pattern, text_lower):
            print(f"[FILESYSTEM_QUERY] Detected find under known folder: {text_stripped[:80]}")
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.FILESYSTEM_QUERY,
                confidence=0.98,
                rule_name="filesystem_find_under_folder",
                reason=f"Filesystem find query with known folder keyword",
            )
    
    # Pattern 7: "Find files with X in the name" 
    # ONLY if also has Windows path OR known folder keyword
    find_files_with_patterns = [
        r"^find\s+files?\s+(?:with|named|containing)\s+.+\s+(?:in\s+(?:the\s+)?name|in\s+their\s+name)",
    ]
    
    for pattern in find_files_with_patterns:
        if re.match(pattern, text_lower):
            # Must have path OR folder keyword to avoid false positives
            if _has_windows_path(text_lower) or _has_known_folder_keyword(text_lower):
                if _is_within_allowed_roots(text_lower):
                    print(f"[FILESYSTEM_QUERY] Detected find files with pattern: {text_stripped[:80]}")
                    return Tier0RuleResult(
                        matched=True,
                        intent=CanonicalIntent.FILESYSTEM_QUERY,
                        confidence=0.95,
                        rule_name="filesystem_find_files_pattern",
                        reason=f"Filesystem find files query",
                    )
    
    # Pattern 8: Generic "find files with X" - requires Windows path explicitly in query
    generic_find_with_path = [
        r"^find\s+(?:all\s+)?files?\s+(?:with|containing|named)\s+.+\s+(?:under|in|on|inside)\s+[A-Za-z]:[/\\]",
    ]
    
    for pattern in generic_find_with_path:
        if re.match(pattern, text_lower):
            if _is_within_allowed_roots(text_lower):
                print(f"[FILESYSTEM_QUERY] Detected generic find with path: {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=1.0,
                    rule_name="filesystem_generic_find_path",
                    reason=f"Filesystem find query with explicit path",
                )
    
    # Pattern 9: List with known folder keyword (no explicit path)
    # "List everything in my Desktop", "List folders in OneDrive"
    list_with_folder_keyword = [
        r"^list\s+(?:everything|all|contents?|files?(?:\s+and\s+folders?)?|folders?(?:\s+and\s+files?)?|top[- ]?level\s+folders?)\s+(?:on|in|at|under|inside)\s+(?:my\s+)?(?:desktop|onedrive|documents|downloads)",
    ]
    
    for pattern in list_with_folder_keyword:
        if re.match(pattern, text_lower):
            print(f"[FILESYSTEM_QUERY] Detected list with folder keyword: {text_stripped[:80]}")
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.FILESYSTEM_QUERY,
                confidence=0.98,
                rule_name="filesystem_list_folder_keyword",
                reason=f"Filesystem list query with known folder keyword",
            )
    
    # Pattern 10: "What's in my Desktop/OneDrive" (no explicit path)
    whats_in_folder_keyword = [
        r"^what(?:'s|\s+is)\s+(?:in|on)\s+(?:my\s+)?(?:desktop|onedrive|documents|downloads)",
    ]
    
    for pattern in whats_in_folder_keyword:
        if re.match(pattern, text_lower):
            print(f"[FILESYSTEM_QUERY] Detected what's in folder keyword: {text_stripped[:80]}")
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.FILESYSTEM_QUERY,
                confidence=0.98,
                rule_name="filesystem_whats_in_folder",
                reason=f"Filesystem 'what's in' query with known folder keyword",
            )
    
    # ==========================================================================
    # v1.6: READ FILE PATTERNS (from zobie_tools.py _parse_filesystem_query)
    # Routes file read queries to FILESYSTEM_QUERY for DB-first content lookup
    # Handles apostrophe variants: straight ' and curly '
    # ==========================================================================
    
    # Normalize curly apostrophes to straight for consistent matching
    text_normalized = text_lower.replace("'", "'").replace("'", "'")
    
    # Pattern 11: "what's written in <path>" / "whats written in <path>" / "whats inside <path>"
    # Requires Windows path
    whats_written_patterns = [
        r"^what'?s\s+(?:written|inside)\s+(?:in\s+)?[A-Za-z]:[/\\]",
    ]
    
    for pattern in whats_written_patterns:
        if re.match(pattern, text_normalized):
            if _is_within_allowed_roots(text_lower):
                print(f"[FILESYSTEM_QUERY] Detected READ (what's written): {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=1.0,
                    rule_name="filesystem_read_whats_written",
                    reason="Filesystem READ query: 'what's written in'",
                )
    
    # Pattern 12: "read <path>" / "read file <path>" / "read the file <path>"
    # Requires Windows path
    read_file_patterns = [
        r"^read\s+(?:the\s+)?(?:file\s+)?[A-Za-z]:[/\\]",
    ]
    
    for pattern in read_file_patterns:
        if re.match(pattern, text_normalized):
            if _is_within_allowed_roots(text_lower):
                print(f"[FILESYSTEM_QUERY] Detected READ (read file): {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=1.0,
                    rule_name="filesystem_read_file",
                    reason="Filesystem READ query: 'read file'",
                )
    
    # Pattern 13: "show contents of <path>" / "show the contents of <path>"
    # Requires Windows path
    show_contents_of_patterns = [
        r"^show\s+(?:the\s+)?contents?\s+of\s+[A-Za-z]:[/\\]",
    ]
    
    for pattern in show_contents_of_patterns:
        if re.match(pattern, text_normalized):
            if _is_within_allowed_roots(text_lower):
                print(f"[FILESYSTEM_QUERY] Detected READ (show contents of): {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=1.0,
                    rule_name="filesystem_read_show_contents",
                    reason="Filesystem READ query: 'show contents of'",
                )
    
    # Pattern 14: "view/display/cat/open <path>" (Unix-style + generic)
    # Requires Windows path
    view_display_patterns = [
        r"^(?:view|display|cat|output|print)\s+(?:the\s+)?(?:file\s+)?[A-Za-z]:[/\\]",
    ]
    
    for pattern in view_display_patterns:
        if re.match(pattern, text_normalized):
            if _is_within_allowed_roots(text_lower):
                print(f"[FILESYSTEM_QUERY] Detected READ (view/display/cat): {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=1.0,
                    rule_name="filesystem_read_view_display",
                    reason="Filesystem READ query: 'view/display/cat'",
                )
    
    # Pattern 15: "open <path>" - only if path looks like a file (has extension)
    # Requires Windows path with file extension
    open_file_patterns = [
        r"^open\s+(?:the\s+)?(?:file\s+)?[A-Za-z]:[/\\].+\.\w+",
    ]
    
    for pattern in open_file_patterns:
        if re.match(pattern, text_normalized):
            if _is_within_allowed_roots(text_lower):
                print(f"[FILESYSTEM_QUERY] Detected READ (open file): {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=0.98,
                    rule_name="filesystem_read_open",
                    reason="Filesystem READ query: 'open file'",
                )
    
    # Pattern 16: "what does <path> say/contain" 
    # Requires Windows path
    what_does_say_patterns = [
        r"^what\s+does\s+[A-Za-z]:[/\\].+\s+(?:say|contain)",
    ]
    
    for pattern in what_does_say_patterns:
        if re.match(pattern, text_normalized):
            if _is_within_allowed_roots(text_lower):
                print(f"[FILESYSTEM_QUERY] Detected READ (what does say/contain): {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=0.98,
                    rule_name="filesystem_read_what_does",
                    reason="Filesystem READ query: 'what does X say/contain'",
                )
    
    return Tier0RuleResult(matched=False)


# =============================================================================
# USER CHAT PATTERN LIBRARY
# =============================================================================
# Common user chat patterns that should be learned over time.
# These are seed patterns that can be augmented by the phrase cache.

USER_CHAT_PATTERNS = [
    # Exploratory questions about the system
    r"tell me (?:about|more about)",
    r"what (?:is|are) your",
    r"describe your",
    r"explain (?:your|the|how)",
    r"show me (?:your|the|how)",
    r"how does (?:your|the)",
    r"what does (?:your|the|this|that|it)",
    
    # Casual conversation
    r"^(?:hi|hello|hey|yo|sup)",
    r"^(?:thanks|thank you|cheers)",
    r"^(?:ok|okay|sure|got it|understood)",
    r"^(?:hmm|huh|interesting|cool|nice)",
    
    # Questions about capabilities
    r"can you (?:tell|show|explain)",
    r"do you (?:have|know|understand)",
    r"are you (?:able|capable)",
]

# Compiled for efficiency
_COMPILED_USER_CHAT = [re.compile(p, re.IGNORECASE) for p in USER_CHAT_PATTERNS]


def is_user_chat_pattern(text: str) -> bool:
    """
    Check if text matches known user chat patterns.
    Used for additional chat short-circuiting.
    """
    for pattern in _COMPILED_USER_CHAT:
        if pattern.search(text):
            return True
    return False


# Legacy alias for backwards compatibility
TAZISH_CHAT_PATTERNS = USER_CHAT_PATTERNS
_COMPILED_TAZISH_CHAT = _COMPILED_USER_CHAT
is_tazish_chat = is_user_chat_pattern
