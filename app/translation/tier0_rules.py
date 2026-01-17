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