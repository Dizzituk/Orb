# FILE: app/translation/_tier0_trigger_checks.py
"""
Tier 0 special-case trigger check functions.
Each function checks a specific command domain and returns Tier0RuleResult.

v2.3 (2026-03-10): Added "Astra" prefix stripping to all trigger checks.
    Added natural pipeline trigger phrases to check_weaver_trigger.
"""
from __future__ import annotations
import re
from typing import Optional
from .schemas import CanonicalIntent
from .tier0_rules import Tier0RuleResult


# =============================================================================
# v2.3: ASTRA ADDRESS STRIPPING
# =============================================================================

_ASTRA_PREFIX = re.compile(r"^(?:hey[,.]?\s+)?astr[aoe][,.]?\s+", re.IGNORECASE)


def _strip_astra(text: str) -> str:
    """Strip 'Astra,' / 'Hey Astra,' / 'Astro,' prefix from text for trigger matching."""
    return _ASTRA_PREFIX.sub("", text.strip()).strip()


# =============================================================================
# ARCHITECTURE & SANDBOX TRIGGERS
# =============================================================================

def check_architecture_map_variant(text: str) -> Tier0RuleResult:
    """
    Special handler for architecture map variants.
    - ALL CAPS "CREATE ARCHITECTURE MAP" -> with files
    - Normal case "Create architecture map" -> structure only
    """
    text_stripped = _strip_astra(text)

    if text_stripped.startswith("CREATE ARCHITECTURE MAP"):
        return Tier0RuleResult(
            matched=True,
            intent=CanonicalIntent.ARCHITECTURE_MAP_WITH_FILES,
            confidence=1.0,
            rule_name="archmap_all_caps",
            reason="ALL CAPS trigger for full architecture map with files",
        )

    if re.match(r"^[Cc]reate [Aa]rchitecture [Mm]ap", text_stripped):
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
    """Special handler for zombie/sandbox start commands."""
    text_lower = _strip_astra(text).lower()

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
    """Special handler for architecture update commands."""
    text_lower = _strip_astra(text).lower()

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
    """Special handler for sandbox scanning commands."""
    text_lower = _strip_astra(text).lower()

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
# SPEC GATE FLOW HANDLERS (v1.1, v2.3)
# =============================================================================

def check_weaver_trigger(text: str) -> Tier0RuleResult:
    """Special handler for Weaver (spec building) triggers.

    v2.3: Strips 'Astra' prefix before matching.
    v2.3: Added natural pipeline trigger phrases:
        - "send that to the build pipeline"
        - "start building" / "build this" / "build it" / "let's build"
        - "can we start building now"
    """
    text_lower = _strip_astra(text).lower()

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
        # v2.3: Natural pipeline triggers
        "send that to the build pipeline",
        "send this to the build pipeline",
        "send it to the build pipeline",
        "send that to the pipeline",
        "send this to the pipeline",
        "send it to the pipeline",
        "start building",
        "start building that",
        "start building this",
        "build this",
        "build that",
        "build it",
        "go build it",
        "go build that",
        "go ahead and build it",
        "go ahead and build that",
        "can we start building now",
        "can you start building now",
        "can we actually start building the app now",
        "can we start building the app now",
        "can you build that",
        "can you build this",
        "let's build it",
        "let's build this",
        "let's build that",
        "okay build it",
        "ok build it",
        "okay build that",
        "ok build that",
        "right, build it",
        "right build it",
        "make it happen",
        "kick off the build",
        "start the build",
    ]

    for phrase in exact_phrases:
        if text_lower == phrase or text_lower.rstrip("?!.") == phrase.rstrip("?!."):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.WEAVER_BUILD_SPEC,
                confidence=1.0,
                rule_name="weaver_exact_phrase",
                reason=f"Weaver trigger: '{phrase}'",
            )

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
        # v2.3: Natural pipeline triggers (pattern-based)
        r"^send (?:that|this|it) to (?:the )?(?:build )?pipeline",
        r"^(?:can (?:we|you) )?(?:actually )?start building",
        r"^(?:go (?:ahead (?:and )?)?)?build (?:it|that|this|the app)",
        r"^(?:let's|lets) (?:go (?:ahead (?:and )?)?)?build",
        r"^(?:right,? )?(?:go (?:ahead (?:and )?)?)?build (?:it|that|this)",
        r"^kick off (?:the )?build",
        r"^start (?:the )?build",
        r"^make (?:it|that|this) happen",
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
    """Special handler for Spec Gate (validation) triggers."""
    text_lower = _strip_astra(text).lower()

    simple_affirmatives = [
        "yes", "yes please", "yes, please", "yep", "yeah",
        "sure", "go ahead", "do it", "proceed", "send it",
        "ok", "okay", "affirmative", "confirmed", "confirm", "y",
    ]

    if text_lower in simple_affirmatives:
        return Tier0RuleResult(
            matched=True,
            intent=CanonicalIntent.SEND_TO_SPEC_GATE,
            confidence=0.95,
            rule_name="specgate_affirmative",
            reason=f"Spec Gate affirmative confirmation: '{text_lower}'",
        )

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


# =============================================================================
# PIPELINE & EXECUTION TRIGGERS
# =============================================================================

def check_critical_pipeline_trigger(text: str) -> Tier0RuleResult:
    """Special handler for critical pipeline triggers."""
    text_lower = _strip_astra(text).lower()

    patterns = [
        r"^run (?:the )?pipeline$",
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
                intent=CanonicalIntent.RUN_PIPELINE,
                confidence=1.0,
                rule_name="run_pipeline",
                reason="Pipeline command detected (unified)",
            )

    return Tier0RuleResult(matched=False)


def check_overwatcher_trigger(text: str) -> Tier0RuleResult:
    """Special handler for Overwatcher execution triggers."""
    text_lower = _strip_astra(text).lower()

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


def check_segment_loop_trigger(text: str) -> Tier0RuleResult:
    """Special handler for segment loop execution triggers."""
    text_lower = _strip_astra(text).lower()

    patterns = [
        r"^run segments?$",
        r"^run (?:the )?segments?$",
        r"^execute segments?$",
        r"^run segment loop$",
        r"^execute segment loop$",
        r"^segment loop$",
        r"^run segmented job$",
        r"^(?:ok(?:ay)?,?\s*)?run (?:the )?segments?$",
    ]

    for pattern in patterns:
        if re.match(pattern, text_lower):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.RUN_PIPELINE,
                confidence=1.0,
                rule_name="run_pipeline",
                reason="Pipeline command detected (unified)",
            )

    return Tier0RuleResult(matched=False)


# =============================================================================
# DATA & REPORT TRIGGERS
# =============================================================================

def check_codebase_report_trigger(text: str) -> Tier0RuleResult:
    """Special handler for codebase report commands."""
    text_lower = _strip_astra(text).lower()

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


def check_latest_report_trigger(text: str) -> Tier0RuleResult:
    """Special handler for latest report resolver commands."""
    text_lower = _strip_astra(text).lower()

    archmap_patterns = [
        r"^latest\s+architecture\s+map$",
    ]

    for pattern in archmap_patterns:
        if re.match(pattern, text_lower):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.LATEST_ARCHITECTURE_MAP,
                confidence=1.0,
                rule_name="latest_architecture_map",
                reason="Latest architecture map command detected",
            )

    codebase_patterns = [
        r"^latest\s+codebase\s+report(?:\s+full)?$",
    ]

    for pattern in codebase_patterns:
        if re.match(pattern, text_lower):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.LATEST_CODEBASE_REPORT_FULL,
                confidence=1.0,
                rule_name="latest_codebase_report_full",
                reason="Latest codebase report (FULL) command detected",
            )

    return Tier0RuleResult(matched=False)


def check_project_scope_trigger(text: str) -> Tier0RuleResult:
    """Detect 'start a new project' style commands."""
    text_lower = _strip_astra(text).lower().strip()

    scope_patterns = [
        r"^(?:i want to |let'?s |let me )?(?:start|begin|create)(?:\s+off)?\s+a new (?:project|app|build)",
        r"^new (?:project|app|android app|build)$",
        r"^i'?ve got (?:a |an )?new (?:idea|project|app)",
        r"^i want to build (?:a |an )?new",
        r"^let'?s build (?:a |an )?new",
        r"^(?:i want to |let'?s )?(?:kick off|spin up|set up) a new (?:project|app|build)",
    ]

    for pattern in scope_patterns:
        if re.match(pattern, text_lower):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.PROJECT_SCOPE_START,
                confidence=0.95,
                rule_name="project_scope_start",
                reason="Project scoping command detected",
            )

    return Tier0RuleResult(matched=False)


def check_embedding_commands(text: str) -> Tier0RuleResult:
    """Special handler for embedding management commands."""
    text_lower = _strip_astra(text).lower()

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
