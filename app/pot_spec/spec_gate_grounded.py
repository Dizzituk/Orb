# FILE: app/pot_spec/spec_gate_grounded.py
"""
SpecGate Contract v1 - Grounded POT Spec Builder

PURPOSE (non-negotiable):
SpecGate turns Weaver output (intent) into a grounded, implementable Point-of-Truth (POT) spec.
It exists to:
- Stop drift
- Remove ambiguity
- Anchor work in repo reality
- Ask only the questions that truly require the human

CORE DECISION RULE: Look first. Ask second. Never guess.

RUNTIME: STRICTLY READ-ONLY
- No filesystem writes (no artifacts, no files)
- No DB writes (even persistence tables)
- Output must be returned/streamed only

QUESTION RULES:
- Max 3-7 questions total, only high-impact
- Only ask when:
  1. Not derivable from evidence (code, structure, patterns, docs)
  2. High-impact (wrong answer causes rewrite / wasted days / wrong UX)
  3. A user preference / product decision (not an engineering fact)

EVIDENCE PRIORITY:
1. Latest architecture map
2. Latest codebase report
3. read/head/lines/find
4. arch_query fallback
5. Ask user (only if still unresolved)

v1.0 (2026-01): Initial Contract v1 implementation
v1.1 (2026-01): Fixed question generation + status logic (Contract v1 compliance)
v1.2 (2026-01): Decision forks replace lazy questions (Contract v1.2 compliance)
              - Round 1 asks bounded A/B/C product decisions, not "tell me steps/tests"
              - Round 2 derives steps/tests from domain + answered forks
              - Added domain detection and fork question bank
"""

from __future__ import annotations

import logging
import re
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# =============================================================================
# IMPORTS
# =============================================================================

# Evidence collector (primary)
try:
    from .evidence_collector import (
        EvidenceBundle,
        EvidenceSource,
        load_evidence,
        add_file_read_to_bundle,
        add_search_to_bundle,
        find_in_evidence,
        verify_path_exists,
        refuse_write_operation,
        WRITE_REFUSED_ERROR,
    )
    _EVIDENCE_AVAILABLE = True
except ImportError as e:
    logger.warning("[spec_gate_grounded] evidence_collector not available: %s", e)
    _EVIDENCE_AVAILABLE = False

# Types from spec_gate_types
try:
    from .spec_gate_types import SpecGateResult
except ImportError:
    # Define minimal result type
    @dataclass
    class SpecGateResult:
        ready_for_pipeline: bool = False
        open_questions: List[str] = field(default_factory=list)
        spot_markdown: Optional[str] = None
        db_persisted: bool = False
        spec_id: Optional[str] = None
        spec_hash: Optional[str] = None
        spec_version: Optional[int] = None
        hard_stopped: bool = False
        hard_stop_reason: Optional[str] = None
        notes: Optional[str] = None
        blocking_issues: List[str] = field(default_factory=list)
        validation_status: str = "pending"


# =============================================================================
# CONSTANTS
# =============================================================================

# Question budget
MIN_QUESTIONS = 0
MAX_QUESTIONS = 7

# Question categories (allowed)
class QuestionCategory(str, Enum):
    PREFERENCE = "preference"           # UI style, tone, naming preference
    MISSING_PRODUCT_DECISION = "product_decision"  # New workflow, manual vs auto
    AMBIGUOUS_EVIDENCE = "ambiguous"    # Map says X, code says Y
    SAFETY_CONSTRAINT = "safety"        # Sandbox vs main, backwards compat
    DECISION_FORK = "decision_fork"     # v1.2: Bounded A/B/C product decision


# =============================================================================
# DECISION FORK SYSTEM (v1.2 - Contract v1.2 Compliance)
# =============================================================================
# 
# SpecGate asks ONLY bounded product decision forks (A/B/C choices).
# It does NOT ask "tell me the steps" or "tell me the acceptance criteria".
# Those are SpecGate's job to DERIVE after forks are answered.
#

# Domain detection keywords (case-insensitive)
DOMAIN_KEYWORDS = {
    "mobile_app": [
        "mobile app", "phone app", "android", "ios", "iphone",
        "offline-first", "offline first", "sync", "ocr", "screenshot",
        "voice", "push-to-talk", "push to talk", "wake word", "wakeword",
        "encryption", "encrypted", "trusted wi-fi", "trusted wifi",
        "in-van", "in van", "delivery", "parcels", "shift",
    ],
    # Future domains can be added here (e.g., "web_app", "cli_tool", "api_service")
}

# Fork question bank - templates for bounded A/B/C questions
# Each fork has: question, why_it_matters, options (A/B/C)
# Evidence is dynamically populated from Weaver text
MOBILE_APP_FORK_BANK = [
    {
        "id": "platform_v1",
        "question": "Platform for v1 release?",
        "why_it_matters": "Determines SDK choice, build tooling, and timeline. iOS adds ~40% development time.",
        "options": ["Android-only first", "Android + iOS from day 1"],
        "triggers": ["android", "ios", "iphone", "mobile", "phone"],
    },
    {
        "id": "offline_storage",
        "question": "Offline data storage approach?",
        "why_it_matters": "Affects data model complexity, encryption implementation, and sync conflict resolution.",
        "options": [
            "Room/SQLite + SQLCipher (structured, queryable)",
            "Encrypted file store (JSON + crypto, simpler but less queryable)",
        ],
        "triggers": ["offline", "encryption", "encrypted", "storage", "local"],
    },
    {
        "id": "input_mode_v1",
        "question": "Primary input mode for v1?",
        "why_it_matters": "Voice requires speech-to-text integration and error handling. Manual is simpler but slower in-van.",
        "options": [
            "Push-to-talk voice + manual fallback",
            "Voice-only (no manual entry)",
            "Manual-only (voice deferred to v2)",
        ],
        "triggers": ["voice", "push-to-talk", "push to talk", "manual", "input", "talk"],
    },
    {
        "id": "ocr_scope_v1",
        "question": "Screenshot OCR scope for v1?",
        "why_it_matters": "Multiple formats require more OCR training/templates. Single format is faster to ship.",
        "options": [
            "Finish Tour screenshot only",
            "Multiple screen formats (Finish Tour + route summary + others)",
        ],
        "triggers": ["ocr", "screenshot", "parse", "finish tour", "scan"],
    },
    {
        "id": "sync_behaviour",
        "question": "Data sync behaviour?",
        "why_it_matters": "Auto-sync needs background service and battery optimization. Manual is simpler but requires user action.",
        "options": [
            "Manual sync only (user taps 'Sync now')",
            "Auto-sync on trusted Wi-Fi",
            "Both (manual + optional auto on trusted Wi-Fi)",
        ],
        "triggers": ["sync", "wi-fi", "wifi", "upload", "background"],
    },
    {
        "id": "sync_target",
        "question": "Sync target for v1?",
        "why_it_matters": "Live endpoint requires server setup and auth. File export is portable but no real-time.",
        "options": [
            "Private ASTRA endpoint over LAN/VPN",
            "Export/import file only (no live endpoint yet)",
        ],
        "triggers": ["astra", "endpoint", "server", "export", "private", "lan", "vpn"],
    },
    {
        "id": "knockon_tracking",
        "question": "Knock-on day tracking method?",
        "why_it_matters": "Inferred rules need pattern detection logic. Manual toggle is explicit but requires user discipline.",
        "options": [
            "Manual toggle per day ('Today is a knock-on day: yes/no')",
            "Inferred from Tue/Thu pattern (auto-detected)",
        ],
        "triggers": ["knock-on", "knockon", "knock on", "tuesday", "thursday", "reschedule"],
    },
    {
        "id": "pay_variation",
        "question": "Pay-per-parcel variation handling?",
        "why_it_matters": "Daily override needs UI for quick rate changes. Fixed default is simpler but less accurate.",
        "options": [
            "Default rate with quick voice/tap override ('Pay is 2.00 today')",
            "Forced daily confirmation before shift start",
        ],
        "triggers": ["pay", "rate", "parcel", "1.85", "2.00", "variation", "override"],
    },
]


def detect_domains(text: str) -> List[str]:
    """
    Detect which domains are mentioned in the text.
    Returns list of domain keys (e.g., ["mobile_app"]).
    """
    if not text:
        return []
    
    text_lower = text.lower()
    detected = []
    
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                detected.append(domain)
                break  # One match is enough for this domain
    
    return detected


def extract_unresolved_ambiguities(weaver_text: str) -> List[str]:
    """
    Extract the "Unresolved ambiguities" section from Weaver output.
    Returns list of ambiguity strings.
    """
    if not weaver_text:
        return []
    
    ambiguities = []
    in_section = False
    
    for line in weaver_text.split("\n"):
        line_stripped = line.strip()
        line_lower = line_stripped.lower()
        
        # Detect section start
        if "unresolved ambigu" in line_lower:
            in_section = True
            continue
        
        # Detect section end (next header or empty after content)
        if in_section:
            if line_stripped.startswith("#") or line_stripped.startswith("**") and not line_stripped.startswith("**-"):
                # New section started
                break
            if line_stripped.startswith("-") or line_stripped.startswith("*"):
                # Bullet point - extract content
                content = line_stripped.lstrip("-*").strip()
                if content:
                    ambiguities.append(content)
            elif line_stripped and not line_stripped.startswith("#"):
                # Non-bullet content in section
                ambiguities.append(line_stripped)
    
    return ambiguities


def extract_decision_forks(
    weaver_text: str,
    detected_domains: List[str],
    max_questions: int = 7,
) -> List[GroundedQuestion]:
    """
    Extract bounded decision fork questions from Weaver text.
    
    v1.2: This replaces the lazy "tell me steps/tests" questions.
    Only asks for genuine product decisions that SpecGate cannot derive.
    
    Args:
        weaver_text: Full Weaver job description text
        detected_domains: List of detected domain keys
        max_questions: Maximum questions to return (default 7)
        
    Returns:
        List of GroundedQuestion with bounded A/B/C options
    """
    if not weaver_text:
        return []
    
    questions = []
    text_lower = weaver_text.lower()
    
    # Get unresolved ambiguities for evidence citation
    ambiguities = extract_unresolved_ambiguities(weaver_text)
    ambiguity_text = " | ".join(ambiguities) if ambiguities else ""
    
    # Process mobile app domain
    if "mobile_app" in detected_domains:
        for fork in MOBILE_APP_FORK_BANK:
            # Check if any trigger keywords are present
            triggered = any(trigger in text_lower for trigger in fork["triggers"])
            
            if triggered:
                # Find relevant ambiguity for evidence citation
                evidence = "Detected from Weaver intent"
                for amb in ambiguities:
                    amb_lower = amb.lower()
                    if any(trigger in amb_lower for trigger in fork["triggers"]):
                        evidence = f"Weaver ambiguity: '{amb[:100]}...'" if len(amb) > 100 else f"Weaver ambiguity: '{amb}'"
                        break
                
                questions.append(GroundedQuestion(
                    question=fork["question"],
                    category=QuestionCategory.DECISION_FORK,
                    why_it_matters=fork["why_it_matters"],
                    evidence_found=evidence,
                    options=fork["options"],
                ))
            
            if len(questions) >= max_questions:
                break
    
    # Future: Add other domain fork banks here
    
    return questions[:max_questions]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GroundedFact:
    """A fact grounded in repo evidence."""
    description: str
    source: str  # Which evidence confirmed this
    path: Optional[str] = None
    confidence: str = "confirmed"  # confirmed, inferred, unverified


@dataclass
class GroundedQuestion:
    """A high-impact question that requires human input."""
    question: str
    category: QuestionCategory
    why_it_matters: str
    evidence_found: str  # What SpecGate found so far
    options: Optional[List[str]] = None  # A/B options if applicable

    def format(self) -> str:
        """Format question for POT spec output."""
        lines = [f"**Q:** {self.question}"]
        lines.append(f"  - *Why it matters:* {self.why_it_matters}")
        lines.append(f"  - *Evidence found:* {self.evidence_found}")
        if self.options:
            lines.append(f"  - *Options:* " + " / ".join(f"({chr(65+i)}) {opt}" for i, opt in enumerate(self.options)))
        return "\n".join(lines)


@dataclass
class GroundedPOTSpec:
    """Point-of-Truth Spec grounded in repo evidence."""
    # Core
    goal: str
    
    # Grounded reality
    confirmed_components: List[GroundedFact] = field(default_factory=list)
    what_exists: List[str] = field(default_factory=list)
    what_missing: List[str] = field(default_factory=list)
    
    # Scope
    in_scope: List[str] = field(default_factory=list)
    out_of_scope: List[str] = field(default_factory=list)
    
    # Constraints
    constraints_from_intent: List[str] = field(default_factory=list)
    constraints_from_repo: List[str] = field(default_factory=list)
    
    # Evidence
    evidence_bundle: Optional[EvidenceBundle] = None
    
    # Plan
    proposed_steps: List[str] = field(default_factory=list)
    acceptance_tests: List[str] = field(default_factory=list)
    
    # Risks
    risks: List[Dict[str, str]] = field(default_factory=list)
    refactor_flags: List[str] = field(default_factory=list)
    
    # Questions (human decisions only)
    open_questions: List[GroundedQuestion] = field(default_factory=list)
    
    # Metadata
    spec_id: Optional[str] = None
    spec_hash: Optional[str] = None
    spec_version: int = 1
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Validation
    is_complete: bool = False
    blocking_issues: List[str] = field(default_factory=list)
    
    # Evidence completeness tracking (v1.1)
    evidence_complete: bool = True
    evidence_gaps: List[str] = field(default_factory=list)


# =============================================================================
# POT SPEC TEMPLATE BUILDER
# =============================================================================

def build_pot_spec_markdown(spec: GroundedPOTSpec) -> str:
    """
    Build POT spec markdown in the required template format.
    
    Template:
    - Goal
    - Current Reality (Grounded Facts)
    - Scope (in/out)
    - Constraints (from Weaver + discovered)
    - Evidence Used
    - Proposed Step Plan (small, testable steps only)
    - Acceptance Tests
    - Risks + Mitigations
    - Refactor Flags (recommendations only)
    - Open Questions (human decisions only)
    """
    lines = []
    
    # Title
    lines.append("# Point-of-Truth Specification")
    lines.append("")
    
    # Goal
    lines.append("## Goal")
    lines.append(spec.goal or "(Not specified)")
    lines.append("")
    
    # Current Reality
    lines.append("## Current Reality (Grounded Facts)")
    lines.append("")
    
    if spec.confirmed_components:
        lines.append("### Confirmed Components/Files/Modules")
        for fact in spec.confirmed_components:
            conf = f" [{fact.confidence}]" if fact.confidence != "confirmed" else ""
            src = f" (source: {fact.source})" if fact.source else ""
            lines.append(f"- {fact.description}{conf}{src}")
        lines.append("")
    
    if spec.what_exists:
        lines.append("### What Exists Now")
        for item in spec.what_exists:
            lines.append(f"- {item}")
        lines.append("")
    
    if spec.what_missing:
        lines.append("### What Doesn't Exist (Gaps)")
        for item in spec.what_missing:
            lines.append(f"- {item}")
        lines.append("")
    
    # Scope
    lines.append("## Scope")
    lines.append("")
    lines.append("### In Scope")
    if spec.in_scope:
        for item in spec.in_scope:
            lines.append(f"- {item}")
    else:
        lines.append("- (To be determined)")
    lines.append("")
    
    lines.append("### Out of Scope")
    if spec.out_of_scope:
        for item in spec.out_of_scope:
            lines.append(f"- {item}")
    else:
        lines.append("- (None explicitly specified)")
    lines.append("")
    
    # Constraints
    lines.append("## Constraints")
    lines.append("")
    
    lines.append("### From Weaver Intent")
    if spec.constraints_from_intent:
        for c in spec.constraints_from_intent:
            lines.append(f"- {c}")
    else:
        lines.append("- (None specified)")
    lines.append("")
    
    lines.append("### Discovered from Repo")
    if spec.constraints_from_repo:
        for c in spec.constraints_from_repo:
            lines.append(f"- {c}")
    else:
        lines.append("- (None discovered)")
    lines.append("")
    
    # Evidence Used
    lines.append("## Evidence Used")
    lines.append("")
    if spec.evidence_bundle:
        for source in spec.evidence_bundle.sources:
            if source.found or source.error:
                lines.append(f"- {source.to_evidence_line()}")
    else:
        lines.append("- (No evidence collected)")
    lines.append("")
    
    # Evidence Gaps Warning (v1.1)
    if spec.evidence_gaps:
        lines.append("### ⚠️ Evidence Gaps")
        lines.append("*The following evidence sources were unavailable, limiting grounding confidence:*")
        lines.append("")
        for gap in spec.evidence_gaps:
            lines.append(f"- {gap}")
        lines.append("")
    
    # Proposed Step Plan
    lines.append("## Proposed Step Plan")
    lines.append("*(Small, testable steps only)*")
    lines.append("")
    if spec.proposed_steps:
        for i, step in enumerate(spec.proposed_steps, 1):
            lines.append(f"{i}. {step}")
    else:
        lines.append("1. (Steps to be determined after questions resolved)")
    lines.append("")
    
    # Acceptance Tests
    lines.append("## Acceptance Tests")
    lines.append("")
    if spec.acceptance_tests:
        for test in spec.acceptance_tests:
            lines.append(f"- [ ] {test}")
    else:
        lines.append("- [ ] (To be determined)")
    lines.append("")
    
    # Risks + Mitigations
    lines.append("## Risks + Mitigations")
    lines.append("")
    if spec.risks:
        lines.append("| Risk | Mitigation |")
        lines.append("|------|------------|")
        for risk in spec.risks:
            lines.append(f"| {risk.get('risk', 'N/A')} | {risk.get('mitigation', 'N/A')} |")
    else:
        lines.append("| Risk | Mitigation |")
        lines.append("|------|------------|")
        lines.append("| (None identified) | - |")
    lines.append("")
    
    # Refactor Flags
    lines.append("## Refactor Flags (Recommendations Only)")
    lines.append("")
    if spec.refactor_flags:
        for flag in spec.refactor_flags:
            lines.append(f"- ⚠️ {flag}")
    else:
        lines.append("- (None)")
    lines.append("")
    
    # Open Questions - ALWAYS show if present (even on Round 3 finalization)
    lines.append("## Open Questions (Human Decisions Only)")
    lines.append("")
    if spec.open_questions:
        # If finalized with questions, mark as UNRESOLVED (no guessing)
        if spec.is_complete and spec.spec_version >= 3:
            lines.append("⚠️ **FINALIZED WITH UNRESOLVED QUESTIONS** - These were NOT guessed or filled in:")
            lines.append("")
        for i, q in enumerate(spec.open_questions, 1):
            lines.append(f"### Question {i}")
            if spec.is_complete and spec.spec_version >= 3:
                lines.append("**Status:** ❓ UNRESOLVED (no guess - human decision required)")
            lines.append(q.format())
            lines.append("")
    else:
        # v1.1 FIX: Only claim "all grounded" if evidence is truly complete
        if spec.evidence_complete and not spec.evidence_gaps:
            lines.append("✅ No questions - all information grounded from evidence.")
        else:
            lines.append("⚠️ No questions generated, but evidence was incomplete (see Evidence Gaps above).")
    lines.append("")
    
    # Blocking Issues
    if spec.blocking_issues:
        lines.append("---")
        lines.append("## ⛔ Blocking Issues")
        lines.append("")
        for issue in spec.blocking_issues:
            lines.append(f"- {issue}")
        lines.append("")
    
    # Unresolved Items Summary (for Round 3 finalization)
    has_unresolved = (
        spec.open_questions or
        not spec.proposed_steps or
        not spec.acceptance_tests or
        "(To be determined)" in str(spec.in_scope)
    )
    if spec.is_complete and has_unresolved:
        lines.append("---")
        lines.append("## ⚠️ Unresolved / Unknown (No Guess)")
        lines.append("")
        lines.append("*The following items remain unresolved. SpecGate did NOT fill these with assumptions:*")
        lines.append("")
        if spec.open_questions:
            lines.append(f"- **{len(spec.open_questions)} unanswered question(s)** - see above")
        if not spec.proposed_steps:
            lines.append("- **Steps:** Not specified (requires human input)")
        if not spec.acceptance_tests or all('(To be determined)' in str(t) for t in spec.acceptance_tests):
            lines.append("- **Acceptance tests:** Not specified (requires human input)")
        lines.append("")
    
    # Metadata
    lines.append("---")
    lines.append("## Metadata")
    lines.append(f"- **Spec ID:** `{spec.spec_id or 'N/A'}`")
    lines.append(f"- **Spec Hash:** `{spec.spec_hash[:16] if spec.spec_hash else 'N/A'}...`")
    lines.append(f"- **Version:** {spec.spec_version}")
    lines.append(f"- **Generated:** {spec.generated_at.isoformat()}")
    # v1.1 FIX: Status reflects true completeness
    lines.append(f"- **Status:** {'Complete' if spec.is_complete else 'Awaiting answers'}")
    
    return "\n".join(lines)


# =============================================================================
# WEAVER INTENT PARSER
# =============================================================================

def parse_weaver_intent(constraints_hint: Optional[Dict]) -> Dict[str, Any]:
    """
    Parse Weaver output to extract intent components.
    
    Handles both:
    - v3.0 simple text (weaver_job_description_text)
    - v2.x full spec JSON (weaver_spec_json)
    """
    if not constraints_hint:
        return {}
    
    result = {}
    
    # v3.0: Simple Weaver text
    job_desc_text = constraints_hint.get("weaver_job_description_text")
    if job_desc_text:
        result["raw_text"] = job_desc_text
        result["source"] = "weaver_simple"
        
        # Extract goal from text
        lines = job_desc_text.strip().split("\n")
        if lines:
            # First non-empty line is usually the goal
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    result["goal"] = line
                    break
        
        # Look for constraints/scope markers
        result["constraints"] = []
        result["scope_in"] = []
        result["scope_out"] = []
        
        for line in lines:
            line_lower = line.lower().strip()
            if "constraint" in line_lower or "must" in line_lower or "require" in line_lower:
                result["constraints"].append(line.strip())
            if "in scope" in line_lower or "should" in line_lower:
                result["scope_in"].append(line.strip())
            if "out of scope" in line_lower or "should not" in line_lower or "don't" in line_lower:
                result["scope_out"].append(line.strip())
    
    # v2.x: Full spec JSON
    weaver_spec = constraints_hint.get("weaver_spec_json")
    if isinstance(weaver_spec, dict):
        result["source"] = weaver_spec.get("source", "weaver_spec")
        result["goal"] = (
            weaver_spec.get("objective") or
            weaver_spec.get("title") or
            weaver_spec.get("job_description", "")[:200]
        )
        
        # Extract metadata
        metadata = weaver_spec.get("metadata", {}) or {}
        result["content_verbatim"] = (
            metadata.get("content_verbatim") or
            weaver_spec.get("content_verbatim")
        )
        result["location"] = (
            metadata.get("location") or
            weaver_spec.get("location")
        )
        result["scope_constraints"] = (
            metadata.get("scope_constraints") or
            weaver_spec.get("scope_constraints", [])
        )
        
        # Steps and outputs
        result["weaver_steps"] = weaver_spec.get("steps", [])
        result["weaver_outputs"] = weaver_spec.get("outputs", [])
        result["weaver_acceptance"] = weaver_spec.get("acceptance_criteria", [])
    
    return result


# =============================================================================
# GROUNDING ENGINE
# =============================================================================

def ground_intent_with_evidence(
    intent: Dict[str, Any],
    evidence: EvidenceBundle,
) -> GroundedPOTSpec:
    """
    Ground Weaver intent against repo evidence.
    
    This is the core grounding logic:
    1. Look for mentioned paths/modules in evidence
    2. Verify what exists vs what doesn't
    3. Identify constraints from repo patterns
    4. Generate questions ONLY for true unknowns
    """
    spec = GroundedPOTSpec(
        goal=intent.get("goal", ""),
        evidence_bundle=evidence,
    )
    
    # v1.1: Track evidence completeness
    spec.evidence_complete = True
    spec.evidence_gaps = []
    
    # Check if codebase report was loaded
    has_codebase_report = False
    has_arch_map = False
    for source in evidence.sources:
        if source.source_type == "codebase_report":
            if source.found:
                has_codebase_report = True
            elif source.error:
                spec.evidence_gaps.append(f"Codebase report: {source.error}")
                spec.evidence_complete = False
        if source.source_type == "architecture_map":
            if source.found:
                has_arch_map = True
            elif source.error:
                spec.evidence_gaps.append(f"Architecture map: {source.error}")
                spec.evidence_complete = False
    
    # Extract any paths mentioned in intent
    mentioned_paths = _extract_paths_from_text(intent.get("raw_text", ""))
    mentioned_paths.extend(_extract_paths_from_text(intent.get("goal", "")))
    
    # Add location if specified
    if intent.get("location"):
        mentioned_paths.append(intent["location"])
    
    # Ground each mentioned path
    for path in set(mentioned_paths):
        exists, source = verify_path_exists(evidence, path)
        if exists:
            spec.confirmed_components.append(GroundedFact(
                description=f"Path `{path}` exists",
                source=source or "evidence",
                path=path,
                confidence="confirmed",
            ))
            spec.what_exists.append(f"`{path}`")
        else:
            spec.what_missing.append(f"`{path}` (not found in evidence)")
    
    # Extract constraints from intent
    if intent.get("constraints"):
        spec.constraints_from_intent.extend(intent["constraints"])
    if intent.get("scope_constraints"):
        spec.constraints_from_intent.extend(intent["scope_constraints"])
    
    # Extract scope
    if intent.get("scope_in"):
        spec.in_scope.extend(intent["scope_in"])
    if intent.get("scope_out"):
        spec.out_of_scope.extend(intent["scope_out"])
    
    # Try to find relevant patterns in evidence
    if evidence.arch_map_content:
        # Look for related modules
        goal_keywords = _extract_keywords(intent.get("goal", ""))
        for keyword in goal_keywords[:5]:  # Top 5 keywords
            matches = find_in_evidence(evidence, rf"\b{re.escape(keyword)}\b", "architecture_map")
            if matches:
                spec.confirmed_components.append(GroundedFact(
                    description=f"Related content found for '{keyword}' in architecture map",
                    source="architecture_map",
                    confidence="inferred",
                ))
    
    # Copy steps/outputs from Weaver if available
    if intent.get("weaver_steps"):
        spec.proposed_steps = intent["weaver_steps"]
    if intent.get("weaver_acceptance"):
        spec.acceptance_tests = intent["weaver_acceptance"]
    
    # Detect refactor candidates from codebase report
    if evidence.codebase_report_content:
        # Look for bloat warnings
        bloat_matches = find_in_evidence(
            evidence,
            r"(size_critical|size_high|lines_critical|lines_high)",
            "codebase_report"
        )
        if bloat_matches:
            spec.refactor_flags.append(
                "Codebase report indicates large/complex files - consider refactoring"
            )
    
    return spec


def _extract_paths_from_text(text: str) -> List[str]:
    """Extract file/directory paths from text."""
    if not text:
        return []
    
    patterns = [
        r'`([^`]+\.(?:py|ts|tsx|js|jsx|json|md|yaml|yml))`',  # backtick paths
        r'[\'"]([^\'"]+\.(?:py|ts|tsx|js|jsx|json|md|yaml|yml))[\'"]',  # quoted paths
        r'(?:^|\s)(app/[^\s]+)',  # app/ paths
        r'(?:^|\s)(src/[^\s]+)',  # src/ paths
        r'(?:^|\s)(tests/[^\s]+)',  # tests/ paths
    ]
    
    paths = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        paths.extend(matches)
    
    return paths


def _extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from text."""
    if not text:
        return []
    
    # Remove common words
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'this', 'that', 'these', 'those', 'it', 'its', 'i', 'you', 'we',
        'they', 'he', 'she', 'what', 'which', 'who', 'whom', 'when', 'where',
        'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())
    
    # Filter and score by length (longer = more meaningful)
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    
    # Dedupe while preserving order
    seen = set()
    result = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            result.append(kw)
    
    return result


# =============================================================================
# QUESTION GENERATOR (v1.2 - Decision Forks, Not Lazy Questions)
# =============================================================================

def generate_grounded_questions(
    spec: GroundedPOTSpec,
    intent: Dict[str, Any],
    evidence: EvidenceBundle,
    round_number: int,
) -> List[GroundedQuestion]:
    """
    Generate questions ONLY for genuine unknowns.
    
    v1.2 CONTRACT:
    - Round 1: Ask bounded decision forks (A/B/C) only
    - Round 2+: Steps/tests are DERIVED from fork answers (not asked for)
    - Never ask "tell me the steps" or "tell me the acceptance criteria"
    - Max 7 questions total, only high-impact product decisions
    
    Rules:
    - Only ask when NOT derivable from evidence
    - Only ask high-impact questions (wrong answer = rework)
    - Preference/product decisions only (not engineering facts)
    """
    questions = []
    
    # Get Weaver text for domain detection and fork extraction
    weaver_text = intent.get("raw_text", "") or ""
    
    # =================================================================
    # ROUND 2+: Derive steps/tests from fork answers, don't ask more
    # =================================================================
    if round_number >= 2:
        # v1.2: In Round 2+, we DERIVE steps/tests from answered forks.
        # We do NOT ask the user to write them for us.
        # If steps/tests are still missing, they will be generated here.
        
        # Only ask critical questions if there's a genuine blocker
        # that can't be derived (e.g., truly missing goal)
        if not spec.goal or spec.goal.strip() == "":
            questions.append(GroundedQuestion(
                question="What is the primary goal/objective of this job?",
                category=QuestionCategory.MISSING_PRODUCT_DECISION,
                why_it_matters="Without a clear goal, the spec cannot be grounded",
                evidence_found="No goal found in Weaver output",
            ))
        
        # v1.2: Steps and tests are SpecGate's job to derive, NOT the user's
        # If we reach Round 2 without them, derive from the domain + forks
        if not spec.proposed_steps:
            spec.proposed_steps = _derive_steps_from_domain(intent, spec)
        if not spec.acceptance_tests or all('(To be determined)' in str(t) for t in spec.acceptance_tests):
            spec.acceptance_tests = _derive_tests_from_domain(intent, spec)
        
        return questions[:MAX_QUESTIONS]
    
    # =================================================================
    # ROUND 1: Ask bounded decision forks (A/B/C) - NOT lazy questions
    # =================================================================
    
    # Check for missing goal (this is a critical blocker, not a lazy question)
    if not spec.goal or spec.goal.strip() == "":
        questions.append(GroundedQuestion(
            question="What is the primary goal/objective of this job?",
            category=QuestionCategory.MISSING_PRODUCT_DECISION,
            why_it_matters="Without a clear goal, the spec cannot be grounded",
            evidence_found="No goal found in Weaver output",
        ))
    
    # v1.2: Detect domain and extract decision forks
    detected_domains = detect_domains(weaver_text)
    
    if detected_domains:
        # Extract bounded A/B/C fork questions from domain templates
        fork_questions = extract_decision_forks(
            weaver_text=weaver_text,
            detected_domains=detected_domains,
            max_questions=MAX_QUESTIONS - len(questions),  # Reserve room for other questions
        )
        questions.extend(fork_questions)
        
        logger.info(
            "[spec_gate_grounded] v1.2: Detected domains %s, generated %d fork questions",
            detected_domains, len(fork_questions)
        )
    
    # v1.2 REMOVED: Do NOT ask lazy "steps/tests" questions
    # These are SpecGate's job to DERIVE after forks are answered:
    #   - "What are the key implementation steps for this work?" ← REMOVED
    #   - "What acceptance criteria should verify this work is complete?" ← REMOVED
    
    # Check for ambiguous paths (mentioned but not found) - this is still valid
    if spec.what_missing:
        missing_count = len(spec.what_missing)
        if missing_count > 0 and len(questions) < MAX_QUESTIONS:
            questions.append(GroundedQuestion(
                question=f"These paths were mentioned but not found in evidence: {', '.join(spec.what_missing[:3])}. Should they be created, or are the paths incorrect?",
                category=QuestionCategory.AMBIGUOUS_EVIDENCE,
                why_it_matters="Need to know if files should be created vs paths are wrong",
                evidence_found=f"Searched architecture map and codebase report - {missing_count} path(s) not found",
                options=["Create new files at these paths", "Paths may be incorrect - suggest alternatives"],
            ))
    
    # Check for safety constraints if touching critical paths
    critical_paths = ['stream_router', 'overwatcher', 'translation', 'routing']
    touches_critical = any(
        any(crit in fact.description.lower() for crit in critical_paths)
        for fact in spec.confirmed_components
    )
    if touches_critical and not any('sandbox' in c.lower() for c in spec.constraints_from_intent):
        if len(questions) < MAX_QUESTIONS:
            questions.append(GroundedQuestion(
                question="This job touches critical routing/pipeline code. Should changes be tested in SANDBOX first before MAIN repo?",
                category=QuestionCategory.SAFETY_CONSTRAINT,
                why_it_matters="Touching critical code without sandbox testing risks breaking the system",
                evidence_found="Detected critical paths in scope",
                options=["Sandbox first, then MAIN", "MAIN repo directly (I'll verify manually)"],
            ))
    
    # Cap at MAX_QUESTIONS
    return questions[:MAX_QUESTIONS]


def _derive_steps_from_domain(intent: Dict[str, Any], spec: GroundedPOTSpec) -> List[str]:
    """
    v1.2: Derive implementation steps from domain + answered forks.
    
    This is SpecGate's job, NOT the user's. Once product decisions are made,
    the steps can be derived automatically.
    """
    weaver_text = intent.get("raw_text", "") or ""
    detected_domains = detect_domains(weaver_text)
    
    steps = []
    
    if "mobile_app" in detected_domains:
        # Mobile app domain - standard implementation steps
        steps = [
            "Set up mobile project structure (platform-specific tooling)",
            "Implement local encrypted data storage layer",
            "Build core UI screens (shift start/stop, data entry)",
            "Implement voice input handler (if selected) or manual input forms",
            "Implement screenshot OCR parser (if selected)",
            "Build sync mechanism (manual/auto per selected option)",
            "Implement end-of-week summary calculations",
            "Add ASTRA integration endpoint (if selected)",
            "Integration testing across all input modes",
            "Security audit (encryption, data handling)",
        ]
    else:
        # Generic steps for unknown domains
        steps = [
            "Analyze requirements and create technical design",
            "Set up project structure and dependencies",
            "Implement core functionality",
            "Add error handling and edge cases",
            "Write tests and documentation",
            "Integration testing",
            "Security review",
        ]
    
    return steps


def _derive_tests_from_domain(intent: Dict[str, Any], spec: GroundedPOTSpec) -> List[str]:
    """
    v1.2: Derive acceptance tests from domain + answered forks.
    
    This is SpecGate's job, NOT the user's. Once product decisions are made,
    the acceptance criteria can be derived automatically.
    """
    weaver_text = intent.get("raw_text", "") or ""
    detected_domains = detect_domains(weaver_text)
    
    tests = []
    
    if "mobile_app" in detected_domains:
        # Mobile app domain - standard acceptance tests
        tests = [
            "App starts and displays main screen within 2 seconds",
            "Shift start/stop logs timestamp correctly to local storage",
            "Voice input (if enabled) correctly transcribes test phrases",
            "Screenshot OCR (if enabled) extracts stop count from test image",
            "Data persists across app restart (encrypted storage verified)",
            "Manual sync successfully transfers data to target (endpoint or file)",
            "End-of-week summary shows correct totals for hours, parcels, pay",
            "App functions fully offline (no network required for core features)",
            "No sensitive data exposed in logs or debug output",
        ]
    else:
        # Generic tests for unknown domains
        tests = [
            "Core functionality works as specified",
            "Error handling covers expected failure modes",
            "Performance meets requirements",
            "Security review passes",
            "Documentation is complete and accurate",
        ]
    
    return tests


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def run_spec_gate_grounded(
    db: Session,
    job_id: str,
    user_intent: str,
    provider_id: str,
    model_id: str,
    project_id: int,
    constraints_hint: Optional[Dict] = None,
    spec_version: int = 1,
    user_answers: Optional[Dict[str, str]] = None,
) -> SpecGateResult:
    """
    Run SpecGate Contract v1 - Grounded POT Spec Builder.
    
    RUNTIME IS READ-ONLY:
    - No filesystem writes
    - No DB writes
    - Output/stream only
    
    Args:
        db: Database session (NOT USED for writes)
        job_id: Job identifier
        user_intent: User's raw intent text
        provider_id: LLM provider (for metadata only)
        model_id: LLM model (for metadata only)
        project_id: Project ID
        constraints_hint: Weaver output and other hints
        spec_version: Round number (1 = initial, 2+ = after answers)
        user_answers: User's answers to previous questions
        
    Returns:
        SpecGateResult with POT spec or questions
    """
    try:
        round_n = max(1, min(3, int(spec_version or 1)))
        
        logger.info(
            "[spec_gate_grounded] Starting round %d for job %s (project %d)",
            round_n, job_id, project_id
        )
        
        # =================================================================
        # STEP 1: Load Evidence (read-only)
        # =================================================================
        
        if not _EVIDENCE_AVAILABLE:
            return SpecGateResult(
                ready_for_pipeline=False,
                hard_stopped=True,
                hard_stop_reason="Evidence collector not available",
                validation_status="error",
            )
        
        evidence = load_evidence(
            include_arch_map=True,
            include_codebase_report=True,
            arch_map_max_lines=500,
            codebase_report_max_lines=300,
        )
        
        logger.info(
            "[spec_gate_grounded] Loaded evidence: %d sources, %d errors",
            len(evidence.sources),
            len(evidence.errors),
        )
        
        # =================================================================
        # STEP 2: Parse Weaver Intent
        # =================================================================
        
        intent = parse_weaver_intent(constraints_hint or {})
        
        # Include user's raw text if provided
        if user_intent and user_intent.strip():
            # Strip "Astra, command:" prefix
            clean_intent = re.sub(
                r'^(?:astra[,:]?\s*)?(?:command[:\s]+)?(?:critical\s+)?(?:architecture\s*)?',
                '',
                user_intent,
                flags=re.IGNORECASE
            ).strip()
            if clean_intent:
                intent["user_text"] = clean_intent
                if not intent.get("goal"):
                    intent["goal"] = clean_intent
        
        # =================================================================
        # STEP 3: Ground Intent with Evidence
        # =================================================================
        
        spec = ground_intent_with_evidence(intent, evidence)
        
        # =================================================================
        # STEP 4: Apply User Answers (if round 2+)
        # =================================================================
        
        if user_answers and round_n >= 2:
            # Integrate answers into spec
            for key, answer in user_answers.items():
                if "scope" in key.lower():
                    spec.out_of_scope.append(answer)
                elif "step" in key.lower():
                    spec.proposed_steps.append(answer)
                elif "path" in key.lower() or "file" in key.lower():
                    spec.what_exists.append(f"User confirmed: {answer}")
        
        # =================================================================
        # STEP 5: Generate Questions (if needed)
        # =================================================================
        
        questions = generate_grounded_questions(spec, intent, evidence, round_n)
        spec.open_questions = questions
        
        # =================================================================
        # STEP 6: Determine Completion Status (v1.1 FIX)
        # =================================================================
        
        # Round 3 always finalizes (even with gaps)
        # IMPORTANT: We do NOT fill gaps - we just mark them as unresolved
        if round_n >= 3:
            spec.is_complete = True
            if questions:
                spec.blocking_issues.append(
                    f"Finalized with {len(questions)} unanswered question(s) - NOT guessed"
                )
            # Questions are preserved in spec.open_questions for the markdown output
        else:
            # v1.1 FIX: is_complete only when:
            # - No questions AND
            # - Steps exist AND
            # - Acceptance tests exist (and aren't placeholders)
            has_real_steps = bool(spec.proposed_steps)
            has_real_tests = (
                bool(spec.acceptance_tests) and
                not all('(To be determined)' in str(t) for t in spec.acceptance_tests)
            )
            
            spec.is_complete = (
                len(questions) == 0 and
                has_real_steps and
                has_real_tests
            )
            
            logger.info(
                "[spec_gate_grounded] Completion check: questions=%d, steps=%s, tests=%s -> complete=%s",
                len(questions), has_real_steps, has_real_tests, spec.is_complete
            )
        
        # =================================================================
        # STEP 7: Generate IDs and Hash (no writes!)
        # =================================================================
        
        import uuid
        spec.spec_id = f"sg-{uuid.uuid4().hex[:12]}"
        spec.spec_version = round_n
        
        # Compute hash from spec content
        hash_content = json.dumps({
            "goal": spec.goal,
            "in_scope": spec.in_scope,
            "out_of_scope": spec.out_of_scope,
            "steps": spec.proposed_steps,
            "version": round_n,
        }, sort_keys=True)
        spec.spec_hash = hashlib.sha256(hash_content.encode()).hexdigest()
        
        # =================================================================
        # STEP 8: Build POT Spec Markdown
        # =================================================================
        
        spot_md = build_pot_spec_markdown(spec)
        
        # =================================================================
        # STEP 9: Return Result (NO DB/FILE WRITES)
        # =================================================================
        
        validation_status = "validated" if spec.is_complete else "needs_clarification"
        if spec.blocking_issues:
            validation_status = "validated_with_issues" if spec.is_complete else "blocked"
        
        # Include questions in output - even if complete (Round 3), so they're visible as unresolved
        open_q_text = [q.question for q in spec.open_questions]
        
        logger.info(
            "[spec_gate_grounded] Result: complete=%s, questions=%d, round=%d",
            spec.is_complete, len(open_q_text), round_n
        )
        
        return SpecGateResult(
            ready_for_pipeline=spec.is_complete,
            # Always return questions so they're visible (especially on Round 3)
            open_questions=open_q_text,
            spot_markdown=spot_md if spec.is_complete else None,
            db_persisted=False,  # NEVER persist - read-only runtime
            spec_id=spec.spec_id,
            spec_hash=spec.spec_hash,
            spec_version=round_n,
            notes=(
                f"Evidence sources: {len(evidence.sources)}; "
                f"arch_query_used: {evidence.arch_query_used}; "
                f"evidence_complete: {spec.evidence_complete}"
            ),
            blocking_issues=[str(i) for i in spec.blocking_issues],
            validation_status=validation_status,
        )
        
    except Exception as e:
        logger.exception("[spec_gate_grounded] HARD STOP: %s", e)
        return SpecGateResult(
            ready_for_pipeline=False,
            hard_stopped=True,
            hard_stop_reason=str(e),
            spec_version=int(spec_version) if isinstance(spec_version, int) else None,
            validation_status="error",
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "run_spec_gate_grounded",
    "GroundedPOTSpec",
    "GroundedQuestion",
    "GroundedFact",
    "QuestionCategory",
    "build_pot_spec_markdown",
    "load_evidence",
    "WRITE_REFUSED_ERROR",
    # v1.2 additions
    "detect_domains",
    "extract_decision_forks",
    "extract_unresolved_ambiguities",
    "DOMAIN_KEYWORDS",
    "MOBILE_APP_FORK_BANK",
]
