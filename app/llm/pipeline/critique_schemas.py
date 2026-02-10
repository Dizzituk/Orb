# FILE: app/llm/pipeline/critique_schemas.py
"""Structured schemas for machine-driven critique pipeline (Block 5).

The critique output is strict JSON for deterministic pass/fail decisioning.
A parallel markdown artifact is generated for human readability.

v1.3 (2026-02-05): SECTION AUTHORITY - distinguish user requirements from LLM suggestions
- Critique prompt now includes SECTION AUTHORITY LEVELS guidance
- 'Files to Modify', 'Implementation Steps', etc. are LLM-generated suggestions (non-blocking only)
- 'Constraints', 'Goal', user-stated features are hard requirements (blocking if missed)
- Fixes deadlock where critique raised blockers on LLM-suggested file modifications
- See critique-pipeline-fix-jobspec.md for root cause analysis

v1.2 (2026-02-02): GROUNDED CRITIQUE - POT spec as source of truth
- build_json_critique_prompt() now accepts spec_markdown parameter
- Full POT spec with grounded evidence injected into critique prompt
- Critique ONLY flags issues that violate the spec
- Critique DOES NOT invent constraints not in the spec
- Philosophy: "Ground and trust" - spec IS the contract

v1.1 (2026-01):
- Added critique_mode field: "quickcheck" or "deep"
- Added blocker type constants for filtering

v1.0 (2025-12):
- CritiqueIssue: Single blocking or non-blocking issue
- CritiqueResult: Full critique output with pass/fail
- Parsing helpers for LLM JSON output
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Approved Blocker Types (v1.1)
# =============================================================================

# These are the ONLY categories that can block an architecture job.
# Any blocking issue with a category NOT in this set will be downgraded.
APPROVED_ARCHITECTURE_BLOCKER_TYPES = {
    # Core blocker types (match spec contract)
    "spec_mismatch",           # Contradicts PoT spec
    "missing_required",        # Missing component required by spec  
    "broken_flow",             # End-to-end flow has magic steps
    "unsafe_boundary",         # Trust boundary violation
    "repo_mismatch",           # Invented modules/endpoints that don't exist
    "operational_gap",         # Locking/migration/timeout ignored
    "internal_contradiction",  # Architecture contradicts itself
    "contract_violation",      # v5.4: Violates interface contract from Critical Supervisor
    
    # v1.2 (2026-01-22): Spec-Anchored Compliance Types (CRITICAL FIX)
    # These catch architecture drift from SPoT requirements
    "platform_mismatch",       # Architecture platform differs from spec (desktop vs web vs mobile)
    "stack_mismatch",          # Tech stack differs from user-discussed choices (Python vs TS/JS)
    "scope_inflation",         # Architecture adds features/complexity not in spec
    "spec_compliance",         # General spec compliance failure
    
    # Standard critic categories (from LLM output)
    "security",                # Security vulnerabilities
    "correctness",             # Correctness errors / calculations wrong
    "completeness",            # Missing required component (if spec says it's needed)
    
    # v2.0: Evidence-or-Request Contract (CRITICAL_CLAIMS validation)
    "unresolved_critical",     # CRITICAL_CLAIMS register has unresolved/invalid entries
    "missing_claims_register", # Stage did not output CRITICAL_CLAIMS register (transition)
    
    # v2.1: Scope creep detection (endpoint/feature drift)
    "scope_creep",             # Architecture adds endpoints/features not in spec
    "endpoint_rename",         # Architecture renames spec-listed endpoints
    "excluded_feature",        # Architecture includes feature spec explicitly excludes
    
    # Aliases (LLM may output these)
    "drift",                   # Spec drift
    "hallucination",           # Invented requirements
    "contradiction",           # Internal contradiction
    "boundary_violation",      # Trust boundary
}


# =============================================================================
# Known Architecture Issue Types (v2.0 - includes non-blocking transition types)
# =============================================================================
# Types that are RECOGNIZED but not yet blocking. These exist for diagnostics
# and will be promoted to APPROVED_ARCHITECTURE_BLOCKER_TYPES once all stages
# reliably emit the corresponding data.
KNOWN_ARCHITECTURE_ISSUE_TYPES = {
    "missing_claims_register",  # Stage did not output CRITICAL_CLAIMS register (transition period, non_blocking)
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CritiqueIssue:
    """A single issue identified by the critic.
    
    Attributes:
        id: Unique issue identifier (e.g., "ISSUE-001")
        spec_ref: Reference to spec section (e.g., "MUST-3", "SHOULD-1")
        arch_ref: Reference to architecture section being critiqued
        category: Issue category (security, correctness, completeness, clarity, performance)
        severity: blocking or non_blocking
        description: What's wrong
        fix_suggestion: How to fix it
    """
    id: str
    spec_ref: Optional[str]
    arch_ref: Optional[str]
    category: str
    severity: str  # "blocking" or "non_blocking"
    description: str
    fix_suggestion: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CritiqueIssue":
        return cls(
            id=data.get("id", "UNKNOWN"),
            spec_ref=data.get("spec_ref"),
            arch_ref=data.get("arch_ref"),
            category=data.get("category", "general"),
            severity=data.get("severity", "non_blocking"),
            description=data.get("description", ""),
            fix_suggestion=data.get("fix_suggestion", ""),
        )


@dataclass
class CritiqueResult:
    """Full critique result with machine-driven pass/fail.
    
    Attributes:
        blocking_issues: List of issues that MUST be fixed before approval
        non_blocking_issues: List of issues that SHOULD be fixed but don't block
        overall_pass: True iff blocking_issues is empty AND critique didn't fail
        summary: Brief human-readable summary
        spec_coverage: Dict mapping spec requirements to coverage status
        critique_model: Model that generated this critique
        critique_version: Schema version
        critique_failed: True if critique could not be completed (timeout, empty response, etc.)
        critique_mode: "quickcheck" or "deep" - type of critique performed
    """
    blocking_issues: List[CritiqueIssue] = field(default_factory=list)
    non_blocking_issues: List[CritiqueIssue] = field(default_factory=list)
    overall_pass: bool = False
    summary: str = ""
    spec_coverage: Dict[str, str] = field(default_factory=dict)
    critique_model: str = ""
    critique_version: str = "v2"
    critique_failed: bool = False  # FAIL-CLOSED: True if critique could not complete
    critique_mode: str = "deep"    # v1.1: "quickcheck" or "deep"
    
    def __post_init__(self):
        # FAIL-CLOSED: overall_pass is True iff no blocking issues AND critique succeeded
        # If critique_failed=True, overall_pass is ALWAYS False (fail-closed behavior)
        self.overall_pass = (not self.critique_failed) and len(self.blocking_issues) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocking_issues": [i.to_dict() for i in self.blocking_issues],
            "non_blocking_issues": [i.to_dict() for i in self.non_blocking_issues],
            "overall_pass": self.overall_pass,
            "summary": self.summary,
            "spec_coverage": self.spec_coverage,
            "critique_model": self.critique_model,
            "critique_version": self.critique_version,
            "critique_failed": self.critique_failed,
            "critique_mode": self.critique_mode,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def to_markdown(self) -> str:
        """Generate human-readable markdown from critique."""
        lines = ["# Architecture Critique Report", ""]
        
        # Summary
        status = "✅ PASSED" if self.overall_pass else "❌ FAILED (blocking issues)"
        lines.append(f"**Status:** {status}")
        lines.append(f"**Model:** {self.critique_model}")
        lines.append(f"**Mode:** {self.critique_mode}")
        lines.append("")
        
        if self.summary:
            lines.append("## Summary")
            lines.append(self.summary)
            lines.append("")
        
        # Blocking issues
        if self.blocking_issues:
            lines.append("## Blocking Issues (Must Fix)")
            lines.append("")
            for issue in self.blocking_issues:
                lines.append(f"### {issue.id}: {issue.category.title()}")
                if issue.spec_ref:
                    lines.append(f"**Spec Reference:** {issue.spec_ref}")
                if issue.arch_ref:
                    lines.append(f"**Architecture Section:** {issue.arch_ref}")
                lines.append(f"**Problem:** {issue.description}")
                lines.append(f"**Suggested Fix:** {issue.fix_suggestion}")
                lines.append("")
        
        # Non-blocking issues
        if self.non_blocking_issues:
            lines.append("## Non-Blocking Issues (Should Fix)")
            lines.append("")
            for issue in self.non_blocking_issues:
                lines.append(f"### {issue.id}: {issue.category.title()}")
                if issue.spec_ref:
                    lines.append(f"**Spec Reference:** {issue.spec_ref}")
                if issue.arch_ref:
                    lines.append(f"**Architecture Section:** {issue.arch_ref}")
                lines.append(f"**Problem:** {issue.description}")
                lines.append(f"**Suggested Fix:** {issue.fix_suggestion}")
                lines.append("")
        
        # Spec coverage
        if self.spec_coverage:
            lines.append("## Spec Coverage")
            lines.append("")
            lines.append("| Requirement | Status |")
            lines.append("|-------------|--------|")
            for req, status in self.spec_coverage.items():
                lines.append(f"| {req} | {status} |")
            lines.append("")
        
        return "\n".join(lines)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CritiqueResult":
        blocking = [CritiqueIssue.from_dict(i) for i in data.get("blocking_issues", [])]
        non_blocking = [CritiqueIssue.from_dict(i) for i in data.get("non_blocking_issues", [])]
        
        result = cls(
            blocking_issues=blocking,
            non_blocking_issues=non_blocking,
            summary=data.get("summary", ""),
            spec_coverage=data.get("spec_coverage", {}),
            critique_model=data.get("critique_model", ""),
            critique_version=data.get("critique_version", "v2"),
            critique_failed=data.get("critique_failed", False),
            critique_mode=data.get("critique_mode", "deep"),
        )
        # overall_pass is computed in __post_init__
        return result
    
    @classmethod
    def from_json(cls, json_str: str) -> "CritiqueResult":
        data = json.loads(json_str)
        return cls.from_dict(data)


# =============================================================================
# Parsing Helpers
# =============================================================================

def extract_json_from_llm_output(raw_output: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from LLM output that may contain markdown/prose.
    
    Handles:
    - Clean JSON
    - JSON in ```json code blocks
    - JSON with leading/trailing prose
    
    Returns parsed dict or None if no valid JSON found.
    """
    if not raw_output:
        return None
    
    text = raw_output.strip()
    
    # Try 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try 2: Extract from code fence
    fence_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text, re.IGNORECASE)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Try 3: Find JSON object boundaries
    start = text.find("{")
    if start == -1:
        return None
    
    # Find matching closing brace
    depth = 0
    end = -1
    in_string = False
    escape = False
    
    for i, char in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    
    if end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    
    return None


def parse_critique_output(raw_output: str, model: str = "") -> CritiqueResult:
    """Parse LLM critique output into structured CritiqueResult.
    
    Args:
        raw_output: Raw LLM output (may be JSON or mixed)
        model: Model that generated the critique
    
    Returns:
        CritiqueResult with parsed issues, or empty result on parse failure
    """
    data = extract_json_from_llm_output(raw_output)
    
    if data is None:
        logger.warning("[critique] Failed to parse JSON from critique output")
        return CritiqueResult(
            summary="Failed to parse critique output",
            critique_model=model,
        )
    
    result = CritiqueResult.from_dict(data)
    result.critique_model = model
    return result


# =============================================================================
# Prompt Builders for JSON Critique
# =============================================================================

CRITIQUE_JSON_SCHEMA = """{
  "blocking_issues": [
    {
      "id": "ISSUE-001",
      "spec_ref": "MUST-3",
      "arch_ref": "Section 2.1",
      "category": "security|correctness|completeness|clarity|performance",
      "severity": "blocking",
      "description": "What is wrong",
      "fix_suggestion": "How to fix it"
    }
  ],
  "non_blocking_issues": [
    {
      "id": "ISSUE-002",
      "spec_ref": "SHOULD-1",
      "arch_ref": "Section 3.2",
      "category": "...",
      "severity": "non_blocking",
      "description": "...",
      "fix_suggestion": "..."
    }
  ],
  "summary": "Brief overall assessment",
  "spec_coverage": {
    "MUST-1": "covered",
    "MUST-2": "partial",
    "SHOULD-1": "missing"
  }
}"""


def build_json_critique_prompt(
    draft_text: str,
    original_request: str,
    spec_json: Optional[str] = None,
    spec_markdown: Optional[str] = None,
    env_context: Optional[Dict[str, Any]] = None,
    segment_contract_markdown: Optional[str] = None,
) -> str:
    """Build prompt for structured JSON critique output.
    
    v1.2 (2026-02-02): Now accepts spec_markdown for grounded critique.
    The POT spec markdown contains VERIFIED evidence (file paths, line numbers).
    Critique judges ONLY against what's in the spec, not invented constraints.
    
    Args:
        draft_text: The architecture document to critique
        original_request: Original user request
        spec_json: Optional spec JSON for traceability (metadata)
        spec_markdown: Optional POT spec markdown (AUTHORITATIVE source of truth)
        env_context: Optional environment constraints
    
    Returns:
        Prompt string requesting JSON critique output
    """
    # =========================================================================
    # v1.2: POT SPEC MARKDOWN (PRIMARY SOURCE OF TRUTH)
    # =========================================================================
    # This is the AUTHORITATIVE contract. Critique judges ONLY against this.
    # If user requested "OpenAI API", that's in the spec, so it's ALLOWED.
    
    pot_spec_section = ""
    if spec_markdown:
        pot_spec_section = f"""
{'='*70}
POT SPEC - AUTHORITATIVE CONTRACT (CRITIQUE JUDGES AGAINST THIS)
{'='*70}

The following POT spec is the AUTHORITATIVE source of truth for this task.
It contains VERIFIED information from the codebase and user requirements.

Your critique MUST:
1. Judge the architecture ONLY against what's in this spec
2. NOT invent constraints that aren't in the spec
3. NOT flag user-requested features as violations (e.g., if spec says "use OpenAI API", that's ALLOWED)
4. Check that the architecture addresses what the spec requires
5. Check grounding: do referenced file paths exist in the spec evidence?

SECTION AUTHORITY LEVELS:
========================
Not all sections of this spec carry equal authority. Some are USER REQUIREMENTS
(hard constraints from the user), others are LLM-GENERATED SUGGESTIONS
(implementation guidance from automated analysis).

HARD REQUIREMENTS (blocking if missed):
- 'Goal' section — what the user asked for
- 'Constraints' section — explicit limits stated by the user
- 'Scope' section — what's in/out of scope
- 'Acceptance' criteria that map to explicit user requests
- 'Implementation Stack' (if STACK LOCKED) — non-negotiable tech choice

SOFT GUIDANCE (non-blocking only — architecture MAY choose alternatives):
- 'Files to Modify' — LLM-suggested integration points from codebase analysis
- 'Reference Files' — patterns identified by analysis, not mandated by user
- 'Implementation Steps' — LLM-suggested execution order
- 'New Files to Create' — LLM-suggested file structure
- 'Patterns to Follow' / 'Existing Patterns' — LLM-extracted code patterns
- 'LLM Architecture Analysis' — automated analysis output
These sections are implementation SUGGESTIONS. The architecture may choose
completely different files, approaches, or structures if they better serve
the hard requirements. Only flag as NON-BLOCKING if the alternative approach
seems technically problematic.

You should flag as BLOCKING only if:
- Architecture MISSES something the spec REQUIRES (from HARD REQUIREMENT sections)
- Architecture CONTRADICTS something the spec STATES (from HARD REQUIREMENT sections)
- Architecture references files/paths NOT in the spec evidence
- Architecture has internal contradictions

You should NOT flag as blocking:
- Architecture choosing different files than 'Files to Modify' suggests
- Architecture using a different integration approach than the LLM analysis suggested
- User-requested external integrations (if spec says "use X API", that's allowed)
- Technology choices that align with the spec
- Features that the spec explicitly requested

{spec_markdown}

{'='*70}
END OF POT SPEC - Judge architecture against ONLY the HARD REQUIREMENTS above
{'='*70}
"""

    spec_section = ""
    if spec_json and not spec_markdown:
        # Only use spec_json if spec_markdown is not provided (backward compat)
        spec_section = f"""

SPECIFICATION (from Spec Gate):
```json
{spec_json}
```

You MUST check coverage of every MUST requirement from the spec.
"""

    env_section = ""
    if env_context:
        env_section = f"""

ENVIRONMENT CONSTRAINTS:
{json.dumps(env_context, indent=2)}

Flag any architecture decisions that violate these constraints as blocking issues.
NOTE: If the POT spec explicitly requests something (like external API integration),
that OVERRIDES generic environment constraints. The spec is authoritative.
"""

    # v5.4 PHASE 2B: Interface contract compliance check
    _contract_section = ""
    if segment_contract_markdown:
        _contract_section = f"""
INTERFACE CONTRACT COMPLIANCE (v5.4 — BLOCKING):
{'='*60}
This architecture is for ONE SEGMENT of a multi-segment job.
The Critical Supervisor has defined interface contracts that this
segment MUST conform to. Violations are BLOCKING because they
will cause cross-segment integration failures.

{segment_contract_markdown}

CONTRACT COMPLIANCE CHECKS (all BLOCKING if violated):
1. Every interface listed under "MUST EXPOSE" exists in the architecture
2. Names match EXACTLY (class names, function names, file paths)
3. Signatures match EXACTLY (parameters, types, return types)
4. Import paths match EXACTLY
5. Data shapes match EXACTLY (fields, types)
6. Interfaces listed under "CONSUMES" are imported correctly
7. No renaming, no "creative interpretation" of contract names

If the architecture violates ANY contract term, mark it as a BLOCKING issue
with category "contract_violation" and reference the specific contract term.
{'='*60}

"""

    return f"""You are a senior architecture reviewer. Critique the following architecture document.
{pot_spec_section}

Your output MUST be valid JSON matching this schema exactly:
```json
{CRITIQUE_JSON_SCHEMA}
```

CRITICAL CALIBRATION - READ CAREFULLY:
======================================
A "blocking" issue is something that would make the system INCORRECT or UNBUILDABLE.

BLOCKING issues (should catch real problems):
- Security vulnerabilities that could cause data breach or system compromise
- Missing components that are EXPLICITLY REQUIRED in the spec
- Architectural decisions that make the spec requirements IMPOSSIBLE to implement
- INTERNAL CONTRADICTIONS within the document (e.g., says "20 rows" in one place, "22 rows" elsewhere)
- VALUES THAT CONTRADICT USER ANSWERS from the spec (e.g., spec says DAS=150ms but doc says 167ms)
- Off-by-one errors or incorrect calculations that would produce wrong behavior
- Incomplete sections that are critical to implementation (truncated content)

NOT BLOCKING (use non_blocking instead):
- Typos, grammar, formatting issues that don't affect meaning
- Missing OPTIONAL features or nice-to-haves not in spec
- Performance optimizations that aren't spec requirements
- Style preferences or alternative approaches that would still work
- Minor documentation gaps that don't affect implementation
- Suggestions for improvement beyond spec requirements

KEY CHECKS:
1. Internal consistency: Do all numbers/values match throughout the document?
2. Spec compliance: Do values match what the user specified in Q&A?
3. Completeness: Are all MUST requirements from spec addressed?
4. Correctness: Are calculations and references accurate?

ASK YOURSELF: "Would building this produce the CORRECT system per the spec?"
If values contradict or are inconsistent, that's BLOCKING even if technically buildable.

EVIDENCE REQUIREMENT:
=====================
For a blocking issue to be valid, you MUST provide BOTH:
- spec_ref: Which spec requirement is violated
- arch_ref: Which part of the architecture shows the problem

If you cannot cite both, make it non_blocking.

RULES:
1. blocking_issues: Issues that would result in incorrect or broken implementation
2. non_blocking_issues: Style, optimization, and enhancement suggestions
3. Each issue MUST have a unique id (ISSUE-001, ISSUE-002, etc.)
4. Each blocking issue MUST reference spec_ref AND arch_ref (evidence required)
5. category must be one of: security, correctness, completeness, spec_mismatch, broken_flow
6. overall_pass is derived: true if blocking_issues is empty, false otherwise
7. spec_coverage maps each spec requirement to: covered, partial, missing, or not_applicable

{_contract_section}ORIGINAL REQUEST:
{original_request}
{spec_section}{env_section}
ARCHITECTURE DOCUMENT TO CRITIQUE:
```
{draft_text}
```

CRITICAL GROUNDING CHECK:
=========================
If a POT spec was provided above, your critique is BOUND to that spec.
- The spec IS the contract - you cannot add terms to it
- If the user requested an external API, that's allowed per the spec
- Only flag violations of what's ACTUALLY in the spec

Output ONLY valid JSON, no markdown, no explanation, no preamble."""


def build_json_revision_prompt(
    draft_text: str,
    original_request: str,
    critique: CritiqueResult,
    spec_json: Optional[str] = None,
) -> str:
    """Build prompt for revision that addresses blocking issues.
    
    Args:
        draft_text: Current architecture document
        original_request: Original user request
        critique: Parsed critique with blocking issues
        spec_json: Optional spec JSON for reference
    
    Returns:
        Prompt string for revision
    """
    issues_text = ""
    for issue in critique.blocking_issues:
        issues_text += f"""
### {issue.id}
- **Category:** {issue.category}
- **Spec Reference:** {issue.spec_ref or 'N/A'}
- **Problem:** {issue.description}
- **Required Fix:** {issue.fix_suggestion}
"""

    spec_section = ""
    if spec_json:
        spec_section = f"""

SPECIFICATION (for reference):
```json
{spec_json}
```
"""

    return f"""You are revising an architecture document to address blocking issues from review.

RULES:
1. Address EVERY blocking issue listed below
2. For each issue, clearly fix the problem in the relevant section
3. Maintain overall document structure and completeness
4. Do not introduce new problems
5. Output the complete revised architecture document

ORIGINAL REQUEST:
{original_request}
{spec_section}
BLOCKING ISSUES TO ADDRESS:
{issues_text}

CURRENT ARCHITECTURE DOCUMENT:
```
{draft_text}
```

Output the complete revised architecture document. Do not include meta-commentary about the revision process."""


__all__ = [
    # Blocker types
    "APPROVED_ARCHITECTURE_BLOCKER_TYPES",
    # Data classes
    "CritiqueIssue",
    "CritiqueResult",
    # Parsing
    "extract_json_from_llm_output",
    "parse_critique_output",
    # Prompt builders
    "build_json_critique_prompt",
    "build_json_revision_prompt",
    "CRITIQUE_JSON_SCHEMA",
]
