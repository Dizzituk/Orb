# FILE: app/llm/pipeline/critique_schemas.py
# Purpose: Structured schemas for machine-driven critique pipeline (Block 5).
# Called-by: app.llm.critical_pipeline.config, app.llm.pipeline._critique_schemas_utils, app.llm.pipeline.critique, app.llm.pipeline.critique_parts.blocker_filtering (+8 more)
# Depends-on: app.llm.pipeline._critique_schemas_utils
# Last-renovated: 2026-06-11
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
from app.llm.pipeline._critique_schemas_utils import APPROVED_ARCHITECTURE_BLOCKER_TYPES, CRITIQUE_JSON_SCHEMA, KNOWN_ARCHITECTURE_ISSUE_TYPES, build_json_critique_prompt, build_json_revision_prompt, parse_critique_output

logger = logging.getLogger(__name__)


# =============================================================================
# Approved Blocker Types (v1.1)
# =============================================================================

# These are the ONLY categories that can block an architecture job.
# Any blocking issue with a category NOT in this set will be downgraded.


# =============================================================================
# Known Architecture Issue Types (v2.0 - includes non-blocking transition types)
# =============================================================================
# Types that are RECOGNIZED but not yet blocking. These exist for diagnostics
# and will be promoted to APPROVED_ARCHITECTURE_BLOCKER_TYPES once all stages
# reliably emit the corresponding data.


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


# =============================================================================
# Prompt Builders for JSON Critique
# =============================================================================


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
