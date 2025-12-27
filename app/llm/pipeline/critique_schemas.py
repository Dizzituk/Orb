# FILE: app/llm/pipeline/critique_schemas.py
"""Structured schemas for machine-driven critique pipeline (Block 5).

The critique output is strict JSON for deterministic pass/fail decisioning.
A parallel markdown artifact is generated for human readability.

v1 (2025-12):
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
        overall_pass: True iff blocking_issues is empty
        summary: Brief human-readable summary
        spec_coverage: Dict mapping spec requirements to coverage status
        critique_model: Model that generated this critique
        critique_version: Schema version
    """
    blocking_issues: List[CritiqueIssue] = field(default_factory=list)
    non_blocking_issues: List[CritiqueIssue] = field(default_factory=list)
    overall_pass: bool = False
    summary: str = ""
    spec_coverage: Dict[str, str] = field(default_factory=dict)
    critique_model: str = ""
    critique_version: str = "v2"
    
    def __post_init__(self):
        # Enforce: overall_pass is True iff no blocking issues
        self.overall_pass = len(self.blocking_issues) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocking_issues": [i.to_dict() for i in self.blocking_issues],
            "non_blocking_issues": [i.to_dict() for i in self.non_blocking_issues],
            "overall_pass": self.overall_pass,
            "summary": self.summary,
            "spec_coverage": self.spec_coverage,
            "critique_model": self.critique_model,
            "critique_version": self.critique_version,
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
    env_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build prompt for structured JSON critique output.
    
    Args:
        draft_text: The architecture document to critique
        original_request: Original user request
        spec_json: Optional spec JSON for traceability
        env_context: Optional environment constraints
    
    Returns:
        Prompt string requesting JSON critique output
    """
    spec_section = ""
    if spec_json:
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
"""

    return f"""You are a senior architecture reviewer. Critique the following architecture document.

Your output MUST be valid JSON matching this schema exactly:
```json
{CRITIQUE_JSON_SCHEMA}
```

RULES:
1. blocking_issues: Problems that MUST be fixed before approval (security flaws, spec violations, missing critical components)
2. non_blocking_issues: Improvements that SHOULD be made but don't block approval
3. Each issue MUST have a unique id (ISSUE-001, ISSUE-002, etc.)
4. Each issue SHOULD reference spec_ref (which spec requirement) and arch_ref (which section)
5. category must be one of: security, correctness, completeness, clarity, performance
6. overall_pass is derived: true if blocking_issues is empty, false otherwise
7. spec_coverage maps each spec requirement to: covered, partial, missing, or not_applicable

ORIGINAL REQUEST:
{original_request}
{spec_section}{env_section}
ARCHITECTURE DOCUMENT TO CRITIQUE:
```
{draft_text}
```

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
