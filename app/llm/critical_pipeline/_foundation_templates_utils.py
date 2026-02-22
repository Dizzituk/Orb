from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


FOUNDATION_TEMPLATES_BUILD_ID = "2026-02-10-v1.0-foundation-templates"

@dataclass
class FoundationTemplate:
    """A single foundation pattern."""
    id: str                          # Unique ID (e.g. "fastapi-auth-jwt")
    name: str                        # Display name
    category: str                    # auth | persistence | state | api | error | config
    tech_tags: Set[str]              # Match triggers (e.g. {"fastapi", "python"})
    concept_tags: Set[str]           # Concept triggers (e.g. {"auth", "login", "jwt"})
    description: str                 # What this pattern provides
    pattern_markdown: str            # The actual architecture pattern
    file_patterns: List[str] = field(default_factory=list)  # Example file layout
    dependencies: List[str] = field(default_factory=list)    # Template IDs this depends on

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "file_patterns": self.file_patterns,
        }

@dataclass
class MatchedTemplates:
    """Result of template matching against a spec."""
    templates: List[FoundationTemplate] = field(default_factory=list)
    match_reasons: Dict[str, str] = field(default_factory=dict)  # template_id → reason

    @property
    def count(self) -> int:
        return len(self.templates)

    def format_for_prompt(self) -> str:
        """Format matched templates as markdown for architecture prompt injection."""
        if not self.templates:
            return ""

        sections = []
        sections.append("=" * 60)
        sections.append("FOUNDATION PATTERNS — Pre-Validated Reference Architecture")
        sections.append("=" * 60)
        sections.append("")
        sections.append(
            "The following architectural patterns are RECOMMENDED starting points "
            "for this project. They have been pre-validated to work together. "
            "You SHOULD follow these patterns unless the spec explicitly requires "
            "a different approach. If you deviate, document WHY in the architecture."
        )
        sections.append("")

        for tmpl in self.templates:
            reason = self.match_reasons.get(tmpl.id, "")
            sections.append(f"### {tmpl.name} ({tmpl.category})")
            if reason:
                sections.append(f"_Matched because: {reason}_")
            sections.append("")
            sections.append(tmpl.description)
            sections.append("")
            if tmpl.file_patterns:
                sections.append("**Suggested file layout:**")
                for fp in tmpl.file_patterns:
                    sections.append(f"  - `{fp}`")
                sections.append("")
            sections.append(tmpl.pattern_markdown)
            sections.append("")
            sections.append("---")
            sections.append("")

        sections.append("=" * 60)
        sections.append("END FOUNDATION PATTERNS")
        sections.append("=" * 60)

        return "\n".join(sections)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "templates": [t.to_dict() for t in self.templates],
            "match_reasons": self.match_reasons,
        }

def _register(tmpl: FoundationTemplate):
    """Register a template in the global registry."""
    from .foundation_templates import _REGISTRY
    _REGISTRY.append(tmpl)

def match_templates(
    tech_stack: Optional[Dict[str, str]] = None,
    spec_concepts: Optional[List[str]] = None,
    spec_text: Optional[str] = None,
    max_templates: int = 5,
) -> MatchedTemplates:
    """
    Match foundation templates against a job's tech stack and concepts.

    Args:
        tech_stack: Dict with keys like frontend_framework, backend_framework, etc.
        spec_concepts: Extracted concept keywords from the spec
        spec_text: Raw spec text for keyword scanning
        max_templates: Maximum templates to return (highest scoring first)

    Returns:
        MatchedTemplates with ranked matches and reasons
    """
    from .foundation_templates import _REGISTRY
    tech_stack = tech_stack or {}
    spec_concepts = [c.lower() for c in (spec_concepts or [])]

    # Build a set of tech indicators from the stack
    tech_indicators: Set[str] = set()
    for key, value in tech_stack.items():
        if value:
            tech_indicators.add(value.lower())
            # Also add common aliases
            _v = value.lower()
            if "fastapi" in _v:
                tech_indicators.update({"fastapi", "python"})
            elif "express" in _v:
                tech_indicators.update({"express", "node", "javascript"})
            elif "react" in _v:
                tech_indicators.add("react")
            elif "next" in _v:
                tech_indicators.update({"nextjs", "react"})
            elif "vue" in _v:
                tech_indicators.add("vue")
            if "typescript" in _v:
                tech_indicators.add("typescript")
            if "python" in _v:
                tech_indicators.add("python")

    # Extract additional concepts from spec text
    text_concepts: Set[str] = set()
    if spec_text:
        _text_lower = spec_text.lower()
        # Scan for concept keywords
        concept_keywords = [
            "auth", "login", "register", "jwt", "token", "user",
            "database", "db", "persistence", "sql", "crud",
            "api", "endpoint", "route", "rest",
            "state", "store", "context",
            "config", "settings", "environment",
            "error", "exception", "handling",
            "migration", "schema", "model",
        ]
        for kw in concept_keywords:
            if kw in _text_lower:
                text_concepts.add(kw)

    all_concepts = set(spec_concepts) | text_concepts

    # Score each template
    scored: List[tuple] = []  # (score, template, reason)

    for tmpl in _REGISTRY:
        score = 0
        reasons = []

        # Tech match (strong signal)
        tech_overlap = tmpl.tech_tags & tech_indicators
        if tech_overlap:
            score += len(tech_overlap) * 3
            reasons.append(f"tech: {', '.join(tech_overlap)}")

        # Concept match
        concept_overlap = tmpl.concept_tags & all_concepts
        if concept_overlap:
            score += len(concept_overlap) * 2
            reasons.append(f"concepts: {', '.join(concept_overlap)}")

        # Only include if we have BOTH tech AND concept match
        # (prevents auth templates showing for non-auth projects)
        if tech_overlap and concept_overlap:
            scored.append((score, tmpl, " + ".join(reasons)))

    # Sort by score descending, take top N
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:max_templates]

    result = MatchedTemplates()
    for _score, tmpl, reason in top:
        result.templates.append(tmpl)
        result.match_reasons[tmpl.id] = reason

    if result.count:
        logger.info(
            "[foundation_templates] Matched %d templates: %s",
            result.count,
            [t.id for t in result.templates],
        )
    else:
        logger.debug("[foundation_templates] No templates matched")

    return result
