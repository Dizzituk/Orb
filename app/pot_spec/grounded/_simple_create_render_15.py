# Purpose: simple create utils 15 — grounded CREATE spec markdown render (split from _simple_create_utils_15.py).
# Called-by: app.pot_spec.grounded._simple_create_utils_15
# Depends-on: app.pot_spec.grounded._simple_create_utils_12, app.pot_spec.grounded._simple_create_utils_14
# Last-renovated: 2026-06-21
from __future__ import annotations
from typing import List
from app.pot_spec.grounded._simple_create_utils_12 import _extract_acceptance_from_constraints
from app.pot_spec.grounded._simple_create_utils_14 import _sanitize_goal


def build_create_spec(
    goal: str,
    what_to_do: str,
    evidence: CreateEvidence,
    project_paths: List[str],
) -> str:
    """
    v2.0: Build a grounded CREATE spec with evidence and LLM analysis.
    
    If LLM analysis is available, uses it for implementation steps and
    acceptance criteria. Falls back to weaver output if LLM unavailable.
    """
    from ._simple_create_utils_16 import CreateEvidence
    lines = []
    
    sanitized_goal = _sanitize_goal(goal, what_to_do)
    
    # Header
    lines.append("# SPoT Spec — New Feature")
    lines.append("")
    
    # Goal
    lines.append("## Goal")
    lines.append("")
    lines.append(sanitized_goal)
    lines.append("")
    
    # Tech Stack Context
    lines.append("## Tech Stack (Detected)")
    lines.append("")
    stack = evidence.tech_stack
    if stack.frontend_framework:
        lines.append(f"- **Frontend**: {stack.frontend_framework}" + 
                    (f" ({stack.frontend_language})" if stack.frontend_language else ""))
    if stack.backend_framework:
        lines.append(f"- **Backend**: {stack.backend_framework}" +
                    (f" ({stack.backend_language})" if stack.backend_language else ""))
    if stack.styling:
        lines.append(f"- **Styling**: {stack.styling}")
    if stack.state_management:
        lines.append(f"- **State**: {stack.state_management}")
    lines.append("")
    
    # Constraints (v2.0 — NEW)
    if evidence.constraints:
        lines.append("## Constraints")
        lines.append("")
        for c in evidence.constraints:
            lines.append(f"- ⛔ {c}")
        lines.append("")
    
    # Integration Points (WHERE)
    lines.append("## Integration Points")
    lines.append("")
    
    # v3.3: Deduplicated, single list. Architecture decides what to modify.
    seen_names = set()
    deduped_points = []
    for p in evidence.integration_points:
        if p.file_name not in seen_names:
            seen_names.add(p.file_name)
            deduped_points.append(p)
    
    if deduped_points:
        lines.append("### Discovered Files (architecture determines what to modify)")
        lines.append("")
        for p in deduped_points[:10]:
            lines.append(f"- `{p.file_name}` — {p.relevance}")
        lines.append("")
    
    # Suggested New Files
    if evidence.suggested_files:
        lines.append("### Suggested New Files")
        lines.append("")
        for f in evidence.suggested_files:
            lines.append(f"- `{f}`")
        lines.append("")
    
    # v5.2: Resolved Target Files — deterministic, survives LLM analysis collapse.
    # These are the actual files mentioned in the Weaver output, resolved to
    # real filesystem paths by _resolve_mentioned_files(). They MUST appear in
    # the spec markdown so _extract_file_scope_from_spec() can find them for
    # segmentation, regardless of what the LLM analysis contains.
    _resolved = getattr(evidence, 'resolved_target_files', None)
    if _resolved:
        lines.append("## Target Files (Resolved)")
        lines.append("")
        for rtf in _resolved:
            resolved_path = rtf.get('resolved_path', rtf.get('mentioned', ''))
            mentioned = rtf.get('mentioned', '')
            if resolved_path:
                lines.append(f"- `{resolved_path}`")
        lines.append("")
    
    # v2.0: LLM Analysis or Requirements
    if evidence.llm_analysis:
        lines.append("## Architecture Analysis")
        lines.append("")
        lines.append("")
        lines.append("")
        lines.append(evidence.llm_analysis)
        lines.append("")
    else:
        # Fallback: include weaver output as requirements (not "implementation steps")
        lines.append("## Requirements (from Weaver)")
        lines.append("")
        if what_to_do:
            for line in what_to_do.split('\n'):
                stripped = line.strip()
                if stripped and not stripped.lower().startswith('what is being built'):
                    lines.append(stripped)
        lines.append("")
    
    # v3.3: Removed raw code excerpts from spec.
    # The critical pipeline reads referenced files directly during implementation.
    # Pasting stale 400-char snippets adds noise without value.
    
    # v2.0: Task-specific Acceptance Criteria
    lines.append("## Acceptance")
    lines.append("")
    
    # Always include base criteria
    lines.append("- [ ] Feature works as described in requirements")
    lines.append("- [ ] Integrates with existing UI patterns")
    lines.append("- [ ] No console errors")
    lines.append("- [ ] App boots without issues")
    
    # Add constraint-derived criteria (v2.0)
    constraint_criteria = _extract_acceptance_from_constraints(evidence.constraints)
    for cc in constraint_criteria:
        lines.append(f"- [ ] {cc}")
    
    lines.append("")
    
    # Evidence Summary
    lines.append("## Evidence Summary")
    lines.append("")
    lines.append(f"- Integration points found: {len(evidence.integration_points)}")
    lines.append(f"- Patterns extracted: {len(evidence.existing_patterns)}")
    lines.append(f"- Constraints detected: {len(evidence.constraints)}")
    lines.append(f"- LLM analysis: {'Yes' if evidence.llm_analysis else 'No (fallback mode)'}")
    lines.append(f"- Project paths: {', '.join(project_paths)}")
    lines.append("")
    
    return "\n".join(lines)
