from __future__ import annotations
import logging
import os
import re
from app.pot_spec.grounded._simple_create_utils import ARCHITECTURAL_FILE_PATTERNS, _score_integration_point
from app.pot_spec.grounded._simple_create_utils import CONCEPT_DIRECTORY_PATTERNS, _EVIDENCE_MAX_FILE_CHARS, _sanitize_goal
from app.pot_spec.grounded._simple_create_utils import _extract_acceptance_from_constraints
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


def _read_text_any_encoding(file_path: str) -> str:
    """
    v2.1: Read a text file trying multiple encodings.
    
    Some files (e.g., pip freeze output on Windows) are UTF-16 encoded.
    If default encoding fails to produce readable content, try alternatives.
    """
    # Try encodings in order of likelihood
    encodings = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1']
    
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read()
            # Sanity check: UTF-16 files read as UTF-8 will have null bytes
            # appearing as spaces between every character
            if enc == 'utf-8' and '\x00' in content:
                continue  # Try next encoding
            # Another sanity check: if every other char is a space and content
            # is very long, it's probably UTF-16 misread as UTF-8
            if enc == 'utf-8' and len(content) > 100:
                sample = content[:200]
                space_ratio = sample.count(' ') / len(sample) if sample else 0
                if space_ratio > 0.35:  # More than 35% spaces is suspicious
                    continue
            return content
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            continue
    
    # Last resort: read as binary and decode with errors='replace'
    try:
        with open(file_path, 'rb') as f:
            raw = f.read()
        return raw.decode('utf-8', errors='replace')
    except Exception:
        return ""

def _find_integration_points(
    project_path: str,
    concepts: List[str],
    sandbox_client: Any = None,
) -> List[IntegrationPoint]:
    """
    v2.0: Find architecturally relevant integration points.
    
    Uses SPECIFIC file patterns (regex on full filename) instead of
    substring matching. Also searches for existing directories that
    match the task's concepts.
    """
    points = []
    
    try:
        for root, dirs, files in os.walk(project_path):
            # Skip non-source directories
            dirs[:] = [d for d in dirs if d not in {
                'node_modules', '.git', '__pycache__', '.venv', 'venv',
                'dist', 'build', '.next', 'coverage', '.architecture',
                '_backup_before_audit', '_patches',
            }]
            
            rel_root = os.path.relpath(root, project_path)
            
            for filename in files:
                # Only check source files
                if not filename.endswith(('.tsx', '.jsx', '.ts', '.js', '.py', '.css')):
                    continue
                
                full_path = os.path.join(root, filename)
                
                # v2.0: Match against SPECIFIC architectural patterns (regex on full filename)
                for pattern, relevance, action in ARCHITECTURAL_FILE_PATTERNS:
                    if re.match(pattern, filename, re.IGNORECASE):
                        points.append(IntegrationPoint(
                            file_path=full_path,
                            file_name=filename,
                            relevance=relevance,
                            action=action,
                        ))
                        break
                
                # v2.0: Match files in concept-relevant directories
                for concept in concepts:
                    dir_patterns = CONCEPT_DIRECTORY_PATTERNS.get(concept, [])
                    for dir_pat in dir_patterns:
                        # Check if file is under a directory matching the concept
                        if dir_pat in rel_root.lower():
                            if not any(p.file_path == full_path for p in points):
                                points.append(IntegrationPoint(
                                    file_path=full_path,
                                    file_name=filename,
                                    relevance=f"In '{dir_pat}/' directory (relevant to {concept})",
                                    action="reference",
                                ))
                            break
    except Exception as e:
        logger.warning("[simple_create] Error scanning project: %s", e)
    
    # Dedupe and prioritize
    seen = set()
    unique = []
    for p in points:
        if p.file_path not in seen:
            seen.add(p.file_path)
            unique.append(p)
    
    # v3.7: Score each integration point using content signals + path heuristics.
    # Drop negative-scored points (false positives like static/main.py).
    # Sort remainder: modify actions first, then highest score, then filename.
    scored = []
    dropped = []
    for p in unique:
        s = _score_integration_point(p.file_path, project_path)
        if s < 0:
            dropped.append((p.file_name, s))
        else:
            scored.append((p, s))
    
    if dropped:
        print(f"[simple_create] v3.7 DROPPED {len(dropped)} negative-scored integration point(s): "
              f"{[(name, sc) for name, sc in dropped]}")
        logger.info("[simple_create] v3.7 Dropped %d negative-scored points: %s", len(dropped), dropped)
    
    scored.sort(key=lambda x: (0 if x[0].action == "modify" else 1, -x[1], x[0].file_name))
    result = [p for p, _ in scored]
    
    return result[:15]  # Limit to top 15

def _extract_patterns(
    integration_points: List[IntegrationPoint],
    tech_stack: TechStack,
) -> Dict[str, str]:
    """Extract coding patterns from existing files."""
    patterns = {}
    
    for point in integration_points:
        if point.action != "modify":
            continue
        
        try:
            with open(point.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract React component pattern
            if point.file_name.endswith(('.tsx', '.jsx')):
                # Find component definition
                comp_match = re.search(
                    r'((?:export\s+)?(?:const|function)\s+\w+\s*[=:]\s*(?:\([^)]*\)|[^=])*\s*(?:=>|{)[^}]*(?:return\s*\()?[^)]*<)',
                    content[:2000]
                )
                if comp_match:
                    patterns[f"component_pattern:{point.file_name}"] = comp_match.group(0)[:500]
                
                # Find import pattern
                import_match = re.search(r"^(import\s+.+\n)+", content, re.MULTILINE)
                if import_match:
                    patterns[f"import_pattern:{point.file_name}"] = import_match.group(0)[:300]
            
            # Extract API call pattern
            if 'api' in point.file_name.lower():
                fetch_match = re.search(
                    r'((?:export\s+)?(?:async\s+)?(?:function|const)\s+\w+\s*[=:]?\s*(?:async\s*)?\([^)]*\)[^{]*{[^}]*fetch[^}]*})',
                    content,
                    re.DOTALL
                )
                if fetch_match:
                    patterns["api_call_pattern"] = fetch_match.group(0)[:600]
                    
        except Exception as e:
            logger.debug("[simple_create] Could not read %s: %s", point.file_path, e)
    
    return patterns

def _host_read_file(file_path: str, max_chars: int = 0, project_paths: Optional[List[str]] = None) -> Tuple[bool, str]:
    """Read a file from the host filesystem for evidence fulfilment.

    v4.1: Added project_paths parameter for resolving relative paths.
    If file_path is not absolute or doesn't exist, tries resolving against
    each project root (e.g. 'app/llm/stream_router.py' → 'D:\\Orb\\app\\llm\\stream_router.py').

    Returns (success, content_or_error_message).
    Uses _read_text_any_encoding for robust encoding handling.
    """
    if not max_chars:
        max_chars = _EVIDENCE_MAX_FILE_CHARS

    # Normalise path separators for Windows
    file_path = file_path.replace('/', os.sep).replace('\\', os.sep)

    # v4.1: Resolve relative paths against project roots
    if not os.path.exists(file_path) and project_paths:
        for root in project_paths:
            candidate = os.path.join(root, file_path)
            candidate = candidate.replace('/', os.sep).replace('\\', os.sep)
            if os.path.exists(candidate):
                logger.info("[SPEC_GATE_EVIDENCE] Resolved relative path: %s → %s", file_path, candidate)
                file_path = candidate
                break

    if not os.path.exists(file_path):
        logger.info("[SPEC_GATE_EVIDENCE] File not found: %s", file_path)
        return False, f"File not found: {file_path}"

    if not os.path.isfile(file_path):
        logger.info("[SPEC_GATE_EVIDENCE] Not a file: %s", file_path)
        return False, f"Path is not a file: {file_path}"

    try:
        content = _read_text_any_encoding(file_path)
        if not content:
            return False, f"File is empty or unreadable: {file_path}"
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n... [truncated at {max_chars} chars, file has {len(content)} total]"
        logger.info("[SPEC_GATE_EVIDENCE] Read %d chars from %s", min(len(content), max_chars), file_path)
        return True, content
    except Exception as exc:
        logger.warning("[SPEC_GATE_EVIDENCE] Failed to read %s: %s", file_path, exc)
        return False, f"Read error: {exc}"

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
    
    modify_points = [p for p in evidence.integration_points if p.action == "modify"]
    reference_points = [p for p in evidence.integration_points if p.action != "modify"]
    
    if modify_points:
        lines.append("### Files to Modify (Suggested — architecture may choose alternatives)")
        lines.append("")
        lines.append("*These are LLM-suggested integration points from codebase analysis.*")
        lines.append("*The architecture may use different files or approaches if they better serve the requirements.*")
        lines.append("")
        for p in modify_points[:5]:
            lines.append(f"- `{p.file_name}` — {p.relevance}")
        lines.append("")
    
    if reference_points:
        lines.append("### Reference Files (patterns to follow — not mandatory)")
        lines.append("")
        for p in reference_points[:5]:
            lines.append(f"- `{p.file_name}` — {p.relevance}")
        lines.append("")
    
    # Suggested New Files
    if evidence.suggested_files:
        lines.append("### New Files to Create (Suggested — architecture determines final structure)")
        lines.append("")
        for f in evidence.suggested_files:
            lines.append(f"- `{f}`")
        lines.append("")
    
    # v2.0: LLM Analysis or Requirements
    if evidence.llm_analysis:
        lines.append("## LLM Architecture Analysis (Suggested — architecture determines final approach)")
        lines.append("")
        lines.append("*This analysis was generated by LLM codebase review. It is guidance, not a binding requirement.*")
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
    
    # Patterns (HOW)
    if evidence.existing_patterns:
        lines.append("## Existing Patterns to Follow")
        lines.append("")
        for name, pattern in list(evidence.existing_patterns.items())[:3]:
            short_name = name.split(':')[-1] if ':' in name else name
            lines.append(f"### Pattern from `{short_name}`")
            lines.append("```")
            pattern_preview = pattern[:400] + "..." if len(pattern) > 400 else pattern
            lines.append(pattern_preview)
            lines.append("```")
            lines.append("")
    
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
