from __future__ import annotations
import logging
import os
import re
from app.pot_spec.grounded._simple_create_utils_12 import _extract_acceptance_from_constraints
from app.pot_spec.grounded._simple_create_utils_13 import ARCHITECTURAL_FILE_PATTERNS, _score_integration_point
from app.pot_spec.grounded._simple_create_utils_14 import CONCEPT_DIRECTORY_PATTERNS, _EVIDENCE_MAX_FILE_CHARS, _sanitize_goal
from typing import Any, Dict, List, Optional, Tuple
from app.pot_spec.grounded._sbx_fs import _sbx_isfile, _sbx_isdir, _sbx_exists, _sbx_read, _sbx_ls
logger = logging.getLogger(__name__)

# v3.2-fix: Sandbox-aware filesystem checks for codebase paths.
# v4.3: Spec gate evidence reads from HOST filesystem, not sandbox.
# The sandbox controller is only available during execution (Windows Sandbox).
# Using it here causes "file not found" for files that exist on disk.


def _read_text_any_encoding(file_path: str) -> str:
    """
    v10.0: Read a text file from the sandbox.

    The sandbox controller handles encoding internally.
    No host fallbacks. No multi-encoding retry.
    """
    content = _sbx_read(file_path)
    return content if content is not None else ""

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
    from ._simple_create_utils_16 import IntegrationPoint
    points = []
    
    skip_dirs = {
        'node_modules', '.git', '__pycache__', '.venv', 'venv',
        'dist', 'build', '.next', 'coverage', '.architecture',
        '_backup_before_audit', '_patches',
    }
    source_exts = ('.tsx', '.jsx', '.ts', '.js', '.py', '.css')

    def _scan_dir(dir_path: str, rel_root: str) -> None:
        try:
            entries = _sbx_ls(dir_path)
        except Exception as e:
            logger.warning("[simple_create] Error scanning %s: %s", dir_path, e)
            return
        for name in entries:
            if name in skip_dirs:
                continue
            full_path = os.path.join(dir_path, name)
            ext = os.path.splitext(name)[1].lower()
            if ext and ext in source_exts:
                # It's a source file
                for pattern, relevance, action in ARCHITECTURAL_FILE_PATTERNS:
                    if re.match(pattern, name, re.IGNORECASE):
                        points.append(IntegrationPoint(
                            file_path=full_path,
                            file_name=name,
                            relevance=relevance,
                            action=action,
                        ))
                        break
                for concept in concepts:
                    dir_patterns = CONCEPT_DIRECTORY_PATTERNS.get(concept, [])
                    for dir_pat in dir_patterns:
                        if dir_pat in rel_root.lower():
                            if not any(p.file_path == full_path for p in points):
                                points.append(IntegrationPoint(
                                    file_path=full_path,
                                    file_name=name,
                                    relevance=f"In '{dir_pat}/' directory (relevant to {concept})",
                                    action="reference",
                                ))
                            break
            elif not ext:
                # Likely a directory — recurse
                child_rel = os.path.join(rel_root, name) if rel_root != '.' else name
                _scan_dir(full_path, child_rel)

    try:
        _scan_dir(project_path, '.')
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
    from ._simple_create_utils_16 import IntegrationPoint
    from ._simple_create_utils_17 import TechStack
    patterns = {}
    
    for point in integration_points:
        if point.action != "modify":
            continue
        
        content = _sbx_read(point.file_path)
        if content:
            
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
    
    return patterns

def _resolve_evidence_path(
    file_path: str,
    project_paths: Optional[List[str]] = None,
) -> Optional[str]:
    """Resolve a bare or relative file path to an absolute sandbox path.

    v10.0: Uses architecture INDEX.json first (fast, no sandbox round-trips),
    then tries common subdirectory patterns, then direct root+path.
    All existence checks go through the sandbox. No host fallbacks.

    Returns the resolved absolute path, or None if not found.
    """
    basename = os.path.basename(file_path)

    # Strategy 1: Architecture INDEX.json lookup (host operational data)
    try:
        import json as _json
        _idx_path = os.path.join("D:\\Orb", ".architecture", "INDEX.json")
        if os.path.isfile(_idx_path):
            with open(_idx_path, "r", encoding="utf-8") as _f:
                _idx = _json.load(_f)
            for _entry in _idx.get("files", []):
                _name = _entry.get("name", "")
                _path = _entry.get("path", "")
                if _name == basename and _path:
                    if _sbx_isfile(_path):
                        return _path
                # Also try matching the relative path suffix
                if _path and _path.replace("\\", "/").endswith(
                    file_path.replace("\\", "/")
                ):
                    if _sbx_isfile(_path):
                        return _path
    except Exception:
        pass

    # Strategy 2: Try common subdirectory patterns via sandbox
    _subdirs = [
        "src", "app", "src/components", "src/services", "src/types",
        "src/components/chat-panel", "src/components/debug",
        "src/components/builds",
    ]
    if project_paths:
        for root in project_paths:
            for subdir in _subdirs:
                candidate = os.path.join(root, subdir, file_path)
                if _sbx_isfile(candidate):
                    return candidate
            # Direct root + path
            candidate = os.path.join(root, file_path)
            if _sbx_isfile(candidate):
                return candidate

    return None


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

    # v10.0: Resolve relative/bare paths using INDEX.json + sandbox checks
    if not _sbx_exists(file_path):
        resolved = _resolve_evidence_path(file_path, project_paths)
        if resolved:
            logger.info("[SPEC_GATE_EVIDENCE] Resolved path: %s → %s", file_path, resolved)
            file_path = resolved

    # v4.2: TypeScript barrel export fallback.
    # If 'foo.ts' not found, try 'foo/index.ts' and 'foo/index.tsx'.
    # Common TS pattern: `import { X } from './types'` resolves to types/index.ts
    if not _sbx_exists(file_path):
        _tried_barrel = False
        if file_path.endswith(('.ts', '.tsx')):
            _stem = file_path.rsplit('.', 1)[0]
            for _barrel_ext in ('/index.ts', '/index.tsx'):
                _barrel = _stem + _barrel_ext
                if _sbx_exists(_barrel):
                    logger.info(
                        "[SPEC_GATE_EVIDENCE] v4.2 Barrel fallback: %s → %s",
                        file_path, _barrel,
                    )
                    file_path = _barrel
                    _tried_barrel = True
                    break
        if not _tried_barrel:
            logger.info("[SPEC_GATE_EVIDENCE] File not found: %s", file_path)
            return False, f"File not found: {file_path}"

    if not _sbx_isfile(file_path):
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
