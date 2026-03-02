# FILE: app/orchestrator/upstream_evidence.py
"""
Cross-Segment Evidence Propagation.

v1.0 (2026-03-01): Extracts public interfaces from approved/completed
upstream segment architectures and provides them as evidence to
downstream segments during architecture generation.

This solves the problem where seg-04 needs to know the function
signatures from seg-01/02/03 but those files don't exist on disk yet
(they're being created in this job). The evidence gatherer finds 0/5
files and the downstream architecture LLM has to guess.

The fix: after each segment's architecture is approved, parse its
File Inventory and code blocks to extract public interfaces (function
signatures, class names, exports). Store them keyed by segment ID.
Inject into downstream segments' context.

Uses the manifest's dependency graph to determine which upstream
segments are relevant for each downstream segment.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def extract_interfaces_from_architecture(
    arch_text: str,
    segment_id: str,
) -> Dict[str, Any]:
    """Extract public interfaces from an architecture document.

    Parses fenced code blocks to find function signatures, class
    definitions, and export statements. Returns a structured dict
    that can be serialised to JSON and injected into downstream prompts.

    Args:
        arch_text: Architecture markdown text.
        segment_id: For logging and keying.

    Returns:
        Dict with 'segment_id', 'files' (list of file interface dicts).
    """
    files: List[Dict[str, Any]] = []

    # Find all file sections with code blocks
    # Pattern: ## N) `path` (CREATE/MODIFY) ... ```lang ... ```
    section_pattern = re.compile(
        r'^##\s+\d+\)\s+`([^`]+)`\s*\((CREATE|MODIFY)\)',
        re.MULTILINE,
    )

    for match in section_pattern.finditer(arch_text):
        file_path = match.group(1).strip()
        operation = match.group(2)

        # Extract the section text until the next ## heading
        start = match.end()
        next_section = re.search(r'^##\s+\d+\)', arch_text[start:], re.MULTILINE)
        end = start + next_section.start() if next_section else len(arch_text)
        section_text = arch_text[start:end]

        # Extract code blocks from this section
        code_blocks = re.findall(r'```(?:\w+)?\s*\n(.*?)```', section_text, re.DOTALL)
        if not code_blocks:
            continue

        code = '\n\n'.join(code_blocks)
        interfaces = _extract_interfaces_from_code(code, file_path)

        if interfaces:
            files.append({
                'path': file_path,
                'operation': operation,
                'interfaces': interfaces,
            })

    logger.info(
        "[upstream_evidence] Extracted interfaces from %d files in %s",
        len(files), segment_id,
    )

    return {
        'segment_id': segment_id,
        'files': files,
    }


def _extract_interfaces_from_code(
    code: str,
    file_path: str,
) -> List[Dict[str, str]]:
    """Extract function/class signatures from a code block.

    Returns a list of dicts with 'type' ('function'/'class'/'export'),
    'name', and 'signature'.
    """
    interfaces: List[Dict[str, str]] = []
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.py':
        interfaces.extend(_extract_python_interfaces(code))
    elif ext in ('.ts', '.tsx', '.jsx', '.js'):
        interfaces.extend(_extract_typescript_interfaces(code))

    return interfaces


def _extract_python_interfaces(code: str) -> List[Dict[str, str]]:
    """Extract Python function and class signatures."""
    interfaces: List[Dict[str, str]] = []

    # Function/method signatures (including async)
    func_re = re.compile(
        r'^(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*(.+?))?\s*:',
        re.MULTILINE,
    )
    for m in func_re.finditer(code):
        name = m.group(1)
        if name.startswith('_') and not name.startswith('__'):
            continue  # Skip private functions
        params = m.group(2).strip()
        ret = m.group(3).strip() if m.group(3) else None
        sig = f"def {name}({params})"
        if ret:
            sig += f" -> {ret}"
        interfaces.append({
            'type': 'function',
            'name': name,
            'signature': sig,
        })

    # Class definitions
    class_re = re.compile(
        r'^class\s+(\w+)\s*(?:\(([^)]*)\))?\s*:',
        re.MULTILINE,
    )
    for m in class_re.finditer(code):
        name = m.group(1)
        bases = m.group(2).strip() if m.group(2) else ''
        sig = f"class {name}({bases})" if bases else f"class {name}"
        interfaces.append({
            'type': 'class',
            'name': name,
            'signature': sig,
        })

    return interfaces


def _extract_typescript_interfaces(code: str) -> List[Dict[str, str]]:
    """Extract TypeScript/JSX function and interface signatures."""
    interfaces: List[Dict[str, str]] = []

    # Export function declarations
    func_re = re.compile(
        r'^export\s+(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)',
        re.MULTILINE,
    )
    for m in func_re.finditer(code):
        name = m.group(1)
        params = m.group(2).strip()
        interfaces.append({
            'type': 'function',
            'name': name,
            'signature': f"export function {name}({params})",
        })

    # Interface declarations
    iface_re = re.compile(
        r'^(?:export\s+)?interface\s+(\w+)\s*(?:extends\s+([^{]+))?\s*\{',
        re.MULTILINE,
    )
    for m in iface_re.finditer(code):
        name = m.group(1)
        extends = m.group(2).strip() if m.group(2) else ''
        sig = f"interface {name} extends {extends}" if extends else f"interface {name}"
        interfaces.append({
            'type': 'interface',
            'name': name,
            'signature': sig,
        })

    # Type declarations
    type_re = re.compile(
        r'^(?:export\s+)?type\s+(\w+)\s*=',
        re.MULTILINE,
    )
    for m in type_re.finditer(code):
        interfaces.append({
            'type': 'type_alias',
            'name': m.group(1),
            'signature': f"type {m.group(1)}",
        })

    return interfaces


def build_upstream_evidence_text(
    segment_id: str,
    state: Any,
    job_dir_path: str,
    manifest: Any,
) -> str:
    """Build formatted evidence text from upstream segment architectures.

    Called from build_segment_context() to inject upstream interface
    evidence into the downstream segment's architecture prompt.

    Args:
        segment_id: The downstream segment requesting evidence.
        state: JobState.
        job_dir_path: Path to job directory.
        manifest: SegmentManifest.

    Returns:
        Formatted markdown string, or empty string if no evidence.
    """
    from app.orchestrator._segment_loop_utils_6 import _find_latest_arch

    seg_spec = manifest.get_segment(segment_id)
    if seg_spec is None:
        return ""

    parts: List[str] = []

    for dep_id in (seg_spec.dependencies or []):
        dep_state = state.segments.get(dep_id)
        if dep_state is None:
            continue

        # Accept APPROVED, COMPLETE, or IN_PROGRESS segments
        if dep_state.status not in ('approved', 'complete', 'in_progress'):
            continue

        # Find the architecture file for this upstream segment
        seg_dir = os.path.join(job_dir_path, "segments", dep_id)
        arch_path = _find_latest_arch(seg_dir)
        if arch_path is None:
            continue

        try:
            with open(arch_path, 'r', encoding='utf-8') as f:
                arch_text = f.read()
        except Exception as e:
            logger.debug(
                "[upstream_evidence] Cannot read arch for %s: %s", dep_id, e,
            )
            continue

        evidence = extract_interfaces_from_architecture(arch_text, dep_id)
        if not evidence['files']:
            continue

        part = f"### Upstream Segment: {dep_id}\n\n"
        for file_info in evidence['files']:
            part += f"**{file_info['path']}** ({file_info['operation']}):\n"
            for iface in file_info['interfaces']:
                part += f"- `{iface['signature']}`\n"
            part += "\n"

        parts.append(part)

    if not parts:
        return ""

    header = (
        "## Upstream Segment Interfaces\n\n"
        "The following interfaces are defined by upstream segments in this job. "
        "Use these EXACT signatures when importing from these modules.\n\n"
    )

    result = header + "\n".join(parts)
    logger.info(
        "[upstream_evidence] Built evidence for %s: %d upstream segments",
        segment_id, len(parts),
    )
    return result
