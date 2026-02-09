# FILE: app/llm/pipeline/critique_parts/scope_creep.py
"""Block 5e: SCOPE CREEP DETECTION (v2.1 - Endpoint/Feature Drift)"""

import logging
import re
from typing import List, Optional, Tuple

from app.llm.pipeline.critique_schemas import CritiqueIssue

logger = logging.getLogger(__name__)

_ENDPOINT_PATTERN = re.compile(
    r'(?:^|\s)'
    r'(GET|POST|PUT|PATCH|DELETE|WS|WSS|WebSocket)'
    r'\s+'
    r'(/[a-zA-Z0-9/_{}\-]+)',
    re.IGNORECASE | re.MULTILINE
)

# Features that spec constraints might explicitly exclude
_EXCLUDED_FEATURE_KEYWORDS = {
    'wake_word': ['wake_word', 'wake word', 'hotword', 'wakeword'],
    'tts': ['text-to-speech', 'text_to_speech', 'tts', 'piper'],
    'websocket': ['websocket', 'ws /voice', 'ws /audio', '/stream'],
}


def _extract_endpoints(text: str) -> List[Tuple[str, str]]:
    """Extract (METHOD, /path) pairs from text."""
    if not text:
        return []
    results = []
    for match in _ENDPOINT_PATTERN.finditer(text):
        method = match.group(1).upper()
        path = match.group(2).rstrip('.')
        if method in ('WS', 'WSS', 'WEBSOCKET'):
            method = 'WS'
        results.append((method, path.lower()))
    seen = set()
    unique = []
    for ep in results:
        if ep not in seen:
            seen.add(ep)
            unique.append(ep)
    return unique


def build_scope_creep_exclusion_zones(lines: List[str]) -> set:
    """Build a set of line indices that are inside exclusion zones.
    
    Exclusion zones are sections like "Out of Scope", "Not in this phase",
    "Future Work", etc. Returns set of 0-based line indices.
    """
    exclusion_headers = [
        'out of scope', 'not in scope', 'excluded', 'future work',
        'phase 2', 'phase 3', 'not in this phase', 'deferred',
        'not implemented', 'out-of-scope', 'non-goals',
    ]
    zones: set = set()
    in_exclusion = False
    exclusion_depth = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        heading_level = 0
        if stripped.startswith('#'):
            heading_level = len(stripped) - len(stripped.lstrip('#'))
        
        if heading_level > 0:
            heading_text = stripped.lstrip('#').strip()
            if any(eh in heading_text for eh in exclusion_headers):
                in_exclusion = True
                exclusion_depth = heading_level
                zones.add(i)
                continue
            if in_exclusion and heading_level <= exclusion_depth:
                in_exclusion = False
                exclusion_depth = 0
        
        if in_exclusion:
            zones.add(i)
    
    return zones


def run_scope_creep_check(
    arch_content: str,
    spec_markdown: Optional[str] = None,
    spec_json: Optional[str] = None,
) -> List[CritiqueIssue]:
    """
    v2.1: Deterministic scope creep detection.
    
    Compares endpoints and features in architecture against spec.
    Flags extra endpoints, renamed endpoints, and excluded features.
    """
    issues: List[CritiqueIssue] = []
    
    if not spec_markdown or not arch_content:
        return issues
    
    spec_endpoints = _extract_endpoints(spec_markdown)
    arch_endpoints = _extract_endpoints(arch_content)
    
    if not spec_endpoints:
        return issues
    
    spec_paths = {path for _, path in spec_endpoints}
    arch_paths = {path for _, path in arch_endpoints}
    
    # Check 1: Endpoints in architecture not in spec
    extra_paths = arch_paths - spec_paths
    if extra_paths:
        truly_extra = []
        for ep in extra_paths:
            is_subpath = any(ep.startswith(sp + '/') for sp in spec_paths)
            if not is_subpath:
                truly_extra.append(ep)
        
        if truly_extra:
            issue_id = len(issues) + 1
            issues.append(CritiqueIssue(
                id=f"SCOPE-CREEP-{issue_id:03d}",
                spec_ref=f"Spec endpoints: {[f'{m} {p}' for m, p in spec_endpoints]}",
                arch_ref=f"Extra architecture endpoints: {truly_extra}",
                category="scope_creep",
                severity="blocking",
                description=(
                    f"SCOPE CREEP: Architecture adds {len(truly_extra)} endpoint(s) not in spec: "
                    f"{truly_extra}. Spec only lists: {[f'{m} {p}' for m, p in spec_endpoints]}. "
                    f"Remove extra endpoints or get spec approval first."
                ),
                fix_suggestion=f"Remove these endpoints from the architecture: {truly_extra}. Only implement what the spec lists.",
            ))
            print(f"[DEBUG] [critique] v2.1 SCOPE CREEP: {len(truly_extra)} extra endpoint(s): {truly_extra}")
    
    # Check 2: Spec endpoints missing from architecture
    missing_paths = spec_paths - arch_paths
    if missing_paths:
        issue_id = len(issues) + 1
        issues.append(CritiqueIssue(
            id=f"SCOPE-CREEP-{issue_id:03d}",
            spec_ref=f"Missing spec endpoints: {list(missing_paths)}",
            arch_ref=f"Architecture endpoints: {[f'{m} {p}' for m, p in arch_endpoints]}",
            category="endpoint_rename",
            severity="blocking",
            description=(
                f"ENDPOINT MISMATCH: Spec requires {list(missing_paths)} but architecture "
                f"does not include them."
            ),
            fix_suggestion=f"Ensure these spec endpoints are present with the exact paths: {list(missing_paths)}",
        ))
        print(f"[DEBUG] [critique] v2.1 ENDPOINT MISMATCH: Missing {list(missing_paths)}")
    
    # Check 3: Excluded features appearing in architecture
    spec_lower = spec_markdown.lower()
    arch_lower = arch_content.lower()
    
    for feature, keywords in _EXCLUDED_FEATURE_KEYWORDS.items():
        spec_excludes = any(
            re.search(rf'(?:do\s+not|don.t|no)\s+(?:implement)?.*{re.escape(kw)}', spec_lower)
            for kw in keywords
        )
        if not spec_excludes:
            continue
        
        for kw in keywords:
            if kw in arch_lower:
                lines = arch_content.splitlines()
                exclusion_zones = build_scope_creep_exclusion_zones(lines)
                for i, line in enumerate(lines):
                    if kw in line.lower() and i not in exclusion_zones:
                        issue_id = len(issues) + 1
                        issues.append(CritiqueIssue(
                            id=f"SCOPE-CREEP-{issue_id:03d}",
                            spec_ref=f"Spec excludes: {feature}",
                            arch_ref=f"Architecture includes '{kw}' at line {i+1}",
                            category="excluded_feature",
                            severity="blocking",
                            description=(
                                f"EXCLUDED FEATURE: Spec explicitly says do not implement '{feature}', "
                                f"but architecture includes '{kw}' in a non-exclusion context (line {i+1})."
                            ),
                            fix_suggestion=f"Remove all references to '{feature}' from the architecture.",
                        ))
                        print(f"[DEBUG] [critique] v2.1 EXCLUDED FEATURE: '{feature}' found at line {i+1}")
                        break
                break
    
    return issues
