from __future__ import annotations
import os
import re
from app.pot_spec.grounded._spec_runner_utils_10 import _PRODUCT_SYNONYMS_RAW
from typing import Dict, List, Optional


def _parse_product_synonyms() -> Dict[str, List[str]]:
    """Parse product synonyms from env config into a lookup table.

    Returns dict mapping each name to ALL its synonyms (including itself).
    E.g., {"orb": ["orb", "astra"], "astra": ["orb", "astra"]}
    """
    synonyms: Dict[str, List[str]] = {}
    if not _PRODUCT_SYNONYMS_RAW:
        return synonyms

    for pair in _PRODUCT_SYNONYMS_RAW.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        names = [n.strip().lower() for n in pair.split("=") if n.strip()]
        # Each name maps to the full group
        for name in names:
            synonyms[name] = names
    return synonyms

_PRODUCT_SYNONYMS = _parse_product_synonyms()

def _generate_aliases_for_root(folder_name: str, root_path: str) -> List[str]:
    """Generate product name aliases for a discovered root path.

    For folder 'orb-desktop' with synonyms orb=astra:
      → ['orb desktop', 'orb-desktop', 'astra desktop', 'astra-desktop']
    For folder 'Orb' with synonyms orb=astra:
      → ['orb', 'astra']
    For folder 'my-project' with no synonyms:
      → ['my project', 'my-project']
    """
    aliases: List[str] = []
    folder_lower = folder_name.lower()

    # Split folder name into base + suffix: "orb-desktop" → ("orb", "desktop")
    parts = re.split(r'[-_\s]', folder_lower)
    base_name = parts[0]
    suffix = '-'.join(parts[1:]) if len(parts) > 1 else ''

    # Get all name variants (base name + its synonyms)
    name_variants = _PRODUCT_SYNONYMS.get(base_name, [base_name])

    for variant in name_variants:
        if suffix:
            aliases.append(f"{variant} {suffix}")   # "orb desktop" / "astra desktop"
            aliases.append(f"{variant}-{suffix}")    # "orb-desktop" / "astra-desktop"
        else:
            aliases.append(variant)                   # "orb" / "astra"

    return aliases

SCOPE_FRONTEND = {
    # UI modification patterns (user wants to CHANGE the UI)
    'the ui': True, 'on the ui': True, 'in the ui': True, 'to the ui': True,
    'a ui': True, 'ui button': True, 'ui feature': True, 'ui text': True,
    'change the ui': True, 'modify the ui': True, 'update the ui': True,
    'the frontend': True, 'front-end': True, 'frontend ui': True,
    # v5.2: REMOVED 'text input' — too ambiguous.
    # 'text input' in AI/ML context means text-type data ingestion, not a UI field.
    # Also REMOVED 'context window' and 'input window' in v4.7 for same reason.
    # Users requesting frontend changes will say 'the UI', 'the frontend', etc.
    # Electron is specific enough — only mentioned when discussing frontend code
    'electron': True,
    # NOTE: 'the app', 'desktop app', "app's" deliberately REMOVED.
    # These match consumer mentions like "works from the desktop app".
}

SCOPE_BACKEND = {
    'the backend': True, 'back-end': True, 'backend api': True,
    'fastapi': True, 'api endpoint': True, 'server': True,
    # NOTE: Static keyword dicts are a dead end for scope detection.
    # The real fix is to use INDEX.json zone data (every file is already
    # tagged as 'backend' or 'frontend'). When the job mentions files or
    # modules, look up their zone. That's intelligent scope detection.
    # TODO: Replace keyword-based scope with zone-aware detection using
    # the integration points already extracted by the spec runner.
}

def _detect_search_replace_terms(text: str) -> tuple:
    """
    v4.3.2: Detect search and replacement terms to exclude from path matching.
    
    Returns (search_term, replace_term) or (None, None)
    
    v4.3.2: Require at least 3 characters to avoid false positives like 's' -> 'p'
    """
    text_lower = text.lower()
    
    # Pattern: "X to Y" refactor/rename
    # v4.3.2: Use {3,} instead of + to require at least 3 chars
    patterns = [
        r'from\s+["\']?([a-z]{3,})["\']?\s+to\s+["\']?([a-z]{3,})["\']?',
        r'rename\s+["\']?([a-z]{3,})["\']?\s+to\s+["\']?([a-z]{3,})["\']?',
        r'replace\s+["\']?([a-z]{3,})["\']?\s+with\s+["\']?([a-z]{3,})["\']?',
        r'["\']?([a-z]{3,})["\']?\s+to\s+["\']?([a-z]{3,})["\']?\s*\(refactor',
        r'rebrand.*?from\s+["\']?([a-z]{3,})["\']?\s+to\s+["\']?([a-z]{3,})["\']?',
        r'change.*?["\']?([a-z]{3,})["\']?.*?to.*?["\']?([a-z]{3,})["\']?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return (match.group(1), match.group(2))
    
    return (None, None)

def _get_job_dir_for_segmentation(job_id: str) -> str:
    """
    v4.9 PHASE 2: Get job directory path for segmentation manifest references.
    Uses the same path construction as _write_segmentation_output().
    """
    try:
        from ..spec_gate_persistence import artifact_root as _ar, job_dir as _jd
        return _jd(_ar(), job_id)
    except ImportError:
        _root = os.path.abspath(os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"))
        return os.path.join(_root, "jobs", job_id)

def _extract_requirements_from_spec(spec_markdown: Optional[str]) -> List[str]:
    """
    v4.8: Extract requirement lines from spec markdown.
    
    Looks for bullet points under Goal, Requirements, What to do sections.
    """
    requirements: List[str] = []
    if not spec_markdown:
        return requirements
    
    in_section = False
    for line in spec_markdown.split('\n'):
        stripped = line.strip()
        # Detect relevant sections
        if stripped.startswith('## ') and any(
            kw in stripped.lower() for kw in ['goal', 'requirement', 'what to do', 'scope']
        ):
            in_section = True
            continue
        elif stripped.startswith('## '):
            in_section = False
            continue
        
        if in_section and stripped and (stripped.startswith('- ') or stripped.startswith('* ')):
            req_text = stripped.lstrip('- *').strip()
            if req_text and len(req_text) > 5:  # Skip trivially short items
                requirements.append(req_text)
    
    return requirements
