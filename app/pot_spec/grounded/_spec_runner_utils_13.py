from __future__ import annotations
import glob
import json
import os
import re
from app.pot_spec.grounded._spec_runner_utils_10 import _ARCH_INDEX_DIR, _ARCH_REPORT_DIR
from app.pot_spec.grounded._spec_runner_utils_11 import _generate_aliases_for_root
from functools import lru_cache
from typing import Any, Dict, List, Optional

# v3.2-fix: Sandbox-aware filesystem checks for codebase paths.
# v4.3: Spec gate uses HOST filesystem, not sandbox.
# v9.0-fix: _sbx_isfile/_sbx_isdir were never defined — use os.path directly.
_sbx_isfile = os.path.isfile
_sbx_isdir = os.path.isdir


@lru_cache(maxsize=1)
def _discover_project_roots() -> Dict[str, Any]:
    """Dynamically discover project roots from architecture index.

    Reads the ground-truth architecture documents to find:
    - Which directories are project roots
    - Which are frontend vs backend
    - What product names map to which paths

    Sources (in priority order):
    1. INDEX.json from .architecture/ (structured, has zone metadata)
    2. CODEBASE_REPORT_FULL_*.json from .architecture/ (scan metadata)
    3. Hardcoded fallback (last resort)

    Returns dict with:
        roots: list of absolute paths to project roots
        frontend_paths: list of frontend root paths
        backend_paths: list of backend root paths
        all_paths: combined list
        aliases: dict mapping product name patterns → list of root paths
    """
    result: Dict[str, Any] = {
        "roots": [],
        "frontend_paths": [],
        "backend_paths": [],
        "all_paths": [],
        "aliases": {},
        "source": "none",
    }

    # --- Source 1: INDEX.json (best - has per-file zone metadata) ---
    index_path = os.path.join(_ARCH_INDEX_DIR, "INDEX.json")
    if _sbx_isfile(index_path):
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)

            roots = index_data.get("roots", [])
            # Filter to roots that actually exist on disk
            roots = [r for r in roots if _sbx_isdir(r)]

            if roots:
                result["roots"] = roots
                result["all_paths"] = roots
                result["source"] = f"INDEX.json ({index_path})"

                # Classify roots by zone from file entries
                frontend_roots: set = set()
                backend_roots: set = set()
                for file_entry in index_data.get("files", [])[:200]:  # Sample first 200 for speed
                    zone = file_entry.get("zone", "")
                    root = file_entry.get("root", "")
                    if zone == "frontend" and root:
                        frontend_roots.add(root)
                    elif zone == "backend" and root:
                        backend_roots.add(root)

                # Use zone metadata if available, else heuristic
                if frontend_roots:
                    result["frontend_paths"] = [r for r in roots if r in frontend_roots]
                else:
                    result["frontend_paths"] = [r for r in roots if 'desktop' in r.lower()]

                if backend_roots:
                    result["backend_paths"] = [r for r in roots if r in backend_roots]
                else:
                    result["backend_paths"] = [r for r in roots if 'desktop' not in r.lower()]

                print(f"[spec_runner] v4.5 DISCOVERY from INDEX.json: roots={roots}")
        except Exception as e:
            print(f"[spec_runner] v4.5 Failed to read INDEX.json: {e}")

    # --- Source 2: Codebase report JSON (fallback) ---
    if not result["roots"]:
        report_pattern = os.path.join(_ARCH_REPORT_DIR, "CODEBASE_REPORT_FULL_*.json")
        report_files = sorted(glob.glob(report_pattern), reverse=True)
        if report_files:
            try:
                with open(report_files[0], 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                roots = report_data.get("metadata", {}).get("roots_scanned", [])
                roots = [r for r in roots if _sbx_isdir(r)]

                if roots:
                    result["roots"] = roots
                    result["all_paths"] = roots
                    result["source"] = f"CODEBASE_REPORT ({report_files[0]})"
                    result["frontend_paths"] = [r for r in roots if 'desktop' in r.lower()]
                    result["backend_paths"] = [r for r in roots if 'desktop' not in r.lower()]
                    print(f"[spec_runner] v4.5 DISCOVERY from codebase report: roots={roots}")
            except Exception as e:
                print(f"[spec_runner] v4.5 Failed to read codebase report: {e}")

    # --- Source 2b: Merge in external project roots from target_registry ---
    # Android projects and other external targets are registered in the pipeline
    # but not in INDEX.json (which only scans ASTRA's own codebase).
    try:
        from app.pipeline_v2.target_registry import ALL_PROFILES
        for _pid, _profile in ALL_PROFILES.items():
            _proot = _profile.project_root.replace('/', os.sep)
            if _sbx_isdir(_proot) and _proot not in result["roots"]:
                result["roots"].append(_proot)
                result["all_paths"].append(_proot)
                # Android/Kotlin projects are neither frontend nor backend in the web sense
                # but classify as "backend" so the spec runner doesn't skip them
                if _proot not in result["backend_paths"]:
                    result["backend_paths"].append(_proot)
                print(f"[spec_runner] v4.7 Added external project root: {_proot} ({_profile.project_name})")
    except ImportError:
        pass
    except Exception as _ext_err:
        print(f"[spec_runner] v4.7 External root discovery failed: {_ext_err}")

    # --- Source 3: Hardcoded fallback (last resort) ---
    if not result["roots"]:
        print("[spec_runner] v4.5 WARNING: No architecture index found, using hardcoded fallback")
        result["roots"] = ['D:\\orb-desktop', 'D:\\Orb']
        result["frontend_paths"] = ['D:\\orb-desktop']
        result["backend_paths"] = ['D:\\Orb']
        result["all_paths"] = ['D:\\orb-desktop', 'D:\\Orb']
        result["source"] = "hardcoded_fallback"

    # --- Build product name aliases from discovered roots ---
    for root in result["roots"]:
        folder_name = os.path.basename(root)
        if not folder_name:  # Handle "D:\\" edge case
            continue
        aliases = _generate_aliases_for_root(folder_name, root)
        for alias in aliases:
            if alias not in result["aliases"]:
                result["aliases"][alias] = []
            if root not in result["aliases"][alias]:
                result["aliases"][alias].append(root)

    print(f"[spec_runner] v4.5 DISCOVERY COMPLETE: "
          f"roots={result['roots']}, "
          f"frontend={result['frontend_paths']}, "
          f"backend={result['backend_paths']}, "
          f"aliases={list(result['aliases'].keys())}, "
          f"source={result['source']}")

    return result

def _extract_file_scope_from_spec(
    spec_markdown: Optional[str],
    grounding_data: Optional[Dict] = None,
    multi_file_op: Optional[Any] = None,
) -> List[str]:
    """
    v4.9: Extract file paths mentioned as targets in the spec.
    
    Looks for patterns like:
    - Relative paths: `app/foo/bar.py`, `src/components/Foo.tsx`
    - Absolute Windows paths: `D:\\Orb\\app\\foo.py`, `D:/Orb/app/foo.py`
    - Backtick-wrapped paths in markdown
    - Paths from multi_file_op target files
    - Paths from grounding_data if available
    
    v4.9 (2026-02-08): Added absolute path extraction and multi_file_op support.
    Previous version only matched relative paths starting with app/, src/, etc.
    This meant CREATE jobs (which use absolute paths from simple_create.py)
    returned empty file scope and segmentation never triggered.
    """
    paths: List[str] = []
    seen_normalised: set = set()  # Normalised keys for dedup
    
    def _add_path(p: str) -> None:
        """Add path with dedup (case-insensitive, separator-normalised)."""
        key = p.lower().replace('/', '\\').rstrip('\\')
        if key not in seen_normalised:
            seen_normalised.add(key)
            paths.append(p)
    
    if not spec_markdown:
        # Still check multi_file_op even without spec markdown
        if multi_file_op and hasattr(multi_file_op, 'raw_matches'):
            for match_entry in (multi_file_op.raw_matches or []):
                if isinstance(match_entry, dict) and 'file' in match_entry:
                    _add_path(match_entry['file'])
        return paths
    
    # Pattern 1: Relative paths (existing — app/, src/, etc.)
    # v2.3: Added Kotlin/Android extensions: .kt, .kts, .xml, .properties, .gradle
    _KNOWN_EXTENSIONS = r'py|tsx|ts|jsx|js|json|yaml|yml|md|css|kts|kt|xml|properties|gradle'
    rel_pattern = re.compile(
        r'(?:^|[\s`|])'
        r'((?:app|src|orb-desktop|tests|scripts|config)[/\\]'
        r'[\w/\\.-]+\.(?:' + _KNOWN_EXTENSIONS + r'))'
        r'(?:[\s`|,]|$)',
        re.MULTILINE,
    )
    for match in rel_pattern.finditer(spec_markdown):
        _add_path(match.group(1).replace('/', os.sep))
    
    # Pattern 1b: Sub-package paths without app/ prefix (v5.16)
    # Specs often reference paths like "overwatcher/executor.py" instead of
    # "app/overwatcher/executor.py". Detect known app/ subdirectories used
    # as path prefixes and prepend "app/" so segmentation triggers correctly.
    _app_subdirs = set()
    try:
        _app_root = None
        # Discover project roots to find app/ directory
        _disc = _discover_project_roots()
        for _r in _disc.get('roots', []):
            _candidate = os.path.join(_r, 'app')
            if _sbx_isdir(_candidate):
                _app_root = _candidate
                break
        if _app_root:
            _app_subdirs = {
                d.lower() for d in os.listdir(_app_root)
                if _sbx_isdir(os.path.join(_app_root, d)) and not d.startswith('_')
            }
    except Exception:
        pass
    
    if _app_subdirs:
        # Match paths starting with known app subdirs (e.g. overwatcher/foo.py)
        _subdir_pattern_str = '|'.join(re.escape(d) for d in sorted(_app_subdirs))
        subpkg_pattern = re.compile(
            r'(?:^|[\s`|])'
            r'((?:' + _subdir_pattern_str + r')[/\\]'
            r'[\w/\\.-]+\.(?:' + _KNOWN_EXTENSIONS + r'))'
            r'(?:[\s`|,]|$)',
            re.MULTILINE,
        )
        for match in subpkg_pattern.finditer(spec_markdown):
            raw = match.group(1).replace('/', os.sep)
            # Prepend app/ since these are app subdirectory paths
            _add_path(f"app{os.sep}{raw}")
    
    # Pattern 2: Absolute Windows paths (D:\Orb\app\foo.py or D:/Orb/app/foo.py)
    # Grounded CREATE specs from simple_create.py use full absolute paths.
    abs_pattern = re.compile(
        r'(?:^|[\s`|])'
        r'([A-Za-z]:[/\\]'
        r'[\w/\\.-]+\.(?:' + _KNOWN_EXTENSIONS + r'))'
        r'(?:[\s`|,]|$)',
        re.MULTILINE,
    )
    # v5.7: Get known project roots for absolute→relative conversion
    _known_roots = []
    try:
        _disc = _discover_project_roots()
        for _root in _disc.get('roots', []):
            _nr = _root.replace('/', '\\').rstrip('\\').lower()
            _known_roots.append(_nr)
    except Exception:
        pass
    # Sort longest first so D:\Orb\app matches before D:\Orb
    _known_roots.sort(key=len, reverse=True)

    for match in abs_pattern.finditer(spec_markdown):
        abs_path = match.group(1)
        # v5.7: Convert absolute paths to relative using known project roots.
        # This prevents duplicate entries where both absolute and relative forms
        # of the same file end up in scope (e.g. D:\Orb\main.py AND main.py),
        # which causes double-counting in segmentation and skeleton contracts.
        normalised = abs_path.replace('/', '\\')
        _found_relative = False

        # Strategy 1: Strip known project root prefix
        # e.g. D:\Orb\main.py → main.py, D:\Orb\app\foo.py → app\foo.py
        for _root in _known_roots:
            if normalised.lower().startswith(_root + '\\'):
                rel_part = normalised[len(_root) + 1:]  # +1 for the separator
                if rel_part:  # Safety: don't add empty string
                    _add_path(rel_part)
                    _found_relative = True
                break

        # Strategy 2: Find known sub-directory prefix (fallback for unknown roots)
        if not _found_relative:
            for prefix in ('app\\', 'src\\', 'tests\\', 'scripts\\', 'config\\', 'orb-desktop\\'):
                idx = normalised.lower().find(prefix)
                if idx >= 0:
                    rel_part = normalised[idx:]
                    _add_path(rel_part)
                    _found_relative = True
                    break

        # Only keep the absolute path if no relative portion could be extracted
        if not _found_relative:
            _add_path(abs_path)
    
    # Pattern 3: Extract from multi_file_op target files if available
    if multi_file_op:
        if hasattr(multi_file_op, 'raw_matches'):
            for match_entry in (multi_file_op.raw_matches or []):
                if isinstance(match_entry, dict) and 'file' in match_entry:
                    _add_path(match_entry['file'])
        # Also check target_files attribute directly
        if hasattr(multi_file_op, 'target_files'):
            for f in (multi_file_op.target_files or []):
                if isinstance(f, str):
                    _add_path(f)
    
    # Pattern 4: Extract from grounding_data if provided
    if grounding_data and isinstance(grounding_data, dict):
        multi = grounding_data.get('multi_file', {})
        if multi and isinstance(multi, dict):
            for f in multi.get('target_files', []):
                if isinstance(f, str):
                    _add_path(f)

    # --- v5.18: Aggressive fallback for missed paths ---
    # If we found very few files (<3), do a broader scan.
    # The LLM spec often lists paths in formats the stricter regexes miss:
    #   - "D:/Orb/app/debug/models.py — DB models" (em dash breaks boundary)
    #   - Paths inside **bold** or other markdown formatting
    #   - Paths after bullets without proper whitespace
    # This broader pattern catches any plausible file path anywhere in the text.
    if len(paths) < 3 and spec_markdown:
        _broad_rel = re.compile(
            r'((?:app|src|orb-desktop)[/\\]'
            r'[\w/\\.-]+\.(?:' + _KNOWN_EXTENSIONS + r'))',
        )
        _broad_abs = re.compile(
            r'([A-Za-z]:[/\\]'
            r'[\w/\\.-]+\.(?:' + _KNOWN_EXTENSIONS + r'))',
        )
        _before_count = len(paths)
        for m in _broad_rel.finditer(spec_markdown):
            _add_path(m.group(1).replace('/', os.sep))
        for m in _broad_abs.finditer(spec_markdown):
            normalised = m.group(1).replace('/', '\\')
            _found = False
            for _root in _known_roots:
                if normalised.lower().startswith(_root + '\\'):
                    rel_part = normalised[len(_root) + 1:]
                    if rel_part:
                        _add_path(rel_part)
                        _found = True
                    break
            if not _found:
                for prefix in ('app\\', 'src\\', 'tests\\', 'orb-desktop\\'):
                    idx = normalised.lower().find(prefix)
                    if idx >= 0:
                        _add_path(normalised[idx:])
                        _found = True
                        break
            if not _found:
                _add_path(m.group(1))
        if len(paths) > _before_count:
            print(f"[spec_runner] v5.18 BROAD SCAN rescued {len(paths) - _before_count} additional path(s) (total={len(paths)})")

    return paths

def _extract_acceptance_from_spec(spec_markdown: Optional[str]) -> List[str]:
    """
    v4.8: Extract acceptance criteria from spec markdown.
    
    Looks for checkbox items under Acceptance sections.
    """
    criteria: List[str] = []
    if not spec_markdown:
        return criteria
    
    in_section = False
    for line in spec_markdown.split('\n'):
        stripped = line.strip()
        if stripped.startswith('## ') and any(
            kw in stripped.lower() for kw in ['acceptance', 'verification', 'criteria']
        ):
            in_section = True
            continue
        elif stripped.startswith('## '):
            in_section = False
            continue
        
        if in_section and stripped:
            # Match checkbox items: - [ ] or - [x] or just bullet points
            check_match = re.match(r'^[-*]\s*\[.?\]\s*(.*)', stripped)
            if check_match:
                criteria.append(check_match.group(1).strip())
            elif stripped.startswith('- ') or stripped.startswith('* '):
                criteria.append(stripped.lstrip('- *').strip())
    
    return criteria
