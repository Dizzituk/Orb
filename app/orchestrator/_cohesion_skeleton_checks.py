# FILE: app/orchestrator/_cohesion_skeleton_checks.py
"""
Cohesion Check: Extended skeleton compliance checks (4-8).

Extracted from cohesion_check.py to keep file sizes modular.
These are deterministic checks that run after the base scope/reference checks.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Set

from app.orchestrator._cohesion_check_utils_8 import (
    _extract_arch_file_paths,
    _extract_segment_references,
)
from app.orchestrator._cohesion_check_utils_10 import CohesionIssue

logger = logging.getLogger(__name__)


def check_undeclared_dependencies(
    architectures: Dict[str, str],
    manifest_dict: Dict[str, Any],
    issue_counter: int,
) -> tuple[list[CohesionIssue], int]:
    """Check 4: Detect imports from undeclared upstream segments."""
    issues: list[CohesionIssue] = []
    _seg_id_to_files: Dict[str, set] = {}
    for _seg_data in manifest_dict.get("segments", []):
        _sid = _seg_data.get("segment_id", "")
        _seg_id_to_files[_sid] = set(
            f.replace("\\", "/").lower().rsplit("/", 1)[-1].replace(".py", "")
            for f in _seg_data.get("file_scope", [])
        )

    for seg_id, arch_content in architectures.items():
        _imports = re.findall(r'from\s+\.(\w+)\s+import', arch_content)
        _seg_data = next(
            (s for s in manifest_dict.get("segments", []) if s.get("segment_id") == seg_id),
            None,
        )
        if not _seg_data:
            continue
        _seg_deps = set(_seg_data.get("dependencies", []))
        _own_modules = _seg_id_to_files.get(seg_id, set())

        for _imp_module in set(_imports):
            if _imp_module in _own_modules:
                continue
            _owner = None
            for _other_sid, _other_mods in _seg_id_to_files.items():
                if _other_sid != seg_id and _imp_module in _other_mods:
                    _owner = _other_sid
                    break
            if _owner and _owner not in _seg_deps:
                issue_counter += 1
                issues.append(CohesionIssue(
                    issue_id=f"SKEL-{issue_counter:03d}",
                    severity="warning",
                    category="undeclared_dependency",
                    description=(
                        f"{seg_id} imports from '.{_imp_module}' which belongs "
                        f"to {_owner}, but {seg_id} does not declare {_owner} "
                        f"as a dependency."
                    ),
                    source_segment=seg_id,
                    related_segment=_owner,
                    file_path=f"{_imp_module}.py",
                    suggested_fix=f"Add {_owner} to {seg_id}'s dependencies",
                ))
    return issues, issue_counter


def check_missing_stdlib_imports(
    architectures: Dict[str, str],
    issue_counter: int,
) -> tuple[list[CohesionIssue], int]:
    """Check 5: Detect missing stdlib imports (logging, os, etc.)."""
    issues: list[CohesionIssue] = []
    for seg_id, arch_content in architectures.items():
        _has_logger = bool(re.search(r'\blogger\.(info|warning|error|debug|critical)\b', arch_content))
        _has_import = bool(re.search(r'\bimport\s+logging\b', arch_content))
        if _has_logger and not _has_import:
            issue_counter += 1
            issues.append(CohesionIssue(
                issue_id=f"SKEL-{issue_counter:03d}",
                severity="warning",
                category="missing_import",
                description=(
                    f"{seg_id} uses logger.xxx() calls but does not include "
                    f"'import logging' in its architecture."
                ),
                source_segment=seg_id,
                suggested_fix="Add 'import logging' and 'logger = logging.getLogger(__name__)'",
            ))
    return issues, issue_counter


def check_cross_segment_symbols(
    architectures: Dict[str, str],
    manifest_dict: Dict[str, Any],
    issue_counter: int,
) -> tuple[list[CohesionIssue], int]:
    """Check 6: Cross-segment symbol verification."""
    issues: list[CohesionIssue] = []

    _module_to_segment: Dict[str, str] = {}
    _module_exports: Dict[str, Set[str]] = {}
    _seg_package_prefix: Dict[str, str] = {}

    for _seg_data in manifest_dict.get("segments", []):
        _sid = _seg_data.get("segment_id", "")
        _file_paths = [f.replace("\\", "/") for f in _seg_data.get("file_scope", [])]
        if _file_paths:
            _dirs = [fp.rsplit("/", 1)[0] if "/" in fp else "" for fp in _file_paths]
            _seg_package_prefix[_sid] = _dirs[0] if _dirs else ""
        for _fp in _file_paths:
            _full_key = _fp.replace(".py", "").lower()
            _module_to_segment[_full_key] = _sid

    # Extract symbols defined in each segment's architecture
    for seg_id, arch_content in architectures.items():
        _defined = _extract_defined_symbols(arch_content)
        _seg_data = next(
            (s for s in manifest_dict.get("segments", []) if s.get("segment_id") == seg_id),
            None,
        )
        if _seg_data:
            for _fp in _seg_data.get("file_scope", []):
                _full_key = _fp.replace("\\", "/").replace(".py", "").lower()
                _module_exports.setdefault(_full_key, set()).update(_defined)

    # Check imports against exports
    for seg_id, arch_content in architectures.items():
        for _m in re.finditer(r'from\s+\.(\w+)\s+import\s+([^(\n]+)', arch_content):
            _target_mod = _m.group(1).lower()
            _imports_str = _m.group(2).strip().rstrip("\\").strip('`')
            _imported_names = [
                n.strip().strip('`').split(" as ")[0]
                for n in _imports_str.split(",") if n.strip()
            ]

            _pkg_prefix = _seg_package_prefix.get(seg_id, "")
            _full_target = f"{_pkg_prefix}/{_target_mod}".lower() if _pkg_prefix else _target_mod

            _target_seg = _module_to_segment.get(_full_target)
            if not _target_seg or _target_seg == seg_id:
                continue

            _available = _module_exports.get(_full_target, set())
            if not _available:
                continue

            for _imp_name in _imported_names:
                _imp_name = _imp_name.strip().strip('`').strip()
                if not _imp_name or _imp_name.startswith("#") or _imp_name.startswith(")"):
                    continue
                if not re.match(r'^[a-zA-Z_]\w*$', _imp_name):
                    continue
                if _imp_name not in _available:
                    issue_counter += 1
                    issues.append(CohesionIssue(
                        issue_id=f"SKEL-{issue_counter:03d}",
                        severity="blocking",
                        category="missing_symbol",
                        description=(
                            f"{seg_id} imports '{_imp_name}' from '.{_target_mod}' "
                            f"(owned by {_target_seg}), but '{_imp_name}' is not "
                            f"defined in {_target_seg}'s architecture. Available: "
                            f"{sorted(list(_available))[:8]}"
                        ),
                        source_segment=seg_id,
                        related_segment=_target_seg,
                        file_path=f"{_target_mod}.py",
                        suggested_fix=(
                            f"Add '{_imp_name}' to {_target_seg}'s architecture "
                            f"for {_target_mod}.py, or fix the import in {seg_id}"
                        ),
                    ))
    return issues, issue_counter


def check_duplicate_functions(
    architectures: Dict[str, str],
    manifest_dict: Optional[Dict[str, Any]],
    issue_counter: int,
) -> tuple[list[CohesionIssue], int]:
    """Check 7: Duplicate function detection across segments."""
    issues: list[CohesionIssue] = []
    _func_locations: Dict[str, List[tuple]] = {}

    for seg_id, arch_content in architectures.items():
        _seg_files: set = set()
        _seg_source = ""
        if manifest_dict:
            _seg_data = next(
                (s for s in manifest_dict.get("segments", []) if s.get("segment_id") == seg_id),
                None,
            )
            if _seg_data:
                _seg_files = set(_seg_data.get("file_scope", []))
                _seg_source = _seg_data.get("deterministic_source", "")

        for _m in re.finditer(r'(?:async\s+)?def\s+(\w+)\s*\(', arch_content):
            _fname = _m.group(1)
            if _fname.startswith('__') and _fname.endswith('__'):
                continue
            if _fname.startswith('test_'):
                continue

            _start = _m.start()
            _next_def = re.search(
                r'\n(?:async\s+)?def\s+\w+\s*\(|\nclass\s+\w+',
                arch_content[_start + 10:],
            )
            _line_count = 0
            if _next_def:
                _chunk = arch_content[_start:_start + 10 + _next_def.start()]
                _line_count = _chunk.count('\n')
            else:
                _line_count = arch_content[_start:].count('\n')

            _func_locations.setdefault(_fname, []).append(
                (seg_id, ", ".join(sorted(_seg_files)[:3]), _line_count, _seg_source)
            )

    for _fname, _locs in _func_locations.items():
        _by_source: Dict[str, List[tuple]] = {}
        for _loc in _locs:
            _src = _loc[3] if len(_loc) > 3 else ""
            _by_source.setdefault(_src, []).append(_loc)

        for _src_group in _by_source.values():
            _unique_segs = set(loc[0] for loc in _src_group)
            if len(_unique_segs) <= 1:
                continue
            _max_lines = max(loc[2] for loc in _src_group)
            _severity = "blocking" if _max_lines > 100 else "warning"
            _seg_list = ", ".join(sorted(_unique_segs))

            issue_counter += 1
            issues.append(CohesionIssue(
                issue_id=f"SKEL-{issue_counter:03d}",
                severity=_severity,
                category="duplicate_function",
                description=(
                    f"Function '{_fname}' is defined in {len(_unique_segs)} segments: "
                    f"{_seg_list}. Estimated {_max_lines}+ lines."
                ),
                source_segment=sorted(_unique_segs)[0],
                related_segment=sorted(_unique_segs)[1] if len(_unique_segs) > 1 else "",
                suggested_fix=(
                    f"Assign '{_fname}' to one segment only. Others import via "
                    f"'from .module import {_fname}'."
                ),
            ))
    return issues, issue_counter


def check_phantom_symbols(
    architectures: Dict[str, str],
    manifest_dict: Dict[str, Any],
    issues: List[CohesionIssue],
    issue_counter: int,
) -> tuple[list[CohesionIssue], int]:
    """Check 8: Cross-segment missing symbol with monolith verification."""
    _monolith_symbols: set = set()
    _evidence_paths: set = set()
    for _seg_data in manifest_dict.get("segments", []):
        for _ef in _seg_data.get("evidence_files", []):
            _evidence_paths.add(_ef)

    for _ef_path in _evidence_paths:
        for _base in ["D:\\Orb", "D:/Orb"]:
            _full = os.path.join(_base, _ef_path.replace("/", os.sep))
            if os.path.isfile(_full):
                try:
                    with open(_full, "r", encoding="utf-8") as _f:
                        _src = _f.read()
                    for _m in re.finditer(r'(?:async\s+)?def\s+(\w+)\s*\(', _src):
                        _monolith_symbols.add(_m.group(1))
                    for _m in re.finditer(r'class\s+(\w+)\s*[\(:]', _src):
                        _monolith_symbols.add(_m.group(1))
                    for _m in re.finditer(r'^([A-Z][A-Z0-9_]+)\s*=', _src, re.MULTILINE):
                        _monolith_symbols.add(_m.group(1))
                except Exception:
                    pass
                break

    if not _monolith_symbols:
        return issues, issue_counter

    logger.info("[cohesion_check] v4.0 Loaded %d symbols from evidence", len(_monolith_symbols))

    for _issue in issues:
        if _issue.category != "missing_symbol":
            continue
        _sym_match = re.search(r"imports '(\w+)'", _issue.description)
        if not _sym_match:
            continue
        _sym_name = _sym_match.group(1)

        if _sym_name in _monolith_symbols:
            _issue.severity = "warning"
            _issue.auto_fix_note = f"Symbol '{_sym_name}' exists in source monolith"
        else:
            _issue.severity = "blocking"
            _issue.suggested_fix = (
                f"Symbol '{_sym_name}' does not exist anywhere — remove this import"
            )
            _issue.auto_fix_note = f"v4.0: Symbol '{_sym_name}' verified absent from monolith"
            logger.warning(
                "[cohesion_check] v4.0 PHANTOM SYMBOL: '%s' — consuming segment %s",
                _sym_name, _issue.source_segment,
            )
    return issues, issue_counter


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _extract_defined_symbols(arch_content: str) -> Set[str]:
    """Extract all symbol names defined in an architecture text."""
    defined: Set[str] = set()
    for _m in re.finditer(r'(?:async\s+)?def\s+(\w+)\s*\(', arch_content):
        defined.add(_m.group(1))
    for _m in re.finditer(r'class\s+(\w+)\s*[\(:]', arch_content):
        defined.add(_m.group(1))
    for _m in re.finditer(r'^([A-Z][A-Z0-9_]+)\s*=', arch_content, re.MULTILINE):
        defined.add(_m.group(1))
    for _m in re.finditer(r'(?:defines?|contains?|exports?|provides?)\s+`?([A-Z][A-Z0-9_]+)`?', arch_content):
        defined.add(_m.group(1))
    for _m in re.finditer(r'`(\w+)\s*\(', arch_content):
        _name = _m.group(1)
        if _name not in ('import', 'from', 'class', 'def', 'async', 'return', 'if', 'for', 'while', 'with', 'try', 'except'):
            defined.add(_name)
    for _m in re.finditer(r'(?:exports?|imports?|provides?|defines?)\s+`(\w+)`', arch_content):
        defined.add(_m.group(1))
    return defined
