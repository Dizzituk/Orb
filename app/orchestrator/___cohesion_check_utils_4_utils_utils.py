import json
import logging
import os
from app.orchestrator._cohesion_check_utils import CohesionIssue
from app.orchestrator._cohesion_check_utils import _extract_arch_file_paths, _extract_segment_references
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


def run_skeleton_compliance(
    architectures: Dict[str, str],
    skeleton_json: Optional[str] = None,
    manifest_dict: Optional[Dict[str, Any]] = None,
) -> List[CohesionIssue]:
    """
    Deterministic skeleton compliance check.

    Verifies each segment's architecture against the skeleton contract:
    1. File inventory items must be within the segment's file scope
    2. Segment references must exist in the manifest
    3. Exports required by downstream segments should be present

    Args:
        architectures: {segment_id: architecture_content}
        skeleton_json: JSON string from SkeletonContractSet.to_json()
        manifest_dict: Raw manifest dict for additional validation

    Returns:
        List of CohesionIssue objects (may be empty if all clean)
    """
    issues: List[CohesionIssue] = []
    issue_counter = 0

    if not skeleton_json:
        return issues

    try:
        skeleton = json.loads(skeleton_json)
    except (json.JSONDecodeError, TypeError):
        logger.warning("[cohesion_check] Failed to parse skeleton JSON")
        return issues

    all_segment_ids = set()
    scope_by_segment: Dict[str, set] = {}
    exports_by_segment: Dict[str, List[Dict]] = {}

    for skel in skeleton.get("skeletons", []):
        seg_id = skel.get("segment_id", "")
        all_segment_ids.add(seg_id)
        # Normalise scope paths for comparison
        scope_by_segment[seg_id] = {
            p.replace("\\", "/").lower()
            for p in skel.get("file_scope", [])
        }
        exports_by_segment[seg_id] = skel.get("exports", [])

    for seg_id, arch_content in architectures.items():
        seg_scope = scope_by_segment.get(seg_id, set())

        # --- Check 1: File inventory items within scope ---
        # Extract file paths from architecture file inventory tables
        arch_files = _extract_arch_file_paths(arch_content)
        for arch_file in arch_files:
            normalised = arch_file.replace("\\", "/").lower()
            if normalised not in seg_scope:
                # Check if it's a partial match (file might use different prefix)
                basename = normalised.rsplit("/", 1)[-1] if "/" in normalised else normalised
                partial_match = any(s.endswith("/" + basename) or s == basename
                                   for s in seg_scope)
                if not partial_match:
                    issue_counter += 1
                    issues.append(CohesionIssue(
                        issue_id=f"SKEL-{issue_counter:03d}",
                        severity="blocking",
                        category="scope_violation",
                        description=(
                            f"Architecture for {seg_id} includes file "
                            f"'{arch_file}' which is outside its skeleton scope"
                        ),
                        source_segment=seg_id,
                        file_path=arch_file,
                        expected=f"Files in scope: {', '.join(sorted(seg_scope))}",
                        suggested_fix="Remove this file from the architecture or update the manifest scope",
                    ))

        # --- Check 2: Segment references must be valid ---
        seg_refs = _extract_segment_references(arch_content)
        for ref_num in seg_refs:
            # Build possible segment ID patterns
            ref_found = False
            for valid_id in all_segment_ids:
                if f"seg-{ref_num:02d}" in valid_id or f"seg-{ref_num}" in valid_id:
                    ref_found = True
                    break
            if not ref_found:
                issue_counter += 1
                issues.append(CohesionIssue(
                    issue_id=f"SKEL-{issue_counter:03d}",
                    severity="blocking",
                    category="phantom_segment",
                    description=(
                        f"Architecture for {seg_id} references segment {ref_num} "
                        f"which doesn't exist (valid: {sorted(all_segment_ids)})"
                    ),
                    source_segment=seg_id,
                    expected=f"Valid segment numbers: {', '.join(str(i) for i in range(1, len(all_segment_ids)+1))}",
                    suggested_fix="Remove reference to non-existent segment",
                ))

        # --- Check 3: Required exports present in architecture ---
        seg_exports = exports_by_segment.get(seg_id, [])
        for export in seg_exports:
            export_path = export.get("file_path", "").replace("\\", "/").lower()
            if export_path and export_path not in {f.replace("\\", "/").lower() for f in arch_files}:
                # Export file isn't in the architecture's file inventory
                # This is a warning, not blocking — the file might be mentioned
                # elsewhere in the arch or handled implicitly
                consumed_by = export.get("consumed_by", [])
                if consumed_by:
                    issue_counter += 1
                    issues.append(CohesionIssue(
                        issue_id=f"SKEL-{issue_counter:03d}",
                        severity="warning",
                        category="missing_export",
                        description=(
                            f"Segment {seg_id} should export '{export_path}' "
                            f"(consumed by {', '.join(consumed_by)}) but it's not "
                            f"in the file inventory"
                        ),
                        source_segment=seg_id,
                        related_segment=consumed_by[0] if consumed_by else "",
                        file_path=export_path,
                        suggested_fix="Add this file to the architecture's file inventory",
                    ))

    # =========================================================================
    # Check 4 (v2.3 FIX #4): Detect imports from undeclared upstream segments
    # =========================================================================
    # Scan architecture text for "from .module import" patterns and check if
    # the target module lives in an upstream segment that isn't declared.
    if manifest_dict:
        import re as _re
        _seg_id_to_files = {}
        for _seg_data in manifest_dict.get("segments", []):
            _sid = _seg_data.get("segment_id", "")
            _seg_id_to_files[_sid] = set(
                f.replace("\\", "/").lower().rsplit("/", 1)[-1].replace(".py", "")
                for f in _seg_data.get("file_scope", [])
            )

        for seg_id, arch_content in architectures.items():
            # Extract relative imports: from .module import X
            _imports = _re.findall(r'from\s+\.(\w+)\s+import', arch_content)
            _seg_deps = set()
            _seg_data = next((s for s in manifest_dict.get("segments", []) if s.get("segment_id") == seg_id), None)
            if _seg_data:
                _seg_deps = set(_seg_data.get("dependencies", []))
                _own_modules = _seg_id_to_files.get(seg_id, set())

                for _imp_module in set(_imports):
                    if _imp_module in _own_modules:
                        continue  # Same segment, fine
                    # Find which segment owns this module
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
                                f"as a dependency. This may cause build-order issues."
                            ),
                            source_segment=seg_id,
                            related_segment=_owner,
                            file_path=f"{_imp_module}.py",
                            suggested_fix=f"Add {_owner} to {seg_id}'s dependencies",
                        ))

    # =========================================================================
    # Check 5 (v2.3 FIX #5): Detect missing stdlib imports (logging, os, etc.)
    # =========================================================================
    # Scan architecture code blocks for logger.xxx() calls and verify logging
    # is listed in the imports section.
    if architectures:
        import re as _re
        for seg_id, arch_content in architectures.items():
            # Check for logger usage without logging import
            _has_logger_call = bool(_re.search(r'\blogger\.(info|warning|error|debug|critical)\b', arch_content))
            _has_logging_import = bool(_re.search(r'\bimport\s+logging\b', arch_content))
            if _has_logger_call and not _has_logging_import:
                issue_counter += 1
                issues.append(CohesionIssue(
                    issue_id=f"SKEL-{issue_counter:03d}",
                    severity="warning",
                    category="missing_import",
                    description=(
                        f"{seg_id} uses logger.xxx() calls but does not include "
                        f"'import logging' in its architecture. This will cause "
                        f"NameError at runtime."
                    ),
                    source_segment=seg_id,
                    suggested_fix="Add 'import logging' and 'logger = logging.getLogger(__name__)' to the module",
                ))

    # =========================================================================
    # Check 6 (v2.7): Cross-segment symbol verification
    # =========================================================================
    # When segment A's architecture has "from .constants import X, Y, Z" and
    # segment B owns constants.py, verify that X, Y, Z are actually defined
    # in segment B's architecture. This catches the recurring pattern where
    # the LLM generates constants.py but omits some constants.
    if manifest_dict and len(architectures) > 1:
        import re as _re

        # Build a map of full_module_path -> (segment_id, exported_symbols)
        # exported_symbols = set of names defined in that module's architecture
        # v6.1 FIX 22: Key by FULL PATH instead of basename to prevent
        # cross-package collisions when multiple source files produce
        # identically-named submodules (models.py, core.py).
        _module_to_segment: Dict[str, str] = {}  # full_path -> seg_id
        _module_exports: Dict[str, Set[str]] = {}  # full_path -> symbols
        _seg_package_prefix: Dict[str, str] = {}  # seg_id -> package dir

        for _seg_data in manifest_dict.get("segments", []):
            _sid = _seg_data.get("segment_id", "")
            _file_paths = [f.replace("\\", "/") for f in _seg_data.get("file_scope", [])]
            # Derive package prefix from common directory of file_scope
            if _file_paths:
                _dirs = [fp.rsplit("/", 1)[0] if "/" in fp else "" for fp in _file_paths]
                _seg_package_prefix[_sid] = _dirs[0] if _dirs else ""
            for _fp in _file_paths:
                _full_key = _fp.replace(".py", "").lower()
                _module_to_segment[_full_key] = _sid

        # Extract symbols defined in each segment's architecture
        # Look for: def func_name, class ClassName, CONSTANT_NAME =, async def func_name
        for seg_id, arch_content in architectures.items():
            _defined = set()
            # Function/method definitions
            for _m in _re.finditer(r'(?:async\s+)?def\s+(\w+)\s*\(', arch_content):
                _defined.add(_m.group(1))
            # Class definitions
            for _m in _re.finditer(r'class\s+(\w+)\s*[\(:]', arch_content):
                _defined.add(_m.group(1))
            # Constants (ALL_CAPS = value)
            for _m in _re.finditer(r'^([A-Z][A-Z0-9_]+)\s*=', arch_content, _re.MULTILINE):
                _defined.add(_m.group(1))
            # Also check for constants in prose: "defines CONSTANT_NAME"
            for _m in _re.finditer(r'(?:defines?|contains?|exports?|provides?)\s+`?([A-Z][A-Z0-9_]+)`?', arch_content):
                _defined.add(_m.group(1))
            # v3.6: Backtick-quoted function signatures in prose
            # Catches patterns like: `_now_iso() -> str`, `_find_latest_arch(seg_dir: str)`
            for _m in _re.finditer(r'`(\w+)\s*\(', arch_content):
                _name = _m.group(1)
                if _name not in ('import', 'from', 'class', 'def', 'async', 'return', 'if', 'for', 'while', 'with', 'try', 'except'):
                    _defined.add(_name)
            # v3.6: Prose export/import lists: "export _func_name", "imports `_func`"
            for _m in _re.finditer(r'(?:exports?|imports?|provides?|defines?)\s+`(\w+)`', arch_content):
                _defined.add(_m.group(1))

            # Map to modules owned by this segment (v6.1 FIX 22: full path keys)
            _seg_data = next((s for s in manifest_dict.get("segments", []) if s.get("segment_id") == seg_id), None)
            if _seg_data:
                for _fp in _seg_data.get("file_scope", []):
                    _full_key = _fp.replace("\\", "/").replace(".py", "").lower()
                    _module_exports.setdefault(_full_key, set()).update(_defined)

        # Now check: for each segment's imports from other modules,
        # verify the imported symbols exist in the target module's exports
        for seg_id, arch_content in architectures.items():
            # Extract: from .module import name1, name2, name3
            for _m in _re.finditer(
                r'from\s+\.(\w+)\s+import\s+([^(\n]+)',
                arch_content,
            ):
                _target_mod = _m.group(1).lower()
                _imports_str = _m.group(2).strip().rstrip("\\").strip('`')
                _imported_names = [n.strip().strip('`').split(" as ")[0] for n in _imports_str.split(",") if n.strip()]

                # v6.1 FIX 22: Resolve relative import to full path within
                # this segment's package, then check cross-segment ownership.
                _pkg_prefix = _seg_package_prefix.get(seg_id, "")
                _full_target = f"{_pkg_prefix}/{_target_mod}".lower() if _pkg_prefix else _target_mod

                # Only check cross-segment imports
                _target_seg = _module_to_segment.get(_full_target)
                if not _target_seg or _target_seg == seg_id:
                    continue

                _available = _module_exports.get(_full_target, set())
                if not _available:
                    continue  # Can't verify if we don't know the exports

                for _imp_name in _imported_names:
                    _imp_name = _imp_name.strip().strip('`').strip()
                    if not _imp_name or _imp_name.startswith("#") or _imp_name.startswith(")"):
                        continue
                    # v3.4: Skip non-identifier garbage from prose line captures
                    if not _re.match(r'^[a-zA-Z_]\w*$', _imp_name):
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

    # =========================================================================
    # Check 7 (v4.0 Fix 2): Duplicate function detection across segments
    # =========================================================================
    # If the same function name is defined in multiple segments' architectures,
    # it should be assigned to exactly one segment and imported by the others.
    # This catches the sg-8d29f79f bug where run_segmented_job (280+ lines)
    # was placed in both _loop.py (seg-02) and _utils.py (seg-06).
    if len(architectures) > 1:
        import re as _re

        # Build: function_name -> [(segment_id, file_path, line_count_estimate, source)]
        # v6.1 FIX 22: Include deterministic_source to scope duplicates
        # to the same source monolith. Functions with the same name from
        # different source files (e.g. to_dict in conduct_policy vs
        # sandbox_build_validator) are NOT duplicates.
        _func_locations: Dict[str, List[tuple]] = {}

        for seg_id, arch_content in architectures.items():
            # Get this segment's file scope and source from manifest
            _seg_files = set()
            _seg_source = ""
            if manifest_dict:
                _seg_data = next(
                    (s for s in manifest_dict.get("segments", [])
                     if s.get("segment_id") == seg_id), None
                )
                if _seg_data:
                    _seg_files = set(_seg_data.get("file_scope", []))
                    _seg_source = _seg_data.get("deterministic_source", "")

            # Extract function definitions from architecture
            # Match: def func_name( or async def func_name(
            for _m in _re.finditer(
                r'(?:async\s+)?def\s+(\w+)\s*\(',
                arch_content,
            ):
                _fname = _m.group(1)
                # Skip dunder methods and test functions
                if _fname.startswith('__') and _fname.endswith('__'):
                    continue
                if _fname.startswith('test_'):
                    continue

                # Estimate function size by counting lines until next def/class
                _start = _m.start()
                _next_def = _re.search(
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

        # Flag functions that appear in more than one segment
        # v6.1 FIX 22: Only flag duplicates from the SAME source monolith.
        for _fname, _locs in _func_locations.items():
            # Group by source monolith — only flag within same source
            _by_source: Dict[str, List[tuple]] = {}
            for _loc in _locs:
                _src = _loc[3] if len(_loc) > 3 else ""
                _by_source.setdefault(_src, []).append(_loc)

            # Check each source group independently
            for _src_group in _by_source.values():
                _unique_segs = set(loc[0] for loc in _src_group)
                if len(_unique_segs) <= 1:
                    continue

                # Get max estimated line count
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
                        f"{_seg_list}. Estimated {_max_lines}+ lines. "
                        f"It should be defined in exactly one segment and imported by others."
                    ),
                    source_segment=sorted(_unique_segs)[0],
                    related_segment=sorted(_unique_segs)[1] if len(_unique_segs) > 1 else "",
                    suggested_fix=(
                        f"Assign '{_fname}' to one segment only. The other segment(s) "
                        f"should import it via 'from .module import {_fname}'."
                    ),
                ))
                logger.warning(
                    "[cohesion_check] v4.0 DUPLICATE FUNCTION: '%s' in segments %s (~%d lines)",
                    _fname, _seg_list, _max_lines,
                )

    # =========================================================================
    # Check 8 (v4.0 Fix 3): Cross-segment missing symbol with monolith check
    # =========================================================================
    # Extends Check 6: when a symbol isn't found in the producing segment's
    # architecture, also check if it exists in the source monolith. If it
    # doesn't exist anywhere, flag as blocking on the CONSUMING segment
    # (the one that invented the import), not the producing segment.
    if manifest_dict and len(architectures) > 1:
        import re as _re

        # Try to load source file evidence for monolith verification
        _monolith_symbols: set = set()
        _job_dir_segments = os.path.join(job_dir, "segments") if 'job_dir' in dir() else ""
        # Check if source_file_evidence was passed to the parent function
        # (it's available via the cohesion_check caller but not directly here).
        # We'll scan the manifest for evidence_files and try to load them.
        _evidence_paths: set = set()
        for _seg_data in manifest_dict.get("segments", []):
            for _ef in _seg_data.get("evidence_files", []):
                _evidence_paths.add(_ef)

        # Try loading evidence files to extract defined symbols
        for _ef_path in _evidence_paths:
            # Try common base paths
            for _base in ["D:\\Orb", "D:/Orb"]:
                _full = os.path.join(_base, _ef_path.replace("/", os.sep))
                if os.path.isfile(_full):
                    try:
                        with open(_full, "r", encoding="utf-8") as _f:
                            _src = _f.read()
                        # Extract all defined names
                        for _m in _re.finditer(r'(?:async\s+)?def\s+(\w+)\s*\(', _src):
                            _monolith_symbols.add(_m.group(1))
                        for _m in _re.finditer(r'class\s+(\w+)\s*[\(:]', _src):
                            _monolith_symbols.add(_m.group(1))
                        for _m in _re.finditer(r'^([A-Z][A-Z0-9_]+)\s*=', _src, _re.MULTILINE):
                            _monolith_symbols.add(_m.group(1))
                    except Exception:
                        pass
                    break

        if _monolith_symbols:
            logger.info(
                "[cohesion_check] v4.0 Loaded %d symbols from evidence/monolith files",
                len(_monolith_symbols),
            )

        # Now re-check the missing_symbol issues from Check 6
        # For any missing symbol that ALSO doesn't exist in the monolith,
        # upgrade to blocking and target the consuming segment for regen
        for _issue in list(issues):
            if _issue.category != "missing_symbol":
                continue
            # Extract the symbol name from the description
            _sym_match = _re.search(r"imports '(\w+)'", _issue.description)
            if not _sym_match:
                continue
            _sym_name = _sym_match.group(1)

            if _sym_name in _monolith_symbols:
                # Symbol exists in monolith — downgrade to warning
                # The producing segment just needs to include it
                _issue.severity = "warning"
                _issue.auto_fix_note = (
                    f"Symbol '{_sym_name}' exists in source monolith — "
                    f"producing segment should extract it"
                )
            else:
                # Symbol doesn't exist ANYWHERE — the consuming segment
                # invented this import. Flag as blocking on consumer.
                _issue.severity = "blocking"
                _issue.source_segment = _issue.source_segment  # Consumer
                _issue.suggested_fix = (
                    f"Symbol '{_sym_name}' does not exist anywhere — "
                    f"not in the producing segment's architecture and not in "
                    f"the source monolith. Remove this import or use an "
                    f"alternative that actually exists."
                )
                _issue.auto_fix_note = (
                    f"v4.0: Symbol '{_sym_name}' verified absent from monolith"
                )
                logger.warning(
                    "[cohesion_check] v4.0 PHANTOM SYMBOL: '%s' does not exist "
                    "anywhere — consuming segment %s invented this import",
                    _sym_name, _issue.source_segment,
                )

    return issues
