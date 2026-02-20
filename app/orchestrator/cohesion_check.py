"""
Cohesion Check — Cross-Segment Architecture Verification.

Two-layer verification:
  Layer 1: Deterministic skeleton compliance (free, instant)
  Layer 2: LLM-based cross-segment cohesion (Opus 4.6, deep analysis)

Layer 1 runs first and catches mechanical violations:
  - File inventory items outside the segment's skeleton scope
  - References to segments that don't exist
  - Missing exports that downstream segments depend on
  - Architecture files that couldn't be loaded

Layer 2 runs second and catches semantic issues:
  - Import resolution failures across segments
  - Interface signature mismatches
  - Data shape incompatibilities
  - Naming convention inconsistencies

v1.0 (2026-02-10): Initial LLM-based cohesion check
v2.0 (2026-02-12): Added deterministic skeleton compliance (Layer 1),
                    fixed file corruption from v1.0, clean rewrite.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

COHESION_CHECK_BUILD_ID = "2026-02-18-v3.7-sectional-tier2-fix"
print(f"[COHESION_CHECK_LOADED] BUILD_ID={COHESION_CHECK_BUILD_ID}")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CohesionIssue:
    """A single cohesion issue found between segments."""
    issue_id: str = ""
    severity: str = "warning"  # "blocking" or "warning"
    category: str = ""         # import_mismatch, naming_mismatch, shape_mismatch,
                               # missing_export, contract_violation, scope_violation,
                               # phantom_segment, endpoint_mismatch
    description: str = ""
    source_segment: str = ""
    related_segment: str = ""
    file_path: str = ""
    expected: str = ""
    actual: str = ""
    suggested_fix: str = ""
    auto_fix_tier: int = 3          # 1=deterministic, 2=micro-LLM, 3=full-regen
    auto_fixed: bool = False        # True if this issue was auto-resolved
    auto_fix_note: str = ""         # What the auto-fixer did

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "source_segment": self.source_segment,
            "related_segment": self.related_segment,
            "file_path": self.file_path,
            "expected": self.expected,
            "actual": self.actual,
            "suggested_fix": self.suggested_fix,
            "auto_fix_tier": self.auto_fix_tier,
            "auto_fixed": self.auto_fixed,
            "auto_fix_note": self.auto_fix_note,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CohesionIssue":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CohesionResult:
    """Result of the cohesion check."""
    status: str = "pass"  # "pass", "fail", "error"
    issues: List[CohesionIssue] = field(default_factory=list)
    segments_checked: List[str] = field(default_factory=list)
    notes: str = ""
    layer1_ran: bool = False
    layer2_ran: bool = False

    @property
    def blocking_issues(self) -> List[CohesionIssue]:
        return [i for i in self.issues if i.severity == "blocking"]

    @property
    def warning_issues(self) -> List[CohesionIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def segments_needing_regen(self) -> List[str]:
        segs = set()
        for i in self.blocking_issues:
            if i.source_segment:
                segs.add(i.source_segment)
        return sorted(segs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "issues": [i.to_dict() for i in self.issues],
            "segments_checked": self.segments_checked,
            "notes": self.notes,
            "layer1_ran": self.layer1_ran,
            "layer2_ran": self.layer2_ran,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CohesionResult":
        result = cls(
            status=data.get("status", "pass"),
            segments_checked=data.get("segments_checked", []),
            notes=data.get("notes", ""),
            layer1_ran=data.get("layer1_ran", False),
            layer2_ran=data.get("layer2_ran", False),
        )
        for issue_data in data.get("issues", []):
            result.issues.append(CohesionIssue.from_dict(issue_data))
        return result


# =============================================================================
# LAYER 1: DETERMINISTIC SKELETON COMPLIANCE
# =============================================================================

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

        # Build a map of module_name -> (segment_id, exported_symbols)
        # exported_symbols = set of names defined in that module's architecture
        _module_to_segment: Dict[str, str] = {}
        _module_exports: Dict[str, Set[str]] = {}

        for _seg_data in manifest_dict.get("segments", []):
            _sid = _seg_data.get("segment_id", "")
            for _fp in _seg_data.get("file_scope", []):
                _mod_name = _fp.replace("\\", "/").rsplit("/", 1)[-1].replace(".py", "").lower()
                _module_to_segment[_mod_name] = _sid

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

            # Map to modules owned by this segment
            _seg_files = set()
            _seg_data = next((s for s in manifest_dict.get("segments", []) if s.get("segment_id") == seg_id), None)
            if _seg_data:
                _seg_files = {
                    f.replace("\\", "/").rsplit("/", 1)[-1].replace(".py", "").lower()
                    for f in _seg_data.get("file_scope", [])
                }
            for _mod in _seg_files:
                _module_exports.setdefault(_mod, set()).update(_defined)

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

                # Only check cross-segment imports
                _target_seg = _module_to_segment.get(_target_mod)
                if not _target_seg or _target_seg == seg_id:
                    continue

                _available = _module_exports.get(_target_mod, set())
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

        # Build: function_name -> [(segment_id, file_path, line_count_estimate)]
        _func_locations: Dict[str, List[tuple]] = {}

        for seg_id, arch_content in architectures.items():
            # Get this segment's file scope from manifest
            _seg_files = set()
            if manifest_dict:
                _seg_data = next(
                    (s for s in manifest_dict.get("segments", [])
                     if s.get("segment_id") == seg_id), None
                )
                if _seg_data:
                    _seg_files = set(_seg_data.get("file_scope", []))

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
                    (seg_id, ", ".join(sorted(_seg_files)[:3]), _line_count)
                )

        # Flag functions that appear in more than one segment
        for _fname, _locs in _func_locations.items():
            _unique_segs = set(loc[0] for loc in _locs)
            if len(_unique_segs) <= 1:
                continue

            # Get max estimated line count
            _max_lines = max(loc[2] for loc in _locs)
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


def _extract_arch_file_paths(arch_content: str) -> List[str]:
    """Extract file paths from architecture document File Inventory tables only.
    
    IMPORTANT: Only extracts from the File Inventory section to avoid false
    positives from paths mentioned in prose, docstrings, and import examples.
    """
    paths = []
    seen = set()

    # Find the File Inventory section
    inv_match = re.search(r'(?:^|\n)#+\s*File Inventory', arch_content)
    if not inv_match:
        return paths
    
    inv_start = inv_match.start()
    # Find the end of the inventory section (next ## heading or ---)
    inv_end_match = re.search(r'\n(?:##[^#]|---)', arch_content[inv_start + 20:])
    if inv_end_match:
        inv_section = arch_content[inv_start:inv_start + 20 + inv_end_match.start()]
    else:
        # Take a reasonable chunk
        inv_section = arch_content[inv_start:inv_start + 3000]

    # Extract paths from table rows in the inventory section.
    # Only match the FIRST backtick-wrapped path in each table row
    # to avoid picking up filenames from description columns.
    for line in inv_section.split("\n"):
        # Must be a table row (starts with |) and not a header separator
        if not line.strip().startswith("|") or line.strip().startswith("|---"):
            continue
        # Skip rows with "none" or "N/A" markers (in either first cell or description)
        line_lower = line.lower()
        if "*(none" in line_lower or "*(n/a" in line_lower or "_(none" in line_lower or "_(n/a" in line_lower:
            continue
        # Find FIRST backtick-wrapped path in this row
        match = re.search(
            r'`((?:app|src|tests|config|orb-desktop)[/\\][\w/\\._-]+\.[a-z]+)`',
            line
        )
        if not match:
            # Try root-level file (e.g. main.py)
            match = re.search(
                r'`([\w_-]+\.(?:py|ts|tsx|js|jsx|json|yaml|yml|md|css))`',
                line
            )
            # Only accept if it's truly the first cell content
            if match and ("/" in match.group(1) or "\\" in match.group(1)):
                match = None
        if match:
            p = match.group(1)
            key = p.replace("\\", "/").lower()
            if key not in seen:
                seen.add(key)
                paths.append(p)

    return paths


def _extract_segment_references(arch_content: str) -> List[int]:
    """Extract segment number references from architecture content."""
    refs = set()
    for match in re.finditer(r'[Ss]egment[\s_-]*(\d+)', arch_content):
        refs.add(int(match.group(1)))
    return sorted(refs)


# =============================================================================
# LAYER 2: LLM-BASED CROSS-SEGMENT COHESION
# =============================================================================

def _build_cohesion_prompt(
    architectures: Dict[str, str],
    contract_json: Optional[str] = None,
    source_file_evidence: Optional[Dict[str, str]] = None,
) -> str:
    """Build the prompt for the LLM cohesion check."""
    parts = []
    parts.append("# Cross-Segment Architecture Cohesion Check\n")
    parts.append("You are reviewing multiple segment architectures for a single job.")
    parts.append("Check for cross-segment compatibility issues:\n")
    parts.append("1. **Import resolution**: Do imports between segments resolve correctly?")
    parts.append("   - Consider the DIRECTORY STRUCTURE: files in a sub-package use `..` to import from parent package")
    parts.append("2. **Naming matches**: Do function/class/variable names match across boundaries?")
    parts.append("3. **Signature compatibility**: Do function signatures match what callers expect?")
    parts.append("4. **Data shape compatibility**: Do data structures match across segment boundaries?")
    parts.append("5. **Contract compliance**: Do segments fulfil their skeleton contract obligations?")
    parts.append("6. **Endpoint consistency**: Do API endpoints and router prefixes align?")
    parts.append("")
    parts.append("Severity rules:")
    parts.append("- 'blocking': Would cause import errors, type errors, or runtime crashes")
    parts.append("- 'warning': Might cause issues or indicates suboptimal design")
    parts.append("- If the SOURCE FILE EVIDENCE below confirms a claim in the architecture, it is NOT blocking.")
    parts.append("  Only flag as blocking if the architecture CONTRADICTS the source evidence or would cause runtime errors.")
    parts.append("")

    if contract_json:
        parts.append("## Skeleton Contract\n")
        parts.append("```json")
        # Truncate if very large
        if len(contract_json) > 8000:
            parts.append(contract_json[:8000] + "\n... (truncated)")
        else:
            parts.append(contract_json)
        parts.append("```\n")

    # v2.2: Project structure context for import validation
    if source_file_evidence:
        # Derive directory structure from file paths
        _dirs = set()
        for _sf_path in source_file_evidence.keys():
            _path_parts = _sf_path.replace("\\", "/").split("/")
            for _depth in range(1, len(_path_parts)):
                _dirs.add("/".join(_path_parts[:_depth]))
        if _dirs:
            parts.append("## Project Directory Structure\n")
            parts.append(
                "The following directories exist in the project. Use this to determine "
                "correct relative import paths (e.g. files in `app/overwatcher/architecture_executor/` "
                "must use `from ..spec_resolution import ...` to reach `app/overwatcher/spec_resolution.py`, "
                "NOT `from .spec_resolution import ...`).\n"
            )
            for _d in sorted(_dirs):
                parts.append(f"- `{_d}/`")
            parts.append("")

    # v2.2: Source file evidence for verification
    if source_file_evidence:
        parts.append("## Source File Evidence (GROUND TRUTH)\n")
        parts.append(
            "The following file(s) are the ORIGINAL source code being refactored. "
            "Use these to VERIFY claims in the architectures. If an architecture "
            "states a function signature or constant value, check it against this evidence. "
            "Only flag mismatches between architecture and THIS evidence as issues.\n"
        )
        for _sf_path, _sf_content in source_file_evidence.items():
            # Cap at 60K per file for cohesion check (less than critical pipeline)
            _sf_inject = _sf_content[:60_000]
            parts.append(f"**`{_sf_path}`** ({len(_sf_content):,} chars)")
            parts.append(f"```python\n{_sf_inject}\n```\n")
            if len(_sf_content) > 60_000:
                parts.append(f"... (truncated from {len(_sf_content):,} chars)\n")

    for seg_id, arch in architectures.items():
        parts.append(f"## Architecture: {seg_id}\n")
        # Truncate each architecture to avoid context overflow
        if len(arch) > 15000:
            parts.append(arch[:15000])
            parts.append(f"\n... (truncated from {len(arch)} chars)")
        else:
            parts.append(arch)
        parts.append("")

    parts.append("## Response Format\n")
    parts.append("Respond with a JSON object:")
    parts.append("```json")
    parts.append("""{
  "status": "pass" | "fail",
  "issues": [
    {
      "issue_id": "COH-001",
      "severity": "blocking" | "warning",
      "category": "import_mismatch|naming_mismatch|shape_mismatch|missing_export|contract_violation|endpoint_mismatch",
      "description": "What the issue is",
      "source_segment": "seg-01-...",
      "related_segment": "seg-02-...",
      "file_path": "app/foo/bar.py",
      "suggested_fix": "How to fix it"
    }
  ],
  "notes": "Optional overall notes"
}""")
    parts.append("```")
    parts.append("")
    parts.append("If all segments are compatible, return status 'pass' with an empty issues array.")
    parts.append("Only report REAL issues — do not invent problems.")

    return "\n".join(parts)


def _parse_cohesion_response(llm_output: str) -> CohesionResult:
    """Parse the LLM's cohesion check response."""
    # Strip markdown fences
    cleaned = llm_output.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last fence lines
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    # Clean trailing commas (common LLM output issue)
    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("[cohesion_check] Failed to parse LLM response: %s", e)
        return CohesionResult(
            status="error",
            notes=f"Failed to parse LLM response: {e}",
        )

    result = CohesionResult(
        status=data.get("status", "pass"),
        notes=data.get("notes", ""),
    )

    for issue_data in data.get("issues", []):
        result.issues.append(CohesionIssue(
            issue_id=issue_data.get("issue_id", ""),
            severity=issue_data.get("severity", "warning"),
            category=issue_data.get("category", ""),
            description=issue_data.get("description", ""),
            source_segment=issue_data.get("source_segment", ""),
            related_segment=issue_data.get("related_segment", ""),
            file_path=issue_data.get("file_path", ""),
            suggested_fix=issue_data.get("suggested_fix", ""),
        ))

    # Ensure status reflects issues
    if result.blocking_issues and result.status == "pass":
        result.status = "fail"

    return result


# =============================================================================
# ARCHITECTURE LOADING
# =============================================================================

def load_segment_architectures(
    job_dir: str,
    segment_ids: List[str],
) -> Dict[str, str]:
    """
    Load architecture files for the given segments.

    v5.8: Dynamically finds the highest arch_v{N}.md instead of a hardcoded
    fallback list.  This is consistent with segment_loop._find_latest_arch()
    so that cohesion checking and execution always read the same version.

    Returns {segment_id: architecture_content} for segments that have architectures.
    """
    architectures = {}
    for seg_id in segment_ids:
        seg_dir = os.path.join(job_dir, "segments", seg_id)
        arch_dir = os.path.join(seg_dir, "arch")

        if not os.path.isdir(arch_dir):
            continue

        # v5.8: Find the highest version dynamically
        max_version = 0
        best_path = None
        for fname in os.listdir(arch_dir):
            if fname.startswith("arch_v") and fname.endswith(".md"):
                try:
                    v = int(fname.replace("arch_v", "").replace(".md", ""))
                    if v > max_version:
                        max_version = v
                        best_path = os.path.join(arch_dir, fname)
                except ValueError:
                    pass

        if best_path and os.path.isfile(best_path):
            try:
                with open(best_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if content.strip():
                    architectures[seg_id] = content
                    logger.debug("[cohesion_check] Loaded %s for %s (%d chars)",
                                 os.path.basename(best_path), seg_id, len(content))
            except Exception as e:
                logger.warning("[cohesion_check] Failed to read %s: %s", best_path, e)

    return architectures


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def run_cohesion_check(
    job_id: str,
    job_dir: str,
    segment_ids: List[str],
    contract_json: Optional[str] = None,
    provider_id: Optional[str] = None,
    model_id: Optional[str] = None,
    source_file_evidence: Optional[Dict[str, str]] = None,
    skip_llm_layer: bool = False,
) -> CohesionResult:
    """
    Run the cross-segment cohesion check (both layers).

    Layer 1: Deterministic skeleton compliance (always runs, free)
    Layer 2: LLM-based cross-segment analysis (runs if Layer 1 passes)

    v6.1: skip_llm_layer=True for deterministic refactor jobs — the
    architecture was generated from scan data so Layer 2 adds no value.

    Args:
        job_id: Job identifier
        job_dir: Path to job directory on disk
        segment_ids: List of segment IDs to check (APPROVED segments)
        contract_json: Optional JSON string from SkeletonContractSet.to_json()
        provider_id: Override provider for Layer 2 (default: anthropic)
        model_id: Override model for Layer 2 (default: from stage config)

    Returns:
        CohesionResult with any issues found
    """
    if len(segment_ids) < 2:
        return CohesionResult(
            status="pass",
            segments_checked=segment_ids,
            notes="Skipped: fewer than 2 segments to check",
        )

    # Load architectures from disk
    architectures = load_segment_architectures(job_dir, segment_ids)

    if len(architectures) < 2:
        return CohesionResult(
            status="pass",
            segments_checked=list(architectures.keys()),
            notes=f"Skipped: only {len(architectures)} architecture(s) found on disk",
        )

    result = CohesionResult(segments_checked=list(architectures.keys()))

    # Also load manifest for additional context
    manifest_dict = None
    manifest_path = os.path.join(job_dir, "segments", "manifest.json")
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_dict = json.load(f)
        except Exception:
            pass

    # =========================================================================
    # LAYER 1: Deterministic skeleton compliance
    # =========================================================================
    logger.info("[cohesion_check] Layer 1: Running skeleton compliance check")
    layer1_issues = run_skeleton_compliance(
        architectures=architectures,
        skeleton_json=contract_json,
        manifest_dict=manifest_dict,
    )
    result.issues.extend(layer1_issues)
    result.layer1_ran = True

    if layer1_issues:
        n_blocking = len([i for i in layer1_issues if i.severity == "blocking"])
        n_warning = len([i for i in layer1_issues if i.severity == "warning"])
        logger.info("[cohesion_check] Layer 1: %d blocking, %d warning", n_blocking, n_warning)
    else:
        logger.info("[cohesion_check] Layer 1: CLEAN — no skeleton violations")

    # If Layer 1 found blocking issues, try auto-fix BEFORE giving up
    if any(i.severity == "blocking" for i in layer1_issues):
        logger.info("[cohesion_check] Layer 1 blocking issues found — attempting auto-fix")
        result.status = "fail"
        result.notes = "Layer 1 (skeleton compliance) found blocking issues"

        # Attempt tiered auto-fix
        result = await attempt_auto_fixes(
            result=result,
            job_dir=job_dir,
            architectures=architectures,
            skeleton_json=contract_json,
            manifest_dict=manifest_dict,
        )

        # If all blocking issues resolved, continue to Layer 2
        if result.blocking_issues:
            result.notes += " — auto-fix could not resolve all blocking issues, Layer 2 skipped"
            return result
        else:
            logger.info("[cohesion_check] Auto-fix resolved all blocking issues — proceeding to Layer 2")
            result.status = "pass"  # Reset for Layer 2 evaluation

    # =========================================================================
    # LAYER 2: LLM-based cross-segment cohesion
    # =========================================================================
    # v6.1: Skip LLM layer for deterministic refactor jobs
    if skip_llm_layer:
        logger.info("[cohesion_check] Layer 2: SKIPPED (deterministic refactor — skip_llm_layer=True)")
        if result.status != "fail":
            result.status = "pass"
            result.notes = (result.notes + " | Layer 2 skipped (deterministic refactor)").strip(" | ")
        return result

    logger.info("[cohesion_check] Layer 2: Running LLM cohesion check")

    # Resolve provider/model
    _provider = provider_id
    _model = model_id

    if not _provider or not _model:
        try:
            from app.llm.stage_models import get_stage_config
            config = get_stage_config("COHESION_CHECK")
            _provider = _provider or config.get("provider", "anthropic")
            _model = _model or config.get("model", "claude-opus-4-6")
        except Exception:
            _provider = _provider or os.getenv("COHERENCE_GUARDIAN_PROVIDER", "anthropic")
            _model = _model or os.getenv("COHERENCE_GUARDIAN_MODEL", "claude-opus-4-6")

    # Build prompt
    prompt = _build_cohesion_prompt(architectures, contract_json, source_file_evidence)

    # Call LLM
    try:
        from app.providers.registry import llm_call

        _messages = [{"role": "user", "content": prompt}]
        _system = (
            "You are a cross-segment architecture reviewer. "
            "Check for interface compatibility issues between segments. "
            "Be precise and only report real issues. "
            "Respond with valid JSON only."
        )

        llm_result_obj = await llm_call(
            provider_id=_provider,
            model_id=_model,
            messages=_messages,
            system_prompt=_system,
            max_tokens=8192,
            timeout_seconds=180,
        )
        llm_response = llm_result_obj.content if llm_result_obj else None

        if llm_response:
            llm_result = _parse_cohesion_response(llm_response)
            result.issues.extend(llm_result.issues)
            if llm_result.notes:
                result.notes = (result.notes + " | " + llm_result.notes).strip(" | ")
            result.layer2_ran = True

            logger.info(
                "[cohesion_check] Layer 2: %d blocking, %d warning",
                len(llm_result.blocking_issues),
                len(llm_result.warning_issues),
            )
        else:
            result.notes = (result.notes + " | Layer 2: empty LLM response").strip(" | ")
            result.layer2_ran = True

    except Exception as llm_err:
        logger.warning("[cohesion_check] Layer 2 LLM call failed: %s", llm_err)
        result.notes = (result.notes + f" | Layer 2 error: {llm_err}").strip(" | ")

    # Determine final status
    if result.blocking_issues:
        # Layer 2 found blocking issues — try auto-fix on those too
        logger.info("[cohesion_check] Layer 2 blocking issues found — attempting auto-fix")

        # Reload architectures in case Layer 1 auto-fix already patched some
        reloaded = load_segment_architectures(job_dir, list(architectures.keys()))

        result = await attempt_auto_fixes(
            result=result,
            job_dir=job_dir,
            architectures=reloaded,
            skeleton_json=contract_json,
            manifest_dict=manifest_dict,
        )

    if result.blocking_issues:
        result.status = "fail"
    else:
        result.status = "pass"

    return result


# =============================================================================
# LAYER 3: TIERED AUTO-FIX
# =============================================================================
# Tier 1: Deterministic regex patches (zero API cost)
# Tier 2: Micro-LLM targeted fixes (tiny API cost, ~500 tokens)
# Tier 3: Full segment regeneration (existing flow, expensive)
#
# v3.0 (2026-02-13): Initial implementation — all three tiers.
# =============================================================================


def _classify_fix_tier(issue: CohesionIssue) -> int:
    """
    Classify an issue into auto-fix tier based on category and content.

    Returns:
        1 = Deterministic fix (regex/string replacement, zero cost)
        2 = Micro-LLM fix (small targeted call, ~500 tokens)
        3 = Full regeneration (existing pipeline, expensive)
    """
    desc_lower = issue.description.lower()
    fix_lower = issue.suggested_fix.lower()
    cat = issue.category

    # ----- TIER 1: Deterministic -----

    # Import depth: from .X → from ..X
    # v3.6: Tightened — only Tier 1 if SAME module name at different depths.
    # Previously matched whenever description had 'from .' and fix had 'from ..'
    # which caused false positives when they referred to DIFFERENT modules
    # (e.g. description mentions 'from ._utils' and fix mentions 'from ..segment_state').
    # v3.7: Removed unconditional "relative import" early return — it bypassed
    # the module-name safety check and allowed different-module replacements
    # (e.g. ._utils → ..segment_state) to be classified as safe Tier 1.
    if cat == "import_mismatch":
        # Only Tier 1 if we can confirm same module name at different depths
        if "from ." in desc_lower and "from .." in fix_lower:
            _old_mod = re.search(r"from\s+\.(\w+)\s+import", issue.description)
            _new_mod = re.search(r"from\s+\.{2,}(\w+)\s+import", issue.suggested_fix)
            if _old_mod and _new_mod and _old_mod.group(1) == _new_mod.group(1):
                return 1  # Same module, different depth -> safe Tier 1
            if _old_mod and _new_mod and _old_mod.group(1) != _new_mod.group(1):
                return 2  # Different modules -> needs LLM judgement
        # v3.3: Import name mismatch with both names known
        if issue.expected and issue.actual:
            return 1

    # Missing stdlib imports (logging, os, etc.)
    if cat == "missing_import":
        if "import logging" in desc_lower or "import logging" in fix_lower:
            return 1

    # Naming mismatch with both names known
    if cat == "naming_mismatch":
        if issue.expected and issue.actual:
            return 1

    # ----- TIER 2: Micro-LLM -----

    # Missing exports that need context-aware insertion
    if cat == "missing_export" and issue.suggested_fix:
        return 2

    # v2.7: Missing symbol (cross-segment import of undefined name)
    # Tier 2: LLM adds the missing definition to the target architecture
    if cat == "missing_symbol" and issue.suggested_fix:
        return 2

    # Contract violations with clear fix description
    if cat == "contract_violation" and issue.suggested_fix and \
       len(issue.suggested_fix) > 20:
        return 2

    # Import mismatch that isn't simple depth (needs LLM judgement)
    if cat == "import_mismatch" and issue.suggested_fix:
        return 2

    # ----- TIER 3: Full regen (default) -----
    return 3


def _extract_import_replacements(issue: CohesionIssue) -> List[tuple]:
    """
    Extract (old_pattern, new_pattern) pairs from an import_mismatch issue.

    Parses the description and suggested_fix for patterns like:
      "from .implementer import" → "from ..implementer import"
    """
    replacements = []
    combined = issue.description + " " + issue.suggested_fix

    # Pattern 1: 'from .X import' → 'from ..X import'
    # Matches: "Change 'from .implementer import ...' to 'from ..implementer import ...'"
    # v3.7: Added module-name guard — only match if the module name is the
    # same at both depths.  Without this, a suggested_fix like
    # "Change 'from ._utils import' to 'from ..segment_state import'"
    # would produce a cross-module replacement that corrupts the architecture.
    pairs = re.findall(
        r"['\"]from\s+(\.+)(\w+)\s+import[^'\"]*['\"]\s*(?:to|→|->)\s*['\"]from\s+(\.{2,})(\w+)\s+import",
        combined,
    )
    for old_dots, old_name, new_dots, new_name in pairs:
        if old_name == new_name:  # v3.7: Same module at different depth — safe
            replacements.append((f"from {old_dots}{old_name} import", f"from {new_dots}{new_name} import"))

    # Pattern 2: Explicit "from .X" / "from ..X" in suggested_fix
    # v3.6: Only pair if the module name matches (different depth of SAME module).
    # Previously paired any single-dot module with any double-dot module,
    # which corrupted architectures when description and fix mentioned
    # different modules (e.g. ._utils in desc, ..segment_state in fix).
    if not replacements:
        old_match = re.search(r"from\s+(\.)(\w+)\s+import", issue.description)
        new_match = re.search(r"from\s+(\.{2,})(\w+)\s+import", issue.suggested_fix)
        if old_match and new_match and old_match.group(2) == new_match.group(2):
            replacements.append((
                f"from {old_match.group(1)}{old_match.group(2)} import",
                f"from {new_match.group(1)}{new_match.group(2)} import",
            ))

    # Pattern 3: General ".module" → "..module" mentioned anywhere
    if not replacements:
        singles = re.findall(r"'\.([a-zA-Z_]\w*)'", issue.description)
        doubles = re.findall(r"'\.\.([a-zA-Z_]\w*)'", issue.suggested_fix)
        for mod in set(singles) & set(doubles):
            replacements.append((f"from .{mod} import", f"from ..{mod} import"))

    return replacements


def _inject_logging_import(arch_text: str) -> Optional[str]:
    """
    Inject 'import logging' + logger line into architecture text.

    Finds the imports section (```python block with import statements)
    and adds logging if missing.
    """
    if "import logging" in arch_text:
        return None  # Already present

    logging_block = "import logging\nlogger = logging.getLogger(__name__)"

    # Strategy 1: Find a python code block with imports and inject after last import
    code_blocks = list(re.finditer(
        r"```python\n(.*?)```",
        arch_text,
        re.DOTALL,
    ))

    for block_match in code_blocks:
        block_content = block_match.group(1)
        # Check if this block has import statements
        if not re.search(r"^(?:import |from )", block_content, re.MULTILINE):
            continue

        # Find the last import line in this block
        import_lines = list(re.finditer(
            r"^(?:import |from )[^\n]+",
            block_content,
            re.MULTILINE,
        ))
        if import_lines:
            last_import = import_lines[-1]
            insert_pos = block_match.start(1) + last_import.end()
            return (
                arch_text[:insert_pos]
                + "\n" + logging_block
                + arch_text[insert_pos:]
            )

    # Strategy 2: Find any "## Imports" or similar heading and inject after
    imports_heading = re.search(
        r"^#+\s*(?:Imports|Dependencies|Module Imports)[^\n]*\n",
        arch_text,
        re.MULTILINE | re.IGNORECASE,
    )
    if imports_heading:
        insert_pos = imports_heading.end()
        # If there's a code fence right after, inject inside it
        after = arch_text[insert_pos:insert_pos + 20]
        if after.strip().startswith("```python"):
            fence_end = arch_text.index("\n", insert_pos + arch_text[insert_pos:].index("```python")) + 1
            return (
                arch_text[:fence_end]
                + logging_block + "\n"
                + arch_text[fence_end:]
            )

    return None  # Couldn't find safe injection point


def _apply_tier1_fix(issue: CohesionIssue, arch_text: str) -> Optional[str]:
    """
    Apply a deterministic Tier 1 fix to architecture text.

    Returns patched text or None if fix couldn't be applied.
    """
    cat = issue.category

    # --- Import depth fixes ---
    if cat == "import_mismatch":
        replacements = _extract_import_replacements(issue)
        if not replacements:
            return None
        patched = arch_text
        applied = []
        for old_pat, new_pat in replacements:
            if old_pat in patched:
                patched = patched.replace(old_pat, new_pat)
                applied.append(f"{old_pat} → {new_pat}")
        if applied:
            issue.auto_fix_note = f"Tier 1: Replaced {'; '.join(applied)}"
            return patched
        return None

    # --- Missing logging import ---
    if cat == "missing_import" and "logging" in issue.description.lower():
        patched = _inject_logging_import(arch_text)
        if patched:
            issue.auto_fix_note = "Tier 1: Injected 'import logging' + logger init"
            return patched
        return None

    # --- Naming mismatch ---
    if cat == "naming_mismatch" and issue.expected and issue.actual:
        if issue.actual in arch_text:
            patched = arch_text.replace(issue.actual, issue.expected)
            issue.auto_fix_note = f"Tier 1: Renamed '{issue.actual}' \u2192 '{issue.expected}'"
            return patched
        return None

    # --- Import name mismatch (v3.3: post-execution cohesion) ---
    # When the cohesion checker identifies a specific wrong→correct name pair
    # in an import statement, fix it deterministically.
    if cat == "import_mismatch" and issue.expected and issue.actual:
        # issue.actual = wrong import name, issue.expected = correct name
        if issue.actual in arch_text:
            patched = arch_text.replace(issue.actual, issue.expected)
            issue.auto_fix_note = f"Tier 1: Fixed import '{issue.actual}' \u2192 '{issue.expected}'"
            return patched
        return None

    return None


async def _apply_tier2_fix(
    issue: CohesionIssue,
    arch_text: str,
    seg_id: str,
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-5-20250929",
) -> Optional[str]:
    """
    Apply a micro-LLM Tier 2 fix to architecture text.

    v3.7: Sends only the relevant SECTION of the architecture (not the
    full document) to minimise tokens and avoid API timeouts. The LLM
    returns just the patched section, which is spliced back in.

    Returns patched full text or None if fix couldn't be applied.
    """
    try:
        from app.providers.registry import llm_call
    except ImportError:
        logger.warning("[cohesion_auto_fix] LLM not available for Tier 2 fix")
        return None

    # =====================================================================
    # v3.7: Extract relevant section instead of sending full architecture.
    # Find section boundaries using markdown headers (## or ###).
    # The issue description usually references a section name or keyword.
    # =====================================================================
    arch_lines = arch_text.split("\n")
    _issue_keywords = set()
    for _word in (issue.description + " " + issue.suggested_fix).lower().split():
        _clean = _word.strip("(),.:'\"`")
        if len(_clean) > 3 and _clean not in {"from", "import", "this", "that", "with", "should", "must", "the", "and", "for"}:
            _issue_keywords.add(_clean)

    # Find section headers and score them by keyword overlap
    _sections: list = []  # [(start_line, end_line, header_text, score)]
    _header_lines: list = []
    for i, line in enumerate(arch_lines):
        if line.strip().startswith("#"):
            _header_lines.append(i)

    for idx, hdr_line in enumerate(_header_lines):
        _end = _header_lines[idx + 1] if idx + 1 < len(_header_lines) else len(arch_lines)
        _section_text = "\n".join(arch_lines[hdr_line:_end]).lower()
        _score = sum(1 for kw in _issue_keywords if kw in _section_text)
        _sections.append((hdr_line, _end, arch_lines[hdr_line].strip(), _score))

    # Pick the best-matching section(s). Include sections with score > 0.
    _sections.sort(key=lambda s: s[3], reverse=True)
    _best_sections = [s for s in _sections if s[3] > 0]

    if _best_sections and len(arch_text) > 4000:
        # Use sectional approach — extract top 1-2 sections
        _selected = _best_sections[:2]
        _selected.sort(key=lambda s: s[0])  # Keep original order

        _extract_start = max(0, _selected[0][0] - 2)
        _extract_end = min(len(arch_lines), _selected[-1][1] + 2)
        _section_text = "\n".join(arch_lines[_extract_start:_extract_end])
        _prefix = "\n".join(arch_lines[:_extract_start])
        _suffix = "\n".join(arch_lines[_extract_end:])
        _using_section = True

        logger.info(
            "[cohesion_auto_fix] Tier 2 sectional: lines %d-%d of %d (%.0f%% of doc)",
            _extract_start, _extract_end, len(arch_lines),
            100 * (_extract_end - _extract_start) / max(1, len(arch_lines)),
        )
    else:
        # Small document or no good section match — send the whole thing
        _section_text = arch_text
        _prefix = ""
        _suffix = ""
        _using_section = False

    prompt = f"""Fix ONE specific issue in this architecture {'section' if _using_section else 'document'}.

ISSUE ({issue.category}, {issue.severity}):
{issue.description}

SUGGESTED FIX:
{issue.suggested_fix}

SEGMENT: {seg_id}

{'ARCHITECTURE SECTION' if _using_section else 'ARCHITECTURE DOCUMENT'}:
{_section_text}

INSTRUCTIONS:
- Apply ONLY the fix described above. Change nothing else.
- Return the {'COMPLETE SECTION' if _using_section else 'COMPLETE DOCUMENT'} with the fix applied.
- Do NOT add commentary, explanations, or markdown fences.
- Preserve ALL existing content, formatting, and structure.
"""

    try:
        _system = (
            "You are a precise architecture editor. Apply the requested fix "
            "and return the complete " + ("section" if _using_section else "document") + ". No commentary."
        )
        _out_budget = min(len(_section_text) // 2 + 2000, 8000)
        response = await llm_call(
            provider_id=provider,
            model_id=model,
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_system,
            max_tokens=_out_budget,
            timeout_seconds=300,
        )

        patched_section = response.content if response else None

        if patched_section and len(patched_section) > len(_section_text) * 0.3:
            # Strip any wrapping markdown fences the LLM might add
            patched_section = patched_section.strip()
            if patched_section.startswith("```") and patched_section.endswith("```"):
                first_nl = patched_section.index("\n") + 1
                patched_section = patched_section[first_nl:-3].strip()

            # Reassemble full document if we used sectional approach
            if _using_section:
                parts = []
                if _prefix:
                    parts.append(_prefix)
                parts.append(patched_section)
                if _suffix:
                    parts.append(_suffix)
                patched = "\n".join(parts)
            else:
                patched = patched_section

            issue.auto_fix_note = f"Tier 2: LLM micro-patch ({provider}/{model})"
            logger.info(
                "[cohesion_auto_fix] Tier 2 fix applied for %s in %s (%d→%d chars, sectional=%s)",
                issue.issue_id, seg_id, len(arch_text), len(patched), _using_section,
            )
            return patched
        else:
            logger.warning(
                "[cohesion_auto_fix] Tier 2 LLM response too short/empty for %s",
                issue.issue_id,
            )
            return None

    except Exception as e:
        logger.warning("[cohesion_auto_fix] Tier 2 LLM call failed: %s", e)
        return None


def _save_patched_architecture(
    job_dir: str,
    seg_id: str,
    patched_text: str,
    fix_notes: List[str],
) -> str:
    """
    Save a patched architecture as the next version number.

    If current is arch_v1.md, saves as arch_v2.md. Preserves history.
    Returns the path of the saved file.
    """
    arch_dir = os.path.join(job_dir, "segments", seg_id, "arch")

    # Find current highest version
    existing = []
    if os.path.isdir(arch_dir):
        for f in os.listdir(arch_dir):
            m = re.match(r"arch_v(\d+)\.md$", f)
            if m:
                existing.append(int(m.group(1)))

    next_ver = max(existing, default=0) + 1
    new_filename = f"arch_v{next_ver}.md"
    new_path = os.path.join(arch_dir, new_filename)

    # Prepend auto-fix header
    header = (
        f"<!-- AUTO-FIX v3.0: {len(fix_notes)} fix(es) applied by cohesion auto-fixer -->\n"
        + "".join(f"<!-- FIX: {note} -->\n" for note in fix_notes)
        + "\n"
    )

    os.makedirs(arch_dir, exist_ok=True)
    with open(new_path, "w", encoding="utf-8") as f:
        f.write(header + patched_text)

    logger.info(
        "[cohesion_auto_fix] Saved patched architecture: %s (%d chars, %d fixes)",
        new_path, len(patched_text), len(fix_notes),
    )
    return new_path


async def attempt_auto_fixes(
    result: CohesionResult,
    job_dir: str,
    architectures: Dict[str, str],
    skeleton_json: Optional[str] = None,
    manifest_dict: Optional[dict] = None,
    tier2_provider: str = "anthropic",
    tier2_model: str = "claude-sonnet-4-5-20250929",
    max_tier2_fixes: int = 3,
) -> CohesionResult:
    """
    Attempt to auto-fix cohesion issues using tiered approach.

    1. Classify each issue into tier (1, 2, or 3)
    2. Apply Tier 1 (deterministic) fixes — zero API cost
    3. Apply Tier 2 (micro-LLM) fixes — tiny API cost
    4. Save patched architectures to disk
    5. Re-validate with skeleton compliance
    6. Return updated result

    Tier 3 issues are left untouched for the existing regen flow.

    Args:
        result: The CohesionResult from initial check
        job_dir: Path to job directory
        architectures: {segment_id: architecture_content}
        skeleton_json: Skeleton contracts JSON
        manifest_dict: Manifest dict for re-validation
        tier2_provider: LLM provider for Tier 2 fixes
        tier2_model: LLM model for Tier 2 fixes
        max_tier2_fixes: Maximum number of Tier 2 LLM calls to make

    Returns:
        Updated CohesionResult with fixed issues marked
    """
    if not result.issues:
        return result

    # =========================================================================
    # Step 1: Classify all issues
    # =========================================================================
    for issue in result.issues:
        issue.auto_fix_tier = _classify_fix_tier(issue)
        logger.debug(
            "[cohesion_auto_fix] %s (%s/%s) → Tier %d",
            issue.issue_id, issue.category, issue.severity, issue.auto_fix_tier,
        )

    tier1_issues = [i for i in result.issues if i.auto_fix_tier == 1]
    tier2_issues = [i for i in result.issues if i.auto_fix_tier == 2]
    tier3_issues = [i for i in result.issues if i.auto_fix_tier == 3]

    logger.info(
        "[cohesion_auto_fix] Classification: %d Tier-1, %d Tier-2, %d Tier-3",
        len(tier1_issues), len(tier2_issues), len(tier3_issues),
    )

    if not tier1_issues and not tier2_issues:
        logger.info("[cohesion_auto_fix] No auto-fixable issues found")
        return result

    # =========================================================================
    # Step 2: Apply Tier 1 fixes (deterministic, per-segment)
    # =========================================================================
    patched_archs: Dict[str, str] = {}  # seg_id → patched text
    fix_log: Dict[str, List[str]] = {}  # seg_id → list of fix notes
    tier1_fixed = 0

    for issue in tier1_issues:
        seg_id = issue.source_segment
        if not seg_id or seg_id not in architectures:
            continue

        current_text = patched_archs.get(seg_id, architectures[seg_id])
        patched = _apply_tier1_fix(issue, current_text)

        if patched and patched != current_text:
            patched_archs[seg_id] = patched
            fix_log.setdefault(seg_id, []).append(issue.auto_fix_note)
            issue.auto_fixed = True
            tier1_fixed += 1
            logger.info(
                "[cohesion_auto_fix] Tier 1 FIX: %s in %s — %s",
                issue.issue_id, seg_id, issue.auto_fix_note,
            )
        else:
            logger.warning(
                "[cohesion_auto_fix] Tier 1 SKIP: %s in %s — pattern not found in arch text",
                issue.issue_id, seg_id,
            )
            # v5.20: When Tier 1 fails for missing_import, cascade to Tier 2
            # instead of immediately escalating to blocking (which triggers
            # expensive Opus regen). Tier 2 can often patch it for ~500 tokens.
            if issue.category == "missing_import" and issue.severity == "warning":
                if issue not in tier2_issues:
                    issue.auto_fix_tier = 2
                    tier2_issues.append(issue)
                    logger.info(
                        "[cohesion_auto_fix] Tier 1→2 CASCADE: %s — will attempt micro-LLM fix",
                        issue.issue_id,
                    )

    print(f"[cohesion_auto_fix] Tier 1: {tier1_fixed}/{len(tier1_issues)} fixes applied")

    # =========================================================================
    # Step 3: Apply Tier 2 fixes (micro-LLM, per-segment)
    # =========================================================================
    tier2_fixed = 0
    tier2_calls = 0

    for issue in tier2_issues:
        if tier2_calls >= max_tier2_fixes:
            logger.info(
                "[cohesion_auto_fix] Tier 2: reached max calls (%d), skipping rest",
                max_tier2_fixes,
            )
            break

        seg_id = issue.source_segment
        if not seg_id or seg_id not in architectures:
            continue

        current_text = patched_archs.get(seg_id, architectures[seg_id])
        tier2_calls += 1

        patched = await _apply_tier2_fix(
            issue=issue,
            arch_text=current_text,
            seg_id=seg_id,
            provider=tier2_provider,
            model=tier2_model,
        )

        if patched and patched != current_text:
            patched_archs[seg_id] = patched
            fix_log.setdefault(seg_id, []).append(issue.auto_fix_note)
            issue.auto_fixed = True
            tier2_fixed += 1
            logger.info(
                "[cohesion_auto_fix] Tier 2 FIX: %s in %s — %s",
                issue.issue_id, seg_id, issue.auto_fix_note,
            )
        else:
            logger.warning(
                "[cohesion_auto_fix] Tier 2 SKIP: %s in %s — LLM fix failed",
                issue.issue_id, seg_id,
            )

    print(
        f"[cohesion_auto_fix] Tier 2: {tier2_fixed}/{len(tier2_issues)} fixes applied "
        f"({tier2_calls} LLM call(s))"
    )

    # v5.20: Escalate any missing_import issues that BOTH Tier 1 and Tier 2 failed to fix.
    # These are guaranteed NameErrors at runtime and must trigger regen.
    for issue in tier2_issues:
        if not issue.auto_fixed and issue.category == "missing_import" and issue.severity == "warning":
            issue.severity = "blocking"
            issue.auto_fix_note = (
                "Tier 1+2 fix FAILED \u2014 escalated to blocking. "
                "Missing import will cause NameError at runtime."
            )
            logger.warning(
                "[cohesion_auto_fix] \u26a0\ufe0f ESCALATED %s to blocking \u2014 Tier 1+2 both failed",
                issue.issue_id,
            )

    # =========================================================================
    # Step 4: Save patched architectures to disk
    # =========================================================================
    if not patched_archs:
        logger.info("[cohesion_auto_fix] No patches applied — skipping save")
        return result

    for seg_id, patched_text in patched_archs.items():
        notes = fix_log.get(seg_id, [])
        saved_path = _save_patched_architecture(job_dir, seg_id, patched_text, notes)
        print(f"[cohesion_auto_fix] 💾 Saved: {saved_path}")

    # =========================================================================
    # Step 5: Re-validate with skeleton compliance
    # =========================================================================
    print("[cohesion_auto_fix] 🔍 Re-validating after auto-fixes...")

    # Reload architectures (now includes patched versions)
    segment_ids = list(architectures.keys())
    reloaded_archs = load_segment_architectures(job_dir, segment_ids)

    recheck_issues = run_skeleton_compliance(
        architectures=reloaded_archs,
        skeleton_json=skeleton_json,
        manifest_dict=manifest_dict,
    )

    # =========================================================================
    # Step 6: Build updated result
    # =========================================================================
    # Keep unfixed issues + any NEW issues from re-validation
    # Remove issues that were fixed and no longer appear in re-check
    recheck_ids = {(i.category, i.source_segment, i.description[:80]) for i in recheck_issues}

    updated_issues = []

    # Add fixed issues as resolved (downgraded to info)
    for issue in result.issues:
        if issue.auto_fixed:
            # Check if it still appears in re-validation
            key = (issue.category, issue.source_segment, issue.description[:80])
            if key in recheck_ids:
                # Fix didn't work — keep as original severity
                issue.auto_fixed = False
                issue.auto_fix_note += " (FIX FAILED — issue persists)"
                updated_issues.append(issue)
                logger.warning(
                    "[cohesion_auto_fix] Fix FAILED for %s — issue persists after patch",
                    issue.issue_id,
                )
            else:
                # Fix worked — downgrade to resolved
                issue.severity = "resolved"
                updated_issues.append(issue)
                logger.info(
                    "[cohesion_auto_fix] ✅ Fix CONFIRMED for %s",
                    issue.issue_id,
                )
        else:
            # Unfixed issue — check if resolved as side-effect of another fix
            key = (issue.category, issue.source_segment, issue.description[:80])
            if key not in recheck_ids:
                # Side-effect fix! Issue no longer appears in re-validation
                issue.severity = "resolved"
                issue.auto_fixed = True
                issue.auto_fix_note = "Resolved as side-effect of related fix"
                updated_issues.append(issue)
                logger.info(
                    "[cohesion_auto_fix] ✅ Side-effect fix for %s — resolved by related patch",
                    issue.issue_id,
                )

                # v3.2: Annotate affected arch text so implementer knows about the change
                _affected_seg = issue.related_segment or issue.source_segment
                if _affected_seg and _affected_seg in patched_archs:
                    _annotation = (
                        f"\n\n<!-- COHESION ANNOTATION ({issue.issue_id}): "
                        f"{issue.description[:200]} "
                        f"Fix: {issue.suggested_fix[:200]} "
                        f"(resolved as side-effect of related fix) -->\n"
                    )
                    patched_archs[_affected_seg] += _annotation
                    fix_log.setdefault(_affected_seg, []).append(
                        f"Side-effect annotation for {issue.issue_id}"
                    )
                    logger.info(
                        "[cohesion_auto_fix] 📝 Annotated %s arch for side-effect fix %s",
                        _affected_seg, issue.issue_id,
                    )
                elif _affected_seg and _affected_seg in architectures:
                    # Segment wasn't patched yet — start a new patch
                    _annotation = (
                        f"\n\n<!-- COHESION ANNOTATION ({issue.issue_id}): "
                        f"{issue.description[:200]} "
                        f"Fix: {issue.suggested_fix[:200]} "
                        f"(resolved as side-effect of related fix) -->\n"
                    )
                    patched_archs[_affected_seg] = architectures[_affected_seg] + _annotation
                    fix_log.setdefault(_affected_seg, []).append(
                        f"Side-effect annotation for {issue.issue_id}"
                    )
                    logger.info(
                        "[cohesion_auto_fix] 📝 Annotated %s arch for side-effect fix %s (new patch)",
                        _affected_seg, issue.issue_id,
                    )
            else:
                updated_issues.append(issue)

    # Add any NEW issues from re-validation that weren't in original
    original_ids = {(i.category, i.source_segment, i.description[:80]) for i in result.issues}
    for new_issue in recheck_issues:
        key = (new_issue.category, new_issue.source_segment, new_issue.description[:80])
        if key not in original_ids:
            new_issue.auto_fix_note = "NEW: appeared after auto-fix patching"
            updated_issues.append(new_issue)
            logger.warning(
                "[cohesion_auto_fix] NEW issue after patching: %s",
                new_issue.issue_id,
            )

    result.issues = updated_issues

    # Recalculate status
    remaining_blocking = [
        i for i in result.issues
        if i.severity == "blocking"
    ]
    result.status = "fail" if remaining_blocking else "pass"

    total_fixed = tier1_fixed + tier2_fixed
    remaining = len(remaining_blocking)
    result.notes = (
        f"Auto-fix: {total_fixed} fixed "
        f"(T1:{tier1_fixed}, T2:{tier2_fixed}), "
        f"{remaining} blocking remain, "
        f"{len(tier3_issues)} deferred to regen"
    )

    print(
        f"[cohesion_auto_fix] ═══ RESULT: {total_fixed} fixed, "
        f"{remaining} blocking remain ═══"
    )

    return result


# =============================================================================
# PERSISTENCE
# =============================================================================

def save_cohesion_result(result: CohesionResult, job_dir: str) -> str:
    """Save cohesion result to disk alongside the manifest."""
    segments_dir = os.path.join(job_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)
    path = os.path.join(segments_dir, "cohesion_check.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("[cohesion_check] Saved: %s", path)
    return path


def load_cohesion_result(job_dir: str) -> Optional[CohesionResult]:
    """Load cohesion result from disk. Returns None if not found."""
    path = os.path.join(job_dir, "segments", "cohesion_check.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return CohesionResult.from_dict(json.load(f))
    except Exception as e:
        logger.warning("[cohesion_check] Failed to load: %s", e)
        return None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "CohesionIssue",
    "CohesionResult",
    "run_skeleton_compliance",
    "run_cohesion_check",
    "attempt_auto_fixes",
    "load_segment_architectures",
    "save_cohesion_result",
    "load_cohesion_result",
    "COHESION_CHECK_BUILD_ID",
]
