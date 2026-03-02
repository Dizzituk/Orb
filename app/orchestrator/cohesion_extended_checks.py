# FILE: app/orchestrator/cohesion_extended_checks.py
"""
Extended Deterministic Cohesion Checks — Layer 1 Extensions.

Five new cross-segment checks that replace the LLM Layer 2:

1. Cross-segment import path validation
   Verify import paths resolve to correct files in providing architecture.

2. Type signature consistency
   Match function signatures across producer/consumer boundaries.

3. Interface completeness
   Every skeleton `exposes` binding has implementation section in
   producing segment's architecture.

4. Circular dependency detection
   Build directed graph from manifest, verify acyclic.

5. Shared state audit
   Flag segments reading/writing files owned by others without
   declared binding.

Zero LLM calls. Pure structural analysis.

v1.0 (2026-02-27): Initial implementation — Stage 2 of deterministic
verification migration.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

COHESION_EXTENDED_BUILD_ID = "2026-02-27-v1.0-extended-cohesion-checks"


# =========================================================================
# SHARED HELPERS
# =========================================================================

def _extract_imports_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract import statements from architecture text."""
    imports: List[Dict[str, Any]] = []
    for m in re.finditer(r'from\s+\.(\w+)\s+import\s+([^(\n]+)', text):
        module = m.group(1).strip()
        names_str = m.group(2).strip().rstrip("\\").strip("`")
        names = [
            n.strip().strip("`").split(" as ")[0].strip()
            for n in names_str.split(",") if n.strip()
        ]
        names = [n for n in names if re.match(r'^[a-zA-Z_]\w*$', n)]
        if module and names:
            imports.append({"module": module, "names": names, "raw": m.group(0)})
    return imports


def _extract_function_sigs(text: str) -> Dict[str, Dict[str, Any]]:
    """Extract function signatures from text."""
    sigs: Dict[str, Dict[str, Any]] = {}
    for m in re.finditer(
        r'(async\s+)?def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*(\S+))?\s*:',
        text,
    ):
        name = m.group(2)
        params_str = m.group(3).strip()
        params = []
        if params_str:
            for p in params_str.split(","):
                p = p.strip()
                if p and p not in ("self", "cls"):
                    has_default = "=" in p
                    pname = p.split("=")[0].split(":")[0].strip()
                    if pname and not pname.startswith("*"):
                        params.append({"name": pname, "has_default": has_default})
        sigs[name] = {
            "params": params,
            "return_type": m.group(4).strip() if m.group(4) else None,
            "is_async": bool(m.group(1)),
        }
    return sigs


# =========================================================================
# CHECK 1: Cross-Segment Import Path Validation
# =========================================================================

def check_import_path_resolution(
    architectures: Dict[str, str],
    manifest_dict: Dict[str, Any],
    issue_counter: int = 0,
) -> Tuple[List[Any], int]:
    """
    Verify that cross-segment import paths resolve to files that
    actually exist in the providing segment's architecture.

    For each 'from .module import X' in segment A's architecture,
    check that:
    - 'module.py' exists in some segment B's file_scope
    - segment B's architecture has a File Inventory mentioning that file
    """
    from app.orchestrator._cohesion_check_utils_10 import CohesionIssue
    issues: List[CohesionIssue] = []

    # Build module -> (segment_id, file_path) map
    module_to_info: Dict[str, Tuple[str, str]] = {}
    seg_own_modules: Dict[str, Set[str]] = {}

    for seg in manifest_dict.get("segments", []):
        sid = seg.get("segment_id", "")
        own: Set[str] = set()
        for fp in seg.get("file_scope", []):
            basename = fp.replace("\\", "/").rsplit("/", 1)[-1].replace(".py", "").lower()
            module_to_info[basename] = (sid, fp)
            own.add(basename)
        seg_own_modules[sid] = own

    # Extract File Inventory from each architecture
    arch_inventories: Dict[str, Set[str]] = {}
    for sid, arch in architectures.items():
        inv_match = re.search(r'(?:^|\n)#+\s*File Inventory', arch)
        if inv_match:
            inv_start = inv_match.start()
            inv_end = re.search(r'\n(?:##[^#]|---)', arch[inv_start + 20:])
            section = arch[inv_start:inv_start + 20 + inv_end.start()] if inv_end else arch[inv_start:inv_start + 3000]
            files: Set[str] = set()
            for line in section.split("\n"):
                m = re.search(r'`([\w/\\._-]+\.py)`', line)
                if m:
                    files.add(m.group(1).replace("\\", "/").rsplit("/", 1)[-1].replace(".py", "").lower())
            arch_inventories[sid] = files
        else:
            arch_inventories[sid] = set()

    for seg_id, arch_content in architectures.items():
        own_modules = seg_own_modules.get(seg_id, set())
        arch_imports = _extract_imports_from_text(arch_content)

        for imp in arch_imports:
            target = imp["module"].lower()
            if target in own_modules:
                continue

            info = module_to_info.get(target)
            if not info:
                continue

            provider_seg, provider_path = info
            if provider_seg == seg_id:
                continue

            # Check if providing segment's architecture mentions this file
            provider_inv = arch_inventories.get(provider_seg, set())
            if target not in provider_inv:
                issue_counter += 1
                issues.append(CohesionIssue(
                    issue_id=f"COH-EXT-{issue_counter:03d}",
                    severity="warning",
                    category="import_path_unresolved",
                    description=(
                        f"{seg_id} imports from '.{imp['module']}' "
                        f"({provider_path} owned by {provider_seg}) but "
                        f"{provider_seg}'s architecture doesn't list this "
                        f"file in its File Inventory."
                    ),
                    source_segment=seg_id,
                    related_segment=provider_seg,
                    file_path=provider_path,
                    suggested_fix=f"Add '{provider_path}' to {provider_seg}'s File Inventory",
                ))

    return issues, issue_counter


# =========================================================================
# CHECK 2: Type Signature Consistency
# =========================================================================

def check_signature_consistency(
    architectures: Dict[str, str],
    manifest_dict: Dict[str, Any],
    issue_counter: int = 0,
) -> Tuple[List[Any], int]:
    """
    Match function signatures across producer/consumer boundaries.

    When segment A imports function 'foo' from segment B's module,
    check that both architectures define 'foo' with compatible
    parameter counts and required param names.
    """
    from app.orchestrator._cohesion_check_utils_10 import CohesionIssue
    issues: List[CohesionIssue] = []

    # Build: segment_id -> {func_name: signature_info}
    seg_sigs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for sid, arch in architectures.items():
        seg_sigs[sid] = _extract_function_sigs(arch)

    # Build module -> segment map
    module_to_seg: Dict[str, str] = {}
    seg_own_modules: Dict[str, Set[str]] = {}
    for seg in manifest_dict.get("segments", []):
        sid = seg.get("segment_id", "")
        own: Set[str] = set()
        for fp in seg.get("file_scope", []):
            basename = fp.replace("\\", "/").rsplit("/", 1)[-1].replace(".py", "").lower()
            module_to_seg[basename] = sid
            own.add(basename)
        seg_own_modules[sid] = own

    for seg_id, arch_content in architectures.items():
        own_modules = seg_own_modules.get(seg_id, set())
        arch_imports = _extract_imports_from_text(arch_content)

        for imp in arch_imports:
            target = imp["module"].lower()
            if target in own_modules:
                continue
            provider_seg = module_to_seg.get(target)
            if not provider_seg or provider_seg == seg_id:
                continue

            provider_sigs = seg_sigs.get(provider_seg, {})
            consumer_sigs = seg_sigs.get(seg_id, {})

            for name in imp["names"]:
                if name not in provider_sigs:
                    continue  # Can't verify if provider doesn't define it
                if name not in consumer_sigs:
                    continue  # Consumer doesn't redefine — just imports

                # Both define the same function — compare
                p_sig = provider_sigs[name]
                c_sig = consumer_sigs[name]

                p_req = [p["name"] for p in p_sig["params"] if not p.get("has_default")]
                c_req = [p["name"] for p in c_sig["params"] if not p.get("has_default")]

                if len(p_req) != len(c_req):
                    issue_counter += 1
                    issues.append(CohesionIssue(
                        issue_id=f"COH-EXT-{issue_counter:03d}",
                        severity="warning",
                        category="signature_mismatch",
                        description=(
                            f"Function '{name}' has {len(p_req)} required params "
                            f"in {provider_seg} ({', '.join(p_req)}) but "
                            f"{len(c_req)} in {seg_id} ({', '.join(c_req)})"
                        ),
                        source_segment=seg_id,
                        related_segment=provider_seg,
                        suggested_fix="Align function signatures between segments",
                    ))

                if p_sig.get("is_async") != c_sig.get("is_async"):
                    issue_counter += 1
                    issues.append(CohesionIssue(
                        issue_id=f"COH-EXT-{issue_counter:03d}",
                        severity="blocking",
                        category="async_mismatch",
                        description=(
                            f"Function '{name}' is "
                            f"{'async' if p_sig.get('is_async') else 'sync'} in "
                            f"{provider_seg} but "
                            f"{'async' if c_sig.get('is_async') else 'sync'} in "
                            f"{seg_id}"
                        ),
                        source_segment=seg_id,
                        related_segment=provider_seg,
                        suggested_fix="Ensure async/sync consistency",
                    ))

    return issues, issue_counter


# =========================================================================
# CHECK 3: Interface Completeness
# =========================================================================

def check_interface_completeness(
    architectures: Dict[str, str],
    skeleton_json: Optional[str] = None,
    issue_counter: int = 0,
) -> Tuple[List[Any], int]:
    """
    Every skeleton `exposes` binding with named exports has matching
    function/class definitions in the producing segment's architecture.
    """
    from app.orchestrator._cohesion_check_utils_10 import CohesionIssue
    issues: List[CohesionIssue] = []

    if not skeleton_json:
        return issues, issue_counter

    try:
        skeleton = json.loads(skeleton_json)
    except (json.JSONDecodeError, TypeError):
        return issues, issue_counter

    for skel in skeleton.get("skeletons", []):
        seg_id = skel.get("segment_id", "")
        arch = architectures.get(seg_id)
        if not arch:
            continue

        # Get all defined symbols in this architecture
        defined = set()
        for m in re.finditer(r'(?:async\s+)?def\s+(\w+)\s*\(', arch):
            defined.add(m.group(1))
        for m in re.finditer(r'class\s+(\w+)\s*[(:[\])]', arch):
            defined.add(m.group(1))
        for m in re.finditer(r'^([A-Z][A-Z0-9_]+)\s*=', arch, re.MULTILINE):
            defined.add(m.group(1))
        # Also check prose mentions
        for m in re.finditer(r'`(\w+)\s*\(', arch):
            defined.add(m.group(1))

        for export in skel.get("exports", []):
            names = export.get("names", [])
            if not isinstance(names, list):
                continue
            consumed_by = export.get("consumed_by", [])
            for name in names:
                if name and name not in defined:
                    issue_counter += 1
                    issues.append(CohesionIssue(
                        issue_id=f"COH-EXT-{issue_counter:03d}",
                        severity="warning",
                        category="incomplete_interface",
                        description=(
                            f"{seg_id} must export '{name}' (consumed by "
                            f"{', '.join(consumed_by)}) but no definition "
                            f"found in its architecture."
                        ),
                        source_segment=seg_id,
                        related_segment=consumed_by[0] if consumed_by else "",
                        suggested_fix=f"Add definition of '{name}' to {seg_id}'s architecture",
                    ))

    return issues, issue_counter


# =========================================================================
# CHECK 4: Circular Dependency Detection
# =========================================================================

def check_circular_dependencies(
    manifest_dict: Dict[str, Any],
    issue_counter: int = 0,
) -> Tuple[List[Any], int]:
    """
    Build directed graph from manifest dependencies, verify acyclic.
    Uses DFS cycle detection.
    """
    from app.orchestrator._cohesion_check_utils_10 import CohesionIssue
    issues: List[CohesionIssue] = []

    segments = manifest_dict.get("segments", [])
    if not segments:
        return issues, issue_counter

    # Build adjacency list
    graph: Dict[str, List[str]] = {}
    for seg in segments:
        sid = seg.get("segment_id", "")
        graph[sid] = seg.get("dependencies", [])

    # DFS cycle detection
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {sid: WHITE for sid in graph}
    cycles: List[List[str]] = []

    def _dfs(node: str, path: List[str]) -> None:
        color[node] = GRAY
        path.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in color:
                continue  # Unknown segment reference
            if color[neighbor] == GRAY:
                # Found cycle — extract it
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycles.append(cycle)
            elif color[neighbor] == WHITE:
                _dfs(neighbor, path)
        path.pop()
        color[node] = BLACK

    for sid in graph:
        if color[sid] == WHITE:
            _dfs(sid, [])

    for cycle in cycles:
        cycle_str = " → ".join(cycle)
        issue_counter += 1
        issues.append(CohesionIssue(
            issue_id=f"COH-EXT-{issue_counter:03d}",
            severity="blocking",
            category="circular_dependency",
            description=f"Circular dependency detected: {cycle_str}",
            source_segment=cycle[0],
            related_segment=cycle[1] if len(cycle) > 1 else "",
            suggested_fix="Break the cycle by removing one dependency edge",
        ))

    return issues, issue_counter


# =========================================================================
# CHECK 5: Shared State Audit
# =========================================================================

def check_shared_state(
    architectures: Dict[str, str],
    manifest_dict: Dict[str, Any],
    issue_counter: int = 0,
) -> Tuple[List[Any], int]:
    """
    Flag segments that read/write files owned by other segments
    without a declared binding.

    Looks for patterns like:
    - open("path/to/file.json", "w")
    - Path("path/to/file").write_text(...)
    - with open(...) as f: json.dump(...)
    """
    from app.orchestrator._cohesion_check_utils_10 import CohesionIssue
    issues: List[CohesionIssue] = []

    # Build file -> owning segment map
    file_owners: Dict[str, str] = {}
    for seg in manifest_dict.get("segments", []):
        sid = seg.get("segment_id", "")
        for fp in seg.get("file_scope", []):
            file_owners[fp.replace("\\", "/").lower()] = sid

    # Declared dependencies
    seg_deps: Dict[str, Set[str]] = {}
    for seg in manifest_dict.get("segments", []):
        sid = seg.get("segment_id", "")
        seg_deps[sid] = set(seg.get("dependencies", []))

    # Patterns indicating file I/O on specific paths
    io_patterns = [
        re.compile(r'open\s*\(\s*["\']([^"\']+\.(?:json|yaml|yml|py|txt))["\']'),
        re.compile(r'Path\s*\(\s*["\']([^"\']+\.(?:json|yaml|yml|py|txt))["\']'),
        re.compile(r'read_text\s*\(\s*["\']([^"\']+\.(?:json|yaml|yml|py|txt))["\']'),
        re.compile(r'write_text\s*\(\s*["\']([^"\']+\.(?:json|yaml|yml|py|txt))["\']'),
    ]

    for seg_id, arch_content in architectures.items():
        deps = seg_deps.get(seg_id, set())

        for pattern in io_patterns:
            for m in pattern.finditer(arch_content):
                path = m.group(1).replace("\\", "/").lower()
                owner = file_owners.get(path)

                if owner and owner != seg_id and owner not in deps:
                    issue_counter += 1
                    issues.append(CohesionIssue(
                        issue_id=f"COH-EXT-{issue_counter:03d}",
                        severity="warning",
                        category="shared_state",
                        description=(
                            f"{seg_id} accesses file '{path}' which is "
                            f"owned by {owner}, without declaring {owner} "
                            f"as a dependency."
                        ),
                        source_segment=seg_id,
                        related_segment=owner,
                        file_path=path,
                        suggested_fix=(
                            f"Add {owner} as a dependency, or use an "
                            f"interface to access this data."
                        ),
                    ))

    return issues, issue_counter


# =========================================================================
# ORCHESTRATOR — Run all five checks
# =========================================================================

def run_extended_cohesion_checks(
    architectures: Dict[str, str],
    manifest_dict: Optional[Dict[str, Any]] = None,
    skeleton_json: Optional[str] = None,
    issue_counter: int = 0,
) -> Tuple[List[Any], int]:
    """
    Run all five extended cohesion checks.

    Returns (issues_list, updated_issue_counter).
    """
    all_issues: List = []

    if not manifest_dict:
        return all_issues, issue_counter

    # Check 1: Import path resolution
    try:
        new_issues, issue_counter = check_import_path_resolution(
            architectures, manifest_dict, issue_counter,
        )
        all_issues.extend(new_issues)
    except Exception as e:
        logger.warning("[cohesion_ext] Check 1 (import paths) failed: %s", e)

    # Check 2: Signature consistency
    try:
        new_issues, issue_counter = check_signature_consistency(
            architectures, manifest_dict, issue_counter,
        )
        all_issues.extend(new_issues)
    except Exception as e:
        logger.warning("[cohesion_ext] Check 2 (signatures) failed: %s", e)

    # Check 3: Interface completeness
    try:
        new_issues, issue_counter = check_interface_completeness(
            architectures, skeleton_json, issue_counter,
        )
        all_issues.extend(new_issues)
    except Exception as e:
        logger.warning("[cohesion_ext] Check 3 (interface) failed: %s", e)

    # Check 4: Circular dependencies
    try:
        new_issues, issue_counter = check_circular_dependencies(
            manifest_dict, issue_counter,
        )
        all_issues.extend(new_issues)
    except Exception as e:
        logger.warning("[cohesion_ext] Check 4 (circular deps) failed: %s", e)

    # Check 5: Shared state audit
    if len(architectures) > 1:
        try:
            new_issues, issue_counter = check_shared_state(
                architectures, manifest_dict, issue_counter,
            )
            all_issues.extend(new_issues)
        except Exception as e:
            logger.warning("[cohesion_ext] Check 5 (shared state) failed: %s", e)

    if all_issues:
        blocking = sum(1 for i in all_issues if getattr(i, 'severity', '') == 'blocking')
        logger.info(
            "[cohesion_ext] Extended checks: %d issues (%d blocking)",
            len(all_issues), blocking,
        )

    return all_issues, issue_counter


__all__ = [
    "check_import_path_resolution",
    "check_signature_consistency",
    "check_interface_completeness",
    "check_circular_dependencies",
    "check_shared_state",
    "run_extended_cohesion_checks",
    "COHESION_EXTENDED_BUILD_ID",
]
