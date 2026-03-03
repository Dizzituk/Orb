# FILE: app/orchestrator/cross_segment_interfaces.py
"""
Cross-Segment Interface Binding — v1.0

Fixes three interface-consistency problems in the segment pipeline:

Fix 1 (Sibling Architecture Injection):
    When generating architecture for seg-02, read seg-01's already-approved
    architecture and extract its export names + prop interfaces. Inject
    these as a deterministic section in seg-02's template so the LLM
    knows the exact interface to target.

Fix 2 (Cohesion Interface Validator):
    After all architectures are approved, do a deterministic cross-check:
    compare what each segment exports (from its code blocks) with what
    downstream segments import. Flag mismatches as blocking issues.

Fix 3 (Skeleton Back-Propagation):
    After a segment's architecture is approved, parse its code structure
    blocks and write the extracted export names back into the skeleton
    contracts JSON. This means later segments' Import Maps show real
    symbol names instead of "(exported symbols to be specified by architecture)".

2026-03-01: Initial implementation.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =========================================================================
# CODE BLOCK PARSER — shared by all three fixes
# =========================================================================

# Match ```typescript / ```tsx / ```ts / ```css code fences
_CODE_FENCE_RE = re.compile(
    r"```(?:typescript|tsx|ts|css|python|py)\s*\n(.*?)```",
    re.DOTALL,
)

# Match export statements in TypeScript/TSX
_TS_EXPORT_RE = re.compile(
    r"^export\s+(?:"
    r"(?:function|const|let|class|enum|type|interface)\s+(\w+)"  # named export
    r"|(?:default\s+(?:function|class)\s+(\w+))"                # default export
    r")",
    re.MULTILINE,
)

# Match React component prop interfaces
_PROPS_RE = re.compile(
    r"interface\s+(\w+Props)\s*\{([^}]+)\}",
    re.DOTALL,
)

# Match function signature with typed params
_FUNC_SIG_RE = re.compile(
    r"(?:export\s+)?function\s+(\w+)\s*\(([^)]*)\)",
)


def extract_exports_from_arch(arch_text: str) -> Dict[str, Dict[str, Any]]:
    """Parse architecture markdown and extract exports per file.

    Returns:
        {
            "src/components/education/educationData.ts": {
                "exports": ["Course", "Lesson", "mockCourses"],
                "interfaces": {"Course": "id: string; title: string; ..."},
                "props": {},
            },
            "src/components/education/EducationList.tsx": {
                "exports": ["EducationList"],
                "interfaces": {},
                "props": {"EducationListProps": "courses: Course[]; selectedCourseId: string | null; ..."},
            },
        }
    """
    result: Dict[str, Dict[str, Any]] = {}

    # Split architecture into per-file sections
    # v1.1 (2026-03-01): Handle both "### File: `path`" and "### `path`" formats.
    # The template engine generates the File: prefix but the LLM often drops it.
    file_sections = re.split(r"###\s+(?:File:\s*)?`([^`]+)`", arch_text)

    # file_sections[0] is the preamble, then alternating (path, content) pairs
    for i in range(1, len(file_sections) - 1, 2):
        file_path = file_sections[i].strip()
        section_content = file_sections[i + 1]

        # Normalise path — strip orb-desktop/ prefix for matching
        norm_path = file_path.replace("\\", "/")
        if norm_path.startswith("orb-desktop/"):
            norm_path = norm_path[len("orb-desktop/"):]

        file_info: Dict[str, Any] = {
            "exports": [],
            "interfaces": {},
            "props": {},
        }

        # Extract from code blocks within this section
        for code_match in _CODE_FENCE_RE.finditer(section_content):
            code = code_match.group(1)

            # Extract export names
            for exp_match in _TS_EXPORT_RE.finditer(code):
                name = exp_match.group(1) or exp_match.group(2)
                if name and name not in file_info["exports"]:
                    file_info["exports"].append(name)

            # Extract interface definitions (especially Props)
            for iface_match in _PROPS_RE.finditer(code):
                iface_name = iface_match.group(1)
                iface_body = iface_match.group(2).strip()
                if iface_name.endswith("Props"):
                    file_info["props"][iface_name] = iface_body
                else:
                    file_info["interfaces"][iface_name] = iface_body

            # Also extract non-Props interfaces
            for iface_match in re.finditer(
                r"(?:export\s+)?interface\s+(\w+)\s*\{([^}]+)\}",
                code, re.DOTALL,
            ):
                iface_name = iface_match.group(1)
                iface_body = iface_match.group(2).strip()
                if iface_name not in file_info["props"]:
                    file_info["interfaces"][iface_name] = iface_body

        if file_info["exports"] or file_info["props"]:
            result[norm_path] = file_info

    return result


# =========================================================================
# FIX 1: Sibling Architecture Injection
# =========================================================================

def build_sibling_interface_section(
    segment_id: str,
    dependencies: List[str],
    job_dir: str,
) -> str:
    """Build a deterministic section showing upstream segments' interfaces.

    Reads approved architecture files from dependency segments and extracts
    their exports, interfaces, and prop types. Returns a markdown section
    to inject into the architecture template.

    Args:
        segment_id: Current segment being built.
        dependencies: List of upstream segment IDs.
        job_dir: Path to the job directory (contains segments/).

    Returns:
        Markdown string with interface contracts, or empty string if none found.
    """
    if not dependencies:
        return ""

    parts: List[str] = []
    parts.append("## Upstream Interface Contracts [DETERMINISTIC]\n")
    parts.append(
        "The following interfaces are FIXED by upstream segment architectures. "
        "Your code MUST match these exactly — do not rename exports, "
        "change prop types, or alter function signatures.\n"
    )

    found_any = False

    for dep_id in dependencies:
        dep_arch = _load_segment_arch(dep_id, job_dir)
        if not dep_arch:
            continue

        exports = extract_exports_from_arch(dep_arch)
        if not exports:
            continue

        found_any = True
        parts.append(f"### From `{dep_id}`:\n")

        for file_path, file_info in exports.items():
            export_names = file_info.get("exports", [])
            props = file_info.get("props", {})
            interfaces = file_info.get("interfaces", {})

            if export_names:
                parts.append(
                    f"**`{file_path}`** exports: `{', '.join(export_names)}`"
                )

            for iface_name, iface_body in interfaces.items():
                parts.append(f"\n```typescript\ninterface {iface_name} {{\n{iface_body}\n}}\n```")

            for props_name, props_body in props.items():
                parts.append(f"\n```typescript\ninterface {props_name} {{\n{props_body}\n}}\n```")

            parts.append("")

    if not found_any:
        return ""

    return "\n".join(parts)


def _load_segment_arch(segment_id: str, job_dir: str) -> Optional[str]:
    """Load the latest architecture markdown for a segment."""
    # Try segments/{seg_id}/arch/arch_v1.md first (primary location)
    for version in ("arch_v3.md", "arch_v2.md", "arch_v1.md"):
        path = os.path.join(job_dir, "segments", segment_id, "arch", version)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.warning(
                    "[cross_seg] Failed to read arch for %s: %s",
                    segment_id, e,
                )
    return None


# =========================================================================
# FIX 2: Cohesion Interface Validator
# =========================================================================

def validate_cross_segment_interfaces(
    job_dir: str,
    manifest_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Deterministic cross-segment interface validation.

    For each segment pair (producer → consumer), extracts the producer's
    exports and the consumer's imports and checks they match.

    Returns a list of issue dicts compatible with cohesion_check.json format.
    """
    issues: List[Dict[str, Any]] = []

    # Build dependency map: consumer_id -> [producer_ids]
    dep_map: Dict[str, List[str]] = {}
    for seg in manifest_segments:
        sid = seg.get("segment_id", "")
        deps = seg.get("dependencies", [])
        if deps:
            dep_map[sid] = deps

    # Load all architectures and extract exports
    all_exports: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for seg in manifest_segments:
        sid = seg.get("segment_id", "")
        arch = _load_segment_arch(sid, job_dir)
        if arch:
            all_exports[sid] = extract_exports_from_arch(arch)

    # For each consumer, check its imports match producer exports
    for consumer_id, producer_ids in dep_map.items():
        consumer_arch = _load_segment_arch(consumer_id, job_dir)
        if not consumer_arch:
            continue

        consumer_imports = _extract_imports_from_arch(consumer_arch)

        for producer_id in producer_ids:
            producer_exports = all_exports.get(producer_id, {})

            for import_path, import_names in consumer_imports.items():
                # Find matching producer file
                producer_file = _find_matching_export_file(
                    import_path, producer_exports,
                )
                if producer_file is None:
                    continue

                producer_names = set(
                    producer_exports[producer_file].get("exports", [])
                )

                for imp_name in import_names:
                    if imp_name not in producer_names:
                        issues.append({
                            "issue_id": f"IFACE-{len(issues)+1:03d}",
                            "severity": "blocking",
                            "category": "interface_mismatch",
                            "description": (
                                f"Consumer '{consumer_id}' imports '{imp_name}' "
                                f"from '{import_path}' but producer "
                                f"'{producer_id}' exports: "
                                f"{sorted(producer_names) if producer_names else '(none detected)'}"
                            ),
                            "source_segment": producer_id,
                            "related_segment": consumer_id,
                            "file_path": import_path,
                            "expected": imp_name,
                            "actual": ", ".join(sorted(producer_names)),
                            "suggested_fix": (
                                f"Either rename export in {producer_id} to "
                                f"'{imp_name}' or update import in {consumer_id}"
                            ),
                            "auto_fix_tier": 1,
                            "auto_fixed": False,
                            "auto_fix_note": "",
                        })

        # Check prop interface compatibility
        _check_prop_compatibility(
            consumer_id, producer_ids, consumer_arch,
            all_exports, issues,
        )

    return issues


def _extract_imports_from_arch(arch_text: str) -> Dict[str, List[str]]:
    """Extract import statements from architecture code blocks.

    Returns: {relative_path: [imported_names]}
    """
    imports: Dict[str, List[str]] = {}

    for code_match in _CODE_FENCE_RE.finditer(arch_text):
        code = code_match.group(1)

        # Match: import { X, Y } from './path'
        for m in re.finditer(
            r"import\s+(?:type\s+)?\{\s*([^}]+)\}\s+from\s+['\"]([^'\"]+)['\"]",
            code,
        ):
            names_str = m.group(1)
            path = m.group(2)
            names = [n.strip().split(" as ")[0].strip()
                     for n in names_str.split(",") if n.strip()]
            imports.setdefault(path, []).extend(names)

        # Match: import X from './path'
        for m in re.finditer(
            r"import\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]",
            code,
        ):
            name = m.group(1)
            path = m.group(2)
            if name not in ("type", "React"):
                imports.setdefault(path, []).append(name)

    return imports


def _find_matching_export_file(
    import_path: str,
    producer_exports: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """Find which producer file matches an import path."""
    # Normalise import path: ./educationData -> educationData
    norm_import = import_path.lstrip("./").replace("\\", "/")

    for export_path in producer_exports:
        norm_export = export_path.replace("\\", "/")
        # Strip extension for matching
        export_stem = re.sub(r"\.(tsx?|jsx?|css)$", "", norm_export)
        import_stem = re.sub(r"\.(tsx?|jsx?|css)$", "", norm_import)

        if export_stem.endswith(import_stem) or import_stem.endswith(
            os.path.basename(export_stem)
        ):
            return export_path

    return None


def _check_prop_compatibility(
    consumer_id: str,
    producer_ids: List[str],
    consumer_arch: str,
    all_exports: Dict[str, Dict[str, Dict[str, Any]]],
    issues: List[Dict[str, Any]],
) -> None:
    """Check that consumer's usage of components matches producer's prop interfaces.

    Looks for cases where consumer passes props that don't match the
    producer's declared interface (e.g. passing selectedCourse when
    the prop is selectedCourseId).
    """
    # Extract JSX usage from consumer's code blocks
    for code_match in _CODE_FENCE_RE.finditer(consumer_arch):
        code = code_match.group(1)

        # Find JSX component usage: <ComponentName prop1={...} prop2={...} />
        for jsx_match in re.finditer(
            r"<(\w+)\s+([^/>]+?)(?:/>|>)",
            code,
        ):
            component = jsx_match.group(1)
            props_str = jsx_match.group(2)

            # Extract prop names from JSX
            jsx_props = set(re.findall(r"(\w+)\s*=", props_str))

            # Find the matching producer component and its Props interface
            for producer_id in producer_ids:
                for file_path, file_info in all_exports.get(producer_id, {}).items():
                    if component in file_info.get("exports", []):
                        # Found the producer — check its props interface
                        # v2.0 FIX: Only match the Props interface that belongs
                        # to the EXPORTED component, not internal sub-components.
                        # e.g. CourseGrid -> CourseGridProps, not CourseCardProps
                        # Fallback: if no exact match exists, check all (old behaviour).
                        _expected_props_name = f"{component}Props"
                        _all_props = file_info.get("props", {})
                        _has_exact_match = _expected_props_name in _all_props
                        for props_name, props_body in _all_props.items():
                            # If we have an exact match, skip non-matching internal Props
                            if _has_exact_match and props_name != _expected_props_name:
                                continue
                            declared_props = set(
                                re.findall(r"(\w+)\s*[?:]", props_body)
                            )
                            if not declared_props:
                                continue

                            # Check for mismatches
                            missing_in_producer = jsx_props - declared_props
                            for missing_prop in missing_in_producer:
                                # Skip common React props
                                if missing_prop in (
                                    "key", "ref", "className", "style",
                                    "children", "onClick", "onChange",
                                ):
                                    continue

                                issues.append({
                                    "issue_id": f"IFACE-{len(issues)+1:03d}",
                                    "severity": "blocking",
                                    "category": "prop_mismatch",
                                    "description": (
                                        f"'{consumer_id}' passes prop "
                                        f"'{missing_prop}' to <{component}/> "
                                        f"but '{producer_id}' declares "
                                        f"{props_name} with: "
                                        f"{sorted(declared_props)}"
                                    ),
                                    "source_segment": producer_id,
                                    "related_segment": consumer_id,
                                    "file_path": file_path,
                                    "expected": missing_prop,
                                    "actual": ", ".join(
                                        sorted(declared_props)
                                    ),
                                    "suggested_fix": (
                                        f"Align prop name: consumer uses "
                                        f"'{missing_prop}', producer "
                                        f"declares {sorted(declared_props)}"
                                    ),
                                    "auto_fix_tier": 1,
                                    "auto_fixed": False,
                                    "auto_fix_note": "",
                                })


# =========================================================================
# FIX 3: Skeleton Back-Propagation
# =========================================================================

def backpropagate_exports_to_skeleton(
    segment_id: str,
    arch_text: str,
    skeleton_path: str,
) -> int:
    """After architecture approval, write exports back to skeleton contracts.

    Parses the approved architecture's code blocks, extracts export names,
    and updates the skeleton contract JSON so that downstream segments'
    Import Maps show real symbol names.

    Args:
        segment_id: The just-approved segment's ID.
        arch_text: The approved architecture markdown.
        skeleton_path: Path to skeleton_contract.json.

    Returns:
        Number of export bindings updated.
    """
    if not os.path.isfile(skeleton_path):
        logger.warning(
            "[cross_seg] Skeleton not found at %s — skipping backprop",
            skeleton_path,
        )
        return 0

    exports = extract_exports_from_arch(arch_text)
    if not exports:
        return 0

    try:
        with open(skeleton_path, "r", encoding="utf-8") as f:
            skeleton_data = json.load(f)
    except Exception as e:
        logger.warning("[cross_seg] Failed to read skeleton: %s", e)
        return 0

    updated = 0
    for skel in skeleton_data.get("skeletons", []):
        if skel.get("segment_id") != segment_id:
            continue

        for export_binding in skel.get("exports", []):
            exp_path = export_binding.get("file_path", "").replace("\\", "/")

            # Find matching parsed exports
            for parsed_path, parsed_info in exports.items():
                norm_parsed = parsed_path.replace("\\", "/")
                if (
                    exp_path.endswith(norm_parsed)
                    or norm_parsed.endswith(exp_path)
                    or os.path.basename(exp_path) == os.path.basename(norm_parsed)
                ):
                    export_names = parsed_info.get("exports", [])
                    if export_names:
                        export_binding["names"] = export_names
                        updated += 1
                        logger.info(
                            "[cross_seg] Backprop: %s/%s -> %s",
                            segment_id, exp_path, export_names,
                        )
                    break

    if updated:
        try:
            with open(skeleton_path, "w", encoding="utf-8") as f:
                json.dump(skeleton_data, f, indent=2, ensure_ascii=False)
            logger.info(
                "[cross_seg] Backprop: updated %d exports in skeleton for %s",
                updated, segment_id,
            )
        except Exception as e:
            logger.warning("[cross_seg] Failed to write skeleton: %s", e)
            return 0

    return updated
