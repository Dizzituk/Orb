"""
Skeleton Contracts — Deterministic Interface Binding for Segments.

v5.6 of Pipeline Evolution.

Generates interface contracts DETERMINISTICALLY from the manifest alone.
Zero LLM calls. Runs between segmentation and architecture generation.

For each segment, the skeleton defines:
  - File scope constraint (ONLY these files may be touched)
  - Export contracts (files that downstream segments depend on)
  - Import contracts (files from upstream segments this segment needs)
  - Cross-segment bindings (the dependency graph edges)

The contract markdown is injected into each segment's Critical Pipeline
prompt as a hard constraint, preventing:
  - Scope creep (touching files outside the segment's scope)
  - Phantom segments (referencing segments that don't exist)
  - Interface drift (inventing alternative imports)

v1.0 (2026-02-12): Initial implementation — deterministic skeleton.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SKELETON_CONTRACTS_BUILD_ID = "2026-02-18-v2.4-do-not-define-prohibition"
print(f"[SKELETON_CONTRACTS_LOADED] BUILD_ID={SKELETON_CONTRACTS_BUILD_ID}")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExportBinding:
    """A file that this segment creates and downstream segments depend on."""
    file_path: str
    consumed_by: List[str] = field(default_factory=list)
    # v2.0: Named symbols this file MUST export (populated from enrichment).
    # When non-empty, the architecture generator is told explicitly which
    # functions/classes/constants to define, and the cohesion checker validates
    # their presence.  Empty means "export contract unknown — LLM decides."
    names: List[str] = field(default_factory=list)
    # v2.0: Full signatures for exported functions (optional — richer context).
    # Format: ["def func_name(arg: Type) -> ReturnType", ...]
    signatures: List[str] = field(default_factory=list)
    # v2.3: Symbols that should be re-exported from a sibling module rather
    # than defined locally.  Populated by augment_skeleton_with_enrichment()
    # when a symbol is canonically defined in an upstream segment's file_scope.
    # Format: [(symbol_name, source_module_path), ...]
    # e.g. [("_save_execution_trace", "app/orchestrator/segment_loop/_arch_utils.py")]
    re_exports: List[List[str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"file_path": self.file_path, "consumed_by": self.consumed_by}
        if self.names:
            d["names"] = self.names
        if self.signatures:
            d["signatures"] = self.signatures
        if self.re_exports:
            d["re_exports"] = self.re_exports
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExportBinding":
        return cls(
            file_path=data.get("file_path", ""),
            consumed_by=data.get("consumed_by", []),
            names=data.get("names", []),
            signatures=data.get("signatures", []),
            re_exports=data.get("re_exports", []),
        )


@dataclass
class SegmentSkeleton:
    """Skeleton contract for a single segment."""
    segment_id: str
    title: str = ""
    file_scope: List[str] = field(default_factory=list)
    exports: List[ExportBinding] = field(default_factory=list)
    imports_from: Dict[str, List[str]] = field(default_factory=dict)  # seg_id -> [file_paths]
    dependencies: List[str] = field(default_factory=list)
    peer_imports_from: Dict[str, List[str]] = field(default_factory=dict)  # v1.1: peer seg_id -> [file_paths]
    total_segments_in_job: int = 0
    all_segment_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "title": self.title,
            "file_scope": self.file_scope,
            "exports": [e.to_dict() for e in self.exports],
            "imports_from": self.imports_from,
            "dependencies": self.dependencies,
            "peer_imports_from": self.peer_imports_from,
            "total_segments_in_job": self.total_segments_in_job,
            "all_segment_ids": self.all_segment_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SegmentSkeleton":
        return cls(
            segment_id=data.get("segment_id", ""),
            title=data.get("title", ""),
            file_scope=data.get("file_scope", []),
            exports=[ExportBinding.from_dict(e) for e in data.get("exports", [])],
            imports_from=data.get("imports_from", {}),
            dependencies=data.get("dependencies", []),
            peer_imports_from=data.get("peer_imports_from", {}),
            total_segments_in_job=data.get("total_segments_in_job", 0),
            all_segment_ids=data.get("all_segment_ids", []),
        )


@dataclass
class SkeletonContractSet:
    """Complete skeleton contract set for a segmented job."""
    job_id: str
    total_segments: int = 0
    skeletons: List[SegmentSkeleton] = field(default_factory=list)
    cross_segment_bindings: List[Dict[str, str]] = field(default_factory=list)
    generated_at: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "total_segments": self.total_segments,
            "skeletons": [s.to_dict() for s in self.skeletons],
            "cross_segment_bindings": self.cross_segment_bindings,
            "generated_at": self.generated_at,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkeletonContractSet":
        return cls(
            job_id=data.get("job_id", ""),
            total_segments=data.get("total_segments", 0),
            skeletons=[SegmentSkeleton.from_dict(s) for s in data.get("skeletons", [])],
            cross_segment_bindings=data.get("cross_segment_bindings", []),
            generated_at=data.get("generated_at", ""),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "SkeletonContractSet":
        return cls.from_dict(json.loads(json_str))

    def format_contract_for_segment(self, segment_id: str) -> str:
        """
        Format the skeleton contract as markdown for injection into
        a segment's architecture generation prompt.

        This is the key output — it tells the Critical Pipeline LLM
        exactly what files this segment owns, what it exports, what
        it imports, and what the overall segment structure looks like.
        """
        skeleton = None
        for s in self.skeletons:
            if s.segment_id == segment_id:
                skeleton = s
                break

        if skeleton is None:
            return ""

        parts = []
        parts.append("## Skeleton Contract (BINDING — DO NOT VIOLATE)\n")

        # --- Job structure awareness ---
        parts.append(f"**This job has exactly {skeleton.total_segments_in_job} segments.** "
                     f"Do NOT reference any other segment numbers.\n")
        parts.append("Segment IDs in this job:")
        for sid in skeleton.all_segment_ids:
            marker = " ← (this segment)" if sid == segment_id else ""
            parts.append(f"  - `{sid}`{marker}")
        parts.append("")

        # --- File scope constraint ---
        parts.append("### File Scope Constraint\n")
        parts.append("**You may ONLY design architecture for these files. "
                     "Do NOT add, modify, or reference any other files in your file inventory.**\n")
        for fp in skeleton.file_scope:
            parts.append(f"  - `{fp}`")
        parts.append("")

        # --- Exports ---
        if skeleton.exports:
            parts.append("### This Segment EXPORTS\n")
            parts.append("The following files are consumed by downstream segments. "
                        "You MUST create/modify them with stable, importable interfaces.\n")
            for exp in skeleton.exports:
                if exp.consumed_by == ["__self__"]:
                    parts.append(f"  - `{exp.file_path}` → (contract-enforced exports)")
                else:
                    consumers = ", ".join(f"`{c}`" for c in exp.consumed_by)
                    parts.append(f"  - `{exp.file_path}` → consumed by {consumers}")
                # v2.0 / v2.3: Show required export names and signatures.
                # v2.3: Distinguish locally-defined exports from re-exports.
                # Re-exports are symbols canonically defined in another
                # segment's file_scope — this segment should import and
                # re-export them, NOT redefine them.
                if exp.names:
                    # Build lookup maps
                    _sig_map = {}
                    for sig in exp.signatures:
                        _sig_name = sig.split("(")[0].replace("def ", "").replace("class ", "").replace("async def ", "").strip()
                        _sig_map[_sig_name] = sig
                    _re_export_map = {}
                    for _re in exp.re_exports:
                        if len(_re) >= 2:
                            _re_export_map[_re[0]] = _re[1]

                    # Separate into local defines and re-exports
                    _local_names = [n for n in exp.names if n not in _re_export_map]
                    _reexp_names = [n for n in exp.names if n in _re_export_map]

                    if _local_names:
                        parts.append(f"    **MUST DEFINE AND EXPORT these symbols** (downstream segments depend on them):")
                        for name in _local_names:
                            if name in _sig_map:
                                parts.append(f"      - `{_sig_map[name]}`")
                            else:
                                parts.append(f"      - `{name}`")
                    if _reexp_names:
                        parts.append(f"    **MUST RE-EXPORT these symbols** (defined in another module, import and expose):")
                        for name in _reexp_names:
                            _src = _re_export_map[name]
                            _src_stem = os.path.splitext(os.path.basename(_src))[0]
                            parts.append(f"      - `{name}` — import from `.{_src_stem}` and re-export")
                            parts.append(f"        Pattern: `from .{_src_stem} import {name}`")
                            parts.append(f"        Do NOT redefine this function locally. It is canonical in `{_src}`.")
                    if not _local_names and not _reexp_names:
                        # Fallback: all names, no classification
                        parts.append(f"    **MUST EXPORT these symbols** (downstream segments depend on them):")
                        for name in exp.names:
                            if name in _sig_map:
                                parts.append(f"      - `{_sig_map[name]}`")
                            else:
                                parts.append(f"      - `{name}`")
            parts.append("")

        # --- Imports ---
        if skeleton.imports_from:
            parts.append("### This Segment IMPORTS FROM\n")
            parts.append("These files are created by upstream segments. "
                        "When you need functionality from them, import from these exact paths.\n")
            # v2.0: Build a lookup of upstream export names by file path
            _upstream_exports: Dict[str, ExportBinding] = {}
            for _other_skel in self.skeletons:
                for _exp in _other_skel.exports:
                    if _exp.names:
                        _upstream_exports[_exp.file_path] = _exp
            for upstream_seg, files in skeleton.imports_from.items():
                parts.append(f"  From `{upstream_seg}`:")
                for fp in files:
                    _exp_info = _upstream_exports.get(fp)
                    if _exp_info and _exp_info.names:
                        _avail = ", ".join(f"`{n}`" for n in _exp_info.names)
                        parts.append(f"    - `{fp}` — available symbols: {_avail}")
                    else:
                        parts.append(f"    - `{fp}`")
        # --- v2.4: DO NOT DEFINE section ---
        # Lists all symbols exported by upstream segments that this segment
        # imports from. These functions MUST be imported, NEVER redefined
        # locally. This is the #1 cause of implementation failures: the LLM
        # sees the function body in source evidence and copies it in instead
        # of importing from the upstream segment module.
        _do_not_define: List[tuple] = []  # (symbol_name, source_module, source_segment)
        if skeleton.imports_from:
            for upstream_seg, files in skeleton.imports_from.items():
                for fp in files:
                    _exp_info = _upstream_exports.get(fp)
                    if _exp_info and _exp_info.names:
                        _src_stem = os.path.splitext(os.path.basename(fp))[0]
                        for _sym_name in _exp_info.names:
                            _do_not_define.append((_sym_name, f".{_src_stem}", upstream_seg))
        if _do_not_define:
            parts.append("### ⛔ DO NOT DEFINE These Functions (v2.4)\n")
            parts.append("The following symbols are ALREADY DEFINED in upstream segment modules. "
                        "You MUST `import` them — NEVER redefine, copy, or re-implement them "
                        "in your files. Defining a function that already exists in an upstream "
                        "segment creates duplicate definitions, burns strikes, and breaks the package.\n")
            for _sym, _src, _seg in _do_not_define:
                parts.append(f"  - ❌ `{_sym}` — defined in `{_seg}`, import via `from {_src} import {_sym}`")
            parts.append("")
            parts.append("If your code needs to CALL any of these functions, write the import statement. "
                        "Do NOT copy the function body from the source monolith evidence.\n")
        parts.append("")

        # --- Peer imports (v1.1) ---
        if skeleton.peer_imports_from:
            parts.append("### Peer Segment Imports (OPTIONAL)\n")
            parts.append("These segments build before yours and are NOT your direct dependencies, ")
            parts.append("but their exported files are available for import if needed. ")
            parts.append("Using these can avoid unnecessary workarounds like callable injection ")
            parts.append("when a direct import would be simpler and preserve original signatures.\n")
            for peer_seg, files in skeleton.peer_imports_from.items():
                parts.append(f"  From `{peer_seg}`:")
                for fp in files:
                    parts.append(f"    - `{fp}`")
            parts.append("")

        # --- Dependencies ---
        if skeleton.dependencies:
            parts.append("### Dependencies\n")
            parts.append("This segment depends on these segments completing first:")
            for dep in skeleton.dependencies:
                parts.append(f"  - `{dep}`")
            parts.append("")

        # --- Package structure for imports ---
        _packages = set()
        _parent_files = []
        for fp in skeleton.file_scope:
            fp_norm = fp.replace("\\", "/")
            parts_list = fp_norm.split("/")
            if len(parts_list) >= 2:
                _pkg = "/".join(parts_list[:-1])
                _packages.add(_pkg)
            # Detect files in parent package vs sub-package
            if len(parts_list) >= 3:
                _parent_pkg = "/".join(parts_list[:-2])
                _packages.add(_parent_pkg)

        if len(_packages) > 1:
            parts.append("### Import Path Rules\n")
            parts.append("Files in this segment span multiple directory levels:")
            for _pkg in sorted(_packages):
                _pkg_files = [fp for fp in skeleton.file_scope if fp.replace('\\', '/').startswith(_pkg + '/')]
                if _pkg_files:
                    parts.append(f"  - `{_pkg}/`: {len(_pkg_files)} file(s)")
            parts.append("")
            parts.append("**Import rules**:")
            parts.append("- Files in the SAME directory use single-dot: `from .module import ...`")
            parts.append("- Files importing from a PARENT directory use double-dot: `from ..module import ...`")
            parts.append("- Files importing from a SIBLING sub-package use: `from ..subpkg.module import ...`")
            parts.append("")

        # --- v5.32: Complete Package Module Map ---
        # Prevents the architecture model from guessing filenames (e.g. ._main
        # instead of ._loop). Lists every file across every segment so the
        # model always knows exact import targets for deferred/circular imports.
        parts.append("### Complete Package Module Map (ALL segments)\n")
        parts.append("**Every file in this job, grouped by segment.** "
                     "When writing imports — including deferred/circular imports to "
                     "later segments — use ONLY these exact filenames. "
                     "NEVER invent module names like `_main.py` if the map shows `_loop.py`.\n")
        for _map_skel in self.skeletons:
            _marker = " ← (this segment)" if _map_skel.segment_id == segment_id else ""
            parts.append(f"  **{_map_skel.segment_id}**{_marker}:")
            for _map_fp in _map_skel.file_scope:
                _map_basename = os.path.basename(_map_fp)
                _map_stem = _map_basename.replace(".py", "")
                parts.append(f"    - `{_map_fp}` → import as `from .{_map_stem} import ...`")
        parts.append("")

        # --- Rules ---
        parts.append("### Rules\n")
        parts.append("1. Your file inventory MUST only contain files listed in File Scope Constraint above.")
        parts.append("2. Do NOT invent files, test files, or helper files outside the scope.")
        parts.append("3. Do NOT reference segment numbers that don't exist in this job.")
        parts.append(f"4. This job has {skeleton.total_segments_in_job} segments total — "
                     f"not more, not fewer.")
        parts.append("5. If you need to import from upstream segments, use the exact file paths listed above.")
        parts.append("6. Use correct relative import depth — see Import Path Rules above if present.")
        parts.append("7. NEVER invent module filenames. Use ONLY names from the Package Module Map.")
        parts.append("")

        return "\n".join(parts)


# =============================================================================
# GENERATOR — Pure logic, no LLM calls
# =============================================================================

def generate_skeleton_contract(
    manifest_dict: Dict[str, Any],
    job_id: str,
) -> SkeletonContractSet:
    """
    Generate skeleton contracts deterministically from a segment manifest.

    Reads the manifest's segments, file_scopes, evidence_files, and
    dependencies to produce binding contracts for each segment.

    Zero LLM calls. Pure Python logic.
    """
    segments_raw = manifest_dict.get("segments", [])
    if not segments_raw:
        return SkeletonContractSet(job_id=job_id, total_segments=0)

    total_segments = len(segments_raw)
    all_seg_ids = [s.get("segment_id", "") for s in segments_raw]

    # Build a map: file_path -> owning segment_id
    file_to_segment: Dict[str, str] = {}
    for seg in segments_raw:
        seg_id = seg.get("segment_id", "")
        for fp in seg.get("file_scope", []):
            file_to_segment[fp] = seg_id

    # Build a map: segment_id -> evidence_files
    seg_evidence: Dict[str, List[str]] = {}
    for seg in segments_raw:
        seg_id = seg.get("segment_id", "")
        seg_evidence[seg_id] = seg.get("evidence_files", [])

    skeletons = []
    all_bindings = []

    for seg in segments_raw:
        seg_id = seg.get("segment_id", "")
        title = seg.get("title", "")
        file_scope = seg.get("file_scope", [])
        dependencies = seg.get("dependencies", [])

        # --- Determine exports ---
        # A file is "exported" if it appears in another segment's evidence_files
        exports = []
        for fp in file_scope:
            consumers = []
            for other_seg in segments_raw:
                other_id = other_seg.get("segment_id", "")
                if other_id == seg_id:
                    continue
                if fp in other_seg.get("evidence_files", []):
                    consumers.append(other_id)
            if consumers:
                exports.append(ExportBinding(file_path=fp, consumed_by=consumers))
                for consumer_id in consumers:
                    all_bindings.append({
                        "from_segment": seg_id,
                        "to_segment": consumer_id,
                        "file_path": fp,
                        "binding_type": "evidence_dependency",
                    })

        # --- Determine imports ---
        # Group evidence_files by which segment owns them
        imports_from: Dict[str, List[str]] = {}
        for ev_file in seg_evidence.get(seg_id, []):
            owning_seg = file_to_segment.get(ev_file)
            if owning_seg and owning_seg != seg_id:
                if owning_seg not in imports_from:
                    imports_from[owning_seg] = []
                if ev_file not in imports_from[owning_seg]:
                    imports_from[owning_seg].append(ev_file)

        # --- Determine peer imports (v1.1) ---
        # Peer = segments that build before this one (earlier in order) but
        # are NOT listed as direct dependencies. They share a common consumer
        # but don't depend on each other. Their exports are available for import.
        seg_index = all_seg_ids.index(seg_id) if seg_id in all_seg_ids else -1
        peer_imports_from: Dict[str, List[str]] = {}
        if seg_index > 0:
            for earlier_id in all_seg_ids[:seg_index]:
                if earlier_id in dependencies:
                    continue  # already in imports_from, not a peer
                if earlier_id in imports_from:
                    continue  # already imported via evidence
                # Find what this earlier segment exports
                for other_seg in segments_raw:
                    if other_seg.get("segment_id") == earlier_id:
                        for fp in other_seg.get("file_scope", []):
                            # Check if this file is consumed by anyone downstream
                            for consumer_seg in segments_raw:
                                if fp in consumer_seg.get("evidence_files", []):
                                    if earlier_id not in peer_imports_from:
                                        peer_imports_from[earlier_id] = []
                                    if fp not in peer_imports_from[earlier_id]:
                                        peer_imports_from[earlier_id].append(fp)
                                    break
                        break

        skeleton = SegmentSkeleton(
            segment_id=seg_id,
            title=title,
            file_scope=file_scope,
            exports=exports,
            imports_from=imports_from,
            dependencies=dependencies,
            peer_imports_from=peer_imports_from,
            total_segments_in_job=total_segments,
            all_segment_ids=all_seg_ids,
        )
        skeletons.append(skeleton)

    # Deduplicate bindings
    seen_bindings = set()
    unique_bindings = []
    for b in all_bindings:
        key = (b["from_segment"], b["to_segment"], b["file_path"])
        if key not in seen_bindings:
            seen_bindings.add(key)
            unique_bindings.append(b)

    contract_set = SkeletonContractSet(
        job_id=job_id,
        total_segments=total_segments,
        skeletons=skeletons,
        cross_segment_bindings=unique_bindings,
    )

    logger.info(
        "[skeleton_contracts] Generated: %d segments, %d bindings for job %s",
        total_segments, len(unique_bindings), job_id,
    )

    return contract_set


# =============================================================================
# POST-ENRICHMENT AUGMENTATION
# =============================================================================


def augment_skeleton_with_enrichment(
    contract_set: SkeletonContractSet,
    enrichment_data: Dict[str, Any],
    job_dir: Optional[str] = None,
) -> int:
    """
    v2.0: Wire enrichment-extracted symbol names into skeleton export bindings.

    The skeleton is generated BEFORE enrichment (deterministic, zero LLM calls).
    Enrichment runs AFTER (AST extraction + optional LLM resolution).
    This function bridges the gap: it reads the enrichment's `exports` and
    `functions` lists and populates each ExportBinding's `names` and
    `signatures` fields.

    This means the architecture generator prompt will now say:
        "_evidence.py must export: build_evidence_bundle, verify_contracts_fulfilled"
    instead of just:
        "_evidence.py is consumed by seg-05, seg-06, seg-07"

    Args:
        contract_set: The skeleton contract set to augment (modified in place).
        enrichment_data: Dict of {segment_id: enrichment_dict} from enrich_segments().
        job_dir: Optional job directory — if provided, re-saves the augmented skeleton.

    Returns:
        Number of export bindings that were augmented with names.
    """
    augmented_count = 0

    # v2.3: Build a cross-segment symbol ownership map.
    # For each function/class in every segment's enrichment, record which
    # segment's file_scope it canonically belongs to. This lets us detect
    # when a segment's "export" is actually a re-export from a sibling.
    # Key: symbol_name, Value: (owning_segment_id, canonical_file_path)
    _symbol_ownership: Dict[str, tuple] = {}
    for _map_skel in contract_set.skeletons:
        _map_enr = enrichment_data.get(_map_skel.segment_id)
        if not _map_enr:
            continue
        _map_funcs = _map_enr.get("functions", [])
        _map_classes = _map_enr.get("classes", [])
        # For each function/class, check if its name matches a file in this
        # segment's file_scope (the canonical home after refactor).
        # We also check enrichment-level source_file if available.
        for _sym in (_map_funcs + _map_classes):
            _sym_name = _sym.get("name", "")
            if not _sym_name:
                continue
            # If this symbol is already owned by another segment, the
            # first-registered owner wins (earlier segments take priority
            # since they're upstream).
            if _sym_name in _symbol_ownership:
                continue
            _symbol_ownership[_sym_name] = (
                _map_skel.segment_id,
                _map_skel.file_scope[0] if len(_map_skel.file_scope) == 1 else "",
            )
    # For multi-file segments, try to refine the canonical file using
    # the file-stem heuristic (e.g. "build_evidence_bundle" → "_evidence.py")
    for _map_skel in contract_set.skeletons:
        if len(_map_skel.file_scope) <= 1:
            continue
        _map_enr = enrichment_data.get(_map_skel.segment_id)
        if not _map_enr:
            continue
        for _sym in (_map_enr.get("functions", []) + _map_enr.get("classes", [])):
            _sym_name = _sym.get("name", "")
            if not _sym_name:
                continue
            if _symbol_ownership.get(_sym_name, ("",))[0] != _map_skel.segment_id:
                continue  # Only refine if we own this symbol
            # Try file-stem match
            _name_lower = _sym_name.lower()
            for _fp in _map_skel.file_scope:
                _stem = os.path.splitext(os.path.basename(_fp))[0].lstrip("_").lower()
                if _stem in _name_lower or _name_lower in _stem:
                    _symbol_ownership[_sym_name] = (_map_skel.segment_id, _fp)
                    break

    if _symbol_ownership:
        logger.info(
            "[skeleton_contracts] v2.3 Symbol ownership map: %d symbol(s) across %d segment(s)",
            len(_symbol_ownership),
            len(set(v[0] for v in _symbol_ownership.values())),
        )

    for skeleton in contract_set.skeletons:
        seg_id = skeleton.segment_id
        seg_enrichment = enrichment_data.get(seg_id)
        if not seg_enrichment:
            continue

        # Get the enrichment's exports list (symbol names) and functions list
        enriched_exports: List[str] = seg_enrichment.get("exports", [])
        enriched_functions: List[Dict[str, Any]] = seg_enrichment.get("functions", [])
        enriched_classes: List[Dict[str, Any]] = seg_enrichment.get("classes", [])
        enriched_constants: List[Dict[str, Any]] = seg_enrichment.get("constants", [])

        if not enriched_exports:
            continue

        # Build a signature lookup: name -> signature string
        sig_lookup: Dict[str, str] = {}
        for func in enriched_functions:
            fname = func.get("name", "")
            fsig = func.get("signature", "")
            if fname and fsig:
                sig_lookup[fname] = fsig
        for cls in enriched_classes:
            cname = cls.get("name", "")
            if cname:
                sig_lookup[cname] = f"class {cname}"

        # Now determine which exports belong to which file in this segment.
        # Enrichment gives us a flat list of export names for the segment,
        # and the skeleton has per-file ExportBindings.  For single-file
        # segments (most common), all exports go to that one file.  For
        # multi-file segments, we use the function's source_file or line_range
        # to match.

        # v2.1: Terminal segments (no downstream consumers) have zero
        # ExportBindings from generate_skeleton_contract().  But they still
        # need contract enforcement — especially for segments like the main
        # orchestration loop that define critical functions.  Create
        # self-referencing ExportBindings for each file in scope so the
        # contract injection system can enforce function signatures.
        if len(skeleton.exports) == 0 and enriched_exports:
            for fp in skeleton.file_scope:
                skeleton.exports.append(ExportBinding(
                    file_path=fp,
                    consumed_by=["__self__"],
                ))
            logger.info(
                "[skeleton_contracts] v2.1 Created %d self-referencing export(s) "
                "for terminal segment %s",
                len(skeleton.exports), seg_id,
            )

        if len(skeleton.exports) == 1:
            # Simple case: all exports belong to the single exported file
            binding = skeleton.exports[0]
            binding.names = enriched_exports[:]
            binding.signatures = [
                sig_lookup[name] for name in enriched_exports
                if name in sig_lookup
            ]
            # v2.3: Detect re-exports — symbols canonically owned by another segment
            binding.re_exports = []
            for _name in enriched_exports:
                _owner = _symbol_ownership.get(_name)
                if _owner and _owner[0] != seg_id and _owner[1]:
                    binding.re_exports.append([_name, _owner[1]])
                    logger.info(
                        "[skeleton_contracts] v2.3 %s/%s: '%s' is re-export from %s (%s)",
                        seg_id, binding.file_path, _name, _owner[0], _owner[1],
                    )
            augmented_count += 1
            logger.info(
                "[skeleton_contracts] v2.0 Augmented %s: %s with %d export name(s)"
                " (%d re-export(s))",
                seg_id, binding.file_path, len(binding.names), len(binding.re_exports),
            )
        elif len(skeleton.exports) > 1:
            # Multi-file segment: try to assign exports to specific files.
            # Use function body/line_range to guess which file each symbol
            # belongs to.  Enrichment functions have a 'line_range' field
            # and the skeleton has file_scope.  If we can't determine the
            # file, distribute evenly (better than nothing).
            #
            # Build file -> function-name mapping from enrichment functions.
            # Each function has a source context but no explicit file assignment
            # (they all come from the monolith).  The segment's file_scope tells
            # us which files this segment owns.  For refactor jobs, each file
            # typically handles one responsibility area.
            #
            # Heuristic: match function names to file names.
            # e.g. "build_evidence_bundle" -> "_evidence.py" (contains "evidence")
            # First pass: try strong matches (name contains file stem or vice versa)
            _assigned_names: set = set()
            for binding in skeleton.exports:
                _file_stem = os.path.splitext(os.path.basename(binding.file_path))[0]
                _file_stem_clean = _file_stem.lstrip("_").lower()
                matched_names = []
                matched_sigs = []
                for name in enriched_exports:
                    _name_lower = name.lower()
                    if _file_stem_clean in _name_lower or _name_lower in _file_stem_clean:
                        matched_names.append(name)
                        _assigned_names.add(name)
                        if name in sig_lookup:
                            matched_sigs.append(sig_lookup[name])

                if matched_names:
                    binding.names = matched_names
                    binding.signatures = matched_sigs
                    # v2.3: Detect re-exports for multi-file segments
                    binding.re_exports = []
                    for _name in matched_names:
                        _owner = _symbol_ownership.get(_name)
                        if _owner and _owner[0] != seg_id and _owner[1]:
                            binding.re_exports.append([_name, _owner[1]])
                    augmented_count += 1
                    logger.info(
                        "[skeleton_contracts] v2.0 Augmented %s: %s with %d export name(s)"
                        " (%d re-export(s))",
                        seg_id, binding.file_path, len(binding.names),
                        len(binding.re_exports),
                    )

            # Second pass: log unassigned exports but DO NOT blindly assign them.
            # v2.2 FIX: The previous logic dumped unmatched function names onto
            # whatever file binding happened to be empty or first. This caused
            # functions defined in one file (e.g. cohesion.py) to appear as
            # required exports of a different file (e.g. job_runner.py), which
            # then failed signature checking because the function was only
            # re-imported, not defined there.
            #
            # If a function name doesn't match any file stem, we simply skip it.
            # The function is still enforced on the file where it IS matched by
            # the first-pass heuristic or by the single-file fast path.
            _unassigned = [n for n in enriched_exports if n not in _assigned_names]
            if _unassigned:
                logger.info(
                    "[skeleton_contracts] v2.2 %d unassigned export(s) for %s "
                    "(skipped, not blindly assigned): %s",
                    len(_unassigned), seg_id, _unassigned,
                )

            # If heuristic didn't match anything, put all exports on the first binding
            # as a fallback (still better than empty)
            _any_matched = any(exp.names for exp in skeleton.exports)
            if not _any_matched and skeleton.exports:
                skeleton.exports[0].names = enriched_exports[:]
                skeleton.exports[0].signatures = [
                    sig_lookup[name] for name in enriched_exports
                    if name in sig_lookup
                ]
                augmented_count += 1
                logger.info(
                    "[skeleton_contracts] v2.0 Fallback augment %s: all %d exports on %s",
                    seg_id, len(enriched_exports), skeleton.exports[0].file_path,
                )

    # Re-save if job_dir provided
    if job_dir and augmented_count > 0:
        try:
            save_skeleton_contract(contract_set, job_dir)
            logger.info(
                "[skeleton_contracts] v2.0 Re-saved augmented skeleton: %d binding(s) enriched",
                augmented_count,
            )
        except Exception as e:
            logger.warning("[skeleton_contracts] v2.0 Failed to re-save augmented skeleton: %s", e)

    return augmented_count


# =============================================================================
# PERSISTENCE
# =============================================================================

def save_skeleton_contract(contract_set: SkeletonContractSet, job_dir: str) -> str:
    """Save skeleton contracts to disk alongside the segment manifest."""
    segments_dir = os.path.join(job_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)
    path = os.path.join(segments_dir, "skeleton_contract.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(contract_set.to_json(indent=2))
    logger.info("[skeleton_contracts] Saved: %s", path)
    return path


def load_skeleton_contract(job_dir: str) -> Optional[SkeletonContractSet]:
    """Load skeleton contracts from disk. Returns None if not found."""
    path = os.path.join(job_dir, "segments", "skeleton_contract.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return SkeletonContractSet.from_json(f.read())
    except Exception as e:
        logger.warning("[skeleton_contracts] Failed to load: %s", e)
        return None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ExportBinding",
    "SegmentSkeleton",
    "SkeletonContractSet",
    "generate_skeleton_contract",
    "augment_skeleton_with_enrichment",
    "save_skeleton_contract",
    "load_skeleton_contract",
    "SKELETON_CONTRACTS_BUILD_ID",
]
