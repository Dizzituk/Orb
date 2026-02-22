import json
import os
from app.orchestrator._skeleton_contracts_utils import SegmentSkeleton
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


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
