# FILE: app/pot_spec/grounded/segment_validation.py
# Purpose: Derek phase 4 — segment manifest validation: size budgets, cycle/unknown-dep detection, deterministic re-split.
# Called-by: app.pot_spec.grounded._spec_runner_segmentation
# Depends-on: app.pot_spec.grounded.segment_schemas, app.pot_spec.grounded._spec_runner_utils_13
# Last-renovated: 2026-07-04
"""
Capability-sized segments, made enforceable.

A segment must fit a 26B-class worker's reliable working set:
    ASTRA_SEGMENT_MAX_FILES (default 5)   — files in file_scope
    ASTRA_SEGMENT_MAX_KB    (default 60)  — on-disk KB of EXISTING file_scope files

Validation also makes the dependency graph trustworthy:
    - unknown depends_on ids  -> hard failure
    - dependency cycles       -> hard failure (the Rubik's-cube order must be computable)

Oversized segments are split deterministically (greedy chunking, parts
chained part1 <- part2 <- ...), with a bounded number of passes. A segment
that STILL busts the KB budget after splitting (single giant file) fails
loudly — never silently oversized. Deterministic-refactor segments are
KB-exempt: their source monolith is large by definition.

Old manifests without depends_on/budgets validate clean (absent = safe).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MAX_SPLIT_PASSES = 2


class SegmentValidationError(RuntimeError):
    """Manifest failed validation after bounded re-splitting. Message says why."""


@dataclass
class ValidationIssue:
    kind: str          # "cycle" | "unknown_dependency" | "too_many_files" | "too_many_kb"
    segment_id: str
    detail: str


@dataclass
class ValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def of_kind(self, *kinds: str) -> List[ValidationIssue]:
        return [i for i in self.issues if i.kind in kinds]

    def summary(self) -> str:
        return "; ".join(f"{i.kind}[{i.segment_id}]: {i.detail}" for i in self.issues) or "ok"


def _max_files() -> int:
    try:
        return max(1, int(os.getenv("ASTRA_SEGMENT_MAX_FILES", "5")))
    except ValueError:
        return 5


def _max_kb() -> int:
    try:
        return max(1, int(os.getenv("ASTRA_SEGMENT_MAX_KB", "60")))
    except ValueError:
        return 60


def _project_roots() -> List[str]:
    try:
        from app.pot_spec.grounded._spec_runner_utils_13 import _discover_project_roots
        roots = _discover_project_roots().get("roots", [])
        return roots or [r"D:\Orb", r"D:\orb-desktop"]
    except Exception:
        return [r"D:\Orb", r"D:\orb-desktop"]


def _segment_kb(seg: Any, roots: List[str]) -> float:
    """On-disk KB of a segment's EXISTING file_scope files.

    Prefers grounding_data.verified_files sizes; falls back to stat under the
    project roots. CREATE targets (no file on disk) contribute 0 — the file
    COUNT budget bounds those.
    """
    sizes: Dict[str, int] = {}
    grounding = getattr(seg, "grounding_data", None)
    for vf in (getattr(grounding, "verified_files", None) or []):
        path = getattr(vf, "path", None) or (vf.get("path") if isinstance(vf, dict) else None)
        size = getattr(vf, "size_bytes", None) or (vf.get("size_bytes") if isinstance(vf, dict) else None)
        if path and size:
            sizes[str(path).replace("\\", "/").lower()] = int(size)

    total = 0
    for rel in (getattr(seg, "file_scope", None) or []):
        key = str(rel).replace("\\", "/").lower()
        if key in sizes:
            total += sizes[key]
            continue
        for root in roots:
            candidate = os.path.join(root, str(rel).replace("/", os.sep))
            try:
                if os.path.isfile(candidate):
                    total += os.path.getsize(candidate)
                    break
            except OSError:
                continue
    return total / 1024.0


def _find_cycle(segments: List[Any]) -> Optional[List[str]]:
    """DFS three-colour cycle detection over depends_on edges."""
    ids = {getattr(s, "segment_id", "") for s in segments}
    graph = {
        getattr(s, "segment_id", ""): [d for d in (getattr(s, "dependencies", None) or []) if d in ids]
        for s in segments
    }
    WHITE, GREY, BLACK = 0, 1, 2
    colour = {sid: WHITE for sid in graph}
    stack_path: List[str] = []

    def dfs(node: str) -> Optional[List[str]]:
        colour[node] = GREY
        stack_path.append(node)
        for dep in graph.get(node, []):
            if colour[dep] == GREY:
                return stack_path[stack_path.index(dep):] + [dep]
            if colour[dep] == WHITE:
                found = dfs(dep)
                if found:
                    return found
        colour[node] = BLACK
        stack_path.pop()
        return None

    for sid in graph:
        if colour[sid] == WHITE:
            found = dfs(sid)
            if found:
                return found
    return None


def validate_manifest(manifest: Any) -> ValidationReport:
    """Validate segments against the DAG rules and size budgets."""
    report = ValidationReport()
    segments = list(getattr(manifest, "segments", None) or [])
    if not segments:
        return report

    ids = {getattr(s, "segment_id", "") for s in segments}
    max_files, max_kb = _max_files(), _max_kb()
    roots = _project_roots()

    for seg in segments:
        sid = getattr(seg, "segment_id", "?")
        for dep in (getattr(seg, "dependencies", None) or []):
            if dep not in ids:
                report.issues.append(ValidationIssue(
                    "unknown_dependency", sid, f"depends_on {dep!r} which is not a segment id",
                ))
        n_files = len(getattr(seg, "file_scope", None) or [])
        if n_files > max_files:
            report.issues.append(ValidationIssue(
                "too_many_files", sid,
                f"{n_files} files > ASTRA_SEGMENT_MAX_FILES={max_files}",
            ))
        if not getattr(seg, "deterministic_refactor", False):
            kb = _segment_kb(seg, roots)
            if kb > max_kb:
                report.issues.append(ValidationIssue(
                    "too_many_kb", sid,
                    f"{kb:.0f}KB on disk > ASTRA_SEGMENT_MAX_KB={max_kb}",
                ))

    cycle = _find_cycle(segments)
    if cycle:
        report.issues.append(ValidationIssue(
            "cycle", cycle[0], "dependency cycle: " + " -> ".join(cycle),
        ))
    return report


def split_oversized_segments(manifest: Any) -> Tuple[Any, int]:
    """Deterministically split segments whose file_scope busts the budgets.

    Greedy chunking by (file count, KB); parts are chained sequentially
    (part2 depends on part1) and external edges are remapped: dependants of
    the original now depend on the LAST part. Returns (manifest, n_splits).
    """
    segments = list(getattr(manifest, "segments", None) or [])
    max_files, max_kb = _max_files(), _max_kb()
    roots = _project_roots()

    new_segments: List[Any] = []
    remap_last: Dict[str, str] = {}   # original id -> last part id
    remap_first: Dict[str, str] = {}  # original id -> first part id
    splits = 0

    for seg in segments:
        sid = getattr(seg, "segment_id", "?")
        scope = list(getattr(seg, "file_scope", None) or [])
        kb_exempt = bool(getattr(seg, "deterministic_refactor", False))
        seg_kb = 0.0 if kb_exempt else _segment_kb(seg, roots)

        if len(scope) <= max_files and (kb_exempt or seg_kb <= max_kb):
            new_segments.append(seg)
            continue

        # Greedy chunking: pack files until either budget would overflow.
        file_kb: List[Tuple[str, float]] = []
        for rel in scope:
            single = replace(seg, file_scope=[rel])
            file_kb.append((rel, 0.0 if kb_exempt else _segment_kb(single, roots)))

        chunks: List[List[str]] = []
        current: List[str] = []
        current_kb = 0.0
        for rel, kb in file_kb:
            over_files = len(current) + 1 > max_files
            over_kb = (not kb_exempt) and current and (current_kb + kb > max_kb)
            if current and (over_files or over_kb):
                chunks.append(current)
                current, current_kb = [], 0.0
            current.append(rel)
            current_kb += kb
        if current:
            chunks.append(current)

        if len(chunks) <= 1:
            new_segments.append(seg)  # cannot split further (single file) — validation will flag
            continue

        splits += 1
        part_ids: List[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            part_id = f"{sid}-p{idx}"
            part_ids.append(part_id)
            deps = list(getattr(seg, "dependencies", None) or []) if idx == 1 else [f"{sid}-p{idx-1}"]
            part = replace(
                seg,
                segment_id=part_id,
                title=f"{getattr(seg, 'title', sid)} (part {idx}/{len(chunks)})",
                file_scope=chunk,
                dependencies=deps,
                estimated_files=len(chunk),
            )
            new_segments.append(part)
        remap_first[sid] = part_ids[0]
        remap_last[sid] = part_ids[-1]
        logger.info(
            "[segment_validation] Split %s into %d parts (%d files, budgets %d files / %dKB)",
            sid, len(chunks), len(scope), max_files, max_kb,
        )

    if not splits:
        return manifest, 0

    # Remap external references to split segments.
    for seg in new_segments:
        deps = getattr(seg, "dependencies", None) or []
        new_deps = [remap_last.get(d, d) for d in deps]
        if new_deps != deps:
            try:
                seg.dependencies = new_deps
            except Exception:
                pass

    manifest.segments = new_segments
    try:
        manifest.total_segments = len(new_segments)
        manifest.total_files = sum(len(getattr(s, "file_scope", []) or []) for s in new_segments)
    except Exception:
        pass
    # requirement_map values reference segment ids — a requirement spanning
    # a split segment is implemented across ALL its parts (p7 review fix:
    # previously only the boundary parts were listed).
    parts_of: Dict[str, List[str]] = {}
    for seg in new_segments:
        sid = getattr(seg, "segment_id", "")
        for orig, first in remap_first.items():
            if sid == first or (sid.startswith(orig + "-p")):
                parts_of.setdefault(orig, []).append(sid)
    req_map = getattr(manifest, "requirement_map", None)
    if isinstance(req_map, dict):
        for req, sids in req_map.items():
            expanded: List[str] = []
            for s in sids:
                for p in parts_of.get(s, [s]):
                    if p not in expanded:
                        expanded.append(p)
            req_map[req] = expanded
    edges = getattr(manifest, "cross_target_edges", None)
    if isinstance(edges, list):
        for e in edges:
            if isinstance(e, dict):
                if e.get("from") in remap_last:
                    e["from"] = remap_last[e["from"]]
                if e.get("to") in remap_first:
                    e["to"] = remap_first[e["to"]]
    return manifest, splits


def enforce_segment_budgets(manifest: Any, job_id: str = "") -> Any:
    """Validate -> split (bounded) -> revalidate. Raises SegmentValidationError
    on cycles, unknown dependencies, or residual size violations."""
    report = validate_manifest(manifest)
    hard = report.of_kind("cycle", "unknown_dependency")
    if hard:
        raise SegmentValidationError(
            f"Segment manifest for job {job_id or '?'} has a broken dependency graph: "
            + "; ".join(i.detail for i in hard)
        )

    passes = 0
    while not report.ok and passes < MAX_SPLIT_PASSES:
        manifest, n = split_oversized_segments(manifest)
        passes += 1
        report = validate_manifest(manifest)
        if n == 0:
            break

    if not report.ok:
        residual = report.of_kind("too_many_files", "too_many_kb")
        if residual:
            raise SegmentValidationError(
                f"Segment manifest for job {job_id or '?'} is oversized after "
                f"{passes} split pass(es) — never silently oversized. "
                + report.summary()
                + f" (budgets: ASTRA_SEGMENT_MAX_FILES={_max_files()}, ASTRA_SEGMENT_MAX_KB={_max_kb()})"
            )
        hard = report.of_kind("cycle", "unknown_dependency")
        if hard:
            raise SegmentValidationError(
                f"Segment split broke the dependency graph for job {job_id or '?'}: "
                + report.summary()
            )
    logger.info("[segment_validation] Manifest valid after %d split pass(es)", passes)
    return manifest


__all__ = [
    "SegmentValidationError",
    "ValidationIssue",
    "ValidationReport",
    "validate_manifest",
    "split_oversized_segments",
    "enforce_segment_budgets",
]
