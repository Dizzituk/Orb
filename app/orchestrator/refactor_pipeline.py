# FILE: app/orchestrator/refactor_pipeline.py
"""
Deterministic Refactor Pipeline — the routing and orchestration layer.

Detects refactor jobs and runs them through the deterministic path:
  Scanner → Segmenter → Architecture Generator → Compiler → Implementer

For greenfield jobs, falls back to the existing LLM-driven pipeline.

This module provides:
  - is_refactor_job(): detect whether a job should use deterministic path
  - run_deterministic_refactor(): full pipeline for refactor jobs  
  - bridge_to_enrichment(): convert scanner output to enrichment format

BUILD_ID: 2026-02-20-v1.0-deterministic-refactor-pipeline
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from app.orchestrator.codebase_scanner import scan_file
from app.orchestrator.codebase_scanner_models import (
    CodebaseGraph,
    FileScanResult,
    SymbolKind,
)
from app.orchestrator.deterministic_architecture import (
    generate_all_architectures,
    generate_segment_architecture,
    save_architecture,
)
from app.orchestrator.refactor_segmenter import (
    build_refactor_plan,
    save_build_plan,
)
from app.orchestrator.refactor_segmenter_models import (
    RefactorBuildPlan,
    SegmentPlan,
)

logger = logging.getLogger(__name__)

REFACTOR_PIPELINE_BUILD_ID = "2026-02-20-v1.0-deterministic-refactor-pipeline"
print(f"[REFACTOR_PIPELINE_LOADED] BUILD_ID={REFACTOR_PIPELINE_BUILD_ID}")


# =============================================================================
# REFACTOR DETECTION
# =============================================================================

# Keywords in spec text that indicate a refactor operation
_REFACTOR_KEYWORDS = frozenset({
    "decompose", "refactor", "restructure", "extract", "split",
    "break up", "break down", "subpackage", "modularise", "modularize",
    "move functions", "reorganize", "reorganise",
})


def is_refactor_job(
    spec_text: str,
    file_scope: List[str],
    project_roots: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """
    Detect whether a job should use the deterministic refactor pipeline.

    Returns (is_refactor, reason).

    A job is a refactor if:
    1. The spec text contains refactor keywords, AND
    2. At least one target file already exists on disk (there's source to scan)

    If neither condition is met, it's greenfield → use existing LLM pipeline.
    """
    if project_roots is None:
        project_roots = ["D:\\Orb", "D:\\orb-desktop"]

    # Check for refactor keywords in spec
    spec_lower = spec_text.lower()
    matched_keywords = [kw for kw in _REFACTOR_KEYWORDS if kw in spec_lower]

    if not matched_keywords:
        return False, "No refactor keywords found in spec"

    # Check if any source file exists on disk
    existing_files = []
    for rel_path in file_scope:
        for root in project_roots:
            abs_path = os.path.join(root, rel_path.replace("/", os.sep))
            if os.path.isfile(abs_path):
                existing_files.append(rel_path)
                break

    if not existing_files:
        return False, f"Refactor keywords found ({matched_keywords}) but no source files exist on disk"

    return True, (
        f"Refactor detected: keywords={matched_keywords}, "
        f"existing_files={len(existing_files)}/{len(file_scope)}"
    )


# =============================================================================
# SCANNER → ENRICHMENT BRIDGE
# =============================================================================

def scan_to_enrichment(source_scan: FileScanResult) -> Dict[str, Any]:
    """
    Convert a FileScanResult into the enrichment dict format
    that the existing pipeline (compiler, segment_loop) expects.

    This bridges the new scanner to the existing pipeline without
    requiring changes to downstream consumers.
    """
    enrichment: Dict[str, Any] = {
        "functions": [],
        "classes": [],
        "constants": [],
        "imports": [],
        "source_extract": {},
        "source_file": source_scan.file_path,
        "extraction_stats": {
            "total_symbols": len(source_scan.symbols),
            "functions": source_scan.function_count,
            "classes": source_scan.class_count,
            "constants": source_scan.constant_count,
            "dead_symbols": source_scan.dead_symbol_count,
            "scan_method": "deterministic_ast",
        },
    }

    for name, info in source_scan.symbols.items():
        enrichment["source_extract"][name] = info.source_code

        if info.kind in (SymbolKind.FUNCTION, SymbolKind.ASYNC_FUNCTION):
            enrichment["functions"].append({
                "name": name,
                "signature": info.signature,
                "docstring": info.docstring,
                "body": info.source_code,
                "line_range": f"{info.line_start}-{info.line_end}",
                "is_async": info.is_async,
                "decorators": info.decorators,
                "calls": info.calls,
                "called_by": info.called_by,
                "references": info.references,
                "referenced_by": info.referenced_by,
            })
        elif info.kind == SymbolKind.CLASS:
            enrichment["classes"].append({
                "name": name,
                "bases": info.bases,
                "methods": info.methods,
                "body": info.source_code,
                "line_range": f"{info.line_start}-{info.line_end}",
                "references": info.references,
                "referenced_by": info.referenced_by,
            })
        else:
            enrichment["constants"].append({
                "name": name,
                "value": info.source_code,
                "line_number": info.line_start,
                "references": info.references,
                "referenced_by": info.referenced_by,
            })

    for imp in source_scan.imports:
        enrichment["imports"].append(imp.raw_statement)

    return enrichment


# =============================================================================
# MANIFEST BRIDGE
# =============================================================================

def plan_to_manifest(
    plan: RefactorBuildPlan,
    spec_id: str,
    spec_hash: str = "",
) -> Dict[str, Any]:
    """
    Convert a RefactorBuildPlan into the manifest dict format that
    the existing segment_loop pipeline expects.

    Produces the same structure as segmentation.py's generate_segments().
    """
    segments = []
    for seg in plan.segments:
        # Build evidence_files: files from earlier segments
        evidence_files = []
        for dep_seg_id in seg.dependencies:
            dep_seg = next((s for s in plan.segments if s.segment_id == dep_seg_id), None)
            if dep_seg:
                evidence_files.extend(dep_seg.file_paths)

        segment_dict = {
            "segment_id": seg.segment_id,
            "title": seg.title,
            "parent_spec_id": spec_id,
            "requirements": [f"Refactor {plan.source_file} into subpackage"],
            "file_scope": seg.file_paths,
            "evidence_files": evidence_files,
            "dependencies": seg.dependencies,
            "exposes": None,
            "consumes": None,
            "acceptance_criteria": [],
            "estimated_files": seg.estimated_files,
            "grounding_data": {
                "verified_files": [],
                "interface_reads": [],
                "stale_entries": [],
                "create_targets": [
                    {"path": fp, "must_expose": [], "reason": "Deterministic refactor target"}
                    for fp in seg.file_paths
                ],
                "new_files": [],
                "verification_errors": [],
            },
            "deterministic_refactor": True,
            "tiers_included": seg.tiers_included,
            "is_facade_segment": seg.is_facade_segment,
        }
        segments.append(segment_dict)

    manifest = {
        "parent_spec_id": spec_id,
        "parent_spec_hash": spec_hash,
        "segments": segments,
        "requirement_map": {
            f"Refactor {plan.source_file}": [s.segment_id for s in plan.segments]
        },
        "total_segments": plan.total_segments,
        "total_files": sum(s.estimated_files for s in plan.segments),
        "generated_at": _now_iso(),
        "manifest_version": "1.0",
        "deferred_consumer_files": [],
        "deterministic_source": plan.source_file,
    }

    return manifest


# =============================================================================
# FULL DETERMINISTIC PIPELINE
# =============================================================================

def run_deterministic_refactor(
    source_file_path: str,
    architecture_file_inventory: str,
    target_package: str,
    job_dir: str,
    spec_id: str = "",
    project_roots: Optional[List[str]] = None,
) -> Tuple[RefactorBuildPlan, Dict[str, str], Dict[str, Any]]:
    """
    Run the full deterministic refactor pipeline.

    Steps:
    1. Scan source file → CodebaseGraph
    2. Convert scan → enrichment format
    3. Build refactor plan (segmenter)
    4. Generate architecture per segment
    5. Save all artifacts

    Args:
        source_file_path: Relative path to source monolith
        architecture_file_inventory: Architecture text with File Inventory table
                                     (can come from initial LLM arch or previous scan)
        target_package: Target package directory
        job_dir: Job directory for persistence
        spec_id: Parent spec ID
        project_roots: Project root directories to search

    Returns:
        (plan, architectures, manifest)
        - plan: The RefactorBuildPlan
        - architectures: {segment_id: architecture_markdown}
        - manifest: Standard manifest dict for segment_loop
    """
    if project_roots is None:
        project_roots = ["D:\\Orb", "D:\\orb-desktop"]

    start = time.time()

    # Step 1: Read and scan source
    source_code = None
    for root in project_roots:
        abs_path = os.path.join(root, source_file_path.replace("/", os.sep))
        if os.path.isfile(abs_path):
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                source_code = f.read()
            break

    if source_code is None:
        raise FileNotFoundError(f"Source file not found: {source_file_path}")

    source_scan = scan_file(source_file_path, source_code)
    logger.info(
        "[refactor_pipeline] Scanned %s: %d symbols in %.1fs",
        source_file_path, len(source_scan.symbols), time.time() - start,
    )

    # Step 2: Bridge to enrichment
    enrichment = scan_to_enrichment(source_scan)

    # Step 3: Build refactor plan
    plan = build_refactor_plan(
        enrichment=enrichment,
        architecture_text=architecture_file_inventory,
        source_file=source_file_path,
        target_package=target_package,
    )
    save_build_plan(plan, job_dir)
    logger.info(
        "[refactor_pipeline] Plan: %d symbols → %d files → %d tiers → %d segments",
        plan.graph.total_symbols, plan.graph.total_files,
        plan.graph.total_tiers, plan.total_segments,
    )

    # v6.1 FIX 23: Save enrichment to each segment directory so the
    # implementation compiler can load it. Without this, the compiler
    # detects profile=GREENFIELD (no enrichment) and the LLM invents
    # fictional code instead of transplanting from the architecture.
    for seg in plan.segments:
        _enrich_dir = os.path.join(job_dir, "segments", seg.segment_id)
        os.makedirs(_enrich_dir, exist_ok=True)
        _enrich_path = os.path.join(_enrich_dir, "enrichment.json")
        try:
            with open(_enrich_path, "w", encoding="utf-8") as _ef:
                json.dump(enrichment, _ef, indent=2, default=str)
        except Exception as _e:
            logger.warning(
                "[refactor_pipeline] Failed to save enrichment for %s: %s",
                seg.segment_id, _e,
            )

    # Step 4: Generate architectures
    architectures = generate_all_architectures(plan, source_scan)
    for seg_id, arch_text in architectures.items():
        save_architecture(arch_text, job_dir, seg_id)

    # Step 5: Build manifest
    manifest = plan_to_manifest(plan, spec_id)

    elapsed = time.time() - start
    logger.info(
        "[refactor_pipeline] Complete in %.1fs — %d segments, %d arch docs, zero LLM calls",
        elapsed, plan.total_segments, len(architectures),
    )

    # Save pipeline summary
    _save_summary(job_dir, plan, source_scan, elapsed)

    return plan, architectures, manifest


# =============================================================================
# PERSISTENCE
# =============================================================================

def _save_summary(
    job_dir: str,
    plan: RefactorBuildPlan,
    source_scan: FileScanResult,
    elapsed: float,
) -> None:
    """Save pipeline execution summary."""
    summary = {
        "pipeline": "deterministic_refactor",
        "build_id": REFACTOR_PIPELINE_BUILD_ID,
        "source_file": plan.source_file,
        "source_lines": source_scan.line_count,
        "source_symbols": len(source_scan.symbols),
        "target_files": plan.graph.total_files,
        "segments": plan.total_segments,
        "tiers": plan.graph.total_tiers,
        "health_issues": len(source_scan.health_issues),
        "warnings": plan.warnings,
        "cycle_detected": plan.graph.cycle_detected,
        "llm_calls": 0,
        "elapsed_seconds": round(elapsed, 2),
    }

    path = os.path.join(job_dir, "deterministic_refactor_summary.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _now_iso() -> str:
    """ISO timestamp string."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
