# FILE: app/orchestrator/construction_planner.py
"""
Construction Planner — Stage 3 Multi-Phase Decomposition.

Analyses a spec to determine if it requires multiple build phases and,
if so, decomposes into an ordered plan with inter-phase contracts.

For single-phase jobs (the majority), returns a trivial plan wrapping
the existing segment flow unchanged.

v1.0 (2026-02-14): Initial implementation — Stage 3.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from .construction_planner_models import (
    ConstructionPlan,
    PhaseContract,
    PhaseDefinition,
)

logger = logging.getLogger(__name__)

CONSTRUCTION_PLANNER_BUILD_ID = "2026-02-14-v1.0-initial"
print(f"[CONSTRUCTION_PLANNER_LOADED] BUILD_ID={CONSTRUCTION_PLANNER_BUILD_ID}")

# Thresholds
MULTI_PHASE_FILE_THRESHOLD = 25
MULTI_PHASE_SEGMENT_THRESHOLD = 10
MAX_FILES_PER_PHASE = 20


def create_construction_plan(
    job_id: str,
    file_scope: List[str],
    spec_markdown: str,
    estimated_segments: int = 0,
    job_dir: Optional[str] = None,
) -> ConstructionPlan:
    """
    Analyse spec and file scope to produce a construction plan.

    For most jobs (< 25 files), returns single-phase. For large jobs,
    decomposes into ordered phases with dependency contracts.
    """
    needs_multi = _should_decompose(file_scope, spec_markdown, estimated_segments)

    if not needs_multi:
        plan = _create_single_phase_plan(job_id, file_scope, spec_markdown)
    else:
        plan = _create_multi_phase_plan(job_id, file_scope, spec_markdown)

    plan.total_files = len(file_scope)
    plan.estimated_total_segments = estimated_segments or len(file_scope)

    if job_dir:
        _save_plan(plan, job_dir)

    logger.info(
        "[construction_planner] %s plan: %d phase(s), %d files",
        "Multi" if plan.is_multi_phase else "Single",
        plan.total_phases, len(file_scope),
    )
    return plan


def _should_decompose(
    file_scope: List[str], spec_markdown: str, estimated_segments: int,
) -> bool:
    """True if job needs multi-phase decomposition."""
    if len(file_scope) <= MULTI_PHASE_FILE_THRESHOLD:
        return False
    if estimated_segments and estimated_segments <= MULTI_PHASE_SEGMENT_THRESHOLD:
        return False
    return _detect_layering_signals(file_scope, spec_markdown) >= 2


def _detect_layering_signals(file_scope: List[str], spec_markdown: str) -> int:
    """Count signals suggesting dependency layering."""
    signals = 0

    # Signal 1: spec mentions phases/ordering
    phase_kw = re.findall(
        r'\b(phase\s*\d|stage\s*\d|step\s*\d|first.*then|before.*after|'
        r'foundation|prerequisite|depends\s+on|build\s+order)\b',
        spec_markdown.lower(),
    )
    if len(phase_kw) >= 2:
        signals += 1

    # Signal 2: infra vs feature directory separation
    infra_dirs = {"models", "schemas", "db", "database", "config", "core", "utils", "common"}
    feature_dirs = {"api", "routes", "views", "features", "services", "handlers"}
    dirs_in_scope = set()
    for fp in file_scope:
        parts = fp.replace("\\", "/").split("/")
        if len(parts) >= 2:
            dirs_in_scope.add(parts[-2].lower())
    if (dirs_in_scope & infra_dirs) and (dirs_in_scope & feature_dirs):
        signals += 1

    # Signal 3: deep package structure
    if sum(1 for f in file_scope if f.endswith("__init__.py")) >= 4:
        signals += 1

    # Signal 4: very large scope
    if len(file_scope) > 40:
        signals += 1

    return signals


def _create_single_phase_plan(
    job_id: str, file_scope: List[str], spec_markdown: str,
) -> ConstructionPlan:
    """Trivial single-phase plan wrapping existing flow."""
    phase = PhaseDefinition(
        phase_id=f"{job_id}-phase-1",
        phase_number=1,
        title="Full Build",
        description="Single-phase build — all files in one pass",
        file_scope=list(file_scope),
        depends_on=[],
        spec_section=spec_markdown,
        estimated_segments=len(file_scope),
    )
    return ConstructionPlan(
        job_id=job_id, total_phases=1, phases=[phase],
        is_multi_phase=False,
        reasoning="File scope within single-phase threshold",
    )


def _create_multi_phase_plan(
    job_id: str, file_scope: List[str], spec_markdown: str,
) -> ConstructionPlan:
    """
    Decompose into phases using directory-based layer classification.

    Phase 1: Infrastructure (models, schemas, config, db, core, utils)
    Phase 2: Business logic (services, handlers, workers)
    Phase 3: API/Interface (routes, api, views, endpoints)
    Phase N: Remaining files
    """
    layers = _classify_files_into_layers(file_scope)
    phases: List[PhaseDefinition] = []
    phase_num = 0

    layer_order = ["infrastructure", "business", "interface", "other"]
    layer_titles = {
        "infrastructure": "Foundation & Infrastructure",
        "business": "Business Logic & Services",
        "interface": "API & Interface Layer",
        "other": "Remaining Components",
    }

    prev_id = None
    for layer_name in layer_order:
        layer_files = layers.get(layer_name, [])
        if not layer_files:
            continue

        phase_num += 1
        pid = f"{job_id}-phase-{phase_num}"

        contract = PhaseContract(
            phase_id=pid, exports=list(layer_files),
            description=f"Delivers {layer_titles.get(layer_name, layer_name)} files",
        )
        phase = PhaseDefinition(
            phase_id=pid, phase_number=phase_num,
            title=layer_titles.get(layer_name, f"Phase {phase_num}"),
            description=f"Build {len(layer_files)} {layer_name} files",
            file_scope=layer_files,
            depends_on=[prev_id] if prev_id else [],
            contract=contract,
            estimated_segments=max(1, len(layer_files) // 3),
        )
        phases.append(phase)
        prev_id = pid

    if not phases:
        return _create_single_phase_plan(job_id, file_scope, spec_markdown)

    return ConstructionPlan(
        job_id=job_id, total_phases=len(phases), phases=phases,
        is_multi_phase=len(phases) > 1,
        reasoning=(
            f"Decomposed into {len(phases)} phases: "
            + ", ".join(f"{p.title} ({len(p.file_scope)})" for p in phases)
        ),
    )


def _classify_files_into_layers(file_scope: List[str]) -> Dict[str, List[str]]:
    """Classify files into dependency layers by directory name."""
    infra_kw = {"models", "schemas", "db", "database", "config", "core",
                "utils", "common", "constants", "types", "enums"}
    business_kw = {"services", "handlers", "workers", "tasks", "processors",
                   "managers", "engines", "pipelines"}
    interface_kw = {"api", "routes", "views", "endpoints", "controllers",
                    "middleware", "routers"}

    layers: Dict[str, List[str]] = {
        "infrastructure": [], "business": [], "interface": [], "other": [],
    }

    for fp in file_scope:
        parts = fp.replace("\\", "/").lower().split("/")
        parent = parts[-2] if len(parts) >= 2 else ""

        if parent in infra_kw or any(kw in fp.lower() for kw in ["__init__", "constants", "config"]):
            layers["infrastructure"].append(fp)
        elif parent in business_kw:
            layers["business"].append(fp)
        elif parent in interface_kw:
            layers["interface"].append(fp)
        else:
            layers["other"].append(fp)

    return layers


def _save_plan(plan: ConstructionPlan, job_dir: str) -> None:
    """Save construction plan to job directory."""
    path = os.path.join(job_dir, "construction_plan.json")
    try:
        os.makedirs(job_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(plan.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("[construction_planner] Plan saved to %s", path)
    except Exception as exc:
        logger.warning("[construction_planner] Failed to save plan: %s", exc)


def load_construction_plan(job_dir: str) -> Optional[ConstructionPlan]:
    """Load a previously saved construction plan."""
    path = os.path.join(job_dir, "construction_plan.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return ConstructionPlan.from_dict(json.load(f))
    except Exception as exc:
        logger.warning("[construction_planner] Failed to load plan: %s", exc)
        return None
