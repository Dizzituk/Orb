# FILE: app/pipeline_v2/stages/architect.py
"""
Stage 3 — The Architect.

Takes the verified spec from SpecGate and produces a layered build plan.
It does NOT write code. It thinks like a project architect:
  - What's the file structure?
  - What gets built first?
  - What depends on what?
  - How does the project layer up from foundation to finished product?

Uses GPT-5.4 for structural reasoning with large context.
Has tool access to read the existing codebase from the sandbox.

v1.0 (2026-03-06): Initial implementation for ASTRA v2.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

from app.pipeline_v2.config import ARCHITECT_PROVIDER, ARCHITECT_MODEL, ARCHITECT_MAX_OUTPUT
from app.pipeline_v2.models import BuildLayer, BuildPlan

logger = logging.getLogger(__name__)

ARCHITECT_SYSTEM = """You are ASTRA's Architect. You plan builds. You NEVER write code.

You receive a verified specification and produce a layered build plan.
Think like a project architect:
- What files need to exist?
- What order do they get built in?
- What depends on what?
- How does the project layer up from foundation to finished product?

OUTPUT FORMAT (JSON):

{
  "layers": [
    {
      "tier": 0,
      "description": "Foundation — data models and base configuration",
      "segments": [
        {
          "segment_id": "seg-01-models",
          "title": "Core Data Models",
          "file_scope": ["app/models.py"],
          "action": "CREATE or MODIFY",
          "dependencies": [],
          "requirements": ["Define User and Project models"]
        }
      ]
    },
    {
      "tier": 1,
      "description": "Services — business logic built on the foundation",
      "segments": [
        {
          "segment_id": "seg-02-crud",
          "title": "CRUD Operations",
          "file_scope": ["app/crud.py"],
          "action": "CREATE",
          "dependencies": ["seg-01-models"],
          "requirements": ["CRUD for User and Project"]
        }
      ]
    }
  ]
}

RULES:
- Tier 0 segments have NO dependencies.
- Tier N segments depend ONLY on segments from tier 0..N-1.
- Each segment's file_scope lists the exact files it produces.
- Keep segments focused: 1-3 files per segment.
- Order within a tier doesn't matter (they can be built in parallel).
- Use existing file paths from the codebase. Don't invent new structures.
"""


async def run_architect(
    spec: Dict[str, Any],
    manifest: Dict[str, Any],
    job_dir: str,
    on_progress: Optional[Callable[[str], None]] = None,
) -> BuildPlan:
    """Run the Architect stage.

    Takes the spec and manifest from SpecGate, reads the codebase,
    and produces a layered BuildPlan.
    """
    emit = on_progress or (lambda msg: None)
    t_start = time.time()

    emit("📐 Architect: Planning the build...")

    job_id = manifest.get("parent_spec_id", os.path.basename(job_dir))
    segments_data = manifest.get("segments", [])

    # --- Gather evidence from sandbox ---
    emit("   Reading existing codebase...")
    from app.agentic_pipeline.preflight_evidence import (
        gather_preflight_evidence, format_preflight_for_prompt,
    )
    all_files = list(set(
        f for s in segments_data for f in s.get("file_scope", [])
    ))
    preflight = gather_preflight_evidence(all_files)

    # --- Generate skeleton contracts ---
    emit("   Generating skeleton contracts...")
    skeleton_contract: Dict[str, Any] = {}
    try:
        from app.orchestrator.skeleton_contracts import generate_skeleton_contract
        skel = generate_skeleton_contract(manifest, job_id)
        if skel and skel.skeletons:
            skeleton_contract = {
                "skeletons": [
                    s.to_dict() if hasattr(s, "to_dict") else vars(s)
                    for s in skel.skeletons
                ]
            }
            emit(f"   Generated {len(skel.skeletons)} skeleton contracts")
    except Exception as e:
        emit(f"   Skeleton generation failed (non-fatal): {e}")

    # --- Generate deterministic templates ---
    emit("   Generating deterministic templates...")
    templates: Dict[str, str] = {}
    try:
        from app.orchestrator.arch_template.engine import generate_architecture_template
        from app.agentic_pipeline.loop_controller import number_fill_markers
        skeletons_list = skeleton_contract.get("skeletons", [])
        for seg_data in segments_data:
            seg_id = seg_data.get("segment_id", "")
            seg_skeleton = next(
                (s for s in skeletons_list if s.get("segment_id") == seg_id),
                {},
            )
            try:
                tmpl = generate_architecture_template(
                    segment_spec=seg_data,
                    skeleton=seg_skeleton,
                    all_skeletons=skeletons_list,
                    requirements=seg_data.get("requirements", []),
                )
                if tmpl:
                    numbered, _ = number_fill_markers(tmpl, seg_id)
                    templates[seg_id] = numbered
            except Exception:
                pass
        emit(f"   Templates: {len(templates)}/{len(segments_data)}")
    except ImportError:
        emit("   Template engine not available")

    # --- Compute dependency tiers ---
    from app.agentic_pipeline.segment_printer import compute_dependency_tiers
    tiers = compute_dependency_tiers(segments_data)
    emit(f"   Dependency tiers: {len(tiers)}")

    # --- Build the plan ---
    layers: List[BuildLayer] = []
    for tier_idx, tier_segments in enumerate(tiers):
        layer = BuildLayer(
            tier=tier_idx,
            segment_ids=[s.get("segment_id", "") for s in tier_segments],
            file_scope=[
                f for s in tier_segments for f in s.get("file_scope", [])
            ],
            description=f"Tier {tier_idx}: {len(tier_segments)} segment(s)",
            dependencies=[
                d for s in tier_segments
                for d in s.get("dependencies", [])
            ],
        )
        layers.append(layer)
        emit(f"   Layer {tier_idx}: {layer.segment_ids}")

    plan = BuildPlan(
        job_id=job_id,
        layers=layers,
        total_files=len(all_files),
        total_segments=len(segments_data),
        templates=templates,
        skeleton_contract=skeleton_contract,
    )

    duration = time.time() - t_start
    emit(f"📐 Architect complete: {plan.layer_count} layers, "
         f"{plan.total_segments} segments, {plan.total_files} files ({duration:.1f}s)")

    # Save plan to disk
    _save_build_plan(plan, manifest, job_dir)

    return plan


def _save_build_plan(
    plan: BuildPlan, manifest: Dict[str, Any], job_dir: str,
) -> None:
    """Save the build plan to the job directory."""
    os.makedirs(job_dir, exist_ok=True)
    plan_path = os.path.join(job_dir, "build_plan.json")
    try:
        plan_data = {
            "job_id": plan.job_id,
            "layer_count": plan.layer_count,
            "total_files": plan.total_files,
            "total_segments": plan.total_segments,
            "layers": [
                {
                    "tier": layer.tier,
                    "segment_ids": layer.segment_ids,
                    "file_scope": layer.file_scope,
                    "description": layer.description,
                    "dependencies": layer.dependencies,
                }
                for layer in plan.layers
            ],
        }
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan_data, f, indent=2)
        logger.info("[architect] Saved build plan: %s", plan_path)
    except Exception as e:
        logger.error("[architect] Failed to save build plan: %s", e)
