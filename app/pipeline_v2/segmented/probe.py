# FILE: app/pipeline_v2/segmented/probe.py
# Purpose: Derek phase 5 gate probe — a real two-file CLI toy built end-to-end on the segmented stand-ins.
# Called-by: app.endpoints.cost_dashboard (POST /api/cost/probe/segmented), scripts/derek/segmented_probe.py
# Depends-on: app.pipeline_v2.segmented, app.pipeline_v2.build_targets, app.cost.cost_report
# Last-renovated: 2026-07-04
"""
The trivial greenfield rehearsal: two files, one dependency edge.

seg-01-core exposes add(a, b); seg-02-cli consumes it. Real BUILDER_WORKER
stand-in models build it into data/derek_probe/toy-cli/ on the host lane;
the integrator verifies contracts + syntax; per-worker cost lands in the
ledger under job_id=derek-p5-probe. Run in-process (keys live there):

    curl -X POST http://localhost:8000/api/cost/probe/segmented -H "Authorization: Bearer <token>"

Cost: a few worker calls on HANDS-tier models — pennies.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

PROBE_JOB_ID = "derek-p5-probe"


def _probe_root() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "derek_probe" / "toy-cli"


def build_toy_manifest():
    from app.pot_spec.grounded.segment_schemas import SegmentSpec
    from app.pot_spec.grounded._segment_schemas_utils_1 import InterfaceContract, SegmentManifest

    seg1 = SegmentSpec(
        segment_id="seg-01-core", title="Core maths module",
        file_scope=["core.py"],
        requirements=[
            "Create core.py containing exactly one function: add(a, b) that returns a + b.",
            "Pure stdlib, no third-party imports.",
        ],
        acceptance_criteria=["core.py compiles", "add(2, 3) == 5"],
        exposes=InterfaceContract(class_names=[], method_signatures=["def add(a, b)"],
                                  endpoint_paths=[], export_names=["add"]),
    )
    seg2 = SegmentSpec(
        segment_id="seg-02-cli", title="CLI wrapper",
        file_scope=["cli.py"],
        dependencies=["seg-01-core"],
        requirements=[
            "Create cli.py: a command-line tool that takes two integer arguments "
            "and prints their sum using add() imported from core.",
            "Pure stdlib (sys or argparse).",
        ],
        acceptance_criteria=["cli.py compiles", "python cli.py 2 3 prints 5"],
        consumes=InterfaceContract(class_names=[], method_signatures=[],
                                   endpoint_paths=[], export_names=["add"]),
    )
    manifest = SegmentManifest(segments=[seg1, seg2])
    manifest.total_segments = 2
    manifest.total_files = 2
    return manifest


def _toy_profile(root: Path):
    from app.pipeline_v2.build_targets import BuildTargetProfile
    return BuildTargetProfile(
        project_id="derek-probe-toy",
        project_name="Derek Probe Toy CLI",
        project_root=str(root),
        language="python",
        build_system="pip",
        framework="cli",
        source_root="",
        package_name="",
        architecture_pattern="flat",
    )


async def run_toy_probe(clean: bool = True) -> Dict[str, Any]:
    """Run the two-file toy through the REAL segmented builder on the
    configured stand-ins. Returns build outcome + per-stage cost rollup."""
    from app.pipeline_v2.segmented import run_segmented_builder

    root = _probe_root()
    if clean and root.exists():
        shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    job_dir = str(root / "_job")
    os.makedirs(job_dir, exist_ok=True)

    events = []
    result = await run_segmented_builder(
        spec={"goal": "two-file CLI adder toy (Derek phase 5 probe)"},
        manifest=build_toy_manifest(),
        scaffold=None,
        job_dir=job_dir,
        job_id=PROBE_JOB_ID,
        on_progress=lambda m: events.append(m),
        profile=_toy_profile(root),
    )

    cost: Dict[str, Any] = {}
    try:
        from app.cost.cost_report import rollup
        cost = rollup(by="stage", days=1, job_id=PROBE_JOB_ID)
    except Exception as exc:
        cost = {"error": str(exc)}

    files = {}
    for name in ("core.py", "cli.py"):
        p = root / name
        files[name] = p.read_text(encoding="utf-8")[:500] if p.exists() else None

    return {
        "green": bool(result.success),
        "job_id": PROBE_JOB_ID,
        "root": str(root),
        "files": files,
        "files_written": result.all_files_written,
        "tool_calls": result.total_tool_calls,
        "tokens": result.total_input_tokens + result.total_output_tokens,
        "errors": result.errors,
        "cost_by_stage": cost.get("buckets", cost),
        "events_tail": events[-15:],
    }


__all__ = ["run_toy_probe", "build_toy_manifest", "PROBE_JOB_ID"]
