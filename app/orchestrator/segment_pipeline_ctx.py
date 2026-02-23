# FILE: app/orchestrator/segment_pipeline_ctx.py
"""
Shared context objects for the decomposed segment loop.

PipelineCtx carries state through the per-segment pipeline stages.
JobCtx carries state through the multi-segment job orchestration.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


ProgressCallback = Optional[Callable[[str], None]]


@dataclass
class PipelineCtx:
    """State threaded through run_segment_through_pipeline sub-stages."""

    # --- Inputs (set once, read-only during pipeline) ---
    segment: Any                        # SegmentSpec
    segment_context: Dict[str, Any]
    job_id: str
    db: Any
    project_id: int
    on_progress: ProgressCallback = None
    contract_set: Any = None
    job_dir_path: str = ""
    manifest: Any = None
    parent_spec: Any = None
    quarantine_result: Any = None

    # --- Derived (set in init, reused) ---
    seg_id: str = ""
    seg_job_id: str = ""
    emit: Callable = field(default=lambda msg: None)

    # --- Mutable state (written by stages, read by later stages) ---
    arch_text: str = ""
    critique_passed: bool = False
    is_deterministic: bool = False
    result: Dict[str, Any] = field(default_factory=lambda: {
        "success": False,
        "output_files": [],
        "error": None,
        "critique_warnings": [],
    })

    def __post_init__(self):
        self.seg_id = self.segment.segment_id if self.segment else ""
        self.seg_job_id = f"{self.job_id}__{self.seg_id}"
        self.emit = self.on_progress or (lambda msg: None)
        self.is_deterministic = self.segment_context.get(
            "segment_spec", {}
        ).get("deterministic_refactor", False)


@dataclass
class JobCtx:
    """State threaded through run_segmented_job sub-stages."""

    # --- Inputs ---
    job_id: str
    manifest_path: str
    parent_spec: dict
    db: Any = None
    project_id: int = 0
    on_progress: ProgressCallback = None
    implement_only: bool = False

    # --- Derived ---
    emit: Callable = field(default=lambda msg: None)
    job_dir_path: str = ""

    # --- Loaded during init stages ---
    manifest: Any = None                # SegmentManifest
    state: Any = None                   # JobState
    contract_set: Any = None
    source_evidence: Dict[str, str] = field(default_factory=dict)
    enrichment_data: Dict[str, Any] = field(default_factory=dict)
    ledger: Any = None
    quarantine_result: Any = None
    execution_order: List[str] = field(default_factory=list)

    # --- Cohesion tracking ---
    cohesion_passed: bool = False
    cohesion_halted: bool = False
    cohesion_retry_count: int = 0

    def __post_init__(self):
        self.emit = self.on_progress or (lambda msg: None)
        from app.orchestrator.segment_state import get_job_dir
        self.job_dir_path = get_job_dir(self.job_id)
