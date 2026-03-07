# FILE: app/agentic_pipeline/comparison_runner.py
"""
Phase 3: Parallel Comparison Runner.

Runs the agentic pipeline alongside the existing pipeline on the same
job and compares results. The existing pipeline runs as normal (it's
the production path). The agentic pipeline runs in parallel for
comparison only — its outputs are NOT written to the live sandbox.

Comparison metrics:
  - Output quality: do arch docs match or exceed current quality?
  - Cost: total tokens/calls (agentic ~4 calls vs existing ~61)
  - Cycle count: how many check cycles needed inside the loop?
  - False positive rate: did any HARD_BLOCK check incorrectly block?

v1.0 (2026-03-05): Initial implementation.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Results from running both pipelines on the same job."""
    job_id: str = ""
    segment_count: int = 0

    # Agentic pipeline results
    agentic_success: bool = False
    agentic_llm_calls: int = 0
    agentic_duration_seconds: float = 0.0
    agentic_check_cycles: int = 0
    agentic_hard_blocks: int = 0
    agentic_warnings: int = 0
    agentic_arch_doc_count: int = 0
    agentic_confidence: float = 0.0
    agentic_errors: List[str] = field(default_factory=list)

    # Existing pipeline (from state file — it runs independently)
    existing_success: Optional[bool] = None
    existing_segments_completed: int = 0
    existing_segments_failed: int = 0

    # Comparison
    comparison_notes: List[str] = field(default_factory=list)

    def to_report(self) -> str:
        """Generate a human-readable comparison report."""
        lines = [
            f"# Pipeline Comparison Report — {self.job_id}",
            f"Segments: {self.segment_count}",
            "",
            "## Agentic Pipeline",
            f"  Success: {self.agentic_success}",
            f"  LLM calls: {self.agentic_llm_calls}",
            f"  Duration: {self.agentic_duration_seconds:.1f}s",
            f"  Check cycles: {self.agentic_check_cycles}",
            f"  Hard blocks (final): {self.agentic_hard_blocks}",
            f"  Warnings (final): {self.agentic_warnings}",
            f"  Arch docs produced: {self.agentic_arch_doc_count}",
            f"  Confidence: {self.agentic_confidence:.2f}",
        ]
        if self.agentic_errors:
            lines.append(f"  Errors: {self.agentic_errors}")

        lines.extend([
            "",
            "## Existing Pipeline",
            f"  Success: {self.existing_success}",
            f"  Segments completed: {self.existing_segments_completed}",
            f"  Segments failed: {self.existing_segments_failed}",
        ])

        if self.comparison_notes:
            lines.extend(["", "## Notes"])
            for note in self.comparison_notes:
                lines.append(f"  - {note}")

        return "\n".join(lines)


async def run_agentic_comparison(
    job_id: str,
    manifest_path: str,
    job_dir: str,
    llm_call_fn: Callable,
    on_progress: Optional[Callable[[str], None]] = None,
) -> ComparisonResult:
    """Run the agentic pipeline for comparison against the existing pipeline.

    The existing pipeline is NOT run by this function — it runs independently
    through the normal segment_loop path. This function only runs the agentic
    pipeline and reads the existing pipeline's state for comparison.

    Args:
        job_id: Job identifier.
        manifest_path: Path to the segment manifest.json.
        job_dir: Path to the job directory.
        llm_call_fn: Async LLM call function.
        on_progress: Optional progress callback.
    """
    result = ComparisonResult(job_id=job_id)

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        logger.info("[comparison] %s", msg)

    _progress(f"Starting agentic comparison for job {job_id}")

    # --- Load manifest and skeleton contract ---
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        result.agentic_errors.append(f"Failed to load manifest: {e}")
        return result

    result.segment_count = len(manifest.get("segments", []))

    # Load skeleton contract
    skeleton_path = os.path.join(os.path.dirname(manifest_path), "skeleton_contract.json")
    skeleton_contract = {}
    try:
        if os.path.isfile(skeleton_path):
            with open(skeleton_path, "r", encoding="utf-8") as f:
                skeleton_contract = json.load(f)
    except Exception as e:
        _progress(f"Warning: skeleton contract load failed: {e}")

    # --- Run agentic pipeline ---
    _progress("Running agentic pipeline...")
    t0 = time.time()

    try:
        from app.agentic_pipeline.pipeline import run_agentic_pipeline

        pipeline_result = await run_agentic_pipeline(
            job_id=job_id,
            manifest=manifest,
            skeleton_contract=skeleton_contract,
            job_dir=job_dir,
            llm_call_fn=llm_call_fn,
            sandbox_client=None,  # Don't write to sandbox — comparison only
            on_progress=on_progress,
        )

        result.agentic_success = pipeline_result.success
        result.agentic_llm_calls = pipeline_result.total_llm_calls
        result.agentic_duration_seconds = pipeline_result.total_duration_seconds
        result.agentic_arch_doc_count = len(pipeline_result.arch_docs)
        result.agentic_errors = pipeline_result.errors

        # Note: check_cycles, hard_blocks, warnings, confidence are
        # per-batch metrics from the loop_controller. The pipeline_result
        # doesn't surface these directly — they'd need to be added to
        # PipelineResult if we want them in comparison reports.

    except Exception as e:
        result.agentic_errors.append(f"Agentic pipeline crashed: {e}")
        result.agentic_duration_seconds = time.time() - t0
        logger.error("[comparison] Agentic pipeline failed: %s", e, exc_info=True)

    # --- Read existing pipeline state for comparison ---
    state_path = os.path.join(job_dir, "state.json")
    try:
        if os.path.isfile(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            segments = state.get("segments", {})
            result.existing_segments_completed = sum(
                1 for s in segments.values()
                if s.get("status") in ("completed", "approved")
            )
            result.existing_segments_failed = sum(
                1 for s in segments.values()
                if s.get("status") == "failed"
            )
            result.existing_success = (
                result.existing_segments_completed == len(segments)
                and result.existing_segments_failed == 0
            )
    except Exception as e:
        _progress(f"Could not read existing pipeline state: {e}")

    # --- Save comparison report ---
    report = result.to_report()
    report_path = os.path.join(job_dir, "agentic_comparison.md")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        _progress(f"Comparison report saved to {report_path}")
    except Exception as e:
        _progress(f"Could not save comparison report: {e}")

    _progress(f"Comparison complete: agentic={'OK' if result.agentic_success else 'FAIL'}, "
              f"existing={'OK' if result.existing_success else 'FAIL/pending'}")

    return result


async def run_comparison_on_latest_job(
    llm_call_fn: Callable,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Optional[ComparisonResult]:
    """Convenience: find the latest job with a manifest and run comparison."""
    jobs_root = os.path.join("D:\\Orb", "jobs", "jobs")
    if not os.path.isdir(jobs_root):
        logger.warning("[comparison] Jobs root not found: %s", jobs_root)
        return None

    # Find latest job with a manifest
    candidates = []
    for name in sorted(os.listdir(jobs_root), reverse=True):
        job_dir = os.path.join(jobs_root, name)
        if not os.path.isdir(job_dir):
            continue
        if name.startswith("sg-") and "__" not in name:
            manifest_path = os.path.join(job_dir, "segments", "manifest.json")
            if os.path.isfile(manifest_path):
                candidates.append((name, job_dir, manifest_path))
        if len(candidates) >= 1:
            break

    if not candidates:
        logger.warning("[comparison] No jobs with manifests found")
        return None

    job_id, job_dir, manifest_path = candidates[0]
    return await run_agentic_comparison(
        job_id=job_id,
        manifest_path=manifest_path,
        job_dir=job_dir,
        llm_call_fn=llm_call_fn,
        on_progress=on_progress,
    )
