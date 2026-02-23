# FILE: app/orchestrator/seg_pipeline_step0.py
"""Step 0.5: Load previous implementation failure feedback."""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, Any

from app.orchestrator.segment_state import get_job_dir

logger = logging.getLogger(__name__)


def load_failure_feedback(
    seg_id: str,
    job_id: str,
    segment_context: Dict[str, Any],
    emit: Any,
) -> None:
    """
    v5.25: Load previous implementation failure feedback (if any).

    When a previous attempt failed at the Implementer stage, the execution
    trace contains the exact strike errors. Inject these into segment_context
    so the Critical Pipeline can avoid producing architectures that cause
    the same implementation failures.

    Mutates segment_context in place.
    """
    try:
        prev_trace_path = os.path.join(
            get_job_dir(job_id), "segments", seg_id, "execution_trace", "trace.json",
        )
        if not os.path.isfile(prev_trace_path):
            return

        with open(prev_trace_path, "r", encoding="utf-8") as tf:
            prev_trace = json.load(tf)

        if prev_trace.get("success", True):
            return

        feedback_parts = []
        feedback_parts.append(f"Overall error: {prev_trace.get('error', 'Unknown')}")

        for evt in prev_trace.get("trace_events", []):
            if evt.get("stage", "") in (
                "FILE_TASK_STRIKE", "FILE_TASK_FAILED",
                "JOB_CHECK_FAIL", "SIGNATURE_CHECK_FAIL",
            ):
                det = evt.get("details", {})
                path = det.get("path", "")
                err = det.get("error", det.get("last_error", ""))
                if err:
                    feedback_parts.append(
                        f"- [{evt['stage']}] {path}: {err[:300]}"
                    )

        if len(feedback_parts) > 1:
            impl_feedback = "\n".join(feedback_parts)
            segment_context["implementation_feedback"] = impl_feedback
            emit(
                f"  📊 Loaded previous implementation failure feedback "
                f"({len(feedback_parts)-1} issue(s))"
            )
            logger.info(
                "[SEGMENT_LOOP] v5.25 Implementation feedback loaded for %s: %d issue(s)",
                seg_id, len(feedback_parts) - 1,
            )
    except Exception as fb_err:
        logger.warning(
            "[SEGMENT_LOOP] v5.25 Failed to load implementation feedback (non-fatal): %s",
            fb_err,
        )
