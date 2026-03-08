# FILE: app/orchestrator/seg_pipeline_step2.py
"""Step 2: Human Approval Gate (v3.0)."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from app.orchestrator.segment_state import get_job_dir

logger = logging.getLogger(__name__)


def check_approval_gate(
    seg_id: str,
    job_id: str,
    segment_context: Dict[str, Any],
    seg_arch_path: str,
    emit: Any,
) -> Optional[Dict[str, Any]]:
    """
    v3.0: Architecture is generated and critique-approved. STOP here and
    wait for explicit human approval before executing any writes.

    Returns a result dict if the segment should stop (awaiting approval),
    or None if execution should proceed.
    """
    auto_execute = os.getenv("ASTRA_SEGMENT_AUTO_EXECUTE", "0").strip()
    is_cohesion_regen = bool(
        segment_context and segment_context.get("cohesion_feedback")
    )
    is_facade_auto = bool(
        segment_context and segment_context.get("_facade_auto_execute")
    )

    if auto_execute != "1" and not is_cohesion_regen and not is_facade_auto:
        emit(f"  ⏸️ AWAITING APPROVAL: Architecture ready for {seg_id}")
        emit(
            f"  📄 Review: jobs/{os.path.basename(get_job_dir(job_id))}"
            f"/segments/{seg_id}/arch/arch_v1.md"
        )
        emit(f"  💡 To implement: say 'Astra, command: implement segments'")
        return {
            "success": True,
            "output_files": [],
            "error": None,
            "critique_warnings": [],
            "awaiting_approval": True,
            "architecture_path": seg_arch_path,
        }

    if is_facade_auto:
        emit(
            f"  🏗️ Facade auto-execute — bypassing approval gate (implement_only mode)"
        )
        logger.info("[SEGMENT_LOOP] v5.26 Facade approval bypass for %s", seg_id)

    if is_cohesion_regen:
        emit(
            f"  🧩 Cohesion regen — bypassing approval gate (was previously approved)"
        )
        logger.info("[SEGMENT_LOOP] v5.8 Cohesion regen bypass for %s", seg_id)

    return None  # Proceed to execution
