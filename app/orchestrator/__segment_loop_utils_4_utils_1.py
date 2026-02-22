import json
import logging
import os
from app.orchestrator.__segment_loop_utils_4_utils import _ARCH_EXECUTOR_AVAILABLE, _RECONCILIATION_AVAILABLE
from app.orchestrator._segment_loop_utils import _find_latest_arch, _load_source_file_evidence, collect_segment_outputs, is_segment_blocked
from app.orchestrator._segment_loop_utils import _is_facade_segment, _save_execution_trace, build_segment_context, can_execute_segment, mark_dependents_blocked, unblock_recovered_segments, verify_contracts_fulfilled
from app.orchestrator._segment_loop_utils import run_segment_through_pipeline
from app.orchestrator._segment_loop_utils import update_segment_status
from app.orchestrator.segment_state import JobState, SegmentState, get_job_dir, load_or_init_state, save_state
from app.pot_spec.grounded.segment_schemas import SegmentManifest, SegmentStatus
from typing import Any, Dict, List
from typing import Callable, Optional
from app.orchestrator.___segment_loop_utils_4_utils_1_utils import run_segmented_job
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
ProgressCallback = Optional[Callable[[str], None]]
