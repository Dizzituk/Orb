import json
import logging
import os
from app.orchestrator.__segment_loop_utils_3_utils import _ARCH_EXECUTOR_AVAILABLE, _CRITICAL_PIPELINE_AVAILABLE, _RECONCILIATION_AVAILABLE
from app.orchestrator._segment_loop_utils import _clear_stale_arch_versions
from app.orchestrator._segment_loop_utils import _is_facade_segment, _save_execution_trace
from app.orchestrator.segment_state import get_job_dir, load_or_init_state
from app.pot_spec.grounded.segment_schemas import SegmentSpec
from typing import Any, Dict, List
from typing import Callable, Optional
from app.orchestrator.___segment_loop_utils_3_utils_1_utils import run_segment_through_pipeline
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
ProgressCallback = Optional[Callable[[str], None]]
