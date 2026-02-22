import asyncio
import logging
from app.llm._spec_gate_stream_utils import _USE_GROUNDED_SPEC_GATE, _get_weaver_vision_context_from_flow, _load_latest_weaver_spec_json, _resolve_spec_gate_model, _safe_json_event
from sqlalchemy.orm import Session
from typing import Any, AsyncGenerator, Optional
from app.llm.__spec_gate_stream_utils_1_utils import _FLOW_STATE_AVAILABLE, _SPEC_GATE_GROUNDED_AVAILABLE, _SPEC_GATE_V2_AVAILABLE, _SPEC_PERSISTENCE_AVAILABLE
from app.llm.__spec_gate_stream_utils_1_utils import generate_spec_gate_stream
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
run_spec_gate_v2 = None
SpecGateResult = None
run_spec_gate_grounded = None
persist_spec = None
build_spec_schema = None
safe_summary_from_objective = None
get_active_flow = None
advance_to_spec_gate_questions = None
advance_to_spec_validated = None
advance_to_spec_segmented = None
cancel_flow = None
get_active_job_for_project = None
specs_service = None
memory_service = None
memory_schemas = None
