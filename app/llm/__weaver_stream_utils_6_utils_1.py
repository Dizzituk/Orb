import logging
import uuid
from app.llm.__weaver_stream_utils_6_utils import _FLOW_STATE_AVAILABLE, _MEMORY_AVAILABLE, _STREAMING_AVAILABLE
from app.llm._weaver_stream_utils import MICRO_TASK_SYSTEM_PROMPT, REFACTOR_TASK_SYSTEM_PROMPT, _enforce_design_pref_hygiene, _is_refactor_task, _is_vision_context
from app.llm._weaver_stream_utils import _enforce_deduplication, _format_execution_mode, _format_ramble, _get_blocking_questions, _get_weaver_config, _serialize_sse
from app.llm._weaver_stream_utils import _extract_meta_mode, _extract_vision_context, _gather_ramble_messages, _is_micro_file_task, _user_dismissed_questions
from app.llm._weaver_stream_utils import _get_streaming_function, _hash_message, _normalize_typos, _sanitize_weaver_output
from app.llm._weaver_stream_utils import _has_core_goal
from app.llm._weaver_stream_utils import _hash_messages
from sqlalchemy.orm import Session
from typing import Any, AsyncIterator, Dict, List, Optional, Set
from app.llm.___weaver_stream_utils_6_utils_1_utils import generate_weaver_stream
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
memory_service = None
memory_schemas = None
start_weaver_flow = None
clear_weaver_design_questions = None
save_confirmed_design_prefs = None
get_confirmed_design_prefs = None
save_weave_checkpoint = None
get_weave_checkpoint = None
save_woven_user_hashes = None
get_woven_user_hashes = None
