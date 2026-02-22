import asyncio
import json
import logging
from ..config import CODE_SCAN_ROOTS, FULL_ARCHMAP_OUTPUT_DIR, FULL_ARCHMAP_OUTPUT_FILE, FULL_CODEBASE_OUTPUT_FILE, MAX_CONTENT_FILE_SIZE, SANDBOX_CONTROLLER_URL
from ..db_ops import save_scan_with_contents_to_db
from ..rag_helpers import generate_codebase_md, generate_index_for_rag, generate_signatures_json, signatures_to_db
from ..sandbox_client import call_fs_contents, call_fs_tree
from ..sse import sse_done, sse_error, sse_token
from .archmap_db import _build_db_archmap_prompt
from app.llm.audit_logger import RoutingTrace
from app.llm.local_tools.archmap_helpers import ARCHMAP_MAX_CONTINUATION_ROUNDS, ARCHMAP_MODEL, ARCHMAP_PROVIDER, ARCHMAP_SYSTEM_PROMPT, build_continuation_prompt, has_sentinel
from app.memory import memory_schemas
from app.memory import memory_service
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy.orm import Session
from typing import Any, AsyncGenerator, Dict, List, Optional
from app.llm.local_tools.zobie.streams.__archmap_full_utils_utils import _STAGE_MODELS_AVAILABLE
from app.llm.local_tools.zobie.streams.__archmap_full_utils_utils import generate_full_architecture_map_stream
logger = logging.getLogger(__name__)
get_archmap_config = None
logger = logging.getLogger(__name__)
