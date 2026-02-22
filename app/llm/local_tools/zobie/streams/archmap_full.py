# FILE: app/llm/local_tools/zobie/streams/archmap_full.py
"""CREATE ARCHITECTURE MAP - FULL (ALL CAPS) stream generator.

Scan + out folder + map generation.

Extracted from zobie_tools.py for modularity.
No logic changes - exact same behavior and SSE output format.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from sqlalchemy.orm import Session

from app.llm.audit_logger import RoutingTrace
from app.memory import schemas as memory_schemas
from app.memory import service as memory_service

from app.llm.local_tools.archmap_helpers import (
from app.llm.local_tools.zobie.streams._archmap_full_utils import generate_full_architecture_map_stream
    ARCHMAP_PROVIDER,
    ARCHMAP_MODEL,
    ARCHMAP_SYSTEM_PROMPT,
    ARCHMAP_SENTINEL,
    ARCHMAP_MAX_CONTINUATION_ROUNDS,
    build_continuation_prompt,
    has_sentinel,
)

# Import stage config for proper max_tokens/timeout
try:
    from app.llm.stage_models import get_archmap_config
    _STAGE_MODELS_AVAILABLE = True
except ImportError:
    _STAGE_MODELS_AVAILABLE = False
    get_archmap_config = None

from ..config import (
    SANDBOX_CONTROLLER_URL,
    CODE_SCAN_ROOTS,
    MAX_CONTENT_FILE_SIZE,
    FULL_ARCHMAP_OUTPUT_DIR,
    FULL_ARCHMAP_OUTPUT_FILE,
    FULL_CODEBASE_OUTPUT_FILE,
)
from ..sse import sse_token, sse_error, sse_done
from ..sandbox_client import call_fs_tree, call_fs_contents
from ..db_ops import save_scan_with_contents_to_db
from ..rag_helpers import (
    generate_signatures_json,
    generate_index_for_rag,
    generate_codebase_md,
    signatures_to_db,
)

# Reuse the prompt builder from archmap_db
from .archmap_db import _build_db_archmap_prompt

logger = logging.getLogger(__name__)
