import logging
import os
import re
import time
from ..sandbox_client import SandboxClient, get_sandbox_client
from ..spec_resolution import ResolvedSpec
from .constants import IMPLEMENTER_MAX_TOKENS, MAX_STRIKES_PER_TASK, MODIFY_EDIT_MODE_THRESHOLD
from .context import _extract_existing_imports, _extract_file_interfaces, _extract_router_registrations, _format_job_context, _read_existing_file, _read_source_context
from .helpers import _check_python_syntax, _extract_llm_content, _sanitise_python_content, _strip_markdown_fences
from .parsing import _extract_verbatim_code_from_architecture, extract_section_for_file, parse_file_inventory
from .path_resolution import _ensure_python_init_files, _infer_lang_from_path, _resolve_multi_root_path
from .process_boot_strike_loop import BOOT_MAX_STRIKES, _parse_broken_file_from_traceback, _run_boot_check
from .prompts import IMPLEMENTER_MODIFY_EDIT_SYSTEM, IMPLEMENTER_MODIFY_FILE_SYSTEM, IMPLEMENTER_NEW_FILE_SYSTEM, _parse_edit_pairs
from .sandbox_ops import _resolve_sandbox_base, _verify_file_via_sandbox
from .source_extraction import _detect_source_files_from_architecture
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from app.overwatcher.architecture_executor.___orchestrator_utils_utils_utils import run_architecture_execution
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
