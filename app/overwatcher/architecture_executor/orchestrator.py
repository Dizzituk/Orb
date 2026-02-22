"""
Main orchestrator for architecture execution.

Contains run_architecture_execution() — the primary entry point that
supervises the full architecture-to-code pipeline:
1. Parses architecture documents for file operations
2. Calls the Implementer LLM to generate file content
3. Delegates writes via run_implementer_task()
4. Verifies results independently via sandbox reads
5. Implements three-strike error handling per task

Extracted from the original architecture_executor.py monolith.
All utility functions are imported from sibling modules.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..spec_resolution import ResolvedSpec
from ..sandbox_client import SandboxClient, get_sandbox_client

from .constants import (
from app.overwatcher.architecture_executor._orchestrator_utils import run_architecture_execution
    ARCHITECTURE_EXECUTOR_BUILD_ID,
    MAX_STRIKES_PER_TASK,
    IMPLEMENTER_MAX_TOKENS,
    VERIFY_READ_TIMEOUT,
    SOURCE_CONTEXT_MAX_CHARS,
    MODIFY_EDIT_MODE_THRESHOLD,
    INTERFACE_SUMMARY_MAX_CHARS,
)
from .prompts import (
    IMPLEMENTER_NEW_FILE_SYSTEM,
    IMPLEMENTER_MODIFY_FILE_SYSTEM,
    IMPLEMENTER_MODIFY_EDIT_SYSTEM,
    _parse_edit_pairs,
)
from .parsing import (
    parse_file_inventory,
    extract_section_for_file,
    _extract_verbatim_code_from_architecture,
)
from .context import (
    _read_existing_file,
    _read_source_context,
    _format_job_context,
    _extract_file_interfaces,
    _extract_existing_imports,
    _extract_router_registrations,
    _build_resolved_endpoints,
)
from .helpers import _extract_llm_content, _strip_markdown_fences, _sanitise_python_content, _check_python_syntax
from .sandbox_ops import _verify_file_via_sandbox, _resolve_sandbox_base
from .path_resolution import _resolve_multi_root_path, _ensure_python_init_files, _infer_lang_from_path
from .source_extraction import _detect_source_files_from_architecture
from .process_boot_strike_loop import BOOT_MAX_STRIKES, _run_boot_check, _parse_broken_file_from_traceback

logger = logging.getLogger(__name__)


__all__ = [
    "run_architecture_execution",
    "parse_file_inventory",
    "extract_section_for_file",
    "_extract_file_interfaces",
    "_extract_existing_imports",
    "_extract_router_registrations",
    "_build_resolved_endpoints",
    "_format_job_context",
    "_ensure_python_init_files",
    "ARCHITECTURE_EXECUTOR_BUILD_ID",
]
