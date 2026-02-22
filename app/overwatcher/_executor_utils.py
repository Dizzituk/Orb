import logging
import os
import re
from typing import Any, Dict, List
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


SANDBOX_ROOT = os.getenv("ORB_SANDBOX_ROOT", "D:\\SandboxOrb")

IMPLEMENTER_MAX_OUTPUT_TOKENS = int(os.getenv("ORB_IMPLEMENTER_MAX_OUTPUT_TOKENS", "16000"))

FILE_HEADER_PATTERN = re.compile(r'^#\s*FILE:\s*(.+?)\s*$', re.MULTILINE)

CODE_BLOCK_PATTERN = re.compile(r'```(?:\w+)?\n(.*?)```', re.DOTALL)

def emit_chunk_implemented(
    job_artifact_root: str,
    job_id: str,
    chunk_id: str,
    files_added: List[str],
    files_modified: List[str],
    model: str,
) -> None:
    """Emit chunk implementation event."""
    logger.info(f"[executor] Event: chunk_implemented {chunk_id}")

def emit_boundary_violation(
    job_artifact_root: str,
    job_id: str,
    chunk_id: str,
    violations: List[Dict[str, Any]],
) -> None:
    """Emit boundary violation event."""
    logger.warning(f"[executor] Event: boundary_violation {chunk_id}: {violations}")

def emit_stage_failed(
    job_artifact_root: str,
    job_id: str,
    stage_id: str,
    error_type: str,
    error_message: str,
) -> None:
    """Emit stage failure event."""
    logger.error(f"[executor] Event: stage_failed {stage_id}: {error_type} - {error_message}")

def emit_provider_fallback(
    job_artifact_root: str,
    job_id: str,
    from_provider: str,
    from_model: str,
    to_provider: str,
    to_model: str,
    reason: str,
) -> None:
    """Emit provider fallback event."""
    logger.info(f"[executor] Event: provider_fallback {from_provider}/{from_model} -> {to_provider}/{to_model}")
