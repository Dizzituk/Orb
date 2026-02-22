# FILE: app/pot_spec/spec_gate_persistence.py
"""
Spec Gate v2 - Persistence Layer (v2.3.2)

Contains:
- Filesystem artifact writing
- Database persistence
- Spec schema building
- Markdown generation

v2.3.2 (2026-01-30): CRITICAL FIX - Always persist multi-target files
    - Changed condition from `if is_multi_target_read` to `if is_multi_target_read or multi_target_files`
    - Persists multi_target_files if they EXIST, regardless of flag state
    - Sets is_multi_target_read=True if files exist (defensive fix)
    - Changed logger.info to print() for terminal visibility
    - Fixes data loss when flag doesn't propagate but file data does

v2.3.1 (2026-01-30): DIAGNOSTIC - Added logging for multi-target status
    - Added diagnostic logging for is_multi_target_read and multi_target_files_count
    - Logs grounding_data keys for debugging

v2.3 (2026-01-30): CRITICAL FIX - Persist multi-target read fields
    - Added is_multi_target_read to spec_data for Critical Pipeline
    - Added multi_target_files to spec_data for Critical Pipeline
    - Fixes MICRO-CHECK-001 where quickcheck couldn't see multi-target data
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session
from app.pot_spec._spec_gate_persistence_utils_2 import SPEC_GATE_PERSISTENCE_BUILD_ID, artifact_root, build_spec_schema, build_spot_markdown, compute_spec_hash, job_dir, persist_spec, safe_summary_from_objective

logger = logging.getLogger(__name__)

# =============================================================================
# v2.3 BUILD VERIFICATION
# =============================================================================
print(f"[SPEC_GATE_PERSISTENCE_LOADED] BUILD_ID={SPEC_GATE_PERSISTENCE_BUILD_ID}")


# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    from app.specs import service as specs_service
except Exception:
    specs_service = None

try:
    from app.specs.schema import Spec as SpecSchema
    from app.specs.schema import SpecProvenance, SpecStatus
except Exception:
    SpecSchema = None
    SpecProvenance = None
    SpecStatus = None


# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------


def write_spec_artifacts(
    *,
    job_id: str,
    spec_version: int,
    spec_payload: Dict[str, Any],
    spot_markdown: str,
    spec_hash: str,
) -> Tuple[bool, Optional[str]]:
    """Write spec artifacts to filesystem."""
    try:
        job_root = artifact_root()
        job_path = job_dir(job_root, job_id)
        spec_dir = os.path.join(job_path, "spec")
        os.makedirs(spec_dir, exist_ok=True)

        json_path = os.path.join(spec_dir, f"spec_v{spec_version}.json")
        md_path = os.path.join(spec_dir, f"spec_v{spec_version}.md")
        sha_path = os.path.join(spec_dir, f"spec_v{spec_version}.sha256")

        with open(json_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(spec_payload, f, ensure_ascii=False, indent=2)

        with open(md_path, "w", encoding="utf-8", newline="\n") as f:
            f.write((spot_markdown or "").rstrip() + "\n")

        with open(sha_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(spec_hash + "\n")

        logger.info("[spec_gate_persistence] Wrote artifacts to %s", spec_dir)
        return True, None
    except Exception as e:
        logger.exception("[spec_gate_persistence] Failed to write artifacts: %s", e)
        return False, str(e)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Database persistence
# ---------------------------------------------------------------------------


def update_spec_status(
    db: Session,
    spec_id: str,
    *,
    provider_id: str,
    model_id: str,
) -> Tuple[bool, Optional[str]]:
    """Update spec status to validated."""
    if specs_service is None or not hasattr(specs_service, "update_spec_status"):
        return False, "specs_service.update_spec_status not available"

    try:
        validated_status = "validated"
        if SpecStatus is not None:
            try:
                validated_status = SpecStatus.VALIDATED.value
            except Exception:
                pass

        triggered_by = f"spec_gate_v2:{provider_id}/{model_id}"

        specs_service.update_spec_status(
            db=db,
            spec_id=spec_id,
            new_status=validated_status,
            validation_result={"source": "spec_gate_v2", "finalized": True},
            triggered_by=triggered_by,
        )
        logger.info("[spec_gate_persistence] Updated spec status: %s", spec_id)
        return True, None

    except Exception as e:
        logger.warning("[spec_gate_persistence] Failed to update status: %s", e)
        return False, str(e)


__all__ = [
    "artifact_root",
    "job_dir",
    "write_spec_artifacts",
    "compute_spec_hash",
    "build_spot_markdown",
    "safe_summary_from_objective",
    "build_spec_schema",
    "persist_spec",
    "update_spec_status",
]