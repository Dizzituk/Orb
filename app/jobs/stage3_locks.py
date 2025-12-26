# FILE: app/jobs/stage3_locks.py
"""Stage 3 spec-hash locks (refuse-on-mismatch).

Isolated helpers for:
- parsing SPEC_ID/SPEC_HASH header echo
- best-effort ledger event append
- best-effort artifact storage (raw output + meta json)

This keeps app/jobs/engine.py smaller without changing behavior.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def parse_spec_echo_headers(output: str, *, max_lines: int = 10) -> Tuple[Optional[str], Optional[str], str]:
    """Parse the 2-line header echo:

    SPEC_ID: <spec_id>
    SPEC_HASH: <spec_hash>

    Returns (returned_spec_id, returned_spec_hash, parse_note).

    Parsing is strict but tolerates leading BOM/whitespace/newlines (for header detection only).
    """
    if not output:
        return None, None, "empty_output"

    # For header detection only: strip UTF-8 BOM and leading whitespace/newlines.
    s = output
    if s.startswith("\ufeff"):
        s = s.lstrip("\ufeff")
    s = s.lstrip()

    lines = s.splitlines()

    # Take first two non-empty lines (within max_lines)
    non_empty: list[str] = []
    for line in lines[:max_lines]:
        if line.strip():
            non_empty.append(line)
            if len(non_empty) == 2:
                break

    if len(non_empty) < 2:
        return None, None, "missing_header_lines"

    line1 = non_empty[0].strip()
    line2 = non_empty[1].strip()

    if not line1.startswith("SPEC_ID:"):
        return None, None, "missing_SPEC_ID"
    if not line2.startswith("SPEC_HASH:"):
        return None, None, "missing_SPEC_HASH"

    returned_spec_id = line1[len("SPEC_ID:") :].strip() or None
    returned_spec_hash = line2[len("SPEC_HASH:") :].strip() or None

    if not returned_spec_id or not returned_spec_hash:
        return returned_spec_id, returned_spec_hash, "empty_spec_fields"

    return returned_spec_id, returned_spec_hash, "ok"


def write_stage3_artifacts(
    *,
    job_id: str,
    stage: str,
    raw_output: str,
    expected_spec_id: str,
    expected_spec_hash: str,
    returned_spec_id: Optional[str],
    returned_spec_hash: Optional[str],
    verified: bool,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, str]:
    """Write Stage 3 artifacts (raw output + meta JSON). Returns dict of paths written.

    Best-effort: failures here must not crash the job engine.
    """
    try:
        from app.pot_spec.service import get_job_artifact_root
    except Exception:
        return {}

    try:
        root = Path(get_job_artifact_root())
        stage_dir = root / "jobs" / job_id / "stages" / stage
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Choose extension for convenience (architecture maps are typically markdown)
        output_ext = ".md" if stage.lower().startswith("architecture") else ".txt"
        output_path = stage_dir / f"stage_output{output_ext}"
        meta_path = stage_dir / "stage_meta.json"

        # Raw output: store exactly as returned (including header)
        output_path.write_text(raw_output or "", encoding="utf-8", errors="replace")

        meta = {
            "expected_spec_id": expected_spec_id,
            "expected_spec_hash": expected_spec_hash,
            "returned_spec_id": returned_spec_id,
            "returned_spec_hash": returned_spec_hash,
            "verified": verified,
            "provider": provider,
            "model": model,
            "stored_raw_output": str(output_path),
            "raw_output_sha256": hashlib.sha256((raw_output or "").encode("utf-8", errors="replace")).hexdigest(),
            "stored_at_utc": datetime.utcnow().isoformat() + "Z",
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        return {"raw_output": str(output_path), "meta": str(meta_path)}
    except Exception as exc:
        logger.warning(f"[stage3] Failed to write Stage 3 artifacts: {exc}")
        return {}


def append_stage3_ledger_event(job_id: str, event: Dict[str, Any]) -> None:
    """Best-effort ledger append for Stage 3 events."""
    try:
        from app.pot_spec.ledger import append_event
        from app.pot_spec.service import get_job_artifact_root
    except Exception:
        return

    try:
        append_event(job_artifact_root=get_job_artifact_root(), job_id=job_id, event=event)
    except Exception as exc:
        logger.warning(f"[stage3] Failed to append ledger event: {exc}")
