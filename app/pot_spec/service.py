# FILE: app/pot_spec/service.py
"""
PoT Spec service - spec creation, persistence, and verification.

v1.1 (2026-01): Fixed get_job_artifact_root() CWD sensitivity (was causing jobs/jobs/ double path)
"""
from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime
from typing import Any, Optional, Tuple
from uuid import uuid4

from sqlalchemy.orm import Session

from app.pot_spec.schemas import PoTSpec, PoTSpecPayload, Requirements, PoTSpecDraft
from app.pot_spec.models import PoTSpecRecord
from app.pot_spec.errors import SpecNotFoundError, SpecMismatchError
from app.pot_spec.ledger import append_event


# =============================================================================
# Artifact root - v1.1: Fixed CWD sensitivity
# =============================================================================

# v1.1: Cache the artifact root on first call to prevent CWD drift issues
_CACHED_JOB_ARTIFACT_ROOT: Optional[str] = None


def get_job_artifact_root() -> str:
    """Root folder for job artifacts.

    Default is ./jobs to match the Block 0/1 layout:
      jobs/<job_id>/spec/spec_vN.json
      jobs/<job_id>/ledger/events.ndjson
    
    v1.1: Now caches the result to prevent issues if CWD changes during execution.
          The path is computed relative to this file's location if not absolute.
    """
    global _CACHED_JOB_ARTIFACT_ROOT
    
    if _CACHED_JOB_ARTIFACT_ROOT is not None:
        return _CACHED_JOB_ARTIFACT_ROOT
    
    root = os.getenv("ORB_JOB_ARTIFACT_ROOT", "").strip()
    
    if root:
        # If env var is set, use it
        if os.path.isabs(root):
            _CACHED_JOB_ARTIFACT_ROOT = root
        else:
            # Relative path - resolve relative to app root (parent of app/pot_spec/)
            # This file is at app/pot_spec/service.py, so go up 2 levels
            this_file = os.path.abspath(__file__)
            app_root = os.path.dirname(os.path.dirname(os.path.dirname(this_file)))
            _CACHED_JOB_ARTIFACT_ROOT = os.path.join(app_root, root)
    else:
        # Default: "jobs" relative to app root
        this_file = os.path.abspath(__file__)
        app_root = os.path.dirname(os.path.dirname(os.path.dirname(this_file)))
        _CACHED_JOB_ARTIFACT_ROOT = os.path.join(app_root, "jobs")
    
    # Ensure the directory exists
    os.makedirs(_CACHED_JOB_ARTIFACT_ROOT, exist_ok=True)
    
    return _CACHED_JOB_ARTIFACT_ROOT


def _spec_file_path(job_artifact_root: str, job_id: str, spec_version: int) -> str:
    """Build path to spec file.
    
    v1.1: Added validation to prevent job_id containing path separators.
    """
    # v1.1: Sanitize job_id - strip any accidental path prefixes
    if job_id:
        # Remove any leading path components (e.g., "jobs/" prefix)
        job_id = os.path.basename(job_id.replace("\\", "/").rstrip("/"))
    
    spec_dir = os.path.join(job_artifact_root, job_id, "spec")
    os.makedirs(spec_dir, exist_ok=True)
    return os.path.join(spec_dir, f"spec_v{spec_version}.json")


# =============================================================================
# Canonicalization + hashing
# =============================================================================

_ORDER_INSENSITIVE_LIST_PATHS = {
    ("requirements", "must"),
    ("requirements", "should"),
    ("requirements", "can"),
    ("acceptance_tests",),
    ("open_questions",),
    ("recommendations",),
}


def _norm_str(s: str) -> str:
    # Safe normalization only (do not change meaning):
    # - normalize Windows line endings
    # - trim outer whitespace
    return (s or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _canonicalize(value: Any, path: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        return _norm_str(value)

    if isinstance(value, dict):
        out = {}
        for k in sorted(value.keys(), key=lambda x: str(x)):
            out[str(k)] = _canonicalize(value[k], path + (str(k),))
        return out

    if isinstance(value, list):
        items = [_canonicalize(v, path + ("[]",)) for v in value]
        if path in _ORDER_INSENSITIVE_LIST_PATHS and all(isinstance(x, str) for x in items):
            items = sorted(items)
        return items

    return value


def canonical_spec_bytes(payload_dict: dict[str, Any]) -> bytes:
    """Return canonical JSON bytes used for hashing and file write."""
    canon_obj = _canonicalize(payload_dict, tuple())
    text = json.dumps(canon_obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return text.encode("utf-8")


def hash_spec_bytes(canonical_json_bytes: bytes) -> str:
    return hashlib.sha256(canonical_json_bytes).hexdigest()



def get_latest_spec_record(db: Session, job_id: str) -> Optional[PoTSpecRecord]:
    return (
        db.query(PoTSpecRecord)
        .filter(PoTSpecRecord.job_id == job_id)
        .order_by(PoTSpecRecord.spec_version.desc())
        .first()
    )


def create_spec_from_draft(
    db: Session,
    job_id: str,
    draft: PoTSpecDraft,
    *,
    created_by_model: str,
    constraints_override: Optional[dict[str, Any]] = None,
    repo_snapshot: Optional[dict[str, Any]] = None,
) -> PoTSpec:
    """Create a new spec version from a Spec Gate draft and persist.

    Versioning rules:
    - If a prior spec exists for this job_id, increment spec_version and set parent_spec_id.
    - Otherwise, spec_version=1 and parent_spec_id=None.
    """
    latest = get_latest_spec_record(db, job_id)
    next_version = (latest.spec_version + 1) if latest else 1
    parent_spec_id = latest.spec_id if latest else None

    spec_id = str(uuid4())
    created_at = datetime.utcnow()

    payload = PoTSpecPayload(
        job_id=job_id,
        spec_id=spec_id,
        spec_version=next_version,
        parent_spec_id=parent_spec_id,
        created_at=created_at,
        created_by_model=created_by_model,
        goal=_norm_str(draft.goal),
        requirements=draft.requirements or Requirements(must=[], should=[], can=[]),
        constraints=(constraints_override if constraints_override is not None else (draft.constraints or {})),
        acceptance_tests=list(draft.acceptance_tests or []),
        open_questions=list(draft.open_questions or []),
        recommendations=list(draft.recommendations or []),
        repo_snapshot=repo_snapshot if repo_snapshot is not None else draft.repo_snapshot,
    )

    payload_json = payload.model_dump(mode="json")
    canonical_bytes = canonical_spec_bytes(payload_json)
    spec_hash = hash_spec_bytes(canonical_bytes)

    job_root = get_job_artifact_root()
    file_path = _spec_file_path(job_root, job_id, next_version)
    with open(file_path, "wb") as f:
        f.write(canonical_bytes)

    rec = PoTSpecRecord(
        spec_id=spec_id,
        job_id=job_id,
        spec_version=next_version,
        parent_spec_id=parent_spec_id,
        spec_hash=spec_hash,
        created_at=created_at,
        created_by_model=created_by_model,
        spec_json=payload_json,
        file_path=os.path.relpath(file_path, start=job_root),
    )
    db.add(rec)
    db.commit()

    append_event(
        job_artifact_root=job_root,
        job_id=job_id,
        event={
            "event": "SPEC_CREATED",
            "job_id": job_id,
            "spec_id": spec_id,
            "spec_version": next_version,
            "parent_spec_id": parent_spec_id,
            "spec_hash": spec_hash,
            "created_by_model": created_by_model,
            "inputs": {
                "draft_present": True,
                "open_questions_count": len(payload.open_questions or []),
                "repo_snapshot_present": bool(payload.repo_snapshot),
            },
            "outputs": {"spec_file": os.path.relpath(file_path, start=job_root)},
            "status": "ok",
        },
    )

    return PoTSpec(**payload.model_dump(), spec_hash=spec_hash)

# =============================================================================
# Create + persist
# =============================================================================

def create_spec_from_intent(
    db: Session,
    job_id: str,
    user_intent: str,
    *,
    created_by_model: str,
    constraints: Optional[dict[str, Any]] = None,
    repo_snapshot: Optional[dict[str, Any]] = None,
    spec_version: int = 1,
    parent_spec_id: Optional[str] = None,
) -> PoTSpec:
    """Create PoT Spec vN from user intent and persist to DB + file.

    Notes:
    - spec_hash is derived from the exact bytes written to spec_vN.json.
    - spec_vN.json contains ONLY the payload (no spec_hash) to avoid self-reference.
    """
    spec_id = str(uuid4())
    created_at = datetime.utcnow()

    payload = PoTSpecPayload(
        job_id=job_id,
        spec_id=spec_id,
        spec_version=spec_version,
        parent_spec_id=parent_spec_id,
        created_at=created_at,
        created_by_model=created_by_model,
        goal=_norm_str(user_intent),
        requirements=Requirements(must=[], should=[], can=[]),
        constraints=constraints or {},
        acceptance_tests=[],
        open_questions=[],
        recommendations=[],
        repo_snapshot=repo_snapshot,
    )

    payload_json = payload.model_dump(mode="json")
    canonical_bytes = canonical_spec_bytes(payload_json)
    spec_hash = hash_spec_bytes(canonical_bytes)

    # Persist file (exact bytes used for hashing are written)
    job_root = get_job_artifact_root()
    file_path = _spec_file_path(job_root, job_id, spec_version)
    with open(file_path, "wb") as f:
        f.write(canonical_bytes)

    # Persist DB (store payload JSON + derived hash separately)
    rec = PoTSpecRecord(
        spec_id=spec_id,
        job_id=job_id,
        spec_version=spec_version,
        parent_spec_id=parent_spec_id,
        spec_hash=spec_hash,
        created_at=created_at,
        created_by_model=created_by_model,
        spec_json=payload_json,
    )
    db.add(rec)
    db.commit()

    # Ledger event
    append_event(
        job_artifact_root=job_root,
        job_id=job_id,
        event={
            "event": "SPEC_CREATED",
            "job_id": job_id,
            "spec_id": spec_id,
            "spec_version": spec_version,
            "parent_spec_id": parent_spec_id,
            "spec_hash": spec_hash,
            "created_by_model": created_by_model,
            "inputs": {
                "user_intent_chars": len(user_intent or ""),
                "repo_snapshot_present": bool(repo_snapshot),
            },
            "outputs": {
                "spec_file": os.path.relpath(file_path, start=job_root),
            },
            "status": "ok",
        },
    )

    return PoTSpec(**payload.model_dump(), spec_hash=spec_hash)


# =============================================================================
# Load + verify
# =============================================================================

def load_spec(
    db: Session,
    job_id: str,
    spec_id: str,
    *,
    raise_on_mismatch: bool = True,
) -> Tuple[PoTSpec, bool]:
    """Load a spec from DB + file and verify hashes.

    Returns: (spec, hash_ok)

    If raise_on_mismatch=True, raises SpecMismatchError on any mismatch.
    """
    rec = (
        db.query(PoTSpecRecord)
        .filter(PoTSpecRecord.job_id == job_id, PoTSpecRecord.spec_id == spec_id)
        .first()
    )
    if not rec:
        raise SpecNotFoundError(f"Spec not found for job_id={job_id} spec_id={spec_id}")

    # DB hash check
    db_payload = dict(rec.spec_json or {})
    db_hash = hash_spec_bytes(canonical_spec_bytes(db_payload))

    # File hash check
    job_root = get_job_artifact_root()
    file_path = _spec_file_path(job_root, job_id, rec.spec_version)
    if not os.path.exists(file_path):
        if raise_on_mismatch:
            raise SpecMismatchError(f"Spec file missing: {file_path}")
        return PoTSpec(**db_payload, spec_hash=rec.spec_hash), False

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    try:
        file_obj = json.loads(file_bytes.decode("utf-8"))
    except Exception as e:
        if raise_on_mismatch:
            raise SpecMismatchError(f"Spec file is not valid JSON: {file_path} ({e})")
        return PoTSpec(**db_payload, spec_hash=rec.spec_hash), False

    file_hash = hash_spec_bytes(canonical_spec_bytes(file_obj))

    stored_hash = rec.spec_hash
    hash_ok = (db_hash == stored_hash == file_hash)

    if not hash_ok and raise_on_mismatch:
        raise SpecMismatchError(
            "PoT Spec mismatch detected. "
            f"stored_hash={stored_hash} db_hash={db_hash} file_hash={file_hash}"
        )

    # Construct returned object (prefer file payload if valid)
    try:
        payload = PoTSpecPayload(**file_obj).model_dump()
    except Exception:
        payload = PoTSpecPayload(**db_payload).model_dump()

    return PoTSpec(**payload, spec_hash=stored_hash), hash_ok