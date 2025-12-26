# FILE: app/pot_spec/__init__.py
"""PoT Spec (Point-of-Truth Spec) package.

Block 1 delivers:
- Spec schema (Pydantic)
- DB persistence (SQLAlchemy)
- Canonical JSON + sha256 hashing
- File persistence under jobs/<job_id>/spec/spec_vN.json
- Ledger append-only events under jobs/<job_id>/ledger/events.ndjson
"""

from .service import (
    create_spec_from_intent,
    load_spec,
    canonical_spec_bytes,
    hash_spec_bytes,
    get_job_artifact_root,
)

from .schemas import PoTSpec

__all__ = [
    "PoTSpec",
    "create_spec_from_intent",
    "load_spec",
    "canonical_spec_bytes",
    "hash_spec_bytes",
    "get_job_artifact_root",
]

from app.pot_spec.spec_gate import run_spec_gate, detect_user_questions
