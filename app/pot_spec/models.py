# FILE: app/pot_spec/models.py
"""PoT Spec database model.

Block 1 requires:
- Store spec in DB and file
- Canonical hash (sha256) for integrity checks
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, Index, Integer, JSON, String, UniqueConstraint

from app.db import Base


class PoTSpecRecord(Base):
    __tablename__ = "pot_specs"

    # Identifiers
    spec_id = Column(String(36), primary_key=True)
    job_id = Column(String(36), nullable=False, index=True)
    spec_version = Column(Integer, nullable=False, default=1)
    parent_spec_id = Column(String(36), nullable=True)

    # Integrity
    spec_hash = Column(String(64), nullable=False, index=True)

    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_by_model = Column(String(128), nullable=False)

    # Mirror location (relative to job artifact root)
    # Block 1: mirror path stored for traceability; bytes are identical to canonical bytes hashed.
    file_path = Column(String(512), nullable=True)

    # Full spec payload (stored as structured JSON)
    spec_json = Column(JSON, nullable=False)

    __table_args__ = (
        UniqueConstraint("job_id", "spec_version", name="uq_pot_specs_job_version"),
        Index("ix_pot_specs_job_hash", "job_id", "spec_hash"),
    )


__all__ = ["PoTSpecRecord"]
