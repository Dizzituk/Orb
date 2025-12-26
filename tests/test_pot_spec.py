# FILE: tests/test_pot_spec.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.pot_spec.models import PoTSpecRecord  # noqa: F401  (register table)
from app.pot_spec.service import (
    canonical_spec_bytes,
    hash_spec_bytes,
    create_spec_from_intent,
    load_spec,
)


def _make_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        echo=False,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_canonical_hash_stability_for_order_insensitive_lists():
    payload_a = {
        "job_id": "job1",
        "spec_id": "spec1",
        "spec_version": 1,
        "parent_spec_id": None,
        "created_at": "2025-01-01T00:00:00Z",
        "created_by_model": "x",
        "goal": "  Do something\r\n",
        "requirements": {"must": ["b", "a"], "should": [], "can": []},
        "constraints": {},
        "acceptance_tests": ["  test2", "test1 "],
        "open_questions": [],
        "recommendations": [],
        "repo_snapshot": None,
    }

    payload_b = {
        **payload_a,
        "requirements": {"must": ["a", "b"], "should": [], "can": []},
        "acceptance_tests": ["test1", "test2"],
        "goal": "Do something\n",
    }

    h1 = hash_spec_bytes(canonical_spec_bytes(payload_a))
    h2 = hash_spec_bytes(canonical_spec_bytes(payload_b))
    assert h1 == h2


def test_db_vs_file_consistency(tmp_path):
    os.environ["ORB_JOB_ARTIFACT_ROOT"] = str(tmp_path)

    db = _make_session()
    job_id = "job-123"
    spec = create_spec_from_intent(
        db=db,
        job_id=job_id,
        user_intent="Build X",
        created_by_model="test",
        constraints={"needs_internet": False},
        repo_snapshot={"commit": "abc"},
    )

    loaded, ok = load_spec(db=db, job_id=job_id, spec_id=spec.spec_id, raise_on_mismatch=True)
    assert ok is True
    assert loaded.spec_hash == spec.spec_hash
    assert loaded.goal == "Build X"
