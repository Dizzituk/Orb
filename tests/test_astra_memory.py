# FILE: tests/test_astra_memory.py
"""
Tests for ASTRA Memory System (AstraJob 5).
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.astra_memory.models import (
    AstraJob, JobFile, JobEvent, JobChunk,
    OverwatchSummary, GlobalPref, OverwatchPattern,
)


@pytest.fixture
def db_session():
    """Create in-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestJobModel:
    def test_create_job(self, db_session):
        job = AstraJob(job_id="AstraJob-123", user_intent="Build a REST API", status="created")
        db_session.add(job)
        db_session.commit()
        loaded = db_session.query(AstraJob).filter(AstraJob.job_id == "AstraJob-123").first()
        assert loaded is not None
        assert loaded.user_intent == "Build a REST API"

    def test_job_with_spec_link(self, db_session):
        job = AstraJob(job_id="AstraJob-456", spec_id="spec-abc", spec_hash="hash123", spec_version=1, status="spec_gate")
        db_session.add(job)
        db_session.commit()
        loaded = db_session.query(AstraJob).filter(AstraJob.job_id == "AstraJob-456").first()
        assert loaded.spec_id == "spec-abc"

    def test_job_with_arch_link(self, db_session):
        job = AstraJob(job_id="AstraJob-789", arch_id="arch-xyz", arch_hash="archhash", arch_version=2, status="planning")
        db_session.add(job)
        db_session.commit()
        loaded = db_session.query(AstraJob).filter(AstraJob.job_id == "AstraJob-789").first()
        assert loaded.arch_id == "arch-xyz"


class TestJobFileModel:
    def test_record_file_touch(self, db_session):
        job = AstraJob(job_id="AstraJob-file-test", status="executing")
        db_session.add(job)
        db_session.commit()
        file_record = JobFile(
            job_id="AstraJob-file-test", arch_id="arch-001", path="app/api/router.py",
            action="modify", hash_before="abc123", hash_after="def456", chunk_id="chunk-001"
        )
        db_session.add(file_record)
        db_session.commit()
        loaded = db_session.query(JobFile).filter(JobFile.job_id == "AstraJob-file-test").first()
        assert loaded.path == "app/api/router.py"
        assert loaded.action == "modify"

    def test_multiple_files_per_job(self, db_session):
        job = AstraJob(job_id="AstraJob-multi-file", status="executing")
        db_session.add(job)
        db_session.commit()
        files = [
            JobFile(job_id="AstraJob-multi-file", path="file1.py", action="create"),
            JobFile(job_id="AstraJob-multi-file", path="file2.py", action="modify"),
            JobFile(job_id="AstraJob-multi-file", path="file3.py", action="read"),
        ]
        db_session.add_all(files)
        db_session.commit()
        count = db_session.query(JobFile).filter(JobFile.job_id == "AstraJob-multi-file").count()
        assert count == 3


class TestOverwatchSummaryModel:
    def test_create_summary(self, db_session):
        job = AstraJob(job_id="AstraJob-ow-test", status="completed")
        db_session.add(job)
        db_session.commit()
        summary = OverwatchSummary(
            job_id="AstraJob-ow-test", risk_level="medium", risk_score=0.65,
            total_interventions=3, warnings_count=2, blocks_count=1
        )
        db_session.add(summary)
        db_session.commit()
        loaded = db_session.query(OverwatchSummary).filter(OverwatchSummary.job_id == "AstraJob-ow-test").first()
        assert loaded.risk_level == "medium"
        assert loaded.total_interventions == 3

    def test_hard_stopped_summary(self, db_session):
        job = AstraJob(job_id="AstraJob-ow-stopped", status="aborted")
        db_session.add(job)
        db_session.commit()
        summary = OverwatchSummary(
            job_id="AstraJob-ow-stopped", risk_level="critical", current_strikes=3,
            max_strikes_hit=True, hard_stopped=True
        )
        db_session.add(summary)
        db_session.commit()
        loaded = db_session.query(OverwatchSummary).filter(OverwatchSummary.hard_stopped == True).first()
        assert loaded is not None
        assert loaded.max_strikes_hit is True


class TestGlobalPrefModel:
    def test_create_pref(self, db_session):
        pref = GlobalPref(
            key="no_destructive_commands", value="Never use rm -rf",
            category="policy", source="user_declared", applies_to="all", active=True
        )
        db_session.add(pref)
        db_session.commit()
        loaded = db_session.query(GlobalPref).filter(GlobalPref.key == "no_destructive_commands").first()
        assert loaded.category == "policy"

    def test_filter_by_component(self, db_session):
        prefs = [
            GlobalPref(key="ow_pref_1", value="v1", category="preference", applies_to="overwatcher"),
            GlobalPref(key="sg_pref_1", value="v2", category="preference", applies_to="spec_gate"),
            GlobalPref(key="all_pref_1", value="v3", category="preference", applies_to="all"),
        ]
        db_session.add_all(prefs)
        db_session.commit()
        ow_prefs = db_session.query(GlobalPref).filter(GlobalPref.applies_to.in_(["overwatcher", "all"])).all()
        assert len(ow_prefs) == 2


class TestOverwatchPatternModel:
    def test_create_pattern(self, db_session):
        pattern = OverwatchPattern(
            pattern_type="file_fragility", target_path="app/critical/security.py",
            occurrence_count=3, job_ids=["AstraJob-1", "AstraJob-2", "AstraJob-3"], severity="warn"
        )
        db_session.add(pattern)
        db_session.commit()
        loaded = db_session.query(OverwatchPattern).filter(OverwatchPattern.target_path == "app/critical/security.py").first()
        assert loaded.occurrence_count == 3

    def test_model_error_pattern(self, db_session):
        pattern = OverwatchPattern(
            pattern_type="model_error", target_model="gpt-4o", error_signature="sig-timeout-001",
            occurrence_count=5, severity="error", action="require_review"
        )
        db_session.add(pattern)
        db_session.commit()
        loaded = db_session.query(OverwatchPattern).filter(OverwatchPattern.target_model == "gpt-4o").first()
        assert loaded.action == "require_review"


class TestRelationships:
    def test_job_files_relationship(self, db_session):
        job = AstraJob(job_id="AstraJob-rel-test", status="executing")
        db_session.add(job)
        db_session.commit()
        files = [
            JobFile(job_id="AstraJob-rel-test", path="f1.py", action="create"),
            JobFile(job_id="AstraJob-rel-test", path="f2.py", action="modify"),
        ]
        db_session.add_all(files)
        db_session.commit()
        loaded_job = db_session.query(AstraJob).filter(AstraJob.job_id == "AstraJob-rel-test").first()
        assert len(loaded_job.files) == 2

    def test_job_overwatch_relationship(self, db_session):
        job = AstraJob(job_id="AstraJob-ow-rel", status="completed")
        db_session.add(job)
        db_session.commit()
        summary = OverwatchSummary(job_id="AstraJob-ow-rel", risk_level="low")
        db_session.add(summary)
        db_session.commit()
        loaded_job = db_session.query(AstraJob).filter(AstraJob.job_id == "AstraJob-ow-rel").first()
        assert loaded_job.overwatch is not None
        assert loaded_job.overwatch.risk_level == "low"


class TestQueries:
    def test_jobs_by_status(self, db_session):
        jobs = [
            AstraJob(job_id="j1", status="completed"),
            AstraJob(job_id="j2", status="completed"),
            AstraJob(job_id="j3", status="failed"),
        ]
        db_session.add_all(jobs)
        db_session.commit()
        completed = db_session.query(AstraJob).filter(AstraJob.status == "completed").all()
        assert len(completed) == 2

    def test_escalated_jobs_query(self, db_session):
        j1 = AstraJob(job_id="j-esc-1", status="completed")
        j2 = AstraJob(job_id="j-esc-2", status="aborted")
        db_session.add_all([j1, j2])
        db_session.commit()
        db_session.add(OverwatchSummary(job_id="j-esc-1", risk_level="low", escalated=False))
        db_session.add(OverwatchSummary(job_id="j-esc-2", risk_level="high", escalated=True))
        db_session.commit()
        escalated = db_session.query(AstraJob).join(OverwatchSummary).filter(OverwatchSummary.escalated == True).all()
        assert len(escalated) == 1
        assert escalated[0].job_id == "j-esc-2"

    def test_jobs_touching_file(self, db_session):
        j1 = AstraJob(job_id="jf-1", status="completed")
        j2 = AstraJob(job_id="jf-2", status="completed")
        db_session.add_all([j1, j2])
        db_session.commit()
        db_session.add(JobFile(job_id="jf-1", path="app/api.py", action="modify"))
        db_session.add(JobFile(job_id="jf-2", path="app/api.py", action="modify"))
        db_session.add(JobFile(job_id="jf-2", path="app/other.py", action="create"))
        db_session.commit()
        jobs_touching_api = db_session.query(JobFile.job_id).filter(JobFile.path == "app/api.py").distinct().all()
        assert len(jobs_touching_api) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
