# FILE: tests/test_ledger.py
"""Tests for refactored ledger modules.

Tests:
1. Core read/write operations (ledger.py)
2. Pipeline events Blocks 1-6 (ledger_pipeline.py)
3. Overwatcher events Blocks 7-12 (ledger_overwatcher.py)
4. Re-exports work correctly
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import os
import tempfile
import shutil
from datetime import datetime, timezone

import pytest


class TestLedgerCore:
    """Test core ledger operations."""
    
    @pytest.fixture
    def temp_root(self):
        """Create temp directory for artifacts."""
        root = tempfile.mkdtemp(prefix="ledger_test_")
        yield root
        shutil.rmtree(root, ignore_errors=True)
    
    def test_append_event_creates_ledger(self, temp_root):
        """append_event creates ledger directory and file."""
        from app.pot_spec.ledger import append_event
        
        job_id = "test-job-001"
        event = {"event": "TEST_EVENT", "data": "test"}
        
        path = append_event(temp_root, job_id, event)
        
        assert os.path.exists(path)
        assert "events.ndjson" in path
        
        with open(path, "r") as f:
            line = f.readline()
            record = json.loads(line)
        
        assert record["event"] == "TEST_EVENT"
        assert record["job_id"] == job_id
        assert "ts" in record
    
    def test_append_event_appends(self, temp_root):
        """Multiple appends create multiple lines."""
        from app.pot_spec.ledger import append_event
        
        job_id = "test-job-002"
        
        append_event(temp_root, job_id, {"event": "EVENT_1"})
        append_event(temp_root, job_id, {"event": "EVENT_2"})
        append_event(temp_root, job_id, {"event": "EVENT_3"})
        
        ledger_path = os.path.join(temp_root, "jobs", job_id, "ledger", "events.ndjson")
        
        with open(ledger_path, "r") as f:
            lines = f.readlines()
        
        assert len(lines) == 3
        assert json.loads(lines[0])["event"] == "EVENT_1"
        assert json.loads(lines[2])["event"] == "EVENT_3"
    
    def test_read_events(self, temp_root):
        """read_events returns all events."""
        from app.pot_spec.ledger import append_event, read_events
        
        job_id = "test-job-003"
        
        append_event(temp_root, job_id, {"event": "E1", "val": 1})
        append_event(temp_root, job_id, {"event": "E2", "val": 2})
        
        events = read_events(temp_root, job_id)
        
        assert len(events) == 2
        assert events[0]["event"] == "E1"
        assert events[1]["val"] == 2
    
    def test_read_events_empty(self, temp_root):
        """read_events returns empty list for missing ledger."""
        from app.pot_spec.ledger import read_events
        
        events = read_events(temp_root, "nonexistent-job")
        
        assert events == []
    
    def test_read_events_in_range(self, temp_root):
        """read_events_in_range filters by timestamp."""
        from app.pot_spec.ledger import append_event, read_events_in_range
        
        job_id = "test-job-004"
        
        # Manually create events with specific timestamps
        ledger_dir = os.path.join(temp_root, "jobs", job_id, "ledger")
        os.makedirs(ledger_dir, exist_ok=True)
        ledger_path = os.path.join(ledger_dir, "events.ndjson")
        
        events = [
            {"event": "OLD", "ts": "2024-01-01T00:00:00Z", "job_id": job_id},
            {"event": "IN_RANGE", "ts": "2024-06-15T12:00:00Z", "job_id": job_id},
            {"event": "NEW", "ts": "2024-12-31T23:59:59Z", "job_id": job_id},
        ]
        
        with open(ledger_path, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        
        start = datetime(2024, 6, 1, tzinfo=timezone.utc)
        end = datetime(2024, 6, 30, tzinfo=timezone.utc)
        
        filtered = read_events_in_range(temp_root, job_id, start, end)
        
        assert len(filtered) == 1
        assert filtered[0]["event"] == "IN_RANGE"


class TestLedgerPipeline:
    """Test Block 1-6 events."""
    
    @pytest.fixture
    def temp_root(self):
        root = tempfile.mkdtemp(prefix="ledger_pipe_")
        yield root
        shutil.rmtree(root, ignore_errors=True)
    
    def test_emit_job_created(self, temp_root):
        """JOB_CREATED event has required fields."""
        from app.pot_spec.ledger import emit_job_created, read_events
        
        job_id = "job-001"
        emit_job_created(temp_root, job_id, "architecture_design", "Build a REST API")
        
        events = read_events(temp_root, job_id)
        assert len(events) == 1
        assert events[0]["event"] == "JOB_CREATED"
        assert events[0]["job_type"] == "architecture_design"
        assert "REST API" in events[0]["user_request_excerpt"]
    
    def test_emit_spec_created(self, temp_root):
        """SPEC_CREATED event has required fields."""
        from app.pot_spec.ledger import emit_spec_created, read_events
        
        job_id = "job-002"
        emit_spec_created(temp_root, job_id, "SPEC-001", "abc123hash", 1)
        
        events = read_events(temp_root, job_id)
        assert events[0]["event"] == "SPEC_CREATED"
        assert events[0]["spec_id"] == "SPEC-001"
        assert events[0]["spec_hash"] == "abc123hash"
        assert events[0]["spec_version"] == 1
    
    def test_emit_spec_hash_verified(self, temp_root):
        """STAGE_SPEC_HASH_VERIFIED event."""
        from app.pot_spec.ledger import emit_spec_hash_verified, read_events
        
        job_id = "job-003"
        emit_spec_hash_verified(temp_root, job_id, "architecture", "SPEC-001", "hash123")
        
        events = read_events(temp_root, job_id)
        assert events[0]["event"] == "STAGE_SPEC_HASH_VERIFIED"
        assert events[0]["stage_id"] == "architecture"
    
    def test_emit_spec_hash_mismatch(self, temp_root):
        """STAGE_SPEC_HASH_MISMATCH event with severity."""
        from app.pot_spec.ledger import emit_spec_hash_mismatch, read_events
        
        job_id = "job-004"
        emit_spec_hash_mismatch(
            temp_root, job_id, "critique", "SPEC-001",
            expected_spec_hash="expected123",
            observed_spec_hash="observed456",
            reason="drift"
        )
        
        events = read_events(temp_root, job_id)
        assert events[0]["event"] == "STAGE_SPEC_HASH_MISMATCH"
        assert events[0]["severity"] == "ERROR"
        assert events[0]["expected"] == "expected123"
        assert events[0]["observed"] == "observed456"
    
    def test_emit_arch_created(self, temp_root):
        """ARCH_CREATED event."""
        from app.pot_spec.ledger import emit_arch_created, read_events
        
        job_id = "job-005"
        emit_arch_created(
            temp_root, job_id,
            arch_id="ARCH-001",
            arch_version=1,
            arch_hash="archhash",
            spec_id="SPEC-001",
            spec_hash="spechash",
            model="claude-opus-4-5-20251101"
        )
        
        events = read_events(temp_root, job_id)
        assert events[0]["event"] == "ARCH_CREATED"
        assert events[0]["arch_version"] == 1
        assert events[0]["model"] == "claude-opus-4-5-20251101"
    
    def test_emit_critique_pass(self, temp_root):
        """CRITIQUE_PASS event."""
        from app.pot_spec.ledger import emit_critique_pass, read_events
        
        job_id = "job-006"
        emit_critique_pass(temp_root, job_id, "CRIT-001", "ARCH-001", 2)
        
        events = read_events(temp_root, job_id)
        assert events[0]["event"] == "CRITIQUE_PASS"
        assert events[0]["arch_version"] == 2
    
    def test_emit_revision_loop_terminated(self, temp_root):
        """REVISION_LOOP_TERMINATED event."""
        from app.pot_spec.ledger import emit_revision_loop_terminated, read_events
        
        job_id = "job-007"
        emit_revision_loop_terminated(
            temp_root, job_id,
            arch_id="ARCH-001",
            final_version=3,
            reason="pass",
            iterations_used=2,
            final_pass=True
        )
        
        events = read_events(temp_root, job_id)
        assert events[0]["event"] == "REVISION_LOOP_TERMINATED"
        assert events[0]["reason"] == "pass"
        assert events[0]["final_pass"] is True
    
    def test_emit_provider_fallback(self, temp_root):
        """PROVIDER_FALLBACK event."""
        from app.pot_spec.ledger import emit_provider_fallback, read_events
        
        job_id = "job-008"
        emit_provider_fallback(
            temp_root, job_id,
            from_provider="anthropic",
            from_model="claude-opus",
            to_provider="openai",
            to_model="gpt-5.2-pro",
            reason="rate_limited"
        )
        
        events = read_events(temp_root, job_id)
        assert events[0]["event"] == "PROVIDER_FALLBACK"
        assert events[0]["from_provider"] == "anthropic"
        assert events[0]["to_model"] == "gpt-5.2-pro"


class TestLedgerOverwatcher:
    """Test Block 7-12 events."""
    
    @pytest.fixture
    def temp_root(self):
        root = tempfile.mkdtemp(prefix="ledger_ow_")
        yield root
        shutil.rmtree(root, ignore_errors=True)
    
    def test_emit_chunk_plan_created(self, temp_root):
        """CHUNK_PLAN_CREATED event."""
        from app.pot_spec.ledger import emit_chunk_plan_created, read_events
        
        job_id = "job-010"
        emit_chunk_plan_created(
            temp_root, job_id,
            plan_id="PLAN-001",
            arch_id="ARCH-001",
            arch_version=1,
            chunk_count=5
        )
        
        events = read_events(temp_root, job_id)
        assert events[0]["event"] == "CHUNK_PLAN_CREATED"
        assert events[0]["chunk_count"] == 5
    
    def test_emit_chunk_implemented(self, temp_root):
        """CHUNK_IMPLEMENTED event."""
        from app.pot_spec.ledger import emit_chunk_implemented, read_events
        
        job_id = "job-011"
        emit_chunk_implemented(
            temp_root, job_id,
            chunk_id="CHUNK-001",
            files_added=["app/new.py"],
            files_modified=["app/existing.py"],
            model="claude-sonnet-4-5-20250514"
        )
        
        events = read_events(temp_root, job_id)
        assert events[0]["event"] == "CHUNK_IMPLEMENTED"
        assert events[0]["file_count"] == 2
        assert "app/new.py" in events[0]["files_added"]
    
    def test_emit_boundary_violation(self, temp_root):
        """BOUNDARY_VIOLATION event with severity."""
        from app.pot_spec.ledger import emit_boundary_violation, read_events
        
        job_id = "job-012"
        emit_boundary_violation(
            temp_root, job_id,
            chunk_id="CHUNK-001",
            violations=[
                {"file_path": "app/forbidden.py", "action": "modified", "reason": "not in allowed_files"}
            ]
        )
        
        events = read_events(temp_root, job_id)
        assert events[0]["event"] == "BOUNDARY_VIOLATION"
        assert events[0]["severity"] == "ERROR"
        assert events[0]["violation_count"] == 1
    
    def test_emit_verify_pass(self, temp_root):
        """VERIFY_PASS event."""
        from app.pot_spec.ledger import emit_verify_pass, read_events
        
        job_id = "job-013"
        emit_verify_pass(
            temp_root, job_id,
            chunk_id="CHUNK-001",
            tests_passed=15,
            lint_errors=0,
            type_errors=0
        )
        
        events = read_events(temp_root, job_id)
        assert events[0]["event"] == "VERIFY_PASS"
        assert events[0]["tests_passed"] == 15
        assert events[0]["lint_errors"] == 0
    
    def test_emit_verify_fail(self, temp_root):
        """VERIFY_FAIL event."""
        from app.pot_spec.ledger import emit_verify_fail, read_events
        
        job_id = "job-014"
        emit_verify_fail(
            temp_root, job_id,
            chunk_id="CHUNK-001",
            tests_failed=3,
            lint_errors=2,
            type_errors=1,
            failure_summary="3 tests failed, 2 lint errors"
        )
        
        events = read_events(temp_root, job_id)
        assert events[0]["event"] == "VERIFY_FAIL"
        assert events[0]["tests_failed"] == 3
        assert events[0]["severity"] == "ERROR"
    
    def test_emit_quarantine_created(self, temp_root):
        """QUARANTINE_CREATED event."""
        from app.pot_spec.ledger import emit_quarantine_created, read_events
        
        job_id = "job-015"
        emit_quarantine_created(
            temp_root, job_id,
            stage_id="implementation",
            reason="boundary_violation",
            quarantine_path="/quarantine/chunk-001"
        )
        
        events = read_events(temp_root, job_id)
        assert events[0]["event"] == "QUARANTINE_CREATED"
        assert events[0]["status"] == "quarantined"
    
    def test_emit_replay_pack_created(self, temp_root):
        """REPLAY_PACK_CREATED event."""
        from app.pot_spec.ledger import emit_replay_pack_created, read_events
        
        job_id = "job-016"
        emit_replay_pack_created(
            temp_root, job_id,
            pack_id="PACK-001",
            pack_path="/artifacts/replay/pack-001.json"
        )
        
        events = read_events(temp_root, job_id)
        assert events[0]["event"] == "REPLAY_PACK_CREATED"
        assert events[0]["pack_id"] == "PACK-001"


class TestReexports:
    """Test that all events are re-exported from main ledger.py."""
    
    def test_pipeline_events_reexported(self):
        """All pipeline events importable from ledger.py."""
        from app.pot_spec.ledger import (
            emit_job_created,
            emit_spec_created,
            emit_spec_questions_generated,
            emit_spec_hash_verified,
            emit_spec_hash_mismatch,
            emit_spec_hash_missing,
            emit_job_status_changed,
            emit_stage_started,
            emit_stage_output_stored,
            emit_stage_failed,
            emit_provider_fallback,
            emit_job_completed,
            emit_job_failed,
            emit_job_aborted,
            emit_arch_created,
            emit_arch_mirror_written,
            emit_critique_created,
            emit_critique_pass,
            emit_critique_fail,
            emit_revision_loop_started,
            emit_arch_revised,
            emit_revision_loop_terminated,
        )
        
        # All should be callable
        assert callable(emit_job_created)
        assert callable(emit_revision_loop_terminated)
    
    def test_overwatcher_events_reexported(self):
        """All overwatcher events importable from ledger.py."""
        from app.pot_spec.ledger import (
            emit_chunk_plan_created,
            emit_chunk_implemented,
            emit_boundary_violation,
            emit_verify_pass,
            emit_verify_fail,
            emit_quarantine_created,
            emit_quarantine_applied,
            emit_deletion_complete,
            emit_replay_pack_created,
        )
        
        assert callable(emit_chunk_plan_created)
        assert callable(emit_replay_pack_created)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])