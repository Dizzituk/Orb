# FILE: tests/test_overwatcher_schemas.py
"""Unit tests for Overwatcher schemas (Blocks 7-12)."""

import json
import pytest

from app.overwatcher.schemas import (
    # Block 7
    Chunk,
    ChunkPlan,
    ChunkStep,
    ChunkVerification,
    ChunkStatus,
    FileAction,
    # Block 8
    BoundaryViolation,
    DiffCheckResult,
    # Block 9
    CommandResult,
    VerificationResult,
    VerificationStatus,
    # Block 10-11
    QuarantineCandidate,
    QuarantineReport,
    QuarantineReason,
    StaticEvidence,
    DynamicEvidence,
    DeletionReport,
    # Block 12
    ReplayPack,
)


class TestChunkStep:
    def test_to_dict(self):
        step = ChunkStep(
            step_id="STEP-001",
            description="Add auth module",
            file_path="app/auth.py",
            action=FileAction.ADD,
            details="Create basic JWT auth",
        )
        d = step.to_dict()
        assert d["step_id"] == "STEP-001"
        assert d["action"] == "add"

    def test_from_dict(self):
        data = {
            "step_id": "STEP-002",
            "description": "Modify config",
            "file_path": "config.py",
            "action": "modify",
        }
        step = ChunkStep.from_dict(data)
        assert step.action == FileAction.MODIFY


class TestChunk:
    def test_basic_chunk(self):
        chunk = Chunk(
            chunk_id="CHUNK-001",
            title="Add authentication",
            objective="Implement JWT auth",
            spec_refs=["MUST-1", "SHOULD-2"],
            arch_refs=["Section 2.1"],
            allowed_files={
                "add": ["app/auth.py"],
                "modify": ["app/main.py"],
                "delete_candidates": [],
            },
        )
        assert chunk.status == ChunkStatus.PENDING
        assert "app/auth.py" in chunk.get_all_allowed_paths()
        assert "app/main.py" in chunk.get_all_allowed_paths()

    def test_to_json_and_back(self):
        chunk = Chunk(
            chunk_id="CHUNK-001",
            title="Test chunk",
            objective="Test objective",
            steps=[
                ChunkStep(
                    step_id="STEP-001",
                    description="Add file",
                    file_path="test.py",
                    action=FileAction.ADD,
                )
            ],
            verification=ChunkVerification(
                commands=["pytest tests/"],
                timeout_seconds=60,
            ),
        )
        
        json_str = chunk.to_json()
        parsed = Chunk.from_json(json_str)
        
        assert parsed.chunk_id == "CHUNK-001"
        assert len(parsed.steps) == 1
        assert parsed.steps[0].action == FileAction.ADD


class TestChunkPlan:
    def test_plan_with_chunks(self):
        plan = ChunkPlan(
            plan_id="plan-123",
            job_id="job-456",
            arch_id="arch-789",
            arch_version=1,
            spec_id="spec-abc",
            spec_hash="hash123",
            chunks=[
                Chunk(chunk_id="CHUNK-001", title="First", objective="Do first"),
                Chunk(chunk_id="CHUNK-002", title="Second", objective="Do second"),
            ],
        )
        
        d = plan.to_dict()
        assert len(d["chunks"]) == 2
        
        # Round trip
        parsed = ChunkPlan.from_dict(d)
        assert len(parsed.chunks) == 2


class TestDiffCheckResult:
    def test_passed_result(self):
        result = DiffCheckResult(
            passed=True,
            violations=[],
            files_added=["new.py"],
            files_modified=["existing.py"],
            files_deleted=[],
        )
        assert result.passed is True
        assert len(result.violations) == 0

    def test_failed_result(self):
        result = DiffCheckResult(
            passed=False,
            violations=[
                BoundaryViolation(
                    file_path="forbidden.py",
                    action="modified",
                    reason="Not in allowed list",
                )
            ],
        )
        assert result.passed is False
        assert len(result.violations) == 1


class TestVerificationResult:
    def test_passed_verification(self):
        result = VerificationResult(
            chunk_id="CHUNK-001",
            status=VerificationStatus.PASSED,
            tests_passed=10,
            tests_failed=0,
            lint_errors=0,
            type_errors=0,
        )
        assert result.status == VerificationStatus.PASSED

    def test_failed_verification(self):
        result = VerificationResult(
            chunk_id="CHUNK-001",
            status=VerificationStatus.FAILED,
            tests_passed=8,
            tests_failed=2,
            command_results=[
                CommandResult(
                    command="pytest",
                    exit_code=1,
                    stdout="2 failed",
                    stderr="",
                    duration_ms=1000,
                    passed=False,
                )
            ],
        )
        assert result.status == VerificationStatus.FAILED
        assert result.tests_failed == 2


class TestQuarantineCandidate:
    def test_candidate_creation(self):
        candidate = QuarantineCandidate(
            file_path="app/dead_code.py",
            reason=QuarantineReason.NO_REFERENCES,
            confidence=0.85,
            static_evidence=StaticEvidence(
                rg_references=0,
                import_count=0,
            ),
        )
        assert candidate.confidence == 0.85
        assert not candidate.quarantined
        assert not candidate.deleted

    def test_to_dict(self):
        candidate = QuarantineCandidate(
            file_path="test.py",
            reason=QuarantineReason.DEAD_CODE,
            confidence=0.9,
        )
        d = candidate.to_dict()
        assert d["reason"] == "dead_code"
        assert d["confidence"] == 0.9


class TestQuarantineReport:
    def test_report_creation(self):
        report = QuarantineReport(
            report_id="report-123",
            job_id="job-456",
            candidates=[
                QuarantineCandidate(
                    file_path="dead.py",
                    reason=QuarantineReason.NO_IMPORTS,
                    confidence=0.8,
                    quarantined=True,
                    quarantine_path=".quarantine/dead.py",
                )
            ],
            repo_still_passes=True,
        )
        assert len(report.candidates) == 1
        assert report.repo_still_passes is True

    def test_to_json(self):
        report = QuarantineReport(
            report_id="r1",
            job_id="j1",
            repo_still_passes=True,
        )
        json_str = report.to_json()
        data = json.loads(json_str)
        assert data["report_id"] == "r1"


class TestDeletionReport:
    def test_deletion_report(self):
        report = DeletionReport(
            report_id="del-123",
            job_id="job-456",
            quarantine_report_id="quar-789",
            deleted_files=["dead1.py", "dead2.py"],
            deletion_evidence={
                "dead1.py": "No references",
                "dead2.py": "No imports",
            },
            repo_still_passes=True,
            approved_by="user",
        )
        assert len(report.deleted_files) == 2
        assert report.approved_by == "user"


class TestReplayPack:
    def test_replay_pack(self):
        pack = ReplayPack(
            pack_id="pack-123",
            job_id="job-456",
            created_at="2025-12-27T00:00:00Z",
            spec_path="spec/spec_v1.json",
            arch_path="arch/arch_v1.md",
            ledger_path="ledger/events.ndjson",
            model_versions={
                "spec_gate": "gpt-4o",
                "architecture": "claude-opus-4-20250514",
                "critique": "gemini-2.0-flash",
            },
        )
        
        d = pack.to_dict()
        assert d["model_versions"]["critique"] == "gemini-2.0-flash"
        
        json_str = pack.to_json()
        assert "gpt-4o" in json_str
