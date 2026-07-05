# FILE: tests/test_segment_validation.py
# Purpose: Derek phase 4 gate — segment size budgets, DAG validation, deterministic re-split.
# Called-by: pytest
# Depends-on: app.pot_spec.grounded.segment_validation, app.pot_spec.grounded.segment_schemas
# Last-renovated: 2026-07-04
"""Golden-fixture tests: sized segments + valid DAG pass; cycles, unknown
deps and un-splittable oversize are rejected loudly; oversize splits
deterministically with edges remapped."""

import pytest

from app.pot_spec.grounded import segment_validation as sv
from app.pot_spec.grounded.segment_validation import (
    SegmentValidationError,
    enforce_segment_budgets,
    split_oversized_segments,
    validate_manifest,
)
from app.pot_spec.grounded.segment_schemas import SegmentSpec
from app.pot_spec.grounded._segment_schemas_utils_1 import SegmentManifest


def _seg(sid, files=None, deps=None, **kw):
    return SegmentSpec(
        segment_id=sid, title=f"Segment {sid}",
        file_scope=files or [], dependencies=deps or [], **kw,
    )


def _manifest(segments):
    m = SegmentManifest(segments=segments)
    m.total_segments = len(segments)
    m.total_files = sum(len(s.file_scope) for s in segments)
    return m


@pytest.fixture(autouse=True)
def _budgets(monkeypatch, tmp_path):
    monkeypatch.setenv("ASTRA_SEGMENT_MAX_FILES", "5")
    monkeypatch.setenv("ASTRA_SEGMENT_MAX_KB", "60")
    # Keep KB measurement hermetic: no real project roots
    monkeypatch.setattr(sv, "_project_roots", lambda: [str(tmp_path)])
    yield


def test_golden_fixture_valid_dag_passes():
    m = _manifest([
        _seg("seg-01", files=["a.py", "b.py"]),
        _seg("seg-02", files=["c.py"], deps=["seg-01"]),
        _seg("seg-03", files=["d.py", "e.py"], deps=["seg-01", "seg-02"]),
    ])
    report = validate_manifest(m)
    assert report.ok, report.summary()
    out = enforce_segment_budgets(m, "job-golden")
    assert [s.segment_id for s in out.segments] == ["seg-01", "seg-02", "seg-03"]


def test_old_manifest_without_deps_is_safe():
    m = _manifest([_seg("seg-01", files=["a.py"]), _seg("seg-02", files=["b.py"])])
    assert validate_manifest(m).ok


def test_unknown_dependency_rejected():
    m = _manifest([_seg("seg-01", files=["a.py"], deps=["seg-99"])])
    with pytest.raises(SegmentValidationError) as exc:
        enforce_segment_budgets(m, "job-unknown")
    assert "seg-99" in str(exc.value)


def test_cycle_rejected():
    m = _manifest([
        _seg("seg-01", files=["a.py"], deps=["seg-02"]),
        _seg("seg-02", files=["b.py"], deps=["seg-01"]),
    ])
    with pytest.raises(SegmentValidationError) as exc:
        enforce_segment_budgets(m, "job-cycle")
    assert "cycle" in str(exc.value).lower()


def test_oversize_by_count_splits_deterministically():
    files = [f"f{i}.py" for i in range(12)]  # 12 files, budget 5 -> 3 parts
    m = _manifest([
        _seg("seg-big", files=files),
        _seg("seg-after", files=["z.py"], deps=["seg-big"]),
    ])
    out = enforce_segment_budgets(m, "job-split")
    ids = [s.segment_id for s in out.segments]
    assert ids == ["seg-big-p1", "seg-big-p2", "seg-big-p3", "seg-after"]
    parts = {s.segment_id: s for s in out.segments}
    assert all(len(parts[p].file_scope) <= 5 for p in ids[:3])
    # Parts chain sequentially; original deps stay on part 1
    assert parts["seg-big-p1"].dependencies == []
    assert parts["seg-big-p2"].dependencies == ["seg-big-p1"]
    assert parts["seg-big-p3"].dependencies == ["seg-big-p2"]
    # External dependant remapped to the LAST part
    assert parts["seg-after"].dependencies == ["seg-big-p3"]
    # All 12 files preserved, in order, no duplicates
    flat = [f for p in ids[:3] for f in parts[p].file_scope]
    assert flat == files
    # DAG still valid after split
    assert validate_manifest(out).ok
    assert out.total_segments == 4
    assert out.total_files == 13


def test_kb_budget_enforced_and_single_giant_file_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "_project_roots", lambda: [str(tmp_path)])
    big = tmp_path / "giant.py"
    big.write_text("x" * (80 * 1024), encoding="utf-8")  # 80KB > 60KB budget
    m = _manifest([_seg("seg-giant", files=["giant.py"])])
    with pytest.raises(SegmentValidationError) as exc:
        enforce_segment_budgets(m, "job-giant")
    assert "never silently oversized" in str(exc.value)


def test_kb_split_two_files(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "_project_roots", lambda: [str(tmp_path)])
    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text("x" * (40 * 1024), encoding="utf-8")  # 40KB each
    m = _manifest([_seg("seg-heavy", files=["a.py", "b.py"])])  # 80KB combined
    out = enforce_segment_budgets(m, "job-heavy")
    assert [s.segment_id for s in out.segments] == ["seg-heavy-p1", "seg-heavy-p2"]
    assert validate_manifest(out).ok


def test_deterministic_refactor_is_kb_exempt(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "_project_roots", lambda: [str(tmp_path)])
    (tmp_path / "monolith.py").write_text("x" * (200 * 1024), encoding="utf-8")
    m = _manifest([
        _seg("seg-refactor", files=["monolith.py"], deterministic_refactor=True),
    ])
    out = enforce_segment_budgets(m, "job-refactor")
    assert out.segments[0].segment_id == "seg-refactor"


def test_requirement_map_remapped_on_split():
    files = [f"f{i}.py" for i in range(7)]
    m = _manifest([_seg("seg-big", files=files)])
    m.requirement_map = {"REQ-1": ["seg-big"]}
    out, n = split_oversized_segments(m)
    assert n == 1
    assert "seg-big-p1" in out.requirement_map["REQ-1"]
    assert "seg-big-p2" in out.requirement_map["REQ-1"]


def test_env_budget_override(monkeypatch):
    monkeypatch.setenv("ASTRA_SEGMENT_MAX_FILES", "2")
    m = _manifest([_seg("seg-01", files=["a.py", "b.py", "c.py"])])
    out = enforce_segment_budgets(m, "job-env")
    assert len(out.segments) == 2
