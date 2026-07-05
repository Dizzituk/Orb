# FILE: tests/test_segment_routing_fallback.py
# Purpose: live15 — a target_id-less segment only refuses routing when there is real ambiguity.
# Called-by: pytest
# Depends-on: app.pipeline_v2.llm_tool_exec
# Last-renovated: 2026-07-05
"""2026-07-05 11:45: the checkout-repair worker's edits all bounced with
'Segment seg-05-input-handling has no target_id — refusing to route' — the
segment tracker had re-activated a greenfield segment (single-target
manifests stamp no per-segment target ids) and the resolver hard-refused.
Refusal is for multi-target ambiguity; one profile = one answer."""

import pytest

from app.pipeline_v2 import llm_tool_exec as lte
from app.pipeline_v2.build_targets import BuildTargetProfile


class _Seg:
    def __init__(self, segment_id, target_id=None):
        self.segment_id = segment_id
        self.target_id = target_id


def _profile():
    return BuildTargetProfile(
        project_id="tazza-s-tetris", project_name="Tazza's Tetris",
        project_root="C:/Games/T", language="python", build_system="pip",
        framework="generic", source_root="src", package_name="",
        architecture_pattern="flat",
    )


@pytest.fixture(autouse=True)
def _clean_context():
    yield
    lte.set_tool_segment(None)
    lte.set_tool_profile(None)


def test_targetless_segment_falls_back_to_single_job_profile():
    prof = _profile()
    lte.set_tool_profile(prof)
    lte.set_tool_segment(_Seg("seg-05-input-handling", target_id=None))
    assert lte._resolve_active_profile() is prof
    assert lte._resolve_active_profile(path="src/input_handler.py") is prof


def test_targetless_segment_still_refuses_without_job_profile():
    lte.set_tool_profile(None)
    lte.set_tool_segment(_Seg("seg-05-input-handling", target_id=None))
    assert lte._resolve_active_profile() is None


def test_no_segment_no_profile_returns_none():
    lte.set_tool_profile(None)
    lte.set_tool_segment(None)
    assert lte._resolve_active_profile() is None
