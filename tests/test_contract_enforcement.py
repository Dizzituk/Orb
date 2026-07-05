# FILE: tests/test_contract_enforcement.py
# Purpose: live16 — depended-upon segments MUST declare contracts; workers grounded in project root; asset-free boot in greenfield prompts.
# Called-by: pytest
# Depends-on: app.pot_spec.grounded.segmentation, app.pipeline_v2.segmented.worker, app.pot_spec.grounded._greenfield_spec_builder, app.pot_spec.grounded._greenfield_file_planner
# Last-renovated: 2026-07-05
"""Autopsy of the fifth live Tetris build: ten competent workers, zero shared
vocabulary ("move_left" vs "left", inverted gravity return, facades over a
nonexistent assets/ dir, tests run in the wrong folder). live16 turns the
signals the system already emitted into enforcement."""

from app.pot_spec.grounded.segmentation import validate_manifest
from app.pot_spec.grounded.segment_schemas import (
    InterfaceContract, SegmentManifest, SegmentSpec,
)
from app.pot_spec.grounded.smart_segmentation import GROUPING_SYSTEM_PROMPT
from app.pot_spec.grounded._greenfield_spec_builder import build_greenfield_spec
from app.pot_spec.grounded._greenfield_file_planner import _PLANNER_SYSTEM
from app.pipeline_v2.segmented.worker import build_worker_prompt


def _manifest(exposes):
    seg1 = SegmentSpec(segment_id="seg-01-core", title="Core", exposes=exposes)
    seg2 = SegmentSpec(segment_id="seg-02-ui", title="UI", dependencies=["seg-01-core"])
    return SegmentManifest(segments=[seg1, seg2])


class _Profile:
    project_root = "C:/Users/dizzi/OneDrive/Documents/Games/Tazza's Tetris"
    project_name = "Tazza's Tetris"
    language = "python"


# ---------------------------------------------------------------------------
# Check 4: contracts mandatory for depended-upon segments
# ---------------------------------------------------------------------------

def test_depended_upon_segment_without_contracts_fails_validation():
    valid, errors = validate_manifest(_manifest(exposes=None))
    assert valid is False
    assert any("interface contracts" in e for e in errors)

    valid2, errors2 = validate_manifest(_manifest(exposes=InterfaceContract()))
    assert valid2 is False  # empty contract == no contract


def test_declared_contracts_pass_validation():
    exposes = InterfaceContract(
        class_names=["Board"],
        method_signatures=["def lock_piece(piece) -> list[int]  # returns cleared rows"],
        export_names=["move_left", "move_right"],
    )
    valid, errors = validate_manifest(_manifest(exposes=exposes))
    assert valid is True, errors


def test_leaf_segments_may_stay_contractless():
    seg = SegmentSpec(segment_id="seg-01-solo", title="Solo")
    valid, errors = validate_manifest(SegmentManifest(segments=[seg]))
    assert valid is True, errors


# ---------------------------------------------------------------------------
# Prompt pins — the enforcement text must not silently soften
# ---------------------------------------------------------------------------

def test_grouping_prompt_declares_contracts_mandatory():
    assert "INVALID output" in GROUPING_SYSTEM_PROMPT
    assert "SHARED VOCABULARIES ARE CONTRACTS" in GROUPING_SYSTEM_PROMPT
    assert "move_left" in GROUPING_SYSTEM_PROMPT  # the real incident, kept as the example
    assert "RETURN SEMANTICS" in GROUPING_SYSTEM_PROMPT


def test_planner_prompt_forbids_asset_files():
    assert "NO asset files" in _PLANNER_SYSTEM
    assert "assets/" in _PLANNER_SYSTEM


def test_generic_spec_carries_zero_asset_boot_constraint():
    spec = build_greenfield_spec(
        goal="Tetris", what_to_do="build it",
        build_profile={"language": "python", "framework": "generic",
                       "build_system": "pip", "project_root": "C:/x",
                       "project_name": "X", "project_id": "x"},
    )
    assert "ZERO asset files" in spec
    assert "Runtime Constraints" in spec


# ---------------------------------------------------------------------------
# Worker grounding
# ---------------------------------------------------------------------------

def test_worker_prompt_grounds_project_root_and_contract_law():
    seg = SegmentSpec(segment_id="seg-01-core", title="Core", file_scope=["src/board.py"])
    prompt = build_worker_prompt(seg, upstream=[], ledger_block="", profile=_Profile())
    assert "PROJECT ROOT: C:/Users/dizzi/OneDrive/Documents/Games/Tazza's Tetris" in prompt
    assert "cd" in prompt and "pytest" in prompt
    assert "CONTRACTS ARE LAW" in prompt
    assert "move_left" in prompt  # the vocabulary warning example


def test_worker_prompt_without_profile_still_builds():
    seg = SegmentSpec(segment_id="seg-01-core", title="Core", file_scope=["src/board.py"])
    prompt = build_worker_prompt(seg, upstream=[], ledger_block="")
    assert "PROJECT ROOT:" not in prompt
    assert "CONTRACTS ARE LAW" in prompt
