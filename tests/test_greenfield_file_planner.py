# FILE: tests/test_greenfield_file_planner.py
# Purpose: live9 — greenfield specs with no file paths get an APEX-planned, deterministically-validated file plan.
# Called-by: pytest
# Depends-on: app.pot_spec.grounded._greenfield_file_planner, app.pot_spec.grounded._spec_runner_utils_13, app.providers._registry_anthropic_shapes
# Last-renovated: 2026-07-04
"""First live Tetris run (2026-07-04 23:47): python/generic greenfield spec
embedded zero file paths -> 0-file manifest -> 'nothing to scaffold' ->
vacuous 0/0 PASS -> eyes launched nothing -> judge 400'd on temperature.
These tests pin all three live9 fixes at the unit level."""

import asyncio

from app.pot_spec.grounded._greenfield_file_planner import (
    _parse_planned_files,
    plan_greenfield_files,
    render_files_section,
)
from app.pot_spec.grounded._spec_runner_utils_13 import _extract_file_scope_from_spec
from app.providers._registry_anthropic_shapes import (
    is_sampling_param_error,
    strip_sampling_params,
)

GOOD_PLAN = """
- `src/main.py` — entry point, game loop bootstrap
- `src/board.py` — grid state and line clearing
- `src/pieces.py` — tetromino shapes and rotation
- `src/audio.py` — 8-bit chiptune synth
- `tests/test_board.py` — board unit tests
"""

PROFILE = {
    "project_name": "Tazza's Tetris",
    "language": "python",
    "framework": "generic",
    "build_system": "pip",
    "project_id": "tazza-s-tetris",
}


class _FakeRole:
    provider = "anthropic"
    model = "claude-opus-4-8"


# ---------------------------------------------------------------------------
# _parse_planned_files — deterministic containment
# ---------------------------------------------------------------------------

def test_good_plan_parses_all_files():
    files = _parse_planned_files(GOOD_PLAN)
    assert [p for p, _ in files] == [
        "src/main.py", "src/board.py", "src/pieces.py", "src/audio.py",
        "tests/test_board.py",
    ]
    assert files[0][1].startswith("entry point")


def test_garbage_and_dangerous_lines_rejected():
    bad = """
Sure! Here's the plan:
- `src/nested/dir.py` — nested dirs not allowed
- `C:/Windows/evil.py` — absolute path
- `src/../escape.py` — traversal
- `src/binary.exe` — bad extension
- src/no_backticks.py plain token is fine
- `src/ok.py` — good
"""
    files = _parse_planned_files(bad)
    assert [p for p, _ in files] == ["src/no_backticks.py", "src/ok.py"]


def test_dedup_and_cap():
    lines = "\n".join(f"- `src/mod{i}.py` — m" for i in range(40))
    lines += "\n- `src/mod0.py` — duplicate"
    files = _parse_planned_files(lines)
    assert len(files) == 24  # capped
    assert len({p for p, _ in files}) == 24


def test_rendered_section_roundtrips_through_extractor():
    """The whole point: planned paths MUST be visible to segmentation."""
    files = _parse_planned_files(GOOD_PLAN)
    spec = "# SPoT Spec — Greenfield Project\n\n" + render_files_section(files)
    extracted = _extract_file_scope_from_spec(spec)
    extracted_norm = {p.replace("\\", "/") for p in extracted}
    assert {"src/main.py", "src/board.py", "src/pieces.py", "src/audio.py"} <= extracted_norm


# ---------------------------------------------------------------------------
# plan_greenfield_files — fail-soft LLM orchestration
# ---------------------------------------------------------------------------

def _patch_planner(monkeypatch, llm_text=None, llm_exc=None):
    monkeypatch.setattr(
        "app.llm.stage_roles.resolve_stage_role", lambda role: _FakeRole()
    )

    async def fake_call_llm(**kwargs):
        if llm_exc:
            raise llm_exc
        return llm_text

    monkeypatch.setattr("app.pipeline_v2.llm_caller.call_llm", fake_call_llm)


def test_planner_returns_section_on_good_output(monkeypatch):
    _patch_planner(monkeypatch, llm_text=GOOD_PLAN)
    section = asyncio.run(plan_greenfield_files("Tetris", "build it", PROFILE))
    assert section is not None
    assert section.startswith("## Files to create")
    assert "`src/main.py`" in section


def test_planner_none_on_garbage_output(monkeypatch):
    _patch_planner(monkeypatch, llm_text="I cannot plan files, sorry.")
    assert asyncio.run(plan_greenfield_files("Tetris", "build it", PROFILE)) is None


def test_planner_none_on_llm_failure(monkeypatch):
    _patch_planner(monkeypatch, llm_exc=RuntimeError("boom"))
    assert asyncio.run(plan_greenfield_files("Tetris", "build it", PROFILE)) is None


# ---------------------------------------------------------------------------
# Anthropic sampling-param retry helpers (the judge 400)
# ---------------------------------------------------------------------------

def test_live_incident_error_is_sampling_param_error():
    live_err = (
        "Error code: 400 - {'type': 'error', 'error': {'type': "
        "'invalid_request_error', 'message': '`temperature` is deprecated "
        "for this model.'}, 'request_id': 'req_011CchqGZMJ1CL4DqJMfGinz'}"
    )
    assert is_sampling_param_error(live_err) is True


def test_unrelated_400_not_matched():
    assert is_sampling_param_error(
        "Error code: 400 - {'type': 'error', 'error': {'type': "
        "'invalid_request_error', 'message': 'max_tokens too large'}}"
    ) is False
    assert is_sampling_param_error("") is False


def test_strip_sampling_params_removes_all():
    kw = {"model": "m", "temperature": 0.2, "top_p": 0.9, "top_k": 40, "max_tokens": 100}
    strip_sampling_params(kw)
    assert kw == {"model": "m", "max_tokens": 100}
