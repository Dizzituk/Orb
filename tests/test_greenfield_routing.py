# FILE: tests/test_greenfield_routing.py
# Purpose: JOBs 2+3 verify — greenfield jobs stop for scoping, never the sandbox lane.
# Called-by: pytest
# Depends-on: app.llm.weaver_job_class, app.pot_spec.grounded._greenfield_gate, app.pot_spec.grounded.spec_runner
# Last-renovated: 2026-07-04
"""
JOB 2 verify block (greenfield build-target routing fixpack):
- Tetris weaver fixture + no build profile + sandbox offline
  -> v12.0 "No Build Target" stop, NOT the v11.0 sandbox stop,
     and launch_sandbox is NEVER called.
- Regression: a job explicitly about D:\\Orb files with sandbox offline
  still produces the v11.0 sandbox hard stop.

JOB 3 verify block:
- Weaver job_class parser: strict parse, "unknown" default, never raises.
- Weaver text carrying job_class greenfield_new_app (and the hint variant)
  + no build profile -> same scoping-prompt stop, flag alone sufficient.

Run with: pytest tests/test_greenfield_routing.py -v
"""
from unittest.mock import MagicMock

import pytest

from app.llm.weaver_job_class import parse_weaver_job_class
from app.pot_spec.grounded import _sbx_fs
from app.pot_spec.grounded._greenfield_gate import (
    NO_BUILD_TARGET_STOP_REASON,
    is_unrouted_greenfield_job,
    profile_is_greenfield,
)
from app.pot_spec.grounded.spec_runner import run_spec_gate_grounded


@pytest.fixture
def anyio_backend():
    return "asyncio"


# Saved-style Tetris weaver output (the job that triggered this fixpack).
TETRIS_WEAVER_TEXT = """What is being built: Tetris desktop game
Intended outcome: A playable classic Tetris running as a standalone desktop app

Key requirements:
- Create a new project folder for the game
- Classic tetromino gameplay: rotation, soft drop, hard drop
- Line clearing with score and level progression
- Next piece preview and game over screen

SpecGate must resolve:
- Determine rendering approach for the playfield grid

Questions for user: none
"""

# Genuine ASTRA-repo job — no greenfield/game keywords, explicit D:\Orb path.
ASTRA_REPO_WEAVER_TEXT = (
    "What is being built: Retry coalescing hardening\n"
    "Intended outcome: Duplicate sends collapse into one row\n\n"
    "Key requirements:\n"
    "- Update the retry coalescing logic in the backend at D:\\Orb\\app\\bridge "
    "so duplicate sends collapse into one row\n"
)

# Neutral new-app text with NO greenfield/game domain keywords at all —
# proves the Weaver flag ALONE routes it (JOB 3).
FLAG_ONLY_WEAVER_TEXT = (
    "What is being built: Standalone recipe organiser for the kitchen PC\n"
    "Intended outcome: Weekly meal plans stored locally\n\n"
    "**Job class**: greenfield_new_app\n"
)


def _mock_db():
    db = MagicMock()
    db.query.side_effect = Exception("no db in this test")
    return db


def _offline_sandbox(monkeypatch):
    """Sandbox offline; launch attempts are recorded, never really run."""
    launches = []
    monkeypatch.setattr(_sbx_fs, "sandbox_available", lambda: False)
    monkeypatch.setattr(
        _sbx_fs, "launch_sandbox",
        lambda wait_seconds=60: (launches.append(wait_seconds), False)[1],
    )
    return launches


# =============================================================================
# JOB 2 — spec runner routing
# =============================================================================

class TestGreenfieldNoBuildTargetStop:
    @pytest.mark.anyio
    async def test_tetris_without_profile_gets_scoping_stop_not_sandbox(self, monkeypatch):
        launches = _offline_sandbox(monkeypatch)

        result = await run_spec_gate_grounded(
            db=_mock_db(),
            job_id="gf-test-tetris",
            user_intent="send that to spec gate",
            provider_id="test-provider",
            model_id="test-model",
            project_id=987101,
            constraints_hint={
                "project_id": 987101,
                "weaver_job_description_text": TETRIS_WEAVER_TEXT,
                "weaver_source": "weaver_simple",
            },
            spec_version=1,
        )

        assert result.hard_stopped is True
        assert "No Build Target" in (result.hard_stop_reason or "")
        assert "let's start a new project" in (result.hard_stop_reason or "")
        assert "Sandbox Unavailable" not in (result.hard_stop_reason or "")
        assert result.validation_status == "blocked"
        assert launches == [], "launch_sandbox must NEVER be called for greenfield jobs"

    @pytest.mark.anyio
    async def test_astra_repo_job_still_gets_v11_sandbox_stop(self, monkeypatch):
        """Regression: genuine ASTRA-repo jobs keep the v11.0 behaviour."""
        launches = _offline_sandbox(monkeypatch)

        result = await run_spec_gate_grounded(
            db=_mock_db(),
            job_id="gf-test-astra",
            user_intent="send that to spec gate",
            provider_id="test-provider",
            model_id="test-model",
            project_id=987102,
            constraints_hint={
                "project_id": 987102,
                "weaver_job_description_text": ASTRA_REPO_WEAVER_TEXT,
                "weaver_source": "weaver_simple",
            },
            spec_version=1,
        )

        assert result.hard_stopped is True
        assert "Sandbox Unavailable" in (result.hard_stop_reason or "")
        assert "No Build Target" not in (result.hard_stop_reason or "")
        assert launches == [60], "v11.0 auto-start attempt must still happen"


class TestGreenfieldGateUnit:
    def test_domain_hit_stops(self):
        stop, why = is_unrouted_greenfield_job({}, TETRIS_WEAVER_TEXT, "")
        assert stop is True
        assert "domains=" in why

    def test_no_signals_falls_through(self):
        stop, why = is_unrouted_greenfield_job({}, ASTRA_REPO_WEAVER_TEXT, "")
        assert stop is False

    def test_modify_existing_flag_suppresses_domain_fallback(self):
        # Even with Tetris keywords in the text, an explicit Weaver
        # modify_existing classification wins over keyword heuristics.
        hint = {"weaver_job_class": "modify_existing"}
        stop, why = is_unrouted_greenfield_job(hint, TETRIS_WEAVER_TEXT, "")
        assert stop is False
        assert "modify_existing" in why

    def test_hint_flag_alone_stops(self):
        hint = {"weaver_job_class": "greenfield_new_app"}
        stop, why = is_unrouted_greenfield_job(hint, "totally neutral text", "")
        assert stop is True
        assert "greenfield_new_app" in why

    def test_user_intent_fallback_when_no_weaver_text(self):
        stop, why = is_unrouted_greenfield_job({}, "", "build a Tetris desktop game from scratch")
        assert stop is True

    def test_profile_is_greenfield(self, tmp_path):
        empty_root = tmp_path / "NewApp"
        (empty_root / "src").mkdir(parents=True)
        profile = {"project_root": str(empty_root), "source_root": "src"}
        assert profile_is_greenfield(profile) is True

        (empty_root / "src" / "main.py").write_text("print('hi')", encoding="utf-8")
        assert profile_is_greenfield(profile) is False

        # ASTRA self-roots are never greenfield, and no profile is no profile.
        assert profile_is_greenfield({"project_root": "D:/Orb", "source_root": "app"}) is False
        assert profile_is_greenfield(None) is False

    def test_stop_reason_is_distinct_from_v11(self):
        assert "No Build Target" in NO_BUILD_TARGET_STOP_REASON
        assert "Sandbox" not in NO_BUILD_TARGET_STOP_REASON


# =============================================================================
# JOB 3 — Weaver job_class flag
# =============================================================================

class TestJobClassParser:
    @pytest.mark.parametrize("line,expected", [
        ("**Job class**: greenfield_new_app", "greenfield_new_app"),
        ("- **Job class**: modify_existing", "modify_existing"),
        ("Job class: unknown", "unknown"),
        ("JOB_CLASS: greenfield_new_app", "greenfield_new_app"),
        ("job class = modify_existing", "modify_existing"),
        ("**Job class**: **greenfield_new_app**", "greenfield_new_app"),
        ("Job Class: GREENFIELD_NEW_APP", "greenfield_new_app"),
    ])
    def test_valid_variants_parse(self, line, expected):
        text = f"What is being built: something\n{line}\nQuestions for user: none\n"
        assert parse_weaver_job_class(text) == expected

    def test_missing_field_defaults_unknown_without_error(self):
        assert parse_weaver_job_class(TETRIS_WEAVER_TEXT) == "unknown"
        assert parse_weaver_job_class("") == "unknown"
        assert parse_weaver_job_class(None) == "unknown"

    def test_invalid_values_are_rejected(self):
        assert parse_weaver_job_class("Job class: brand_new_thing") == "unknown"
        assert parse_weaver_job_class("Job class: greenfield") == "unknown"
        # Mid-sentence mentions don't count — the line must start with the key.
        assert parse_weaver_job_class("we discussed the job class: modify_existing idea") == "unknown"

    def test_prompts_instruct_the_field(self):
        from app.llm._weaver_prompts import (
            WEAVER_CREATE_SYSTEM_PROMPT,
            WEAVER_UPDATE_SYSTEM_PROMPT,
        )
        for prompt in (WEAVER_CREATE_SYSTEM_PROMPT, WEAVER_UPDATE_SYSTEM_PROMPT):
            assert "Job class" in prompt
            assert "greenfield_new_app" in prompt
            assert "modify_existing" in prompt

    def test_loader_attaches_job_class(self, monkeypatch):
        """_load_latest_weaver_spec_json exposes the parsed flag to SpecGate."""
        from app.llm import _spec_gate_stream_utils_2 as utils2

        class _Flow:
            weaver_job_description = FLAG_ONLY_WEAVER_TEXT

        monkeypatch.setattr(utils2, "_FLOW_STATE_AVAILABLE", True)
        monkeypatch.setattr(utils2, "get_active_flow", lambda pid: _Flow())
        spec_json, prov = utils2._load_latest_weaver_spec_json(MagicMock(), 987103)
        assert spec_json["source"] == "weaver_simple"
        assert spec_json["job_class"] == "greenfield_new_app"


class TestJobClassEndToEnd:
    @pytest.mark.anyio
    async def test_flag_in_weaver_text_alone_routes_to_scoping_stop(self, monkeypatch):
        """No greenfield keywords anywhere — the Weaver flag is the only signal."""
        launches = _offline_sandbox(monkeypatch)

        result = await run_spec_gate_grounded(
            db=_mock_db(),
            job_id="gf-test-flag",
            user_intent="send that to spec gate",
            provider_id="test-provider",
            model_id="test-model",
            project_id=987104,
            constraints_hint={
                "project_id": 987104,
                "weaver_job_description_text": FLAG_ONLY_WEAVER_TEXT,
                "weaver_source": "weaver_simple",
            },
            spec_version=1,
        )

        assert result.hard_stopped is True
        assert "No Build Target" in (result.hard_stop_reason or "")
        assert launches == []

    @pytest.mark.anyio
    async def test_hint_flag_variant_routes_to_scoping_stop(self, monkeypatch):
        """Same stop when spec_gate_stream delivered the flag via the hint."""
        launches = _offline_sandbox(monkeypatch)

        result = await run_spec_gate_grounded(
            db=_mock_db(),
            job_id="gf-test-hint",
            user_intent="send that to spec gate",
            provider_id="test-provider",
            model_id="test-model",
            project_id=987105,
            constraints_hint={
                "project_id": 987105,
                "weaver_job_description_text": (
                    "What is being built: Standalone recipe organiser for the kitchen PC\n"
                ),
                "weaver_job_class": "greenfield_new_app",
            },
            spec_version=1,
        )

        assert result.hard_stopped is True
        assert "No Build Target" in (result.hard_stop_reason or "")
        assert launches == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
