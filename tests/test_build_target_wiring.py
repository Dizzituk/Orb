# FILE: tests/test_build_target_wiring.py
# Purpose: JOB 4 verify — ProjectSession/BuildProject → SpecGate profile wiring.
# Called-by: pytest
# Depends-on: app.llm.project_scoping_stream, app.llm.spec_gate_stream, app.pot_spec.grounded.spec_runner
# Last-renovated: 2026-07-04
"""
JOB 4 verify block (greenfield build-target routing fixpack):

1. Synthetic ProjectSession + registered profile -> generate_spec_gate_stream
   injects build_target_profile into constraints_hint and hands it to
   run_spec_gate_grounded (spy).
2. spec_runner given that hint enters the greenfield lane (v2.4 GREENFIELD
   MODE log line) instead of path-scraping the chat.
3. handle_scoping_confirmation durably stamps BuildProject.build_target_id
   (+ registry persistence), so the wiring survives a backend restart.

Run with: pytest tests/test_build_target_wiring.py -v
"""
import json
from unittest.mock import MagicMock

import pytest

from app.pipeline_v2.build_targets import BuildTargetProfile
from app.pipeline_v2.target_registry import (
    _load_persisted_profiles,
    _REGISTRY,
    get_profile,
    register_profile,
)
from app.shared_context.project_session import (
    clear_project_session,
    get_project_session,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _mock_db():
    db = MagicMock()
    db.query.side_effect = Exception("no db in this test")
    return db


def _sse_text(chunks) -> str:
    """Collapse SSE byte/str chunks into the streamed token text."""
    out = []
    for chunk in chunks:
        raw = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
        for line in raw.split("\n"):
            if line.startswith("data: "):
                try:
                    evt = json.loads(line[6:])
                except Exception:
                    continue
                if evt.get("type") == "token":
                    out.append(evt.get("content") or "")
    return "".join(out)


def _wiring_profile(root: str) -> BuildTargetProfile:
    return BuildTargetProfile(
        project_id="wiring-test-app",
        project_name="Wiring Test App",
        project_root=root,
        language="python",
        build_system="pip",
        framework="generic",
        source_root="src",
        package_name="wiring-test-app",
        architecture_pattern="modular",
    )


class TestProfileInjection:
    @pytest.mark.anyio
    async def test_session_profile_lands_in_constraints_hint(
        self, monkeypatch, tmp_path, capsys,
    ):
        """Active ProjectSession + registered profile -> hint injected + passed on."""
        from app.llm import spec_gate_stream as sgs
        from app.pot_spec.spec_gate_types import SpecGateResult

        monkeypatch.setenv("ASTRA_BUILD_TARGETS_STORE", str(tmp_path / "store.json"))
        pid = 987201
        root = str(tmp_path / "WiringApp")
        (tmp_path / "WiringApp" / "src").mkdir(parents=True)

        register_profile(_wiring_profile(root))
        session = get_project_session(str(pid))
        session.finish_scoping(
            project_id="wiring-test-app", project_name="Wiring Test App",
            project_type="backend", project_root=root,
        )

        captured = {}

        async def _spy(**kwargs):
            captured.update(kwargs)
            return SpecGateResult(
                ready_for_pipeline=False, hard_stopped=True,
                hard_stop_reason="spy stop", validation_status="blocked",
            )

        monkeypatch.setattr(sgs, "run_spec_gate_grounded", _spy)
        monkeypatch.setattr(sgs, "_SPEC_GATE_GROUNDED_AVAILABLE", True)
        monkeypatch.setattr(sgs, "_USE_GROUNDED_SPEC_GATE", True)
        monkeypatch.setattr(sgs, "_SPEC_GATE_V2_AVAILABLE", True)
        monkeypatch.setattr(sgs, "run_spec_gate_v2", lambda **k: None)
        monkeypatch.setattr(sgs, "_resolve_spec_gate_model", lambda: ("test-provider", "test-model"))

        try:
            chunks = [c async for c in sgs.generate_spec_gate_stream(
                project_id=pid, message="trivial job", db=_mock_db(),
            )]
        finally:
            clear_project_session(str(pid))
            _REGISTRY.pop("wiring-test-app", None)

        assert chunks, "stream yielded nothing"
        hint = captured.get("constraints_hint")
        assert hint is not None, "run_spec_gate_grounded never received constraints_hint"
        profile = hint.get("build_target_profile")
        assert profile is not None, "build_target_profile missing from constraints_hint"
        assert profile["project_id"] == "wiring-test-app"
        assert profile["project_root"] == root
        assert profile["source_root"] == "src"
        # JOB 4 lane logging: injection point says which lane the job takes.
        out = capsys.readouterr().out
        assert "BUILD TARGET" in out
        assert "greenfield=yes" in out  # empty src dir -> greenfield lane

    @pytest.mark.anyio
    async def test_spec_runner_receives_profile_and_enters_greenfield_lane(
        self, monkeypatch, tmp_path, capsys,
    ):
        """The injected hint drives spec_runner into greenfield mode (no path scraping)."""
        from app.pot_spec.grounded import spec_runner as sr
        from app.pot_spec.spec_gate_types import SpecGateResult

        monkeypatch.setenv("ORB_JOB_ARTIFACT_ROOT", str(tmp_path / "jobs"))
        root = str(tmp_path / "WiringApp")
        (tmp_path / "WiringApp" / "src").mkdir(parents=True)

        async def _no_segmentation(*args, **kwargs):
            return None, None, False

        def _stub_result(**kwargs):
            return SpecGateResult(
                ready_for_pipeline=True, validation_status="validated",
                spot_markdown=kwargs.get("spot_markdown"),
            )

        monkeypatch.setattr(sr, "run_segmentation_check", _no_segmentation)
        monkeypatch.setattr(sr, "build_spec_result", _stub_result)

        hint = {
            "project_id": 987202,
            "weaver_job_description_text": (
                "What is being built: Standalone recipe organiser for the kitchen PC\n"
            ),
            "build_target_profile": {
                "project_id": "wiring-test-app",
                "project_name": "Wiring Test App",
                "project_root": root,
                "language": "python",
                "framework": "generic",
                "build_system": "pip",
                "source_root": "src",
                "package_name": "wiring-test-app",
                "architecture_pattern": "modular",
            },
        }
        result = await sr.run_spec_gate_grounded(
            db=_mock_db(), job_id="wt-lane-1", user_intent="trivial job",
            provider_id="test-provider", model_id="test-model",
            project_id=987202, constraints_hint=hint, spec_version=1,
        )

        out = capsys.readouterr().out
        assert "GREENFIELD MODE" in out, "spec_runner did not enter the greenfield lane"
        assert result.ready_for_pipeline is True
        assert result.hard_stopped is False


class TestScopingConfirmationStamp:
    @pytest.mark.anyio
    async def test_confirmation_stamps_build_project_and_survives_restart(
        self, monkeypatch, tmp_path,
    ):
        from app.builds import pipeline_bridge
        from app.llm.project_scoping_stream import handle_scoping_confirmation

        monkeypatch.setenv("ASTRA_BUILD_TARGETS_STORE", str(tmp_path / "store.json"))

        class _BP:
            id = "bp-wiring-test"
            build_target_id = None
            target_path = None

        bp = _BP()
        monkeypatch.setattr(
            pipeline_bridge, "get_or_create_build_project",
            lambda db, chat_project_id, brief=None: bp,
        )
        made_dirs = []
        import os as _os
        monkeypatch.setattr(_os, "makedirs", lambda path, exist_ok=False: made_dirs.append(path))

        pid = 987203
        session = get_project_session(str(pid))
        session.start_scoping()
        session.add_scoping_message("user", "Android app")
        session.add_scoping_message("user", "Recipe Organiser")
        session.add_scoping_message(
            "user",
            "It should let me plan weekly meals and store the shopping list locally on the kitchen PC",
        )

        db = MagicMock()
        try:
            chunks = [c async for c in handle_scoping_confirmation(
                project_id=pid, message="yes", db=db,
            )]
            text = _sse_text(chunks)

            assert "Registered build target" in text
            assert "bound to target" in text
            # The BuildProject row is stamped -> SpecGate Path 2 works post-restart.
            assert bp.build_target_id == "recipe-organiser"
            assert bp.target_path == "D:/Astra Android Folder/RecipeOrganiser"
            assert db.commit.called
            assert made_dirs, "project folder creation was not attempted"

            # Registry half: profile registered AND persisted...
            assert get_profile("recipe-organiser") is not None
            store = json.loads((tmp_path / "store.json").read_text(encoding="utf-8"))
            assert "recipe-organiser" in store

            # ...and it survives a simulated restart (memory wiped, then reload).
            _REGISTRY.pop("recipe-organiser", None)
            assert get_profile("recipe-organiser") is None
            assert _load_persisted_profiles() == 1
            restored = get_profile("recipe-organiser")
            assert restored is not None
            assert restored.project_root == "D:/Astra Android Folder/RecipeOrganiser"
        finally:
            clear_project_session(str(pid))
            _REGISTRY.pop("recipe-organiser", None)


class TestGreenfieldOverridesKeywordTargets:
    """v2.7/v3.3 (2026-07-04): the Weaver's greenfield verdict beats keyword-
    scored targets. Live incident: 'desktop app'/'PC' chatter in the Tetris
    ramble stamped astra-frontend on the BuildProject; Path 2 injected it and
    the no-build-target gate never fired."""

    @staticmethod
    def _weaver_json(job_class):
        return ({
            "job_description": "What is being built: Tazza's Tetris desktop game\n",
            "source": "weaver_simple",
            "title": "Job Description from Weaver",
            "job_class": job_class,
        }, {"weaver_source": "flow_state"})

    async def _run_stream(self, monkeypatch, pid, job_class, build_target_id):
        from app.builds import pipeline_bridge
        from app.llm import spec_gate_stream as sgs
        from app.pot_spec.spec_gate_types import SpecGateResult

        captured = {}

        async def _spy(**kwargs):
            captured.update(kwargs)
            return SpecGateResult(
                ready_for_pipeline=False, hard_stopped=True,
                hard_stop_reason="spy stop", validation_status="blocked",
            )

        class _BP:
            id = "bp-noise"
        _BP.build_target_id = build_target_id

        monkeypatch.setattr(sgs, "run_spec_gate_grounded", _spy)
        monkeypatch.setattr(sgs, "_SPEC_GATE_GROUNDED_AVAILABLE", True)
        monkeypatch.setattr(sgs, "_USE_GROUNDED_SPEC_GATE", True)
        monkeypatch.setattr(sgs, "_SPEC_GATE_V2_AVAILABLE", True)
        monkeypatch.setattr(sgs, "run_spec_gate_v2", lambda **k: None)
        monkeypatch.setattr(sgs, "_resolve_spec_gate_model", lambda: ("test-provider", "test-model"))
        monkeypatch.setattr(
            sgs, "_load_latest_weaver_spec_json",
            lambda db, p: self._weaver_json(job_class),
        )
        monkeypatch.setattr(
            pipeline_bridge, "get_or_create_build_project",
            lambda db, cpid, brief=None: _BP(),
        )

        _ = [c async for c in sgs.generate_spec_gate_stream(
            project_id=pid, message="Send to Spec Gate", db=MagicMock(),
        )]
        return captured

    @pytest.mark.anyio
    async def test_greenfield_discards_keyword_scored_builtin(self, monkeypatch, capsys):
        """The exact live failure: stale astra-frontend stamp + greenfield flag."""
        captured = await self._run_stream(
            monkeypatch, 987301, "greenfield_new_app", "astra-frontend",
        )
        hint = captured.get("constraints_hint")
        assert hint is not None
        assert "build_target_profile" not in hint, (
            "keyword-scored built-in target must be discarded for greenfield jobs "
            "(otherwise the no-build-target gate never fires)"
        )
        assert hint["weaver_job_class"] == "greenfield_new_app"
        assert "GREENFIELD OVERRIDE" in capsys.readouterr().out

    @pytest.mark.anyio
    async def test_greenfield_keeps_scoped_dynamic_target(self, monkeypatch, tmp_path):
        """Post-scoping re-send: a dynamic (non-built-in) target must survive."""
        monkeypatch.setenv("ASTRA_BUILD_TARGETS_STORE", str(tmp_path / "store.json"))
        register_profile(_wiring_profile(str(tmp_path / "WiringApp")))
        try:
            captured = await self._run_stream(
                monkeypatch, 987302, "greenfield_new_app", "wiring-test-app",
            )
            profile = captured["constraints_hint"].get("build_target_profile")
            assert profile is not None
            assert profile["project_id"] == "wiring-test-app"
        finally:
            _REGISTRY.pop("wiring-test-app", None)

    @pytest.mark.anyio
    async def test_modify_existing_keeps_builtin_target(self, monkeypatch):
        """ASTRA work is untouched: modify_existing keeps the built-in target."""
        captured = await self._run_stream(
            monkeypatch, 987303, "modify_existing", "astra-frontend",
        )
        profile = captured["constraints_hint"].get("build_target_profile")
        assert profile is not None
        assert profile["project_id"] == "astra-frontend"

    def test_weaver_save_clears_keyword_stamp_for_greenfield(self, monkeypatch):
        """v3.3: the weaver-save hook self-heals a stale built-in stamp."""
        from app.builds import pipeline_bridge

        class _Project:
            build_target_id = "astra-frontend"
            target_path = "D:/orb-desktop"
            target_ids = ["astra-frontend"]
            target_group_id = None
            weaver_extraction = None

        proj = _Project()
        monkeypatch.setattr(pipeline_bridge.build_service, "get_project", lambda db, bpid: proj)
        out = pipeline_bridge.save_weaver_extraction(
            MagicMock(), "bp-x", {"goal": "tetris"}, job_class="greenfield_new_app",
        )
        assert out is proj
        assert proj.build_target_id is None
        assert proj.target_path is None
        assert proj.target_ids is None

    def test_weaver_save_keeps_dynamic_target_for_greenfield(self, monkeypatch):
        from app.builds import pipeline_bridge

        class _Project:
            build_target_id = "recipe-organiser"
            target_path = "D:/Astra Projects/RecipeOrganiser"
            target_ids = ["recipe-organiser"]
            target_group_id = None
            weaver_extraction = None

        proj = _Project()
        monkeypatch.setattr(pipeline_bridge.build_service, "get_project", lambda db, bpid: proj)
        pipeline_bridge.save_weaver_extraction(
            MagicMock(), "bp-y", {"goal": "tetris"}, job_class="greenfield_new_app",
        )
        assert proj.build_target_id == "recipe-organiser"
        assert proj.target_ids == ["recipe-organiser"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
