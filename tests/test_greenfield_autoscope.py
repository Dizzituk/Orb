# FILE: tests/test_greenfield_autoscope.py
# Purpose: Verify greenfield auto-scope — weave with name+location scopes itself, no manual stop.
# Called-by: pytest
# Depends-on: app.llm.greenfield_autoscope, app.llm._sg_target_injection, app.llm.spec_gate_stream
# Last-renovated: 2026-07-04
"""
v2.8 greenfield AUTO-SCOPE verify (2026-07-04):

- Deterministic extraction of {name, root, slug} from the REAL Tazza's Tetris
  weave (apostrophe in folder name, trailing punctuation, repeated path).
- Full auto-scope side effects: folder created, profile registered+persisted,
  ProjectSession set, BuildProject stamped.
- Stream-level: greenfield weave with an explicit location -> profile injected
  (greenfield lane), auto-scope note streamed; weave WITHOUT a location ->
  no profile (falls back to the v12.0 scoping stop).

Run with: pytest tests/test_greenfield_autoscope.py -v
"""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.llm.greenfield_autoscope import (
    autoscope_greenfield_job,
    extract_greenfield_target,
)
from app.pipeline_v2.target_registry import _REGISTRY, get_profile
from app.shared_context.project_session import (
    clear_project_session,
    get_project_session,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


# Condensed from the real 2026-07-04 09:09 weave that hit the scoping stop.
REAL_TETRIS_WEAVE = """What is being built: Tazza's Tetris desktop game
Intended outcome: Double-clickable PC Tetris app with classic gameplay

Key requirements:
- Treat this as a brand-new project, not a modification of an existing app.
- Create the project from scratch inside a new folder.
- Target folder/location: C:\\Users\\dizzi\\OneDrive\\Documents\\Games\\Tazza's Tetris
- Create the project folder immediately/visibly so the user can see it.

SpecGate must resolve:
- Confirm the target folder path exists or can be created: C:\\Users\\dizzi\\OneDrive\\Documents\\Games\\Tazza's Tetris.
- Ensure the project is initialized as a brand-new standalone desktop game.

Questions for user: none

Job class: greenfield_new_app
"""


class TestExtraction:
    def test_extracts_from_real_tetris_weave(self):
        target = extract_greenfield_target(REAL_TETRIS_WEAVE)
        assert target is not None
        assert target["root"] == "C:/Users/dizzi/OneDrive/Documents/Games/Tazza's Tetris"
        assert target["name"] == "Tazza's Tetris"
        assert target["slug"] == "tazza-s-tetris"

    def test_trailing_punctuation_stripped(self):
        text = "Confirm the folder can be created: D:\\Astra Projects\\Foo Game.\n"
        target = extract_greenfield_target(text)
        assert target["root"] == "D:/Astra Projects/Foo Game"
        assert target["name"] == "Foo Game"

    def test_no_path_returns_none(self):
        assert extract_greenfield_target(
            "What is being built: Tazza's Tetris desktop game\n"
        ) is None
        assert extract_greenfield_target("") is None
        assert extract_greenfield_target(None) is None

    def test_builtin_repo_paths_rejected(self):
        text = (
            "Target folder: D:\\Orb\\app\\newthing\n"
            "Also check D:\\orb-desktop\\src\n"
        )
        assert extract_greenfield_target(text) is None

    def test_bare_drive_rejected(self):
        assert extract_greenfield_target("Put it on C:\\ somewhere\n") is None

    def test_prose_embedded_path_rejected(self):
        """Regex runs to end-of-line — swallowed prose must not become a folder."""
        text = "Create the folder C:\\Users\\dizzi\\Games\\Foo and then open it up\n"
        assert extract_greenfield_target(text) is None

    def test_system_locations_rejected(self):
        assert extract_greenfield_target("Target folder: C:\\Windows\\System32\\Games\n") is None
        assert extract_greenfield_target("Target folder: C:\\Program Files\\Games\\Foo\n") is None
        assert extract_greenfield_target("Target folder: C:\\ProgramData\\Foo\n") is None

    def test_parenthesised_name_preserved(self):
        target = extract_greenfield_target("Target folder: C:\\Users\\dizzi\\Games\\Foo (v2)\n")
        assert target["name"] == "Foo (v2)"
        # ...but an unbalanced sentence paren is stripped.
        target = extract_greenfield_target("(see the folder at C:\\Users\\dizzi\\Games\\Bar)\n")
        assert target["name"] == "Bar"

    def test_new_app_in_android_folder_allowed(self):
        """NEW Android apps legitimately live beside the existing ones — only
        the existing app repo roots themselves are rejected."""
        target = extract_greenfield_target("Target folder: D:\\Astra Android Folder\\BrandNewApp\n")
        assert target is not None and target["name"] == "BrandNewApp"
        assert extract_greenfield_target(
            "Target folder: D:\\Astra Android Folder\\Astra-Bridge\\newstuff\n"
        ) is None


class TestReviewHardening:
    """Fixes from the 2026-07-04 adversarial review (15 confirmed findings)."""

    def test_two_paths_one_line_without_hint_ignored(self):
        """Greedy-merge finding: no hint word on the line -> never adopted."""
        text = "Copy assets from C:\\Users\\dizzi\\Pictures\\Sprites into C:\\Users\\dizzi\\Games\\Tetris\n"
        assert extract_greenfield_target(text) is None

    def test_two_paths_recoverable_with_separate_target_line(self):
        text = (
            "Copy assets from C:\\Users\\dizzi\\Pictures\\Sprites into C:\\Users\\dizzi\\Games\\Tetris\n"
            "Target folder/location: C:\\Users\\dizzi\\Games\\Tetris\n"
        )
        target = extract_greenfield_target(text)
        assert target["root"] == "C:/Users/dizzi/Games/Tetris"

    def test_strong_target_line_beats_earlier_weak_asset_line(self):
        text = (
            "- Load sprite pack from folder C:\\Users\\dizzi\\Pictures\\Sprites\n"
            "- Target folder/location: C:\\Users\\dizzi\\Games\\Tetris\n"
        )
        target = extract_greenfield_target(text)
        assert target["root"] == "C:/Users/dizzi/Games/Tetris"
        assert target["name"] == "Tetris"

    def test_incidental_no_hint_path_never_adopted(self):
        text = "- Reuse the sprite pack in C:\\Users\\dizzi\\Downloads\\sprites\n"
        assert extract_greenfield_target(text) is None

    def test_path_traversal_rejected(self):
        assert extract_greenfield_target(
            "Target folder: D:\\Projects\\..\\Orb\\newthing\n"
        ) is None
        assert extract_greenfield_target(
            "Target folder: C:\\Users\\dizzi\\..\\..\\Windows\\Temp\\EvilGame\n"
        ) is None
        assert extract_greenfield_target(
            "Target folder: D:\\.\\Orb\\app\\newthing\n"
        ) is None

    def test_parenthetical_clause_rejected_by_word_cap(self):
        text = "Target folder/location: C:\\Users\\dizzi\\Games\\Snake (create it if missing)\n"
        assert extract_greenfield_target(text) is None

    def test_unbalanced_square_bracket_stripped(self):
        target = extract_greenfield_target("Target folder: [C:\\Users\\dizzi\\Games\\Foo]\n")
        assert target["name"] == "Foo"

    def test_slug_collision_with_builtin_is_deconflicted(self, monkeypatch, tmp_path):
        """'Driver Copilot' folder must NOT hijack the built-in driver-copilot."""
        import hashlib
        from app.builds import pipeline_bridge
        from app.pipeline_v2.target_registry import DRIVER_COPILOT

        monkeypatch.setenv("ASTRA_BUILD_TARGETS_STORE", str(tmp_path / "store.json"))
        root = tmp_path / "Prototypes" / "Driver Copilot"
        weave = f"Target folder/location: {root}\n"

        class _BP:
            id = "bp-collision"
            build_target_id = None
            target_path = None

        monkeypatch.setattr(
            pipeline_bridge, "get_or_create_build_project",
            lambda db, cpid, brief=None: _BP(),
        )
        norm_root = str(root).replace("\\", "/").rstrip("/").lower()
        expected = "driver-copilot-" + hashlib.sha1(norm_root.encode()).hexdigest()[:6]
        pid = 987405
        try:
            profile = autoscope_greenfield_job(MagicMock(), pid, weave)
            assert profile is not None
            assert profile.project_id == expected
            assert get_profile("driver-copilot") is DRIVER_COPILOT, (
                "built-in profile must never be overwritten"
            )
        finally:
            clear_project_session(str(pid))
            _REGISTRY.pop(expected, None)

    def test_registry_front_door_refuses_builtin_overwrite(self, monkeypatch, tmp_path):
        from app.pipeline_v2.build_targets import BuildTargetProfile
        from app.pipeline_v2.target_registry import ASTRA_BRIDGE, register_profile

        monkeypatch.setenv("ASTRA_BUILD_TARGETS_STORE", str(tmp_path / "store.json"))
        impostor = BuildTargetProfile(
            project_id="astra-bridge", project_name="Impostor",
            project_root=str(tmp_path), language="python", build_system="pip",
            framework="generic", source_root="src", package_name="impostor",
            architecture_pattern="modular",
        )
        register_profile(impostor)
        assert get_profile("astra-bridge") is ASTRA_BRIDGE
        assert not (tmp_path / "store.json").exists()

    def test_existing_populated_folder_carries_on_untouched(self, monkeypatch, tmp_path):
        """Taz directive (2026-07-04 evening): an existing populated target
        folder is fine — leave the files be, carry on. Replaces the earlier
        hard refusal that stopped the live Tetris run."""
        from app.builds import pipeline_bridge

        monkeypatch.setenv("ASTRA_BUILD_TARGETS_STORE", str(tmp_path / "store.json"))
        root = tmp_path / "Games" / "Sprites"
        root.mkdir(parents=True)
        (root / "index.html").write_text("<html>old demo</html>", encoding="utf-8")
        monkeypatch.setattr(
            pipeline_bridge, "get_or_create_build_project",
            lambda db, cpid, brief=None: None,
        )
        pid = 987406
        try:
            result = autoscope_greenfield_job(
                MagicMock(), pid, f"Target folder/location: {root}\n",
            )
            assert result is not None, "populated folder must carry on, not refuse"
            assert result.project_id == "sprites"
            # The existing file is untouched — leave them be
            assert (root / "index.html").read_text(encoding="utf-8") == "<html>old demo</html>"
            assert get_profile("sprites") is result
        finally:
            clear_project_session(str(pid))
            _REGISTRY.pop("sprites", None)


class TestKnownFolderResolution:
    """v1.1 (Taz directive after the live Tetris stop): verbal user-folder
    paths resolve via Windows' User Shell Folders mapping — OneDrive-aware,
    deterministic, never guessed."""

    def _fake_known(self, monkeypatch, tmp_path):
        from app.llm import greenfield_autoscope as ga
        docs = tmp_path / "OneDrive" / "Documents"   # OneDrive-redirected Documents
        docs.mkdir(parents=True)

        def fake_root(token):
            return str(docs) if token.lower().strip() in ("documents", "my documents") else None

        monkeypatch.setattr(ga, "_known_folder_root", fake_root)
        return docs

    def test_live_tetris_target_line_resolves(self, monkeypatch, tmp_path):
        """The EXACT line from tonight's live stop now resolves."""
        docs = self._fake_known(monkeypatch, tmp_path)
        target = extract_greenfield_target(
            "Target folder/location: Documents/Games/Tazza’s Tetris\n"
            .replace("’", "'")
        )
        assert target is not None
        assert target["root"] == str(docs).replace("\\", "/") + "/Games/Tazza's Tetris"
        assert target["name"] == "Tazza's Tetris"

    def test_arrow_prose_variant_resolves(self, monkeypatch, tmp_path):
        docs = self._fake_known(monkeypatch, tmp_path)
        target = extract_greenfield_target(
            "Build it into the existing folder: Documents → Games → Retro Blocks\n"
        )
        assert target is not None
        assert target["root"].endswith("/Games/Retro Blocks")

    def test_unknown_token_still_falls_back(self, monkeypatch, tmp_path):
        self._fake_known(monkeypatch, tmp_path)
        assert extract_greenfield_target(
            "Target folder/location: SomeRandomPlace/Games/Foo\n"
        ) is None

    def test_absolute_path_preferred_over_relative(self, monkeypatch, tmp_path):
        docs = self._fake_known(monkeypatch, tmp_path)
        target = extract_greenfield_target(
            f"Target folder/location: {tmp_path}\\Elsewhere\\App (was Documents/Games/App)\n"
        )
        assert target is not None
        assert "elsewhere" in target["root"].lower()

    def test_user_folder_root_itself_rejected(self, monkeypatch, tmp_path):
        from app.llm import greenfield_autoscope as ga
        docs = self._fake_known(monkeypatch, tmp_path)
        monkeypatch.setattr(
            ga, "_protected_user_roots_lower",
            lambda: [str(docs).replace("\\", "/").lower()],
        )
        # Absolute path pointing AT the Documents root — never a destination
        assert extract_greenfield_target(
            f"Target folder/location: {docs}\n"
        ) is None

    def test_live5_descriptor_word_stripped_and_curly_apostrophe_normalised(self, monkeypatch, tmp_path):
        """EXACT replay of the 21:29 live failure: the 'Unresolved ambiguities'
        line minted 'Tazza’s Tetris location' (curly apostrophe + trailing
        descriptor). Must now yield the REAL folder name."""
        docs = self._fake_known(monkeypatch, tmp_path)
        line = (
            "The exact absolute filesystem path is not provided, only the "
            "named Documents/Games/Tazza’s Tetris location."
        )
        target = extract_greenfield_target(line)
        assert target is not None
        assert target["name"] == "Tazza's Tetris"          # straight apostrophe, no 'location'
        assert target["root"] == str(docs).replace("\\", "/") + "/Games/Tazza's Tetris"
        assert target["slug"] == "tazza-s-tetris"

    def test_live5_descriptor_only_name_rejected(self, monkeypatch, tmp_path):
        self._fake_known(monkeypatch, tmp_path)
        assert extract_greenfield_target(
            "Verify the path: Documents/location\n"
        ) is None

    def test_registry_resolver_returns_real_dir_or_none(self):
        """Live probe of the real resolver on this machine — whatever it
        returns must be a real directory (or None on non-redirected setups)."""
        from app.llm.greenfield_autoscope import _known_folder_root
        resolved = _known_folder_root("documents")
        assert resolved is None or Path(resolved).is_dir()


class TestAutoscopeSideEffects:
    def test_full_autoscope(self, monkeypatch, tmp_path):
        from app.builds import pipeline_bridge

        monkeypatch.setenv("ASTRA_BUILD_TARGETS_STORE", str(tmp_path / "store.json"))
        root = tmp_path / "Games" / "Tazza's Tetris"
        weave = f"Target folder/location: {root}\nJob class: greenfield_new_app\n"

        class _BP:
            id = "bp-autoscope"
            build_target_id = "astra-frontend"   # stale keyword stamp
            target_path = "D:/orb-desktop"

        bp = _BP()
        monkeypatch.setattr(
            pipeline_bridge, "get_or_create_build_project",
            lambda db, cpid, brief=None: bp,
        )

        db = MagicMock()
        pid = 987401
        # Hermetic guard (live9): the LIVE registry may already hold a real
        # 'tazza-s-tetris' profile (registered by an actual Tetris run) at a
        # different root — the slug de-conflictor would then correctly mint
        # 'tazza-s-tetris-<suffix>' and fail the exact-slug asserts below.
        # Tests must not depend on what the user built that day.
        _REGISTRY.pop("tazza-s-tetris", None)
        try:
            profile = autoscope_greenfield_job(db, pid, weave)

            assert profile is not None
            assert profile.project_id == "tazza-s-tetris"
            assert Path(root).is_dir(), "project folder must be created visibly"
            assert get_profile("tazza-s-tetris") is profile
            store = json.loads((tmp_path / "store.json").read_text(encoding="utf-8"))
            assert "tazza-s-tetris" in store, "target must persist across restarts"
            session = get_project_session(str(pid))
            assert session.project_id == "tazza-s-tetris"
            assert bp.build_target_id == "tazza-s-tetris"
            assert bp.target_path == profile.project_root
            assert db.commit.called
        finally:
            clear_project_session(str(pid))
            _REGISTRY.pop("tazza-s-tetris", None)

    def test_no_location_means_no_side_effects(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ASTRA_BUILD_TARGETS_STORE", str(tmp_path / "store.json"))
        pid = 987402
        result = autoscope_greenfield_job(
            MagicMock(), pid, "What is being built: something brand new\n",
        )
        assert result is None
        assert not (tmp_path / "store.json").exists()
        assert not get_project_session(str(pid)).is_set
        clear_project_session(str(pid))


class TestAutoscopeStream:
    """Stream-level: the exact live failure now self-scopes and proceeds."""

    async def _run_stream(self, monkeypatch, pid, job_description):
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
            build_target_id = "astra-frontend"   # stale keyword stamp
            target_path = "D:/orb-desktop"

        monkeypatch.setattr(sgs, "run_spec_gate_grounded", _spy)
        monkeypatch.setattr(sgs, "_SPEC_GATE_GROUNDED_AVAILABLE", True)
        monkeypatch.setattr(sgs, "_USE_GROUNDED_SPEC_GATE", True)
        monkeypatch.setattr(sgs, "_SPEC_GATE_V2_AVAILABLE", True)
        monkeypatch.setattr(sgs, "run_spec_gate_v2", lambda **k: None)
        monkeypatch.setattr(sgs, "_resolve_spec_gate_model", lambda: ("test-provider", "test-model"))
        monkeypatch.setattr(
            sgs, "_load_latest_weaver_spec_json",
            lambda db, p: ({
                "job_description": job_description,
                "source": "weaver_simple",
                "title": "Job Description from Weaver",
                "job_class": "greenfield_new_app",
            }, {"weaver_source": "flow_state"}),
        )
        monkeypatch.setattr(
            pipeline_bridge, "get_or_create_build_project",
            lambda db, cpid, brief=None: _BP(),
        )

        chunks = [c async for c in sgs.generate_spec_gate_stream(
            project_id=pid, message="Send to Spec Gate", db=MagicMock(),
        )]
        text = "".join(
            json.loads(line[6:]).get("content") or ""
            for chunk in chunks
            for line in (chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)).split("\n")
            if line.startswith("data: ") and '"token"' in line
        )
        return captured, text

    @pytest.mark.anyio
    async def test_weave_with_location_autoscopes_and_proceeds(
        self, monkeypatch, tmp_path, capsys,
    ):
        monkeypatch.setenv("ASTRA_BUILD_TARGETS_STORE", str(tmp_path / "store.json"))
        root = tmp_path / "Games" / "Tazza's Tetris"
        weave = (
            "What is being built: Tazza's Tetris desktop game\n"
            f"- Target folder/location: {root}\n"
            "Job class: greenfield_new_app\n"
        )
        pid = 987403
        try:
            captured, text = await self._run_stream(monkeypatch, pid, weave)

            profile = captured["constraints_hint"].get("build_target_profile")
            assert profile is not None, "auto-scope must inject the new target"
            assert profile["project_id"] == "tazza-s-tetris"
            assert profile["project_root"] == str(root).replace("\\", "/")
            assert Path(root).is_dir()
            assert "auto-scoped" in text.lower(), "user must see the auto-scope note"

            out = capsys.readouterr().out
            assert "GREENFIELD OVERRIDE" in out      # stale astra-frontend discarded
            assert "AUTO-SCOPED" in out
            assert "greenfield=yes" in out           # correct lane visible
        finally:
            clear_project_session(str(pid))
            _REGISTRY.pop("tazza-s-tetris", None)

    @pytest.mark.anyio
    async def test_weave_without_location_still_stops_for_scoping(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setenv("ASTRA_BUILD_TARGETS_STORE", str(tmp_path / "store.json"))
        pid = 987404
        try:
            captured, _ = await self._run_stream(
                monkeypatch, pid,
                "What is being built: Tazza's Tetris desktop game\n"
                "Job class: greenfield_new_app\n",
            )
            assert "build_target_profile" not in captured["constraints_hint"], (
                "no location in the weave -> no guessing; the v12.0 scoping "
                "stop must remain the fallback"
            )
        finally:
            clear_project_session(str(pid))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
