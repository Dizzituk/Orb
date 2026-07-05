# FILE: tests/test_target_persistence.py
# Purpose: JOB 1 verify — dynamic BuildTargetProfile persistence across restarts.
# Called-by: pytest
# Depends-on: app.pipeline_v2.target_persistence, app.pipeline_v2.target_registry
# Last-renovated: 2026-07-04
"""
JOB 1 verify block (greenfield build-target routing fixpack):

a. register a synthetic profile -> the JSON store contains it
b. clear _REGISTRY of the synthetic entry, load_dynamic_profiles ->
   profile round-trips with ALL fields equal
c. attempt to persist a built-in ID -> store unchanged

Plus: registry init reload (simulated restart), unregister, corrupt-store
tolerance, and forward-compat field filtering.

Run with: pytest tests/test_target_persistence.py -v
"""
import json

import pytest

from app.pipeline_v2.build_targets import BuildTargetProfile
from app.pipeline_v2.target_persistence import (
    BUILTIN_PROFILE_IDS,
    load_dynamic_profiles,
    remove_dynamic_profile,
    save_dynamic_profile,
    _store_path,
)
from app.pipeline_v2.target_registry import (
    _load_persisted_profiles,
    _REGISTRY,
    ASTRA_BACKEND,
    get_profile,
    register_profile,
    unregister_profile,
)

SYNTH_ID = "pytest-synth-app"


def _make_synthetic_profile() -> BuildTargetProfile:
    """A dynamic profile exercising nested-dict, list and optional fields."""
    return BuildTargetProfile(
        project_id=SYNTH_ID,
        project_name="Pytest Synth App",
        project_root="D:/Astra Projects/PytestSynthApp",
        language="kotlin",
        build_system="gradle",
        framework="jetpack-compose",
        source_root="app/src/main/java/com/astra/pytestsynthapp",
        package_name="com.astra.pytestsynthapp",
        architecture_pattern="mvvm",
        key_directories={"views": "ui/", "data": "data/"},
        syntax_check_cmd="echo syntax",
        build_cmd="echo build",
        boot_cmd=None,
        clean_cmd="echo clean",
        verification_mode="emulator",
        emulator_config={"avd_name": "Synth_AVD", "api_level": 35},
        screenshot_method="adb-screencap",
        file_extension=".kt",
        test_extension="Test.kt",
        manifest_file="app/src/main/AndroidManifest.xml",
        dependency_file="app/build.gradle.kts",
        dependency_add_pattern='implementation("{package}")',
        related_targets=["astra-backend"],
        shared_contracts=["app/api/router.py"],
        path_signals=["pytestsynthapp/", "com/astra/pytestsynthapp/"],
    )


@pytest.fixture
def tmp_store(monkeypatch, tmp_path):
    """Point the persistence store at a throwaway file; clean registry after."""
    store = tmp_path / "build_targets.json"
    monkeypatch.setenv("ASTRA_BUILD_TARGETS_STORE", str(store))
    yield store
    _REGISTRY.pop(SYNTH_ID, None)


class TestPersistence:
    def test_register_persists_to_json(self, tmp_store):
        """Verify (a): register_profile writes the profile into the JSON store."""
        profile = _make_synthetic_profile()
        register_profile(profile)

        assert tmp_store.exists(), "store file was not created"
        raw = json.loads(tmp_store.read_text(encoding="utf-8"))
        assert SYNTH_ID in raw
        assert raw[SYNTH_ID]["project_root"] == profile.project_root
        assert raw[SYNTH_ID]["language"] == "kotlin"
        assert raw[SYNTH_ID]["key_directories"] == {"views": "ui/", "data": "data/"}

    def test_round_trip_all_fields(self, tmp_store):
        """Verify (b): full-field round-trip through the store."""
        profile = _make_synthetic_profile()
        assert save_dynamic_profile(profile) is True

        _REGISTRY.pop(SYNTH_ID, None)  # simulate restart memory loss
        loaded = [p for p in load_dynamic_profiles() if p.project_id == SYNTH_ID]
        assert len(loaded) == 1
        # Dataclass equality compares EVERY field, nested containers included.
        assert loaded[0] == profile

    def test_builtin_never_persisted(self, tmp_store):
        """Verify (c): built-in IDs are refused; the store stays unchanged."""
        assert ASTRA_BACKEND.project_id in BUILTIN_PROFILE_IDS
        assert save_dynamic_profile(ASTRA_BACKEND) is False
        assert not tmp_store.exists(), "store should not be created for a built-in"

        # Same guard through the public register path (baseline: synth only).
        register_profile(_make_synthetic_profile())
        register_profile(ASTRA_BACKEND)
        raw = json.loads(tmp_store.read_text(encoding="utf-8"))
        assert ASTRA_BACKEND.project_id not in raw
        assert set(raw.keys()) == {SYNTH_ID}

    def test_registry_reload_after_restart(self, tmp_store):
        """Registry init hook restores persisted profiles; built-ins win collisions."""
        profile = _make_synthetic_profile()
        register_profile(profile)
        _REGISTRY.pop(SYNTH_ID, None)  # simulate restart memory loss
        assert get_profile(SYNTH_ID) is None

        # Plant a malicious/stale built-in entry in the store; it must be ignored.
        raw = json.loads(tmp_store.read_text(encoding="utf-8"))
        raw[ASTRA_BACKEND.project_id] = {"project_name": "EVIL OVERRIDE"}
        tmp_store.write_text(json.dumps(raw), encoding="utf-8")

        restored = _load_persisted_profiles()
        assert restored == 1
        assert get_profile(SYNTH_ID) == profile
        assert get_profile(ASTRA_BACKEND.project_id) is ASTRA_BACKEND

    def test_unregister_removes_registry_and_store(self, tmp_store):
        profile = _make_synthetic_profile()
        register_profile(profile)
        assert unregister_profile(SYNTH_ID) is True
        assert get_profile(SYNTH_ID) is None
        raw = json.loads(tmp_store.read_text(encoding="utf-8"))
        assert SYNTH_ID not in raw
        # Built-ins are refused and stay registered.
        assert unregister_profile(ASTRA_BACKEND.project_id) is False
        assert get_profile(ASTRA_BACKEND.project_id) is ASTRA_BACKEND

    def test_corrupt_store_tolerated(self, tmp_store):
        tmp_store.write_text("{not valid json!!", encoding="utf-8")
        assert load_dynamic_profiles() == []          # no raise
        assert remove_dynamic_profile("whatever") is False
        # A save after corruption rebuilds the store cleanly.
        assert save_dynamic_profile(_make_synthetic_profile()) is True
        raw = json.loads(tmp_store.read_text(encoding="utf-8"))
        assert set(raw.keys()) == {SYNTH_ID}

    def test_unknown_fields_dropped_on_load(self, tmp_store):
        """Forward-compat: extra keys from a newer schema must not break loading."""
        profile = _make_synthetic_profile()
        save_dynamic_profile(profile)
        raw = json.loads(tmp_store.read_text(encoding="utf-8"))
        raw[SYNTH_ID]["field_from_the_future"] = {"nested": True}
        tmp_store.write_text(json.dumps(raw), encoding="utf-8")

        loaded = [p for p in load_dynamic_profiles() if p.project_id == SYNTH_ID]
        assert len(loaded) == 1
        assert loaded[0] == profile

    def test_env_override_controls_store_path(self, tmp_store):
        assert _store_path() == tmp_store


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
