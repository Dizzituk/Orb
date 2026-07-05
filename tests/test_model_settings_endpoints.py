# FILE: tests/test_model_settings_endpoints.py
# Purpose: LANE D endpoint machinery tests — role inventory contract, atomic
#          .env rewrite (temp-then-replace, marker append, parse validation),
#          reload diff, and PUT input validation.
# Called-by: pytest
# Depends-on: app.llm.model_roles, app.settings.env_model_store,
#             app.settings.model_settings
# Last-renovated: 2026-07-02
from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.llm import model_roles
from app.settings import env_model_store


@pytest.fixture(autouse=True)
def _restore_environ():
    """reload_env() writes into os.environ (override=True) — restore the
    whole environment after every test so nothing leaks into other suites."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


# ─────────────────────────────────────────────────────────────────────────────
# Role inventory (GET /settings/models contract)
# ─────────────────────────────────────────────────────────────────────────────

def test_inventory_has_core_roles_and_groups():
    roles = model_roles.list_roles()
    names = {r["role"] for r in roles}
    for expected in ("CHAT", "ARCHITECT", "REASONING", "SPEC_GATE", "WEAVER",
                     "RAG_DEEP", "NAT", "IMAGE_GEN", "JOB_CONTINUATION",
                     "anthropic:frontier-verifier"):
        assert expected in names, f"role inventory lost {expected}"
    groups = {r["group"] for r in roles}
    assert groups <= set(model_roles.GROUP_ORDER), f"unknown group(s): {groups - set(model_roles.GROUP_ORDER)}"


def test_inventory_entry_shape_and_verifier_flags():
    roles = {r["role"]: r for r in model_roles.list_roles()}
    for key in ("role", "group", "description", "provider", "model", "source",
                "provider_env", "model_env", "verifier", "restart_gated", "restart_note"):
        assert key in roles["CHAT"], f"contract field {key} missing"
    assert roles["SPEC_GATE"]["verifier"] is True
    assert roles["OVERWATCHER"]["verifier"] is True
    assert roles["CRITIQUE"]["verifier"] is True
    assert roles["CHAT"]["verifier"] is False
    # Restart-gated audit entries surface with notes
    assert roles["GEMINI_FRONTIER"]["restart_gated"] is True
    assert roles["GEMINI_FRONTIER"]["restart_note"]
    assert roles["anthropic:frontier-verifier"]["restart_gated"] is True


def test_get_role_spec_writable_vars():
    spec = model_roles.get_role_spec("CHAT")
    assert spec is not None
    assert spec["writable_vars"] == ["CHAT_PROVIDER", "CHAT_MODEL"]
    env_shaped = model_roles.get_role_spec("GEMINI_VISION_FAST")
    assert env_shaped is not None
    assert env_shaped["writable_vars"] == ["GEMINI_VISION_MODEL_FAST"]
    assert model_roles.get_role_spec("NOPE_NOT_A_ROLE") is None


# ─────────────────────────────────────────────────────────────────────────────
# env_model_store — atomic rewrite against a SCRATCH .env (never the real one)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def scratch_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# header comment\n"
        "CHAT_PROVIDER=openai\n"
        "CHAT_MODEL=gpt-old\n"
        "OTHER=keep\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(env_model_store, "ENV_PATH", env_file)
    monkeypatch.setattr(env_model_store, "_BACKUP_DIR", tmp_path / "backups")
    return env_file


def test_set_env_vars_rewrites_in_place(scratch_env):
    diff = env_model_store.set_env_vars({"CHAT_MODEL": "gpt-new"})
    assert diff == {"CHAT_MODEL": ("gpt-old", "gpt-new")}
    text = scratch_env.read_text(encoding="utf-8")
    assert "CHAT_MODEL=gpt-new" in text
    assert "gpt-old" not in text
    assert "OTHER=keep" in text            # untouched line survives
    assert text.startswith("# header comment")  # no reorder


def test_set_env_vars_appends_under_ui_marker(scratch_env):
    env_model_store.set_env_vars({"BRAND_NEW_MODEL": "abc-1"})
    text = scratch_env.read_text(encoding="utf-8")
    marker_pos = text.index("## UI-MANAGED MODELS")
    assert text.index("BRAND_NEW_MODEL=abc-1") > marker_pos


def test_set_env_vars_rewrites_last_occurrence(scratch_env):
    # dotenv gives later lines precedence — the LAST occurrence must change.
    with scratch_env.open("a", encoding="utf-8") as fh:
        fh.write("CHAT_MODEL=gpt-later\n")
    env_model_store.set_env_vars({"CHAT_MODEL": "gpt-final"})
    text = scratch_env.read_text(encoding="utf-8")
    assert text.count("CHAT_MODEL=") == 2
    assert "CHAT_MODEL=gpt-old" in text          # earlier occurrence untouched
    assert "CHAT_MODEL=gpt-final" in text        # last occurrence rewritten
    assert "gpt-later" not in text
    from dotenv import dotenv_values
    assert dotenv_values(str(scratch_env))["CHAT_MODEL"] == "gpt-final"


def test_set_env_vars_refuses_bad_values(scratch_env):
    before = scratch_env.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        env_model_store.set_env_vars({"CHAT_MODEL": "evil\nINJECTED=1"})
    with pytest.raises(ValueError):
        env_model_store.set_env_vars({"BAD NAME": "x"})
    assert scratch_env.read_text(encoding="utf-8") == before  # untouched


def test_reload_env_diffs_model_vars(scratch_env, monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "gpt-old")
    scratch_env.write_text("CHAT_MODEL=gpt-new\n", encoding="utf-8")
    diff = env_model_store.reload_env()
    assert diff.get("CHAT_MODEL") == ("gpt-old", "gpt-new")
    assert os.environ["CHAT_MODEL"] == "gpt-new"


def test_is_model_var_excludes_secrets():
    assert env_model_store.is_model_var("CHAT_MODEL")
    assert env_model_store.is_model_var("ARCHITECT_PROVIDER")
    assert env_model_store.is_model_var("ASTRA_FALLBACK_CHAIN_CODE")
    assert not env_model_store.is_model_var("OPENAI_API_KEY")
    assert not env_model_store.is_model_var("SOME_SECRET")


# ─────────────────────────────────────────────────────────────────────────────
# PUT /settings/models/{role} validation (handler called directly)
# ─────────────────────────────────────────────────────────────────────────────

def test_put_rejects_bad_model_id(scratch_env):
    from fastapi import HTTPException
    from app.settings.model_settings import PutRoleRequest, put_role_model
    with pytest.raises(HTTPException) as exc:
        put_role_model("CHAT", PutRoleRequest(model="bad model with spaces"))
    assert exc.value.status_code == 400
    with pytest.raises(HTTPException) as exc:
        put_role_model("TOTALLY_UNKNOWN_ROLE", PutRoleRequest(model="gpt-x"))
    assert exc.value.status_code == 404


def test_put_writes_role_vars_to_scratch_env(scratch_env):
    from app.settings.model_settings import PutRoleRequest, put_role_model
    result = put_role_model("CHAT", PutRoleRequest(provider="openai", model="gpt-new-model"))
    assert result["role"] == "CHAT"
    text = scratch_env.read_text(encoding="utf-8")
    assert "CHAT_MODEL=gpt-new-model" in text
    assert "CHAT_PROVIDER=openai" in text
    assert result["reloaded"] is True
    # live resolution reflects the write (env was reloaded from scratch file)
    assert result["live"]["model"] == "gpt-new-model"


# ─────────────────────────────────────────────────────────────────────────────
# DEBUG_PROVIDER switch (debug provider toggle, 2026-07-02): PUT round-trips
# through set_env_vars -> reload_env and the live inventory flips columns.
# ─────────────────────────────────────────────────────────────────────────────

def test_debug_provider_switch_row_shape():
    roles = {r["role"]: r for r in model_roles.list_roles()}
    switch = roles["DEBUG_PROVIDER"]
    assert switch["group"] == "Debug Assistant"
    assert switch["kind"] == "provider_switch"
    assert switch["values"] == ["openai", "anthropic"]
    assert switch["model_env"] == "DEBUG_PROVIDER"
    assert switch["provider_env"] is None
    # debug role rows live in the same group with column-aware env vars
    assert roles["DEBUG_SUBAGENT"]["group"] == "Debug Assistant"


def test_put_debug_provider_round_trips(scratch_env, monkeypatch):
    from app.settings.model_settings import PutRoleRequest, put_role_model
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_DEFAULT_MODEL", "anthro-default")
    monkeypatch.delenv("DEBUG_PROVIDER", raising=False)
    # hermetic: ambient shells may carry the real .env's anthropic column
    monkeypatch.delenv("DEBUG_SUBAGENT_MODEL_ANTHROPIC", raising=False)

    result = put_role_model("DEBUG_PROVIDER", PutRoleRequest(model="anthropic"))
    text = scratch_env.read_text(encoding="utf-8")
    assert "DEBUG_PROVIDER=anthropic" in text          # written to .env
    assert os.environ["DEBUG_PROVIDER"] == "anthropic"  # hot-reloaded
    assert result["live"]["provider"] == "anthropic"
    # the debug role rows now resolve on the anthropic column
    spec = model_roles.get_role_spec("DEBUG_SUBAGENT")
    assert spec["model_env"] == "DEBUG_SUBAGENT_MODEL_ANTHROPIC"
    assert (spec["provider"], spec["model"]) == ("anthropic", "anthro-default")

    # flip back
    put_role_model("DEBUG_PROVIDER", PutRoleRequest(model="openai"))
    assert os.environ["DEBUG_PROVIDER"] == "openai"
    assert model_roles.get_role_spec("DEBUG_SUBAGENT")["model_env"] == "DEBUG_SUBAGENT_MODEL"


def test_put_debug_provider_rejects_unknown_value(scratch_env):
    from fastapi import HTTPException
    from app.settings.model_settings import PutRoleRequest, put_role_model
    before = scratch_env.read_text(encoding="utf-8")
    with pytest.raises(HTTPException) as exc:
        put_role_model("DEBUG_PROVIDER", PutRoleRequest(model="banana"))
    assert exc.value.status_code == 400
    assert scratch_env.read_text(encoding="utf-8") == before  # .env untouched
