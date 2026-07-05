# FILE: tests/test_web_reliability_policy.py
# Purpose: Amendment A Job 9 — role+name click targeting, wait_for in the chat
#          set, re-orient policy, posting-tool redirect, defs-file split.
# Called-by: pytest
# Depends-on: app.debug.web_tool_definitions, app.web_automation.tool_handlers, app.debug.web_tool_playbooks
# Last-renovated: 2026-07-02
"""
Acceptance criteria under test (Amendment A v2):
  AC15 — web_click supports role+name targeting resolved in-page at click
         time; descriptions rewritten; the "coordinates are more reliable"
         claim is gone.
  AC16 (offline half) — web_wait_for is in the chat tool set and mandated
         after navigations / state-changing clicks; RETRY_POLICY forbids
         blind retries after a timeout.
  Job 9c — META_UPLOAD_PLAYBOOK routes standard posting to the posting
         tools; upload defs split out; every touched file under the 30KB cap.
"""
from __future__ import annotations

import os

import pytest

from app.debug import web_tool_definitions as defs
from app.debug import web_tool_playbooks as playbooks
from app.debug.web_tool_definitions import get_web_tools
from app.web_automation import action_types, tool_handlers
from app.web_automation.tool_schemas import WEB_CLICK


def _tool(name: str) -> dict:
    match = [t for t in get_web_tools() if t["name"] == name]
    assert match, f"{name} missing from get_web_tools()"
    return match[0]


# ── 9a · role+name targeting ─────────────────────────────────────────

def test_click_schemas_expose_role_and_name():
    assert "role" in WEB_CLICK["input"]["properties"]
    assert "name" in WEB_CLICK["input"]["properties"]
    chat_params = _tool("web_click")["parameters"]["properties"]
    assert "role" in chat_params and "name" in chat_params


def test_click_description_prefers_role_name_and_warns_stale_coords():
    desc = _tool("web_click")["description"]
    # Preferred order: role+name leads, coords demoted with the staleness warning.
    assert desc.index("role + name") < desc.index("CSS `selector`") < desc.index("(x, y)")
    assert "immediately before" in desc
    assert "stale" in desc.lower()
    assert "re-render" in desc.lower()
    assert "resolved" in desc


def test_coordinates_more_reliable_claim_is_gone():
    hay = " ".join(t["description"] for t in get_web_tools())
    hay += " ".join(tool_handlers.TOOL_DESCRIPTIONS.values())
    assert "more reliable than selectors" not in hay
    assert "Coordinates are more reliable" not in hay


def test_dom_snapshot_description_routes_to_role_name():
    desc = _tool("web_dom_snapshot")["description"]
    assert "role + name" in desc or "role+name" in desc
    assert "immediately after" in desc  # coords validity window


@pytest.mark.asyncio
async def test_click_handler_forwards_role_name_and_resolved(monkeypatch):
    captured = {}

    class _S:
        id = "sess-1"

    async def fake_exec(sid, action_type, payload, timeout_seconds=30.0):
        captured["action"] = action_type
        captured["payload"] = payload
        return {"ok": True, "result": {
            "clicked_at": {"x": 10, "y": 20},
            "resolved": {"tag": "button", "role": "button", "text": "Publish", "match": "exact"},
            "changed": True,
        }}

    monkeypatch.setattr(tool_handlers, "_resolve_session", lambda ref: _S())
    monkeypatch.setattr(tool_handlers.bridge, "execute_action", fake_exec)

    out = await tool_handlers.click_handler(
        {"session": "meta_business", "role": "button", "name": "Publish"}, None
    )
    assert captured["action"] == "click"
    assert captured["payload"] == {"name": "Publish", "role": "button"}
    assert out["ok"] is True
    assert out["resolved"]["match"] == "exact"


@pytest.mark.asyncio
async def test_click_handler_rejects_targetless_call():
    out = await tool_handlers.click_handler({"session": "meta_business"}, None)
    assert out["ok"] is False
    assert "role+name" in out["error"]


def test_action_types_click_constructor_role_name():
    action, payload = action_types.click(name="Next Item", role="button")
    assert action == "click"
    assert payload == {"name": "Next Item", "role": "button"}
    action, payload = action_types.click(name="Next Item")
    assert payload == {"name": "Next Item"}
    # selector still wins over role/name; targetless still raises
    action, payload = action_types.click("div.ok", name="ignored")
    assert payload == {"selector": "div.ok"}
    with pytest.raises(ValueError):
        action_types.click()


@pytest.mark.asyncio
async def test_chat_executor_forwards_role_name(monkeypatch):
    from app.debug.executors import web_automation as chat_exec

    captured = {}

    async def fake_handler(input_data, context):
        captured.update(input_data)
        return {"ok": True, "changed": True}

    monkeypatch.setattr(chat_exec, "click_handler", fake_handler)
    await chat_exec.execute_web_click(
        {"session": "coursera", "role": "tab", "name": "Transcript"}
    )
    assert captured["role"] == "tab" and captured["name"] == "Transcript"


# ── 9b · wait_for exposure + re-orient policy ────────────────────────

def test_wait_for_is_in_chat_set_and_dispatchable():
    tool = _tool("web_wait_for")
    desc = tool["description"]
    assert "after EVERY web_navigate" in desc
    assert "BEFORE the next read or action" in desc
    assert "do NOT blindly retry" in desc

    from app.debug.action_executor import TOOL_HANDLERS
    assert "web_wait_for" in TOOL_HANDLERS


@pytest.mark.asyncio
async def test_chat_wait_for_executor_forwards_and_reports(monkeypatch):
    from app.debug.executors import web_automation as chat_exec

    captured = {}

    async def fake_handler(input_data, context):
        captured.update(input_data)
        return {"ok": True, "matched": False, "timeout": True, "waited_ms": 1500}

    monkeypatch.setattr(chat_exec, "wait_for_handler", fake_handler)
    out = await chat_exec.execute_web_wait_for(
        {"session": "coursera", "selector": "h1", "state": "visible", "timeout_ms": 1500}
    )
    assert captured["selector"] == "h1" and captured["timeout_ms"] == 1500
    assert '"timeout": true' in out  # timeout surfaced, not an exception


def test_navigate_description_mandates_wait_for():
    desc = _tool("web_navigate")["description"]
    assert "load event" in desc
    assert "web_wait_for" in desc


def test_retry_policy_has_reorient_rule():
    assert "RE-ORIENT" in playbooks.RETRY_POLICY
    assert "Never re-click blind" in playbooks.RETRY_POLICY


# ── 9c · posting-tool redirect + file split ──────────────────────────

def test_meta_playbook_redirects_standard_posting():
    head = playbooks.META_UPLOAD_PLAYBOOK[:800]
    assert "post_image_to_instagram" in head
    assert "post_reel_to_instagram" in head
    assert "FALLBACK" in head  # manual flow demoted, still documented


def test_upload_defs_split_and_still_exposed():
    from app.debug.web_upload_tool_definitions import (
        SYSTEM_KEYS_TOOL,
        WEB_UPLOAD_FILE_TOOL,
    )
    names = {t["name"] for t in get_web_tools()}
    assert {"web_upload_file", "system_keys"} <= names
    assert WEB_UPLOAD_FILE_TOOL["name"] == "web_upload_file"
    assert SYSTEM_KEYS_TOOL["name"] == "system_keys"
    # The redirect flows into the upload tool's live description.
    assert "post_image_to_instagram" in WEB_UPLOAD_FILE_TOOL["description"]


def test_touched_files_under_hard_cap():
    for mod in (defs, playbooks):
        assert os.path.getsize(mod.__file__) <= 30 * 1024, mod.__file__
    from app.debug import web_upload_tool_definitions as upload_defs
    assert os.path.getsize(upload_defs.__file__) <= 30 * 1024


# ── wiring parity (extends the coursera_reader parity test) ──────────

def test_registry_parity_including_study_tools():
    from app.web_automation.coursera_reader import (
        COURSERA_DESCRIPTIONS, COURSERA_HANDLERS, COURSERA_SCHEMAS,
    )
    from app.web_automation.coursera_study_tools import (
        COURSERA_STUDY_DESCRIPTIONS, COURSERA_STUDY_HANDLERS, COURSERA_STUDY_SCHEMAS,
    )
    from app.web_automation.tool_handlers import HANDLERS, TOOL_DESCRIPTIONS
    from app.web_automation.tool_schemas import TOOL_SCHEMAS

    handlers = {**HANDLERS, **COURSERA_HANDLERS, **COURSERA_STUDY_HANDLERS}
    schemas = {**TOOL_SCHEMAS, **COURSERA_SCHEMAS, **COURSERA_STUDY_SCHEMAS}
    descriptions = {**TOOL_DESCRIPTIONS, **COURSERA_DESCRIPTIONS, **COURSERA_STUDY_DESCRIPTIONS}
    assert set(handlers) - set(schemas) == set()
    assert set(handlers) - set(descriptions) == set()


def test_every_chat_web_tool_is_dispatchable():
    from app.debug.action_executor import TOOL_HANDLERS
    missing = [t["name"] for t in get_web_tools() if t["name"] not in TOOL_HANDLERS]
    assert missing == []
