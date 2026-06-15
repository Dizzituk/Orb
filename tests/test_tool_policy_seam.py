# FILE: tests/test_tool_policy_seam.py
# Purpose: Job 7 (2026-06-12) — capability-gating seam at tool dispatch:
#          allow-all by default, deniable per policy, callers unchanged.
# Called-by: pytest
# Depends-on: app.tools.registry
# Last-renovated: 2026-06-12
from __future__ import annotations

import asyncio

import pytest

from app.tools import registry


@pytest.fixture()
def dummy_tool():
    name = "_test_dummy_tool"

    async def _handler(input_data, context=None):
        return {"echo": input_data.get("value", "")}

    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": [],
    }
    out_schema = {
        "type": "object",
        "properties": {"echo": {"type": "string"}},
        "required": [],
    }
    registry.register_tool(registry.ToolDefinition(
        name=name, version="v1", description="test dummy",
        input_schema=schema, output_schema=out_schema, handler=_handler,
    ))
    yield name
    registry._TOOL_DEFS.pop((name, "v1"), None)
    registry.set_tool_policy(None)


def test_default_policy_allows_everything(dummy_tool):
    resp = asyncio.run(registry.execute_tool_async(
        dummy_tool, "v1", {"value": "hi"},
    ))
    assert resp.ok
    assert resp.result == {"echo": "hi"}


def test_policy_can_deny_without_breaking_callers(dummy_tool):
    registry.set_tool_policy(
        lambda name, version, context: name != dummy_tool
    )
    resp = asyncio.run(registry.execute_tool_async(
        dummy_tool, "v1", {"value": "hi"},
    ))
    assert not resp.ok
    assert "capability policy" in resp.error_message
    # Normal ToolResponse shape — no caller changes needed
    assert resp.tool_name == dummy_tool

    registry.set_tool_policy(None)
    resp = asyncio.run(registry.execute_tool_async(
        dummy_tool, "v1", {"value": "hi"},
    ))
    assert resp.ok


def test_failing_policy_fails_open(dummy_tool):
    def _broken(name, version, context):
        raise RuntimeError("policy exploded")

    registry.set_tool_policy(_broken)
    resp = asyncio.run(registry.execute_tool_async(
        dummy_tool, "v1", {"value": "hi"},
    ))
    assert resp.ok, "policy errors must fail open until profiles ship"
