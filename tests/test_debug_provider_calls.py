# FILE: tests/test_debug_provider_calls.py
# Purpose: Debug provider toggle (2026-07-02) — provider_calls adapters:
#          flat-tool -> Anthropic input_schema conversion, OpenAI-shape history
#          -> Anthropic content blocks (merging + thinking round-trip), uniform
#          extraction parity across both providers, and the mocked Anthropic
#          caller contract. No live API calls.
# Called-by: pytest
# Depends-on: app.debug.orchestrator.provider_calls
# Last-renovated: 2026-07-02
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.debug.orchestrator import provider_calls as pc


# ─────────────────────────────────────────────────────────────────────────────
# Tool-schema conversion
# ─────────────────────────────────────────────────────────────────────────────

def test_flat_debug_tool_to_anthropic_input_schema():
    flat = {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]},
    }
    out = pc.to_anthropic_tools([flat])
    assert out == [{
        "name": "read_file",
        "description": "Read a file",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}},
                         "required": ["path"]},
    }]


def test_anthropic_tools_passthrough_junk_and_defaults():
    already = {"name": "x", "description": "d", "input_schema": {"type": "object"}}
    no_params = {"name": "bare"}
    assert pc.to_anthropic_tools([already, no_params, "junk", {}]) == [
        already,
        {"name": "bare", "description": "",
         "input_schema": {"type": "object", "properties": {}}},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-shape history -> Anthropic messages
# ─────────────────────────────────────────────────────────────────────────────

def test_history_system_and_plain_turns():
    system, msgs = pc.openai_history_to_anthropic([
        {"role": "system", "content": "be rigorous"},
        {"role": "user", "content": "investigate the bug"},
        {"role": "assistant", "content": "on it"},
    ])
    assert system == "be rigorous"
    assert msgs == [
        {"role": "user", "content": "investigate the bug"},
        {"role": "assistant", "content": [{"type": "text", "text": "on it"}]},
    ]


def test_history_assistant_tool_calls_become_tool_use_blocks():
    _, msgs = pc.openai_history_to_anthropic([
        {"role": "assistant", "content": "checking",
         "tool_calls": [{"id": "tc1", "type": "function",
                         "function": {"name": "read_file",
                                      "arguments": json.dumps({"path": "a.py"})}}]},
    ])
    assert msgs == [{
        "role": "assistant",
        "content": [
            {"type": "text", "text": "checking"},
            {"type": "tool_use", "id": "tc1", "name": "read_file", "input": {"path": "a.py"}},
        ],
    }]


def test_history_tool_results_merge_into_one_user_message():
    """Consecutive role:'tool' results must land in ONE user message (Anthropic
    rejects non-alternating roles), tool_result blocks in order."""
    _, msgs = pc.openai_history_to_anthropic([
        {"role": "assistant", "content": "",
         "tool_calls": [
             {"id": "tc1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
             {"id": "tc2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
         ]},
        {"role": "tool", "tool_call_id": "tc1", "content": "result one"},
        {"role": "tool", "tool_call_id": "tc2", "content": "result two"},
    ])
    assert [m["role"] for m in msgs] == ["assistant", "user"]
    results = msgs[1]["content"]
    assert [b["type"] for b in results] == ["tool_result", "tool_result"]
    assert [b["tool_use_id"] for b in results] == ["tc1", "tc2"]
    assert results[0]["content"] == "result one"


def test_history_trailing_user_text_appends_after_tool_results():
    """The loop's 'Tool budget exhausted...' user nudge lands as a text block
    AFTER the tool_result blocks in the same user message (results stay first)."""
    _, msgs = pc.openai_history_to_anthropic([
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "tc1", "type": "function",
                         "function": {"name": "a", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "tc1", "content": "res"},
        {"role": "user", "content": "Tool budget exhausted. Produce your findings JSON now."},
    ])
    assert [m["role"] for m in msgs] == ["assistant", "user"]
    blocks = msgs[1]["content"]
    assert [b["type"] for b in blocks] == ["tool_result", "text"]
    assert "budget exhausted" in blocks[1]["text"]


def test_history_anthropic_content_passes_through_verbatim():
    """Assistant turns produced by the Anthropic caller keep their raw blocks —
    including thinking blocks — untouched (API requirement mid tool-use turn)."""
    raw = [
        {"type": "thinking", "thinking": "hmm", "signature": "sig=="},
        {"type": "text", "text": "let me check"},
        {"type": "tool_use", "id": "tc9", "name": "read_file", "input": {"path": "x"}},
    ]
    _, msgs = pc.openai_history_to_anthropic([
        {"role": "assistant", "content": "let me check",
         "tool_calls": [{"id": "tc9", "type": "function",
                         "function": {"name": "read_file", "arguments": "{\"path\": \"x\"}"}}],
         "_anthropic_content": raw},
    ])
    assert msgs == [{"role": "assistant", "content": raw}]


def test_history_empty_assistant_dropped_and_bad_args_tolerated():
    _, msgs = pc.openai_history_to_anthropic([
        {"role": "assistant", "content": ""},  # nothing to say, no tools -> dropped
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "t", "type": "function",
                         "function": {"name": "n", "arguments": "not-json"}}]},
    ])
    assert len(msgs) == 1
    assert msgs[0]["content"] == [{"type": "tool_use", "id": "t", "name": "n", "input": {}}]


# ─────────────────────────────────────────────────────────────────────────────
# Extraction parity (fake SDK responses, both providers)
# ─────────────────────────────────────────────────────────────────────────────

def _fake_openai_resp():
    tc = SimpleNamespace(
        id="tc1",
        function=SimpleNamespace(name="read_file", arguments=json.dumps({"path": "a.py"})),
    )
    msg = SimpleNamespace(content="checking", tool_calls=[tc])
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


class _Block(SimpleNamespace):
    """Anthropic-SDK-ish content block: attribute access + model_dump()."""
    def model_dump(self, exclude_none=True):
        return {k: v for k, v in vars(self).items() if v is not None or not exclude_none}


def _fake_anthropic_resp():
    return SimpleNamespace(
        content=[
            _Block(type="text", text="checking"),
            _Block(type="tool_use", id="tc1", name="read_file", input={"path": "a.py"}),
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def test_extract_tool_calls_parity():
    expected = [{"id": "tc1", "name": "read_file", "args": {"path": "a.py"}}]
    assert pc.extract_tool_calls(_fake_openai_resp(), "openai") == expected
    assert pc.extract_tool_calls(_fake_anthropic_resp(), "anthropic") == expected


def test_extract_final_text_parity():
    assert pc.extract_final_text(_fake_openai_resp(), "openai") == "checking"
    assert pc.extract_final_text(_fake_anthropic_resp(), "anthropic") == "checking"


def test_extract_assistant_message_parity_and_roundtrip():
    oai = pc.extract_assistant_message(_fake_openai_resp(), "openai")
    anth = pc.extract_assistant_message(_fake_anthropic_resp(), "anthropic")
    # Same OpenAI shape the loop appends to its history
    for out in (oai, anth):
        assert out["role"] == "assistant"
        assert out["content"] == "checking"
        assert out["tool_calls"][0]["id"] == "tc1"
        assert out["tool_calls"][0]["function"]["name"] == "read_file"
        assert json.loads(out["tool_calls"][0]["function"]["arguments"]) == {"path": "a.py"}
    # Anthropic stashes raw blocks; converting the history back must reuse them
    assert anth["_anthropic_content"][1]["type"] == "tool_use"
    _, msgs = pc.openai_history_to_anthropic([anth])
    assert msgs == [{"role": "assistant", "content": anth["_anthropic_content"]}]
    assert "_anthropic_content" not in oai


def test_extract_tool_calls_empty_cases():
    no_tools_oai = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="done", tool_calls=None))])
    no_tools_anth = SimpleNamespace(content=[_Block(type="text", text="done")])
    assert pc.extract_tool_calls(no_tools_oai, "openai") == []
    assert pc.extract_tool_calls(no_tools_anth, "anthropic") == []


# ─────────────────────────────────────────────────────────────────────────────
# Mocked Anthropic caller — request contract, no live calls
# ─────────────────────────────────────────────────────────────────────────────

class _FakeAnthropicClient:
    captured: dict = {}

    def __init__(self, api_key=None, timeout=None):
        _FakeAnthropicClient.captured["init"] = {"api_key": api_key, "timeout": timeout}
        self.messages = self

    async def create(self, **kwargs):
        _FakeAnthropicClient.captured["kwargs"] = kwargs
        return _fake_anthropic_resp()


@pytest.fixture()
def fake_anthropic(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    _FakeAnthropicClient.captured = {}
    monkeypatch.setattr(anthropic, "AsyncAnthropic", _FakeAnthropicClient)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return _FakeAnthropicClient


@pytest.mark.asyncio
async def test_call_anthropic_with_tools_contract(fake_anthropic):
    flat_tool = {"name": "read_file", "description": "r",
                 "parameters": {"type": "object", "properties": {}}}
    messages = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "go"},
    ]
    resp, in_tok, out_tok = await pc._call_anthropic_with_tools(
        model="anthro-model-x", messages=messages, tools=[flat_tool],
        max_tokens=8000, reasoning_effort=None,
    )
    kw = fake_anthropic.captured["kwargs"]
    assert kw["model"] == "anthro-model-x"
    assert kw["system"] == "sys prompt"
    assert kw["messages"] == [{"role": "user", "content": "go"}]
    assert kw["tools"][0]["input_schema"] == {"type": "object", "properties": {}}
    assert kw["max_tokens"] == 8000
    assert "thinking" not in kw and "extra_body" not in kw  # no effort -> no thinking cfg
    assert (in_tok, out_tok) == (10, 5)
    assert pc.extract_final_text(resp, "anthropic") == "checking"
    assert fake_anthropic.captured["init"]["timeout"] == pc._LLM_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_call_anthropic_with_tools_maps_reasoning_effort(fake_anthropic):
    await pc._call_anthropic_with_tools(
        model="anthro-model-x",
        messages=[{"role": "user", "content": "go"}],
        tools=[], max_tokens=1000, reasoning_effort="high",
    )
    kw = fake_anthropic.captured["kwargs"]
    # shape_anthropic_create_kwargs applied: effort routed via output_config and
    # max_tokens floored so thinking + output fit. (Exact shape is owned by
    # app.providers._registry_anthropic_shapes — assert the observable facts.)
    assert kw.get("extra_body", {}).get("output_config", {}).get("effort") == "high"
    assert kw["max_tokens"] >= 1000


@pytest.mark.asyncio
async def test_call_anthropic_missing_key_raises(monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        await pc._call_anthropic_with_tools(
            model="anthro-model-x", messages=[{"role": "user", "content": "x"}], tools=[],
        )


@pytest.mark.asyncio
async def test_call_llm_with_tools_dispatches_by_provider(fake_anthropic):
    resp, _, _ = await pc.call_llm_with_tools(
        provider="anthropic", model="anthro-model-x",
        messages=[{"role": "user", "content": "go"}], tools=[],
    )
    assert pc.extract_final_text(resp, "anthropic") == "checking"
    # openai dispatch without a key must fail with the OpenAI key error,
    # proving it reached the OpenAI caller (behaviour unchanged).
    import os
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            await pc.call_llm_with_tools(
                provider="openai", model="oai-model-x",
                messages=[{"role": "user", "content": "go"}], tools=[],
            )
    finally:
        if saved is not None:
            os.environ["OPENAI_API_KEY"] = saved
