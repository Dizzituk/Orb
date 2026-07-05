# FILE: tests/test_local_provider_lane.py
# Purpose: WS1 — local provider lane + background_local locality guard (app/providers/_registry_local.py).
# Called-by: pytest
# Depends-on: app.providers.registry, app.providers._registry_local
# Last-renovated: 2026-07-01
"""
Idle-agents WS1 acceptance:
  - a background_local call against a cloud provider is REFUSED at the registry
  - the same call succeeds against the local (OpenAI-compatible) endpoint
  - provider_id=None under background_local resolves to the local lane
  - local availability keys off LOCAL_LLM_BASE_URL (VLLM_BASE_URL fallback)
  - an exhausted cloud budget never blocks background_local work

Whitelist (documented, enforced by architecture rather than the guard):
embedding calls (app/embeddings/service.py, Gemini) and Brave Search
(app/tools/registry.py:_brave_search) do not ride llm_call chat completions,
so the locality guard cannot touch them.
"""

import pytest
from unittest.mock import patch

from app.providers.registry import (
    LOCALITY_BACKGROUND_LOCAL,
    LlmCallStatus,
    is_provider_available,
    llm_call,
)

_MSGS = [{"role": "user", "content": "ping"}]


class _FakeUsage:
    prompt_tokens = 7
    completion_tokens = 3


class _FakeMessage:
    content = "local says hi"


class _FakeChoice:
    message = _FakeMessage()


class _FakeResp:
    choices = [_FakeChoice()]
    usage = _FakeUsage()

    def model_dump(self):
        return {"fake": True}


def _fake_client_factory(captured: dict):
    class _FakeCompletions:
        async def create(self, **kwargs):
            captured["create_kwargs"] = kwargs
            return _FakeResp()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self, base_url=None, api_key=None, timeout=None):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            captured["timeout"] = timeout

        chat = _FakeChat()

    return _FakeClient


def _set_local_env(monkeypatch, base_url="http://unit.test.local:9/v1", model="unit-test-model"):
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", base_url)
    monkeypatch.setenv("LOCAL_LLM_DEFAULT_MODEL", model)
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)


@pytest.mark.asyncio
async def test_local_roundtrip_against_mocked_endpoint(monkeypatch):
    _set_local_env(monkeypatch)
    captured = {}
    with patch("openai.AsyncOpenAI", _fake_client_factory(captured)):
        result = await llm_call(
            provider_id="local",
            model_id="",  # forces LOCAL_LLM_DEFAULT_MODEL resolution
            messages=_MSGS,
            execution_context=LOCALITY_BACKGROUND_LOCAL,
        )
    assert result.status == LlmCallStatus.SUCCESS
    assert result.provider_id == "local"
    assert result.content == "local says hi"
    assert result.usage.prompt_tokens == 7 and result.usage.completion_tokens == 3
    assert captured["base_url"] == "http://unit.test.local:9/v1"
    assert captured["create_kwargs"]["model"] == "unit-test-model"
    # dummy key accepted (vLLM convention) when LOCAL_LLM_API_KEY unset
    assert captured["api_key"] == "EMPTY" or captured["api_key"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cloud", ["openai", "anthropic", "google", "gemini"])
async def test_background_local_refuses_cloud_providers(monkeypatch, cloud):
    _set_local_env(monkeypatch)
    result = await llm_call(
        provider_id=cloud,
        model_id="any-model",
        messages=_MSGS,
        execution_context=LOCALITY_BACKGROUND_LOCAL,
    )
    assert result.status == LlmCallStatus.LOCALITY_REFUSED
    assert "Locality refused" in (result.error_message or "")


@pytest.mark.asyncio
async def test_background_local_defaults_to_local_provider(monkeypatch):
    _set_local_env(monkeypatch)
    captured = {}
    with patch("openai.AsyncOpenAI", _fake_client_factory(captured)):
        result = await llm_call(
            provider_id=None,
            model_id="",
            messages=_MSGS,
            execution_context=LOCALITY_BACKGROUND_LOCAL,
        )
    assert result.status == LlmCallStatus.SUCCESS
    assert result.provider_id == "local"


@pytest.mark.asyncio
async def test_foreground_cloud_calls_unaffected_by_guard(monkeypatch):
    # Without execution_context the guard must not fire; with no key the
    # provider is simply unavailable (proves no LOCALITY_REFUSED leakage).
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = await llm_call(
        provider_id="openai",
        model_id="any-model",
        messages=_MSGS,
    )
    assert result.status == LlmCallStatus.PROVIDER_UNAVAILABLE


def test_local_unavailable_without_base_url(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    assert is_provider_available("local") is False


def test_local_available_via_vllm_fallback_env(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("VLLM_BASE_URL", "http://unit.test.local:9/v1")
    assert is_provider_available("local") is True


@pytest.mark.asyncio
async def test_exhausted_cloud_budget_never_blocks_background_local(monkeypatch):
    _set_local_env(monkeypatch)
    captured = {}
    with patch("app.cost.cost_recorder.pre_call_budget_check", return_value=(False, "cap reached")):
        # Control: a foreground cloud call IS blocked by the exhausted budget.
        blocked = await llm_call(provider_id="openai", model_id="any-model", messages=_MSGS)
        assert blocked.status == LlmCallStatus.ERROR
        assert "Budget exceeded" in (blocked.error_message or "")

        # Background-local sails through — local compute is free.
        with patch("openai.AsyncOpenAI", _fake_client_factory(captured)):
            result = await llm_call(
                provider_id=None,
                model_id="",
                messages=_MSGS,
                execution_context=LOCALITY_BACKGROUND_LOCAL,
            )
    assert result.status == LlmCallStatus.SUCCESS


@pytest.mark.asyncio
async def test_local_default_model_falls_back_to_nat_model(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://unit.test.local:9/v1")
    monkeypatch.delenv("LOCAL_LLM_DEFAULT_MODEL", raising=False)
    monkeypatch.setenv("NAT_MODEL", "nat-fallback-model")
    captured = {}
    with patch("openai.AsyncOpenAI", _fake_client_factory(captured)):
        result = await llm_call(
            provider_id="local",
            model_id="",
            messages=_MSGS,
            execution_context=LOCALITY_BACKGROUND_LOCAL,
        )
    assert result.status == LlmCallStatus.SUCCESS
    assert captured["create_kwargs"]["model"] == "nat-fallback-model"
