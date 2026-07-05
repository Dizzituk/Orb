# FILE: tests/test_agent_dispatch.py
# Purpose: Derek phase 4 gate — SpecGate agent dispatch on the HANDS stand-in with cost attribution + ledger evidence.
# Called-by: pytest
# Depends-on: app.pot_spec.grounded.agent_dispatch, app.pipeline_v2.ledger
# Last-renovated: 2026-07-04
"""Agent-dispatch tests: off by default, distils via SPECGATE_AGENT with
stage='specgate_agent', bounded task budget, ledger evidence entries, and
raw evidence preserved on any agent failure."""

import asyncio
import json

import pytest

import app.providers.registry as registry
from app.pot_spec.grounded import agent_dispatch
from app.pot_spec.grounded.agent_dispatch import AgentBudget, distill_evidence_results


class _FakeResult:
    def __init__(self, content="DISTILLED-SUMMARY", ok=True):
        self.content = content
        self._ok = ok
        self.error_message = "" if ok else "boom"

    def is_success(self):
        return self._ok


@pytest.fixture()
def dispatch_on(monkeypatch):
    monkeypatch.setenv("ASTRA_SPECGATE_AGENT_DISPATCH", "1")
    monkeypatch.setenv("ASTRA_SPECGATE_AGENT_MIN_CHARS", "100")
    monkeypatch.setenv("ASTRA_SPECGATE_AGENT_MAX_TASKS", "8")
    # HANDS stand-in resolved via the stage-role registry
    monkeypatch.setenv("ASTRA_STAGE_SPECGATE_AGENT_PROVIDER", "openai")
    monkeypatch.setenv("ASTRA_STAGE_SPECGATE_AGENT_MODEL", "gpt-5.4-mini")
    yield


def _results(n=1, size=500):
    return [
        {"tool": "read_file", "file_path": f"app/mod{i}.py",
         "success": True, "content": "x" * size, "error": None}
        for i in range(n)
    ]


def test_disabled_by_default_is_noop(monkeypatch):
    monkeypatch.delenv("ASTRA_SPECGATE_AGENT_DISPATCH", raising=False)
    calls = []

    async def spy(**kw):
        calls.append(kw)
        return _FakeResult()

    monkeypatch.setattr(registry, "llm_call", spy)
    results = _results()
    out = asyncio.run(distill_evidence_results(results, "need", "goal"))
    assert out[0]["content"] == "x" * 500
    assert calls == []


def test_distils_with_hands_standin_and_stage_attribution(dispatch_on, monkeypatch):
    captured = []

    async def fake_llm_call(**kw):
        captured.append(kw)
        return _FakeResult("KEY SIGNATURES: def foo() ...")

    monkeypatch.setattr(registry, "llm_call", fake_llm_call)
    out = asyncio.run(distill_evidence_results(
        _results(), need="what does mod0 expose", goal="build tetris",
    ))
    assert len(captured) == 1
    kw = captured[0]
    assert kw["stage"] == "specgate_agent"        # cost attribution label
    assert kw["model_id"] == "gpt-5.4-mini"        # HANDS stand-in from env
    assert kw["provider_id"] == "openai"
    assert out[0]["content"].startswith("[agent-distilled by SPECGATE_AGENT")
    assert "KEY SIGNATURES" in out[0]["content"]


def test_small_files_stay_raw(dispatch_on, monkeypatch):
    calls = []

    async def spy(**kw):
        calls.append(kw)
        return _FakeResult()

    monkeypatch.setattr(registry, "llm_call", spy)
    out = asyncio.run(distill_evidence_results(_results(size=50), "n", "g"))
    assert out[0]["content"] == "x" * 50
    assert calls == []


def test_task_budget_is_bounded(dispatch_on, monkeypatch):
    calls = []

    async def spy(**kw):
        calls.append(kw)
        return _FakeResult()

    monkeypatch.setattr(registry, "llm_call", spy)
    budget = AgentBudget(max_tasks=2)
    out = asyncio.run(distill_evidence_results(_results(n=5), "n", "g", budget=budget))
    assert len(calls) == 2
    assert budget.dispatched == 2
    assert out[2]["content"] == "x" * 500  # third and later stay raw


def test_agent_failure_keeps_raw_evidence(dispatch_on, monkeypatch):
    async def failing(**kw):
        return _FakeResult(ok=False)

    monkeypatch.setattr(registry, "llm_call", failing)
    out = asyncio.run(distill_evidence_results(_results(), "n", "g"))
    assert out[0]["content"] == "x" * 500  # raw preserved — never lose ground truth


def test_ledger_evidence_entry_lands(dispatch_on, monkeypatch, tmp_path):
    async def fake_llm_call(**kw):
        return _FakeResult("distilled interface facts")

    monkeypatch.setattr(registry, "llm_call", fake_llm_call)

    import app.pot_spec.grounded._spec_runner_utils_11 as utils11
    monkeypatch.setattr(utils11, "_get_job_dir_for_segmentation", lambda job_id: str(tmp_path / job_id))

    asyncio.run(distill_evidence_results(_results(), "the need", "the goal", job_id="job-ledger-1"))

    ledger_file = tmp_path / "job-ledger-1" / "decision_ledger.json"
    assert ledger_file.exists(), "agent evidence must land in the Decision Ledger"
    data = json.loads(ledger_file.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    assert any(
        e.get("type") == "file_read" and e.get("category") == "specgate_agent"
        and "app/mod0.py" in (e.get("path") or "")
        for e in entries
    ), f"no specgate_agent file_read entry in {entries}"
