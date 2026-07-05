# FILE: tests/test_cost_attribution.py
# Purpose: Derek phase 1 — stage/job_id attribution lands in the cost ledger; rollup aggregates it.
# Called-by: pytest
# Depends-on: app.cost.cost_recorder, app.cost.cost_ledger, app.cost.cost_report, app.llm._streaming_utils_3, app.pipeline_v2.llm_tools
# Last-renovated: 2026-07-04
"""Attribution tests: every instrumented path must write stage-labelled rows."""

import asyncio
import json

import pytest

from app.cost import cost_ledger, cost_pricing, cost_recorder
from app.cost.cost_report import rollup


@pytest.fixture()
def temp_ledger(tmp_path, monkeypatch):
    """Point the cost ledger at a temp dir (LEDGER_DIR is read at import time,
    so patch the module attribute, not the env var)."""
    monkeypatch.setattr(cost_ledger, "LEDGER_DIR", str(tmp_path))
    cost_pricing.reload_pricing()
    yield tmp_path / "cost_ledger.jsonl"


def _rows(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_record_with_stage_and_job(temp_ledger):
    cost_recorder.record_llm_cost(
        provider="openai", model="gpt-5.4-mini",
        prompt_tokens=1000, completion_tokens=100,
        stage="builder_worker", job_id="job-123",
    )
    rows = _rows(temp_ledger)
    assert len(rows) == 1
    r = rows[0]
    assert r["stage"] == "builder_worker"
    assert r["job_id"] == "job-123"
    assert r["cost_usd"] is not None and r["cost_usd"] > 0


def test_unknown_model_records_null_cost(temp_ledger):
    ret = cost_recorder.record_llm_cost(
        provider="local", model="totally-unknown-model",
        prompt_tokens=500, completion_tokens=50,
        stage="builder_worker",
    )
    rows = _rows(temp_ledger)
    assert rows[0]["cost_usd"] is None, "unknown model must record null, not 0.00"
    assert rows[0]["prompt_tokens"] == 500  # tokens still recorded
    assert ret == 0.0  # legacy float return for registry callers


def test_missing_stage_still_records_unknown(temp_ledger):
    cost_recorder.record_llm_cost("openai", "gpt-5.4", 10, 10)
    assert _rows(temp_ledger)[0]["stage"] == "unknown"


def test_stream_usage_hook_records(temp_ledger):
    from app.llm._streaming_utils_3 import _record_stream_usage
    _record_stream_usage(
        {"type": "done", "provider": "openai", "model": "gpt-5.4-mini",
         "usage": {"prompt_tokens": 200, "completion_tokens": 20, "total_tokens": 220}},
        provider="openai", model="gpt-5.4-mini",
        stage="spec_gate", job_id="sg-1",
    )
    rows = _rows(temp_ledger)
    assert len(rows) == 1
    assert rows[0]["stage"] == "spec_gate"
    assert rows[0]["job_id"] == "sg-1"


def test_stream_usage_hook_no_usage_no_row(temp_ledger):
    from app.llm._streaming_utils_3 import _record_stream_usage
    _record_stream_usage({"type": "done"}, "openai", "gpt-5.4", "chat", None)
    assert _rows(temp_ledger) == []


def test_tool_loop_records_per_call(temp_ledger, monkeypatch):
    """run_tool_loop must record one stage-labelled row per API call."""
    from app.pipeline_v2 import llm_tools

    async def fake_call_api(**kwargs):
        return {"fake": True}, 111, 22

    monkeypatch.setattr(llm_tools, "_call_api", fake_call_api)
    monkeypatch.setattr(llm_tools, "_extract_tool_calls", lambda r, p: [])
    monkeypatch.setattr(llm_tools, "_extract_text", lambda r, p: "BUILDER_COMPLETE")
    monkeypatch.setattr(llm_tools, "_make_assistant_message", lambda r, p: {"role": "assistant", "content": "x"})

    messages, in_tok, out_tok = asyncio.run(llm_tools.run_tool_loop(
        system_prompt="s", initial_user_message="u",
        provider="openai", model="gpt-5.4-mini",
        max_iterations=3, stage="builder_worker", job_id="job-77",
    ))
    assert (in_tok, out_tok) == (111, 22)
    rows = _rows(temp_ledger)
    assert len(rows) == 1
    assert rows[0]["stage"] == "builder_worker"
    assert rows[0]["job_id"] == "job-77"
    assert rows[0]["prompt_tokens"] == 111


def test_rollup_aggregation(temp_ledger):
    cost_recorder.record_llm_cost("openai", "gpt-5.4-mini", 1000, 100, stage="builder_worker", job_id="j1")
    cost_recorder.record_llm_cost("openai", "gpt-5.4-mini", 2000, 200, stage="builder_worker", job_id="j1")
    cost_recorder.record_llm_cost("anthropic", "claude-sonnet-5", 500, 50, stage="checkout_judge", job_id="j1")
    cost_recorder.record_llm_cost("local", "no-such-model", 100, 10, stage="builder_worker", job_id="j2")

    by_stage = rollup(by="stage", days=1)
    assert set(by_stage["buckets"]) == {"builder_worker", "checkout_judge"}
    bw = by_stage["buckets"]["builder_worker"]
    assert bw["records"] == 3
    assert bw["unpriced_records"] == 1
    assert bw["prompt_tokens"] == 3100
    assert by_stage["totals"]["unpriced_records"] == 1

    by_job = rollup(by="job", days=1)
    assert by_job["buckets"]["j1"]["records"] == 3

    only_j2 = rollup(by="stage", days=1, job_id="j2")
    assert only_j2["totals"]["records"] == 1

    with pytest.raises(ValueError):
        rollup(by="nonsense")
