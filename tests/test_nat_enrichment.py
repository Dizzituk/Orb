# FILE: tests/test_nat_enrichment.py
# Purpose: Guard Nat Job 2 (pre-reply enrichment): suggest->deterministic-fetch,
#          the hard timeout (non-blocking), the circuit breaker, the disable
#          switch, and the sync wrapper. Latency is the make-or-break constraint,
#          so these prove enrichment can NEVER block or over-wait.
# Last-renovated: 2026-06-19
import asyncio
import os
import types

import app.llm.workers.nat_worker as nw
import app.astra_memory.retrieval as retr
from app.memory.nat_jobs import enrichment as enr


def _reset():
    enr._last_fail[0] = 0.0
    os.environ["NAT_ENRICHMENT_ENABLED"] = "1"
    os.environ["NAT_ENRICHMENT_TIMEOUT_MS"] = "400"
    os.environ["NAT_ENRICHMENT_COOLDOWN_S"] = "30"


def _fake_retrieve(records):
    def _f(db, q, **kw):
        return types.SimpleNamespace(records=records)
    return _f


def test_enrichment_suggest_then_deterministic_fetch(monkeypatch):
    _reset()
    calls = {"nat": 0}

    async def fake_nat(system, user, **kw):
        calls["nat"] += 1
        return '["calorie history for last month", "weight trend"]'

    recs = [types.SimpleNamespace(title="2026-05-19 nutrition", content="hit target: 2100 kcal")]
    monkeypatch.setattr(nw, "run_nat_job", fake_nat)
    monkeypatch.setattr(retr, "retrieve_for_query", _fake_retrieve(recs))

    block = asyncio.run(enr.build_enrichment_block(db=object(), project_id=1,
                                                   user_message="worried about my calories today"))
    assert block.startswith("[ENRICHMENT]"), block
    assert "hit target: 2100 kcal" in block
    assert calls["nat"] == 1  # Nat proposed; deterministic retriever fetched


def test_enrichment_hard_timeout_is_nonblocking(monkeypatch):
    _reset()
    os.environ["NAT_ENRICHMENT_TIMEOUT_MS"] = "50"  # tiny budget

    async def slow_nat(system, user, **kw):
        await asyncio.sleep(5)  # far beyond budget
        return "[]"

    monkeypatch.setattr(nw, "run_nat_job", slow_nat)

    import time
    t0 = time.monotonic()
    block = asyncio.run(enr.build_enrichment_block(db=object(), project_id=1, user_message="hi there"))
    elapsed = time.monotonic() - t0
    assert block == ""                 # proceeded without Nat
    assert elapsed < 1.0               # did NOT wait 5s — bounded by the 50ms budget
    assert enr._last_fail[0] != 0.0    # breaker tripped


def test_enrichment_circuit_breaker_skips(monkeypatch):
    _reset()
    enr._last_fail[0] = __import__("time").monotonic()  # just failed
    called = {"nat": 0}

    async def fake_nat(system, user, **kw):
        called["nat"] += 1
        return "[]"

    monkeypatch.setattr(nw, "run_nat_job", fake_nat)
    block = asyncio.run(enr.build_enrichment_block(db=object(), project_id=1, user_message="anything"))
    assert block == ""
    assert called["nat"] == 0  # breaker skipped Nat entirely during cooldown


def test_enrichment_disabled(monkeypatch):
    _reset()
    os.environ["NAT_ENRICHMENT_ENABLED"] = "0"
    called = {"nat": 0}

    async def fake_nat(system, user, **kw):
        called["nat"] += 1
        return "[]"

    monkeypatch.setattr(nw, "run_nat_job", fake_nat)
    assert asyncio.run(enr.build_enrichment_block(db=object(), project_id=1, user_message="x")) == ""
    assert called["nat"] == 0


def test_enrichment_sync_wrapper(monkeypatch):
    _reset()

    async def fake_nat(system, user, **kw):
        return '["q1"]'

    recs = [types.SimpleNamespace(title="T", content="stored fact value")]
    monkeypatch.setattr(nw, "run_nat_job", fake_nat)
    monkeypatch.setattr(retr, "retrieve_for_query", _fake_retrieve(recs))
    # The sync wrapper opens a fresh db via app.db.get_db_session — stub it so the
    # test doesn't touch the real database (retrieve is mocked anyway).
    import app.db as appdb
    monkeypatch.setattr(appdb, "get_db_session", lambda: types.SimpleNamespace(close=lambda: None))

    block = enr.build_enrichment_block_sync(project_id=1, message="worried about calories")
    assert block.startswith("[ENRICHMENT]"), block
    assert "stored fact value" in block


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
