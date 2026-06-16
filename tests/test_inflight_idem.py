# Verifies in-flight idempotency coalescing (app/bridge/inflight_idem.py) — the
# 2026-06-16 retry-storm fix: concurrent same-key /chat-and-speak requests share
# ONE generation instead of each launching a duplicate LLM+TTS run.
import asyncio
import sys
import types

import app.bridge.inflight_idem as ii


def _stub_chat_speak(monkeypatch, fake_replay):
    """Avoid importing the heavy TTS stack: stub the replay function only."""
    fake = types.ModuleType("app.bridge.chat_speak_stream")
    fake.replay_cached_reply = fake_replay
    monkeypatch.setitem(sys.modules, "app.bridge.chat_speak_stream", fake)


def test_claim_is_exclusive_and_releasable():
    ii._inflight.clear()
    assert ii._claim("k") is True       # first in wins
    assert ii._claim("k") is False      # second is blocked
    ii.complete("k")                    # owner releases
    assert ii._claim("k") is True       # key is free again
    ii.complete("k")


def test_complete_is_safe_on_blank_and_unknown_keys():
    ii._inflight.clear()
    ii.complete(None)
    ii.complete("")
    ii.complete("never-claimed")        # none of these raise


def test_stale_owner_is_reclaimable():
    ii._inflight.clear()
    assert ii._claim("k") is True
    event, _ = ii._inflight["k"]
    # Backdate the claim past the TTL → the owner is presumed dead.
    ii._inflight["k"] = (event, ii.time.monotonic() - ii._OWNER_TTL_S - 1.0)
    assert ii._claim("k") is True       # reclaimed despite an existing entry
    ii.complete("k")


def test_waiter_coalesces_onto_owner(monkeypatch):
    ii._inflight.clear()
    store = {}
    replay_calls = []

    async def fake_replay(message_id, db, process_artifacts):
        replay_calls.append(message_id)
        return f"RESP:{message_id}"

    monkeypatch.setattr("app.bridge.tts_cache.idem_get", lambda k: store.get(k))
    _stub_chat_speak(monkeypatch, fake_replay)

    async def body():
        assert ii._claim("k") is True            # pretend the owner is mid-generation

        async def owner():
            await asyncio.sleep(0.05)
            store["k"] = 4242                     # reply persisted + mapped
            ii.complete("k")                      # wake waiters

        async def waiter():
            return await ii.begin("k", db=None, process_artifacts=None)

        result, _ = await asyncio.gather(waiter(), owner())
        return result

    assert asyncio.run(body()) == "RESP:4242"
    assert replay_calls == [4242]                 # exactly one replay, zero duplicate gens


def test_completed_key_replays_immediately(monkeypatch):
    ii._inflight.clear()
    monkeypatch.setattr("app.bridge.tts_cache.idem_get", lambda k: 99 if k == "done" else None)

    async def fake_replay(message_id, db, process_artifacts):
        return f"RESP:{message_id}"

    _stub_chat_speak(monkeypatch, fake_replay)
    assert asyncio.run(ii.begin("done", db=None, process_artifacts=None)) == "RESP:99"
    assert ii._inflight == {}                     # a completed-key replay never claims ownership


def test_waiter_timeout_falls_through_to_generate(monkeypatch):
    ii._inflight.clear()
    monkeypatch.setattr("app.bridge.tts_cache.idem_get", lambda k: None)

    async def fake_replay(message_id, db, process_artifacts):
        raise AssertionError("replay must not run when nothing was produced")

    _stub_chat_speak(monkeypatch, fake_replay)
    monkeypatch.setattr(ii, "_WAIT_SECS", 0.05)

    async def body():
        assert ii._claim("k") is True            # an owner holds it and never completes
        return await ii.begin("k", db=None, process_artifacts=None)

    # Second caller gives up waiting and returns None (= generate normally),
    # and takes ownership so it can release on its own completion.
    assert asyncio.run(body()) is None
    assert "k" in ii._inflight


def test_no_key_is_a_passthrough():
    ii._inflight.clear()
    assert asyncio.run(ii.begin(None, db=None, process_artifacts=None)) is None
    assert ii._inflight == {}
