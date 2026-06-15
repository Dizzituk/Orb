# FILE: tests/test_astra_presence.py
# Purpose: Units for app/astra_presence — state set/get, version monotonicity, alias +
#          validation, subscriber fan-out, GET/POST /astra/state, WS initial+push.
# Called-by: pytest
# Depends-on: app.astra_presence.state, app.astra_presence.router
# Last-renovated: 2026-06-13
from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.astra_presence import state as state_mod
from app.astra_presence.state import presence_state, normalise_state


@pytest.fixture(autouse=True)
def _clean():
    presence_state.reset()
    yield
    presence_state.reset()


@pytest.fixture()
def client():
    from app.astra_presence.router import router
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


# ── state model ───────────────────────────────────────────────────────────────

def test_default_state_is_idle():
    assert presence_state.get_state() == "idle"
    assert presence_state.version == 0


def test_set_state_versioning_monotonic():
    v1 = presence_state.set_state("thinking")
    v2 = presence_state.set_state("speaking")
    assert (v1, v2) == (1, 2)
    assert presence_state.get_state() == "speaking"


def test_deep_research_alias_normalised():
    assert normalise_state("deep_research") == "deep research"
    assert normalise_state("DEEP RESEARCH") == "deep research"
    presence_state.set_state("deep_research")
    assert presence_state.get_state() == "deep research"


def test_unknown_state_rejected():
    assert normalise_state("dancing") is None
    with pytest.raises(ValueError):
        presence_state.set_state("dancing")


@pytest.mark.asyncio
async def test_fanout_to_subscribers():
    q = presence_state.subscribe()
    presence_state.set_state("listening")
    payload = await asyncio.wait_for(q.get(), timeout=1.0)
    assert payload["type"] == "astra_state"
    assert payload["state"] == "listening"
    assert payload["version"] == 1
    presence_state.unsubscribe(q)
    presence_state.set_state("idle")
    assert q.empty()


# ── router ──────────────────────────────────────────────────────────────────────

def test_get_state_endpoint(client):
    body = client.get("/astra/state").json()
    assert body == {"type": "astra_state", "state": "idle", "version": 0}


def test_post_state_endpoint(client):
    resp = client.post("/astra/state", json={"state": "thinking"})
    assert resp.status_code == 200
    assert resp.json()["state"] == "thinking"
    assert client.get("/astra/state").json()["state"] == "thinking"


def test_post_invalid_state_400(client):
    assert client.post("/astra/state", json={"state": "nope"}).status_code == 400


def test_ws_initial_and_push(client):
    presence_state.set_state("thinking")
    with client.websocket_connect("/astra/ws") as ws:
        first = ws.receive_json()
        assert first["state"] == "thinking"
        # a change while connected pushes a frame
        client.post("/astra/state", json={"state": "speaking"})
        nxt = ws.receive_json()
        assert nxt["state"] == "speaking"
