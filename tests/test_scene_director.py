# FILE: tests/test_scene_director.py
# Purpose: Units for app/scene_director — SceneDoc round-trip, catalogue serving +
#          sanitise stripping, deterministic fallback shape, state versioning/fan-out,
#          compose endpoint + websocket handshake with the director mocked.
# Called-by: pytest
# Depends-on: app.scene_director.*
# Last-renovated: 2026-06-13
#
# v3: catalogue-driven — real prefab ids are resolved from the live catalogue at load,
# so the suite is robust to introspection changing the catalogue contents.
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.scene_director import asset_catalog, director
from app.scene_director.schemas import (
    ActorSpec,
    ComposeResponse,
    PropSpec,
    SceneDoc,
    SkyboxSpec,
    TimelineEvent,
)
from app.scene_director.state import scene_state

# Real ids from the live catalogue (introspected). Deterministic pick.
_ENV_ID = sorted(asset_catalog.known_ids("environment"))[0]
_ACTOR_ID = ("person_male_casual_01" if "person_male_casual_01" in asset_catalog.known_ids("actor")
             else sorted(asset_catalog.known_ids("actor"))[0])
_SKY_ID = "day_clear" if "day_clear" in asset_catalog.known_ids("skybox") else sorted(asset_catalog.known_ids("skybox"))[0]


def _make_doc(**overrides) -> SceneDoc:
    base = dict(
        title="Test scene",
        intent="a man walks down a city street",
        skybox=SkyboxSpec(kind="preset", id=_SKY_ID),
        environment=[PropSpec(prop_id="road_a", prefab_id=_ENV_ID, position=(0.0, 0.0, 0.0))],
        actors=[ActorSpec(actor_id="walker", prefab_id=_ACTOR_ID,
                          spawn=(-8.0, 0.0, 2.0), waypoints=[(28.0, 0.0, 2.0)])],
        timeline=[
            TimelineEvent(at_seconds=0.0, action="narrate", text="Here is a street."),
            TimelineEvent(at_seconds=4.0, action="actor_start", actor_id="walker"),
        ],
    )
    base.update(overrides)
    return SceneDoc(**base)


@pytest.fixture(autouse=True)
def _clean_state():
    scene_state.reset()
    yield
    scene_state.reset()


@pytest.fixture()
def client():
    from app.auth import require_auth
    from app.scene_director.router import router

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_auth] = lambda: True
    with TestClient(app) as c:
        yield c


# ── schemas: round-trip + validation ─────────────────────────────────────────

def test_scene_doc_json_round_trip():
    doc = _make_doc()
    restored = SceneDoc.model_validate_json(doc.model_dump_json())
    assert restored.model_dump() == doc.model_dump()


def test_scene_doc_defaults_are_server_generated():
    doc = _make_doc()
    assert doc.scene_id and doc.created_at and doc.version == 1
    other = _make_doc()
    assert other.scene_id != doc.scene_id


def test_timeline_event_rejects_negative_time():
    with pytest.raises(ValidationError):
        TimelineEvent(at_seconds=-1.0, action="wait")


def test_actor_rejects_unknown_animation():
    with pytest.raises(ValidationError):
        ActorSpec(actor_id="a", prefab_id="p", animation="moonwalk")


# ── asset catalogue ──────────────────────────────────────────────────────────

def test_catalog_loads_with_all_three_kinds():
    cat = asset_catalog.load_catalog()
    assert cat["entries"], "catalogue must not be empty"
    assert asset_catalog.known_ids("environment")
    assert asset_catalog.known_ids("actor")
    assert asset_catalog.known_ids("skybox")


def test_catalog_get_and_prompt_block():
    entry = asset_catalog.get(_ENV_ID)
    assert entry and entry["kind"] == "environment"
    assert asset_catalog.get("no_such_prefab") is None
    block = asset_catalog.as_prompt_block()
    assert _ENV_ID in block and _SKY_ID in block


# ── sanitise: the catalogue boundary ─────────────────────────────────────────

def test_sanitise_strips_unknown_ids_into_warnings():
    doc = _make_doc(
        environment=[
            PropSpec(prop_id="road_a", prefab_id=_ENV_ID),
            PropSpec(prop_id="ufo", prefab_id="flying_saucer_99"),
        ],
        actors=[
            ActorSpec(actor_id="walker", prefab_id=_ACTOR_ID, waypoints=[(1.0, 0.0, 0.0)]),
            ActorSpec(actor_id="ghost", prefab_id="spectre_01", waypoints=[(0.0, 0.0, 0.0)]),
        ],
        timeline=[
            TimelineEvent(at_seconds=0.0, action="narrate", text="ok"),
            TimelineEvent(at_seconds=1.0, action="narrate", text="   "),
            TimelineEvent(at_seconds=2.0, action="actor_start", actor_id="ghost"),
            TimelineEvent(at_seconds=3.0, action="actor_start", actor_id="walker"),
        ],
        skybox=SkyboxSpec(kind="preset", id="alien_sky"),
    )
    warnings: list[str] = []
    out = director.sanitise_scene(doc, warnings)
    assert [p.prop_id for p in out.environment] == ["road_a"]
    assert [a.actor_id for a in out.actors] == ["walker"]
    assert [(e.action, e.actor_id) for e in out.timeline] == [
        ("narrate", None), ("actor_start", "walker")]
    assert out.skybox.id == "day_clear"
    assert len(warnings) == 5


def test_sanitise_generated_skybox_coerced_to_preset():
    doc = _make_doc(skybox=SkyboxSpec(kind="generated", id="blockade_xyz"))
    warnings: list[str] = []
    out = director.sanitise_scene(doc, warnings)
    assert out.skybox.kind == "preset" and out.skybox.id == "day_clear"
    assert warnings


def test_sanitise_strips_duplicates_and_pins_waypointless():
    doc = _make_doc(
        environment=[
            PropSpec(prop_id="road_a", prefab_id=_ENV_ID),
            PropSpec(prop_id="road_a", prefab_id=_ENV_ID),
        ],
        actors=[ActorSpec(actor_id="walker", prefab_id=_ACTOR_ID,
                          spawn=(3.0, 0.0, 4.0), waypoints=[])],
    )
    warnings: list[str] = []
    out = director.sanitise_scene(doc, warnings)
    assert len(out.environment) == 1
    assert out.actors[0].waypoints == [(3.0, 0.0, 4.0)]
    assert any("duplicate" in w for w in warnings)
    assert any("waypoints" in w for w in warnings)


def test_sanitise_orders_timeline():
    doc = _make_doc(timeline=[
        TimelineEvent(at_seconds=9.0, action="narrate", text="later"),
        TimelineEvent(at_seconds=0.0, action="narrate", text="first"),
    ])
    out = director.sanitise_scene(doc, [])
    assert [e.at_seconds for e in out.timeline] == [0.0, 9.0]


# ── deterministic fallback ───────────────────────────────────────────────────

def test_fallback_scene_shape():
    doc = director.deterministic_fallback_scene("a man walks down a city street")
    env_ids = asset_catalog.known_ids("environment")
    actor_ids = asset_catalog.known_ids("actor")
    sky_ids = asset_catalog.known_ids("skybox")
    assert all(p.prefab_id in env_ids for p in doc.environment)
    assert all(a.prefab_id in actor_ids for a in doc.actors)
    assert doc.skybox.id in sky_ids
    assert doc.actors and doc.actors[0].waypoints
    actions = [e.action for e in doc.timeline]
    assert "narrate" in actions and "actor_start" in actions
    assert all(e.text for e in doc.timeline if e.action == "narrate")
    starts = {e.actor_id for e in doc.timeline if e.action == "actor_start"}
    assert starts <= {a.actor_id for a in doc.actors}
    assert doc.intent == "a man walks down a city street"


def test_fallback_respects_dusk_keyword():
    doc = director.deterministic_fallback_scene("a man walks at dusk")
    assert doc.skybox.id == "dusk"


# ── state: versioning + fan-out ──────────────────────────────────────────────

def test_state_versioning_monotonic():
    assert scene_state.get_scene() is None
    v1 = scene_state.set_scene(_make_doc())
    v2 = scene_state.set_scene(_make_doc())
    assert (v1, v2) == (1, 2)
    assert scene_state.get_scene().version == 2


@pytest.mark.asyncio
async def test_state_fanout_to_subscribers():
    q = scene_state.subscribe()
    doc = _make_doc()
    scene_state.set_scene(doc)
    pushed = await asyncio.wait_for(q.get(), timeout=1.0)
    assert pushed.scene_id == doc.scene_id
    scene_state.unsubscribe(q)
    scene_state.set_scene(_make_doc())
    assert q.empty()


# ── director: LLM path mocked ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_director_falls_back_on_llm_error(monkeypatch):
    async def boom(messages):
        return "", {"error": "exception"}
    monkeypatch.setattr(director, "_call_director_llm", boom)
    result = await director.compose_scene("anything at all")
    assert result.scene.title  # fallback scene delivered
    assert any("fallback" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_director_retries_once_then_falls_back(monkeypatch):
    calls = {"n": 0}

    async def garbage(messages):
        calls["n"] += 1
        return "this is not json", {}
    monkeypatch.setattr(director, "_call_director_llm", garbage)
    result = await director.compose_scene("anything")
    assert calls["n"] == 2
    assert any("fallback" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_director_accepts_valid_reply_and_enforces_server_fields(monkeypatch):
    reply = json.dumps({
        "scene_id": "model-must-not-control-this",
        "title": "Composed street",
        "skybox": {"kind": "preset", "id": "dusk"},
        "environment": [{"prop_id": "road_a", "prefab_id": _ENV_ID,
                         "position": [0, 0, 0], "rotation_y": 0, "scale": 1.0}],
        "actors": [{"actor_id": "walker", "prefab_id": _ACTOR_ID,
                    "spawn": [-8, 0, 2], "waypoints": [[28, 0, 2]],
                    "animation": "walk", "speed_mps": 1.2, "loop": False}],
        "timeline": [{"at_seconds": 0, "action": "narrate", "text": "A street at dusk."},
                     {"at_seconds": 3, "action": "actor_start", "actor_id": "walker"}],
    })

    async def good(messages):
        return reply, {"total_tokens": 100}
    monkeypatch.setattr(director, "_call_director_llm", good)
    result = await director.compose_scene("a man walks at dusk")
    assert result.scene.title == "Composed street"
    assert result.scene.scene_id != "model-must-not-control-this"
    assert result.scene.intent == "a man walks at dusk"
    assert result.warnings == []


# ── router: compose endpoint + websocket handshake (director mocked) ─────────

def test_compose_endpoint_stores_and_returns_scene(client, monkeypatch):
    monkeypatch.setenv("SCENE_CRITIC_ENABLED", "false")  # focus on router store+return
    canned = ComposeResponse(scene=_make_doc(), warnings=["catalogue guess"])

    async def fake_compose(intent, **kwargs):
        assert intent == "a man walks down a city street"
        return canned
    monkeypatch.setattr(director, "compose_scene", fake_compose)

    resp = client.post("/scene/compose", json={"intent": "a man walks down a city street"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["scene"]["scene_id"] == canned.scene.scene_id
    assert body["warnings"] == ["catalogue guess"]
    assert scene_state.get_scene().scene_id == canned.scene.scene_id

    current = client.get("/scene/current")
    assert current.status_code == 200
    assert current.json()["doc"]["scene_id"] == canned.scene.scene_id


def test_current_404_when_nothing_composed(client):
    assert client.get("/scene/current").status_code == 404


def test_catalog_endpoint_serves_entries(client):
    resp = client.get("/scene/catalog")
    assert resp.status_code == 200
    ids = {e["prefab_id"] for e in resp.json()["entries"]}
    assert _ENV_ID in ids


def test_compose_rejects_empty_intent(client):
    assert client.post("/scene/compose", json={"intent": ""}).status_code == 422


def test_websocket_receives_push_on_recompose(client, monkeypatch):
    monkeypatch.setenv("SCENE_CRITIC_ENABLED", "false")
    canned = ComposeResponse(scene=_make_doc(), warnings=[])

    async def fake_compose(intent, **kwargs):
        return canned
    monkeypatch.setattr(director, "compose_scene", fake_compose)

    with client.websocket_connect("/scene/ws") as ws:
        resp = client.post("/scene/compose", json={"intent": "street please"})
        assert resp.status_code == 200
        msg = ws.receive_json()
        assert msg["type"] == "scene"
        assert msg["doc"]["scene_id"] == canned.scene.scene_id
        assert msg["doc"]["version"] == 1


def test_websocket_initial_push_and_status_breadcrumbs(client):
    doc = _make_doc()
    scene_state.set_scene(doc)
    with client.websocket_connect("/scene/ws") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "scene"
        assert msg["doc"]["scene_id"] == doc.scene_id
        ws.send_json({"type": "status", "event": "scene_loaded", "scene_id": doc.scene_id})
    events = client.get("/scene/events").json()["events"]
    assert any(e.get("event") == "scene_loaded" for e in events)
