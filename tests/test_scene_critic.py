# FILE: tests/test_scene_critic.py
# Purpose: Units for the v3 scene critic + rich director — revision application + re-sanitise,
#          bounded retry, prefab validation after revision, disabled/error degradation,
#          and director v2-structure sanitising (tiles/scatter/vehicle/density).
# Called-by: pytest
# Depends-on: app.scene_director.critic/director/asset_catalog/schemas
# Last-renovated: 2026-06-13
from __future__ import annotations

import json

import pytest

from app.scene_director import asset_catalog, critic, director
from app.scene_director.schemas import (
    ActorSpec,
    PropSpec,
    ScatterRegion,
    SceneDoc,
    SkyboxSpec,
    TileRun,
    TimelineEvent,
)

_ENV_ID = sorted(asset_catalog.known_ids("environment"))[0]
_ENV_ID2 = sorted(asset_catalog.known_ids("environment"))[1]
_ACTOR_ID = ("person_male_casual_01" if "person_male_casual_01" in asset_catalog.known_ids("actor")
             else sorted(asset_catalog.known_ids("actor"))[0])
_VEH_ID = sorted(asset_catalog.known_ids_by_category("vehicle"))[0]
_VEG_ID = sorted(asset_catalog.known_ids_by_category("vegetation"))[0]


def _scene(**ov) -> SceneDoc:
    base = dict(title="draft", intent="a street",
                skybox=SkyboxSpec(kind="preset", id="day_clear"),
                environment=[PropSpec(prop_id="p1", prefab_id=_ENV_ID)],
                actors=[ActorSpec(actor_id="w", prefab_id=_ACTOR_ID, waypoints=[(1, 0, 1)])],
                timeline=[TimelineEvent(at_seconds=0, action="narrate", text="hi"),
                          TimelineEvent(at_seconds=2, action="actor_start", actor_id="w")])
    base.update(ov)
    return SceneDoc(**base)


def _revised_json(**ov):
    """A valid revised SceneDoc the critic could emit."""
    d = {
        "title": "revised richer",
        "skybox": {"kind": "preset", "id": "dusk"},
        "terrain": {"ground": "urban"},
        "environment": [{"prop_id": "p1", "prefab_id": _ENV_ID}],
        "tiles": [{"prefab_id": _ENV_ID, "start": [0, 0, 0], "direction": [1, 0, 0], "count": 6, "spacing_m": 20}],
        "scatter": [{"prefab_ids": [_VEG_ID], "centre": [0, 0, 20], "size": [40, 0, 40], "density_per_100m2": 6, "seed": 1}],
        "actors": [{"actor_id": "w", "prefab_id": _ACTOR_ID, "type": "person", "spawn": [-8, 0, 2], "waypoints": [[28, 0, 2]]}],
        "timeline": [{"at_seconds": 0, "action": "narrate", "text": "Now richer."},
                     {"at_seconds": 3, "action": "actor_start", "actor_id": "w"}],
        "lighting": {"time_of_day": "dusk"},
    }
    d.update(ov)
    return d


# ── critic loop ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_critic_disabled_returns_unchanged(monkeypatch):
    monkeypatch.setenv("SCENE_CRITIC_ENABLED", "false")
    scene = _scene()
    out = await critic.critique_and_revise(scene, "a street", [])
    assert out is scene


@pytest.mark.asyncio
async def test_critic_ok_no_change(monkeypatch):
    monkeypatch.setenv("SCENE_CRITIC_ENABLED", "true")

    async def ok(payload):
        return json.dumps({"ok": True, "issues": []}), {}
    monkeypatch.setattr(critic, "_call_critic_llm", ok)
    scene = _scene()
    warnings: list = []
    out = await critic.critique_and_revise(scene, "a street", warnings)
    assert out.title == "draft"  # unchanged


@pytest.mark.asyncio
async def test_critic_applies_revision_and_resanitises(monkeypatch):
    monkeypatch.setenv("SCENE_CRITIC_ENABLED", "true")
    monkeypatch.setenv("SCENE_CRITIC_MAX_PASSES", "1")

    async def revise(payload):
        # revised scene includes an INVALID prefab that must be stripped on re-sanitise
        rj = _revised_json()
        rj["environment"].append({"prop_id": "bad", "prefab_id": "not_a_real_prefab_zzz"})
        return json.dumps({"ok": False, "issues": ["bare ground", "road too short"], "revised_scene": rj}), {}
    monkeypatch.setattr(critic, "_call_critic_llm", revise)

    scene = _scene()
    warnings: list = []
    out = await critic.critique_and_revise(scene, "a busy street", warnings)
    assert out.title == "revised richer"
    assert out.tiles and out.scatter            # enriched
    assert all(p.prefab_id != "not_a_real_prefab_zzz" for p in out.environment)  # invalid stripped
    assert any("revised the scene" in w for w in warnings)


@pytest.mark.asyncio
async def test_critic_bounded_passes(monkeypatch):
    monkeypatch.setenv("SCENE_CRITIC_ENABLED", "true")
    monkeypatch.setenv("SCENE_CRITIC_MAX_PASSES", "2")
    calls = {"n": 0}

    async def always_revise(payload):
        calls["n"] += 1
        return json.dumps({"ok": False, "issues": ["still bare"], "revised_scene": _revised_json()}), {}
    monkeypatch.setattr(critic, "_call_critic_llm", always_revise)
    await critic.critique_and_revise(_scene(), "x", [])
    assert calls["n"] == 2  # never loops beyond the bound


@pytest.mark.asyncio
async def test_critic_error_keeps_scene(monkeypatch):
    monkeypatch.setenv("SCENE_CRITIC_ENABLED", "true")

    async def boom(payload):
        return "", {"error": "exception"}
    monkeypatch.setattr(critic, "_call_critic_llm", boom)
    scene = _scene()
    warnings: list = []
    out = await critic.critique_and_revise(scene, "x", warnings)
    assert out is scene and any("critic" in w for w in warnings)


@pytest.mark.asyncio
async def test_critic_bad_revision_keeps_prior(monkeypatch):
    monkeypatch.setenv("SCENE_CRITIC_ENABLED", "true")

    async def bad(payload):
        return json.dumps({"ok": False, "issues": ["x"], "revised_scene": {"title": 123}}), {}
    monkeypatch.setattr(critic, "_call_critic_llm", bad)
    scene = _scene()
    out = await critic.critique_and_revise(scene, "x", [])
    assert out.title == "draft"  # kept prior


# ── director: rich v2 sanitise ───────────────────────────────────────────────

def test_sanitise_keeps_valid_tiles_scatter_vehicle():
    doc = _scene(
        tiles=[TileRun(prefab_id=_ENV_ID, count=4), TileRun(prefab_id="ghost_tile", count=3)],
        scatter=[ScatterRegion(prefab_ids=[_VEG_ID, "ghost_veg"], size=(30, 0, 30))],
        actors=[ActorSpec(actor_id="car", prefab_id=_VEH_ID, type="vehicle", waypoints=[(0, 0, 0), (40, 0, 0)]),
                ActorSpec(actor_id="fake", prefab_id="not_real", type="vehicle", waypoints=[(0, 0, 0)])],
    )
    warnings: list = []
    out = director.sanitise_scene(doc, warnings)
    assert [t.prefab_id for t in out.tiles] == [_ENV_ID]          # unknown tile stripped
    assert out.scatter[0].prefab_ids == [_VEG_ID]                 # unknown veg dropped
    # the actors override replaced the person; the fake vehicle is stripped, valid car kept
    assert [a.actor_id for a in out.actors] == ["car"]
    assert any("ghost_tile" in w for w in warnings)


def test_sanitise_keeps_structures_untouched():
    # v3 primitives are raw shapes (no catalogue id) — sanitise must leave them alone.
    from app.scene_director.schemas import StructurePrimitive
    doc = _scene(structures=[
        StructurePrimitive(prop_id="wall", kind="box", size=(8, 3, 0.2)),
        StructurePrimitive(prop_id="board", kind="quad", size=(3, 1.2, 1)),
    ])
    out = director.sanitise_scene(doc, [])
    assert [s.kind for s in out.structures] == ["box", "quad"]


def test_sanitise_density_budget_thins_scatter():
    big = ScatterRegion(prefab_ids=[_VEG_ID], centre=(0, 0, 0), size=(200, 0, 200),
                        density_per_100m2=100.0, seed=1)  # ~40k instances
    doc = _scene(scatter=[big], density_budget=300)
    warnings: list = []
    out = director.sanitise_scene(doc, warnings)
    assert out.scatter[0].density_per_100m2 < 100.0
    assert any("budget" in w for w in warnings)


@pytest.mark.asyncio
async def test_compose_warns_on_uncatalogued_era(monkeypatch):
    # Layer 5: a non-modern era with no catalogued assets must warn + keep the requested era,
    # not silently dress the past in modern blocks.
    async def good(messages):
        return json.dumps(_revised_json()), {"total_tokens": 1}
    monkeypatch.setattr(director, "_call_director_llm", good)
    result = await director.compose_scene("a tudor street", era="tudor")
    assert result.scene.era == "tudor"
    assert any("tudor" in w.lower() and "asset" in w.lower() for w in result.warnings)


@pytest.mark.asyncio
async def test_compose_applies_provenance_from_fact_pack(monkeypatch):
    fact_pack = {"facts": ["The Thames is tidal."], "summary": "London river.",
                 "sources": [{"title": "Wiki", "url": "https://example.org"}]}

    async def good(messages):
        return json.dumps(_revised_json(title="Researched")), {"total_tokens": 1}
    monkeypatch.setattr(director, "_call_director_llm", good)
    result = await director.compose_scene("the thames", era="modern", fact_pack=fact_pack)
    assert result.scene.provenance is not None
    assert result.scene.provenance.sources[0].url == "https://example.org"
    assert result.scene.era == "modern"
