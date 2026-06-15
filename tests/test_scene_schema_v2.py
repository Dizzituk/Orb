# FILE: tests/test_scene_schema_v2.py
# Purpose: Units for the SceneDoc v2 schema (v3 enrichment) — backward-compat with v1,
#          upconvert, and round-trip/validation of terrain, tiling, scatter, vehicles,
#          lighting, era/style, provenance, density_budget.
# Called-by: pytest
# Depends-on: app.scene_director.schemas, app.scene_director.asset_catalog
# Last-renovated: 2026-06-13
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.scene_director import asset_catalog
from app.scene_director.schemas import (
    SCHEMA_VERSION,
    ActorSpec,
    ElevationFeature,
    LightingSpec,
    ProvenanceSpec,
    ScatterRegion,
    SceneDoc,
    SourceRef,
    StructurePrimitive,
    TerrainSpec,
    TileRun,
    detect_schema_version,
    upconvert_to_v2,
)

_ENV_ID = sorted(asset_catalog.known_ids("environment"))[0]


def _min_doc(**ov) -> SceneDoc:
    base = dict(title="t", intent="i")
    base.update(ov)
    return SceneDoc(**base)


# ── backward compatibility (sacred) ──────────────────────────────────────────

def test_v1_dict_validates_as_v2_with_empty_enrichments():
    # A v1 SceneDoc JSON (no schema_version, no terrain/tiles/scatter/lighting).
    v1 = {
        "scene_id": "abc", "version": 3, "title": "v1 scene", "intent": "old",
        "skybox": {"kind": "preset", "id": "dusk"},
        "environment": [{"prop_id": "p", "prefab_id": _ENV_ID}],
        "actors": [{"actor_id": "a", "prefab_id": "x", "waypoints": [[1, 0, 1]]}],
        "timeline": [{"at_seconds": 0, "action": "narrate", "text": "hi"}],
        "created_at": "2026-06-12T00:00:00+00:00",
    }
    doc = SceneDoc.model_validate(v1)
    assert doc.schema_version == SCHEMA_VERSION
    assert doc.terrain is None and doc.lighting is None and doc.provenance is None
    assert doc.tiles == [] and doc.scatter == []
    assert doc.era == "modern" and doc.density_budget > 0
    # v1 actor defaults to a person
    assert doc.actors[0].type == "person"


def test_detect_and_upconvert():
    assert detect_schema_version({"title": "x"}) == 1          # no schema_version → v1
    assert detect_schema_version({"schema_version": 2}) == 2
    up = upconvert_to_v2({"title": "x", "intent": "y"})
    assert up["schema_version"] == SCHEMA_VERSION
    # idempotent
    assert upconvert_to_v2(up)["schema_version"] == SCHEMA_VERSION


# ── v2 structures round-trip + validate ──────────────────────────────────────

def test_tilerun_round_trip_and_bounds():
    t = TileRun(prefab_id="road_01", start=(0, 0, 0), direction=(1, 0, 0), count=12, spacing_m=20.0)
    assert t.count == 12
    with pytest.raises(ValidationError):
        TileRun(prefab_id="road_01", count=0)      # ge=1
    with pytest.raises(ValidationError):
        TileRun(prefab_id="road_01", count=999)     # le=200


def test_scatter_and_terrain_round_trip():
    s = ScatterRegion(prefab_ids=["tree_01", "tree_02"], centre=(0, 0, 0),
                      size=(40, 0, 40), density_per_100m2=12.0, seed=7)
    assert s.align_to_ground is True and s.random_rotation is True
    terr = TerrainSpec(ground="grass", elevation=[
        ElevationFeature(shape="hill", centre=(10, 0, 10), radius=15, height=4)])
    assert terr.elevation[0].shape == "hill"


def test_vehicle_actor_type():
    v = ActorSpec(actor_id="car1", prefab_id="car_01", type="vehicle",
                  waypoints=[(0, 0, 0), (50, 0, 0)], speed_mps=10.0)
    assert v.type == "vehicle"
    p = ActorSpec(actor_id="p", prefab_id="person_male_casual_01")
    assert p.type == "person"   # default
    with pytest.raises(ValidationError):
        ActorSpec(actor_id="x", prefab_id="y", type="spaceship")


def test_lighting_and_provenance():
    lit = LightingSpec(time_of_day="dusk", sun_angle=8.0, sun_intensity=0.7, ambient=0.3)
    assert lit.time_of_day == "dusk"
    prov = ProvenanceSpec(facts=["fact one"], sources=[SourceRef(title="T", url="http://x")])
    assert prov.sources[0].url == "http://x"


def test_structures_round_trip_and_v1_default():
    # v3 primitives: build blocks need no catalogue id; v1 docs default to empty.
    doc = _min_doc(structures=[
        StructurePrimitive(prop_id="wall_n", kind="box", position=(0, 1.5, 5), size=(8, 3, 0.2), color="#eeeeee"),
        StructurePrimitive(prop_id="floor", kind="plane", size=(8, 0, 8), color="#9b8a6a"),
        StructurePrimitive(prop_id="board", kind="quad", position=(0, 1.6, 4.9), size=(3, 1.2, 1), color="#0b3d0b"),
    ])
    restored = SceneDoc.model_validate_json(doc.model_dump_json())
    assert [s.kind for s in restored.structures] == ["box", "plane", "quad"]
    assert restored.structures[2].color == "#0b3d0b"
    # backward-compat: a v1 doc has no structures
    v1 = SceneDoc.model_validate({"title": "t", "intent": "i"})
    assert v1.structures == []
    with pytest.raises(ValidationError):
        StructurePrimitive(prop_id="x", kind="pyramid")


def test_full_v2_scene_round_trips():
    doc = _min_doc(
        terrain=TerrainSpec(ground="urban", elevation=[ElevationFeature(centre=(30, 0, 30))]),
        tiles=[TileRun(prefab_id="road_01", count=8, spacing_m=20)],
        scatter=[ScatterRegion(prefab_ids=["tree_01"], size=(50, 0, 50))],
        actors=[ActorSpec(actor_id="car", prefab_id="car_01", type="vehicle",
                          waypoints=[(0, 0, 0), (40, 0, 0)], speed_mps=9)],
        lighting=LightingSpec(time_of_day="dusk"),
        era="modern", style="low-poly",
        provenance=ProvenanceSpec(facts=["x"], sources=[SourceRef(title="t", url="u")]),
        density_budget=300,
    )
    restored = SceneDoc.model_validate_json(doc.model_dump_json())
    assert restored.model_dump() == doc.model_dump()
    assert restored.schema_version == SCHEMA_VERSION
    assert restored.tiles[0].count == 8
    assert restored.actors[0].type == "vehicle"


# ── catalogue v2 queries ─────────────────────────────────────────────────────

def test_catalogue_has_rich_categories():
    cats = asset_catalog.categories()
    # introspection should have produced real category breadth
    assert {"structure", "vegetation", "vehicle"} <= cats
    assert asset_catalog.known_ids_by_category("vegetation")
    assert asset_catalog.known_ids_by_category("vehicle")


def test_prompt_block_is_grouped_and_bounded():
    block = asset_catalog.as_prompt_block(max_per_group=10)
    assert "VEHICLES" in block or "BUILDINGS" in block
    # bounded: each group capped, so the whole block stays well under the full 537 lines
    assert block.count("\n") < 200


def test_footprint_and_drivable_queries():
    veh = sorted(asset_catalog.known_ids_by_category("vehicle"))[0]
    assert asset_catalog.is_drivable(veh)
    # most environment props have a measured footprint
    fp = asset_catalog.footprint_of(_ENV_ID)
    assert fp is None or (isinstance(fp, list) and len(fp) >= 3)
