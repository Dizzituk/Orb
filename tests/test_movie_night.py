# FILE: tests/test_movie_night.py
# Purpose: Unit tests for app/media/tmdb_client.py + movie_night.py — narrowing
#          session, provider templates, hook seam, honest not-available path.
# Called-by: pytest
# Depends-on: app.media.tmdb_client, app.media.movie_night
# Last-renovated: 2026-06-12
from __future__ import annotations

import pytest

from app.media import movie_night, tmdb_client
from app.media.tmdb_client import Film, WatchProviders


@pytest.fixture(autouse=True)
def fresh_session():
    movie_night._session = None
    yield
    movie_night._session = None


def _films(*titles_years):
    return [Film(id=i + 1, title=t, year=y, rating=7.0, overview=f"Pitch for {t}")
            for i, (t, y) in enumerate(titles_years)]


# ── tmdb client param building (mocked transport) ──────────────────────────

@pytest.mark.asyncio
async def test_discover_builds_params(monkeypatch):
    seen = {}

    async def fake_get(path, params=None):
        seen["path"] = path
        seen["params"] = params
        return {"results": [{"id": 5, "title": "Heat", "release_date": "1995-12-15",
                             "vote_average": 8.2, "overview": "Crime epic",
                             "genre_ids": [80, 18]}]}

    monkeypatch.setattr(tmdb_client, "_get", fake_get)
    films = await tmdb_client.discover(genres=[35], year_from=1990, year_to=1999,
                                       min_rating=7, max_runtime=100, sort="rating")
    assert seen["path"] == "/discover/movie"
    p = seen["params"]
    assert p["with_genres"] == "35"
    assert p["primary_release_date.gte"] == "1990-01-01"
    assert p["primary_release_date.lte"] == "1999-12-31"
    assert p["vote_average.gte"] == 7
    assert p["with_runtime.lte"] == 100
    assert p["sort_by"] == "vote_average.desc"
    assert films[0].title == "Heat" and films[0].year == 1995


@pytest.mark.asyncio
async def test_watch_providers_region_split(monkeypatch):
    async def fake_get(path, params=None):
        return {"results": {"GB": {
            "link": "https://tmdb/x",
            "flatrate": [{"provider_name": "Netflix"}],
            "rent": [{"provider_name": "Apple TV"}],
            "buy": [{"provider_name": "Amazon Video"}],
        }}}

    monkeypatch.setattr(tmdb_client, "_get", fake_get)
    out = await tmdb_client.watch_providers(5)
    assert out.region == "GB"
    assert out.flatrate == ["Netflix"]
    assert out.rent == ["Apple TV"]


def test_missing_key_is_clear_error(monkeypatch):
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    with pytest.raises(tmdb_client.TmdbError, match="TMDB_API_KEY"):
        tmdb_client._auth()


# ── narrowing session ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_find_then_refine_keeps_filters(monkeypatch):
    captured = []

    async def fake_discover(**kwargs):
        captured.append(kwargs)
        return _films(("Groundhog Day", 1993), ("Clueless", 1995))

    monkeypatch.setattr(tmdb_client, "discover", fake_discover)

    first = await movie_night.find_candidates(
        fresh=True, mood="funny", year_from=1990, year_to=1999)
    assert first["ok"] and first["candidates"][0]["pitch"].startswith("Pitch")
    assert captured[0]["genres"] == [tmdb_client.GENRE_IDS["comedy"]]

    second = await movie_night.find_candidates(fresh=False, year_to=1989)
    assert second["ok"]
    # refine kept the comedy genre AND year_from while moving year_to older
    assert captured[1]["genres"] == [tmdb_client.GENRE_IDS["comedy"]]
    assert captured[1]["year_from"] == 1990
    assert captured[1]["year_to"] == 1989


@pytest.mark.asyncio
async def test_mood_map_mind_bending(monkeypatch):
    captured = {}

    async def fake_discover(**kwargs):
        captured.update(kwargs)
        return _films(("Primer", 2004))

    monkeypatch.setattr(tmdb_client, "discover", fake_discover)
    await movie_night.find_candidates(fresh=True, mood="mind-bending")
    assert set(captured["genres"]) == {878, 53}  # sci-fi + thriller


@pytest.mark.asyncio
async def test_resolve_candidate_by_index_and_title(monkeypatch):
    async def fake_discover(**kwargs):
        return _films(("Heat", 1995), ("Ronin", 1998))

    monkeypatch.setattr(tmdb_client, "discover", fake_discover)
    await movie_night.find_candidates(fresh=True, mood="gritty")
    assert movie_night.resolve_candidate(2).title == "Ronin"
    assert movie_night.resolve_candidate("ronin").title == "Ronin"
    assert movie_night.resolve_candidate(None).title == "Heat"
    assert movie_night.resolve_candidate(99) is None


# ── providers + play ───────────────────────────────────────────────────────

def _wire_pick(monkeypatch, flatrate):
    async def fake_discover(**kwargs):
        return _films(("The Big Lebowski", 1998))

    async def fake_providers(movie_id, region=None):
        return WatchProviders(movie_id=movie_id, region="GB", flatrate=flatrate,
                              rent=["Apple TV"], buy=["Amazon Video"])

    async def fake_details(movie_id):
        film = _films(("The Big Lebowski", 1998))[0]
        film.runtime_minutes = 117
        return film

    monkeypatch.setattr(tmdb_client, "discover", fake_discover)
    monkeypatch.setattr(tmdb_client, "watch_providers", fake_providers)
    monkeypatch.setattr(tmdb_client, "details", fake_details)


@pytest.mark.asyncio
async def test_providers_reports_streaming(monkeypatch):
    _wire_pick(monkeypatch, ["Netflix"])
    await movie_night.find_candidates(fresh=True, mood="funny")
    out = await movie_night.providers_for(1)
    assert out["ok"] and out["streaming"] == ["Netflix"]
    assert out["playable"] is True
    assert out["film"]["runtime_minutes"] == 117


@pytest.mark.asyncio
async def test_play_dispatches_to_bed_screen_with_hook(monkeypatch):
    _wire_pick(monkeypatch, ["Netflix"])
    await movie_night.find_candidates(fresh=True, mood="funny")

    hook_ctx = {}

    async def lights_hook(context):
        hook_ctx.update(context)

    opened = {}

    async def fake_open(url, display="main", **kwargs):
        opened["url"] = url
        opened["display"] = display
        opened["session"] = kwargs.get("session")
        return {"ok": True, "result": {"window_id": 7, "note": None}}

    from app.media import display_client
    monkeypatch.setattr(display_client, "open_on_display", fake_open)
    movie_night.pre_play_hooks.append(lights_hook)
    try:
        out = await movie_night.play(1)
    finally:
        movie_night.pre_play_hooks.clear()

    assert out["ok"] and out["provider"] == "Netflix"
    assert opened["display"] == "bed screen"          # spec default
    assert opened["session"] == "streaming"
    assert "netflix.com/search" in opened["url"] and "Big+Lebowski" in opened["url"]
    assert hook_ctx["provider"] == "Netflix"          # the seam fired
    assert "manual_step" in out


@pytest.mark.asyncio
async def test_play_honest_when_on_nothing_he_has(monkeypatch):
    _wire_pick(monkeypatch, ["Some Obscure Service"])
    await movie_night.find_candidates(fresh=True, mood="funny")
    out = await movie_night.play(1)
    assert out["ok"] is False
    assert "where it IS available" in out["error"]
    assert out["rent"] == ["Apple TV"]                # alternatives surfaced


def test_movie_tools_register_cleanly():
    from app.tools import movie_tools
    from app.tools.registry import list_tools
    movie_tools.register_movie_tools()
    names = {t["name"] for t in list_tools()}
    assert {"find_films", "refine_films", "film_providers", "play_film"} <= names
