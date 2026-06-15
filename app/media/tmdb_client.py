# FILE: app/media/tmdb_client.py
# Purpose: TMDB API client — discover/search/watch-providers/similar/details
#          as slim dataclasses (no raw TMDB JSON leaks upward).
# Called-by: app.media.movie_night, app.tools.movie_tools
# Depends-on: httpx; env TMDB_API_KEY (v3 key or v4 read token, free at themoviedb.org)
# Last-renovated: 2026-06-12
"""
TMDB client.

Auth: TMDB_API_KEY env var. A short hex string is treated as a v3 key
(api_key query param); a long eyJ… JWT as a v4 read token (Bearer header).
Either works — whatever Taz pastes in.

Region defaults to GB (env ASTRA_WATCH_REGION for the Portugal future).
Overviews are truncated for TTS at the edge, never upstream.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)

API_BASE = "https://api.themoviedb.org/3"
REQUEST_TIMEOUT = 12.0
OVERVIEW_TTS_LIMIT = 220

# TMDB's documented static movie genre ids.
GENRE_IDS = {
    "action": 28, "adventure": 12, "animation": 16, "comedy": 35, "crime": 80,
    "documentary": 99, "drama": 18, "family": 10751, "fantasy": 14,
    "history": 36, "horror": 27, "music": 10402, "mystery": 9648,
    "romance": 10749, "science fiction": 878, "sci-fi": 878, "thriller": 53,
    "war": 10752, "western": 37,
}

SORT_OPTIONS = {
    "popularity": "popularity.desc",
    "rating": "vote_average.desc",
    "newest": "primary_release_date.desc",
}


class TmdbError(Exception):
    """TMDB unreachable / bad reply / missing key."""


@dataclass
class Film:
    id: int
    title: str
    year: Optional[int]
    rating: float                 # TMDB vote_average 0-10
    overview: str                 # truncated for TTS
    genre_ids: List[int] = field(default_factory=list)
    runtime_minutes: Optional[int] = None   # only filled by details()


@dataclass
class WatchProviders:
    movie_id: int
    region: str
    flatrate: List[str] = field(default_factory=list)   # streaming (included)
    rent: List[str] = field(default_factory=list)
    buy: List[str] = field(default_factory=list)
    link: str = ""                                      # TMDB's provider page


def default_region() -> str:
    return (os.getenv("ASTRA_WATCH_REGION") or "GB").strip().upper() or "GB"


def truncate_overview(text: Optional[str], limit: int = OVERVIEW_TTS_LIMIT) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    cut = clean[:limit]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut + "…"


def _auth() -> tuple[dict, dict]:
    """(headers, params) carrying whichever credential style the key is."""
    key = (os.getenv("TMDB_API_KEY") or "").strip()
    if not key:
        raise TmdbError(
            "TMDB key missing — set TMDB_API_KEY (free key from "
            "themoviedb.org → Settings → API)"
        )
    if key.startswith("eyJ"):                  # v4 read access token
        return {"Authorization": f"Bearer {key}"}, {}
    return {}, {"api_key": key}                # v3 key


async def _get(path: str, params: Optional[dict] = None) -> dict:
    headers, auth_params = _auth()
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.get(
                f"{API_BASE}{path}", params={**(params or {}), **auth_params},
                headers=headers,
            )
        resp.raise_for_status()
        return resp.json()
    except TmdbError:
        raise
    except Exception as exc:
        raise TmdbError(f"TMDB request failed: {exc}") from exc


def _film_from_row(row: dict) -> Film:
    date = str(row.get("release_date") or "")
    return Film(
        id=int(row.get("id") or 0),
        title=str(row.get("title") or "Untitled"),
        year=int(date[:4]) if len(date) >= 4 and date[:4].isdigit() else None,
        rating=float(row.get("vote_average") or 0.0),
        overview=truncate_overview(row.get("overview")),
        genre_ids=list(row.get("genre_ids") or []),
    )


async def discover(genres: Optional[List[int]] = None,
                   year_from: Optional[int] = None,
                   year_to: Optional[int] = None,
                   min_rating: Optional[float] = None,
                   max_runtime: Optional[int] = None,
                   sort: str = "popularity",
                   max_results: int = 5) -> List[Film]:
    """/discover/movie with the narrowing params movie-night uses."""
    params: dict = {
        "sort_by": SORT_OPTIONS.get(sort, SORT_OPTIONS["popularity"]),
        "include_adult": "false",
        "vote_count.gte": 100,    # keeps obscure zero-vote noise out
        "page": 1,
    }
    if genres:
        params["with_genres"] = ",".join(str(g) for g in genres)
    if year_from:
        params["primary_release_date.gte"] = f"{year_from}-01-01"
    if year_to:
        params["primary_release_date.lte"] = f"{year_to}-12-31"
    if min_rating:
        params["vote_average.gte"] = min_rating
    if max_runtime:
        params["with_runtime.lte"] = max_runtime
    data = await _get("/discover/movie", params)
    return [_film_from_row(r) for r in (data.get("results") or [])[:max_results]]


async def search(title: str, max_results: int = 5) -> List[Film]:
    data = await _get("/search/movie", {"query": title, "include_adult": "false"})
    return [_film_from_row(r) for r in (data.get("results") or [])[:max_results]]


async def similar(movie_id: int, max_results: int = 5) -> List[Film]:
    data = await _get(f"/movie/{movie_id}/similar")
    return [_film_from_row(r) for r in (data.get("results") or [])[:max_results]]


async def details(movie_id: int) -> Film:
    row = await _get(f"/movie/{movie_id}")
    film = _film_from_row(row)
    film.runtime_minutes = row.get("runtime")
    film.genre_ids = [g.get("id") for g in (row.get("genres") or []) if g.get("id")]
    return film


async def watch_providers(movie_id: int, region: Optional[str] = None) -> WatchProviders:
    """Streaming/rent/buy provider names for the region (default GB)."""
    region = (region or default_region()).upper()
    data = await _get(f"/movie/{movie_id}/watch/providers")
    row = ((data.get("results") or {}).get(region)) or {}

    def names(kind: str) -> List[str]:
        return [str(p.get("provider_name")) for p in (row.get(kind) or [])
                if p.get("provider_name")]

    return WatchProviders(
        movie_id=movie_id,
        region=region,
        flatrate=names("flatrate"),
        rent=names("rent"),
        buy=names("buy"),
        link=str(row.get("link") or ""),
    )
