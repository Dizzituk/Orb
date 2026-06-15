# FILE: app/media/movie_night.py
# Purpose: Movie-night orchestrator — conversational narrowing session over
#          TMDB, provider resolution (GB), dispatch to the bed screen.
# Called-by: app.tools.movie_tools
# Depends-on: app.media.tmdb_client, app.media.display_client
# Last-renovated: 2026-06-12
"""
Movie night.

Flow: "find me a film" -> narrowing session (mood words -> genre ids; "nah,
older" / "something shorter" adjust the same session) -> pick -> "it's on
Netflix" -> open the provider's page on the bed screen.

Provider URL templates live in PROVIDER_URLS (one constants block): search
pages with the title pre-filled, NOT deep links — deep-link schemes churn,
search pages survive. v1 accepts one manual interaction on the provider
page (Netflix profile click / picking the title from results).

pre_play_hooks: async callables run just before dispatch with
{film, provider, url, display}. EMPTY by design — the Home Assistant
lights call lands here later; nothing else is implemented now.
"""
from __future__ import annotations

import logging
import urllib.parse
from typing import Awaitable, Callable, List, Optional

from app.media import tmdb_client
from app.media.tmdb_client import Film, GENRE_IDS

logger = logging.getLogger(__name__)

# ── provider -> URL template (the ONE constants block) ────────────────────
PROVIDER_URLS = {
    "netflix": "https://www.netflix.com/search?q={title}",
    "amazon prime video": "https://www.amazon.co.uk/s?k={title}&i=instant-video",
    "disney plus": "https://www.disneyplus.com/search?q={title}",
    "bbc iplayer": "https://www.bbc.co.uk/iplayer/search?q={title}",
    "itvx": "https://www.itv.com/watch?q={title}",
    "channel 4": "https://www.channel4.com/search?q={title}",
    "now": "https://www.nowtv.com/gb/search?term={title}",
    "now tv cinema": "https://www.nowtv.com/gb/search?term={title}",
    "apple tv plus": "https://tv.apple.com/gb/search?term={title}",
    "paramount plus": "https://www.paramountplus.com/gb/search/?q={title}",
    "sky go": "https://www.sky.com/watch/search?q={title}",
}

# Mood words -> genre id sets (extends plain genre names from GENRE_IDS).
MOOD_MAP = {
    "funny": ["comedy"],
    "something funny": ["comedy"],
    "laugh": ["comedy"],
    "mind-bending": ["science fiction", "thriller"],
    "mind bending": ["science fiction", "thriller"],
    "trippy": ["science fiction", "mystery"],
    "scary": ["horror"],
    "spooky": ["horror", "mystery"],
    "tense": ["thriller"],
    "gritty": ["crime", "drama"],
    "romantic": ["romance"],
    "feel-good": ["comedy", "family"],
    "feel good": ["comedy", "family"],
    "epic": ["adventure", "fantasy"],
    "true story": ["history", "drama"],
    "space": ["science fiction"],
    "explosions": ["action"],
}

DEFAULT_DISPLAY = "bed screen"

# Hook seam (see module docstring). Signature: async fn(context: dict) -> None.
pre_play_hooks: List[Callable[[dict], Awaitable[None]]] = []


class NarrowingSession:
    """The active 'find me a film' conversation — params + last candidates."""

    def __init__(self):
        self.params: dict = {"sort": "popularity"}
        self.candidates: List[Film] = []

    def apply(self, *, mood=None, genres=None, year_from=None, year_to=None,
              min_rating=None, max_runtime=None, sort=None) -> None:
        genre_ids: List[int] = list(self.params.get("genres") or [])
        for word in (genres or []):
            gid = GENRE_IDS.get(str(word).strip().lower())
            if gid and gid not in genre_ids:
                genre_ids.append(gid)
        if mood:
            for name in MOOD_MAP.get(str(mood).strip().lower(), []):
                gid = GENRE_IDS.get(name)
                if gid and gid not in genre_ids:
                    genre_ids.append(gid)
        if genre_ids:
            self.params["genres"] = genre_ids
        if year_from is not None:
            self.params["year_from"] = year_from
        if year_to is not None:
            self.params["year_to"] = year_to
        if min_rating is not None:
            self.params["min_rating"] = min_rating
        if max_runtime is not None:
            self.params["max_runtime"] = max_runtime
        if sort:
            self.params["sort"] = sort


_session: Optional[NarrowingSession] = None


def _ensure_session(fresh: bool = False) -> NarrowingSession:
    global _session
    if fresh or _session is None:
        _session = NarrowingSession()
    return _session


async def find_candidates(fresh: bool = False, similar_to: Optional[str] = None,
                          **narrowing) -> dict:
    """Start or refine the narrowing session; returns speakable candidates."""
    session = _ensure_session(fresh)
    session.apply(**narrowing)

    if similar_to:
        seeds = await tmdb_client.search(similar_to, max_results=1)
        if not seeds:
            return {"ok": False, "error": f"couldn't find '{similar_to}' to riff on"}
        films = await tmdb_client.similar(seeds[0].id, max_results=5)
    else:
        films = await tmdb_client.discover(
            genres=session.params.get("genres"),
            year_from=session.params.get("year_from"),
            year_to=session.params.get("year_to"),
            min_rating=session.params.get("min_rating"),
            max_runtime=session.params.get("max_runtime"),
            sort=session.params.get("sort", "popularity"),
            max_results=5,
        )
    session.candidates = films
    return {
        "ok": True,
        "candidates": [
            {"index": i + 1, "id": f.id, "title": f.title, "year": f.year,
             "rating": round(f.rating, 1), "pitch": f.overview}
            for i, f in enumerate(films)
        ],
        "active_filters": dict(session.params),
    }


def resolve_candidate(selection) -> Optional[Film]:
    """1-based index or title fragment against the session's candidates."""
    session = _ensure_session()
    if selection is None or selection == "":
        return session.candidates[0] if session.candidates else None
    if isinstance(selection, int) or (isinstance(selection, str) and str(selection).strip().isdigit()):
        idx = int(selection) - 1
        return session.candidates[idx] if 0 <= idx < len(session.candidates) else None
    wanted = str(selection).strip().lower()
    for film in session.candidates:
        if wanted in film.title.lower():
            return film
    return None


async def providers_for(selection, region: Optional[str] = None) -> dict:
    """Where a pick streams in the region — honest when it's nowhere he has."""
    film = resolve_candidate(selection)
    if not film:
        # Maybe they named a film straight out — search TMDB directly.
        if selection:
            hits = await tmdb_client.search(str(selection), max_results=1)
            film = hits[0] if hits else None
        if not film:
            return {"ok": False, "error": "I don't know which film you mean"}
    providers = await tmdb_client.watch_providers(film.id, region)
    details = await tmdb_client.details(film.id)
    return {
        "ok": True,
        "film": {"id": film.id, "title": film.title, "year": film.year,
                 "runtime_minutes": details.runtime_minutes,
                 "pitch": film.overview},
        "streaming": providers.flatrate,
        "rent": providers.rent,
        "buy": providers.buy,
        "region": providers.region,
        "playable": any(p.lower() in PROVIDER_URLS for p in providers.flatrate),
    }


async def play(selection, display: str = DEFAULT_DISPLAY,
               provider: Optional[str] = None, region: Optional[str] = None) -> dict:
    """Open the pick's provider page on the chosen screen (default bed screen)."""
    info = await providers_for(selection, region)
    if not info.get("ok"):
        return info
    film = info["film"]

    chosen = None
    if provider:
        chosen = provider if provider.lower() in PROVIDER_URLS else None
        if not chosen:
            return {"ok": False, "error": f"I don't have a URL pattern for {provider}"}
    else:
        for name in info["streaming"]:
            if name.lower() in PROVIDER_URLS:
                chosen = name
                break
    if not chosen:
        return {
            "ok": False,
            "film": film,
            "streaming": info["streaming"],
            "rent": info["rent"],
            "buy": info["buy"],
            "error": "that one isn't on any service he has — say where it IS "
                     "available and offer an alternative instead",
        }

    url = PROVIDER_URLS[chosen.lower()].format(
        title=urllib.parse.quote_plus(film["title"]))

    context = {"film": film, "provider": chosen, "url": url, "display": display}
    for hook in pre_play_hooks:   # lights etc. land here later — empty today
        try:
            await hook(context)
        except Exception:
            logger.exception("[movie_night] pre_play hook failed (non-fatal)")

    from app.media import display_client
    opened = await display_client.open_on_display(
        url, display, cdp=True, session="streaming",
    )
    if not opened.get("ok"):
        return {"ok": False, "error": f"couldn't open the window: "
                                      f"{opened.get('error', 'desktop unreachable')}"}
    window = opened.get("result") or {}
    return {
        "ok": True,
        "film": film,
        "provider": chosen,
        "window": window,
        "note": window.get("note"),
        "manual_step": "Provider page is open with the film queued in search — "
                       "one click on the title (or a Netflix profile) may be "
                       "needed; that's expected in v1.",
    }
