# FILE: app/tools/movie_tools.py
# Purpose: Movie-night chat tools — conversational film narrowing (TMDB),
#          provider lookup, "put it on the bed screen".
# Called-by: app.tools.registry (_register_defaults)
# Depends-on: app.media.movie_night
# Last-renovated: 2026-06-12
"""
Movie-night chat tools (v1, 2026-06-12).

  find_films        start a narrowing session ("something funny from the 90s")
  refine_films      adjust it ("nah, older", "something shorter", "higher rated")
  film_providers    where a pick streams in GB ("is it on Netflix?")
  play_film         open the provider page on the bed screen

Conversational style the descriptions ask of the model: speak 2-3
candidates with their one-line pitches as prose, iterate naturally, and
when a film is on nothing he has, say where it IS available and offer an
alternative instead. Region is GB (env ASTRA_WATCH_REGION later).

Handler style mirrors vehicle_tools: {ok: false, error}; schemas inline.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
# Handlers
# ────────────────────────────────────────────────────────────────────────

def _narrowing_kwargs(input_data: dict) -> dict:
    return {
        "mood": input_data.get("mood"),
        "genres": input_data.get("genres"),
        "year_from": input_data.get("year_from"),
        "year_to": input_data.get("year_to"),
        "min_rating": input_data.get("min_rating"),
        "max_runtime": input_data.get("max_runtime_minutes"),
        "sort": input_data.get("sort"),
    }


async def find_films_handler(input_data: dict, context: Optional[dict]) -> dict:
    try:
        from app.media import movie_night
        return await movie_night.find_candidates(
            fresh=True,
            similar_to=input_data.get("similar_to"),
            **_narrowing_kwargs(input_data),
        )
    except Exception as exc:
        logger.exception("[movie_tools] find_films failed")
        return {"ok": False, "error": str(exc)}


async def refine_films_handler(input_data: dict, context: Optional[dict]) -> dict:
    try:
        from app.media import movie_night
        return await movie_night.find_candidates(
            fresh=False,
            similar_to=input_data.get("similar_to"),
            **_narrowing_kwargs(input_data),
        )
    except Exception as exc:
        logger.exception("[movie_tools] refine_films failed")
        return {"ok": False, "error": str(exc)}


async def film_providers_handler(input_data: dict, context: Optional[dict]) -> dict:
    try:
        from app.media import movie_night
        return await movie_night.providers_for(
            input_data.get("selection"), input_data.get("region")
        )
    except Exception as exc:
        logger.exception("[movie_tools] film_providers failed")
        return {"ok": False, "error": str(exc)}


async def play_film_handler(input_data: dict, context: Optional[dict]) -> dict:
    try:
        from app.media import movie_night
        return await movie_night.play(
            input_data.get("selection"),
            display=input_data.get("display") or movie_night.DEFAULT_DISPLAY,
            provider=input_data.get("provider"),
            region=input_data.get("region"),
        )
    except Exception as exc:
        logger.exception("[movie_tools] play_film failed")
        return {"ok": False, "error": str(exc)}


# ────────────────────────────────────────────────────────────────────────
# Schemas (inline — self-contained file)
# ────────────────────────────────────────────────────────────────────────

_OK_ERR = {"ok": {"type": "boolean"}, "error": {"type": "string"}}

_NARROWING_PROPS = {
    "mood": {"type": ["string", "null"],
             "description": "Mood words: funny, mind-bending, scary, tense, "
                            "romantic, feel-good, epic, gritty, space…"},
    "genres": {"type": ["array", "null"], "items": {"type": "string"},
               "description": "Plain genre names (comedy, thriller, sci-fi…)"},
    "year_from": {"type": ["integer", "null"], "description": "e.g. 1990"},
    "year_to": {"type": ["integer", "null"], "description": "e.g. 1999"},
    "min_rating": {"type": ["number", "null"],
                   "description": "TMDB 0-10; 7 = solid, 7.5+ = great"},
    "max_runtime_minutes": {"type": ["integer", "null"],
                            "description": "'something shorter' -> e.g. 100"},
    "sort": {"type": ["string", "null"], "enum": ["popularity", "rating", "newest", None]},
    "similar_to": {"type": ["string", "null"],
                   "description": "'like Inception' -> seeds from that film"},
}

_CANDIDATES_OUTPUT = {
    "type": "object",
    "required": ["ok"],
    "properties": {
        **_OK_ERR,
        "candidates": {"type": ["array", "null"], "items": {"type": "object"}},
        "active_filters": {"type": ["object", "null"]},
    },
}

FIND_FILMS_INPUT = {"type": "object", "properties": _NARROWING_PROPS}
REFINE_FILMS_INPUT = {"type": "object", "properties": _NARROWING_PROPS}

_SELECTION_PROPS = {
    "selection": {"type": ["string", "integer", "null"],
                  "description": "1-based index into the candidates, a title "
                                 "fragment, or a film name straight out. Omit "
                                 "for the top candidate."},
    "region": {"type": ["string", "null"], "description": "Default GB"},
}

FILM_PROVIDERS_INPUT = {"type": "object", "properties": _SELECTION_PROPS}
FILM_PROVIDERS_OUTPUT = {
    "type": "object",
    "required": ["ok"],
    "properties": {
        **_OK_ERR,
        "film": {"type": ["object", "null"]},
        "streaming": {"type": ["array", "null"], "items": {"type": "string"}},
        "rent": {"type": ["array", "null"], "items": {"type": "string"}},
        "buy": {"type": ["array", "null"], "items": {"type": "string"}},
        "region": {"type": ["string", "null"]},
        "playable": {"type": ["boolean", "null"]},
    },
}

PLAY_FILM_INPUT = {
    "type": "object",
    "properties": {
        **_SELECTION_PROPS,
        "display": {"type": ["string", "null"],
                    "description": "Default 'bed screen'"},
        "provider": {"type": ["string", "null"],
                     "description": "Force a service (Netflix, BBC iPlayer…) "
                                    "instead of the first streaming match"},
    },
}
PLAY_FILM_OUTPUT = {
    "type": "object",
    "required": ["ok"],
    "properties": {
        **_OK_ERR,
        "film": {"type": ["object", "null"]},
        "provider": {"type": ["string", "null"]},
        "window": {"type": ["object", "null"]},
        "note": {"type": ["string", "null"]},
        "manual_step": {"type": ["string", "null"]},
        "streaming": {"type": ["array", "null"], "items": {"type": "string"}},
        "rent": {"type": ["array", "null"], "items": {"type": "string"}},
        "buy": {"type": ["array", "null"], "items": {"type": "string"}},
    },
}


# ────────────────────────────────────────────────────────────────────────
# Registration
# ────────────────────────────────────────────────────────────────────────

def register_movie_tools() -> None:
    """Register the movie-night tools with the global tool registry.
    Called once from registry._register_defaults at module load."""
    from app.tools.registry import ToolDefinition, register_tool

    register_tool(ToolDefinition(
        name="find_films",
        version="v1",
        description=(
            "Start a movie-night search ('find me something funny from the "
            "nineties'). Map the user's mood words and constraints into the "
            "inputs. Speak 2-3 of the candidates conversationally with their "
            "one-line pitches — prose, not a list. Follow-ups ('nah, older') "
            "go to refine_films, NOT here."
        ),
        input_schema=FIND_FILMS_INPUT,
        output_schema=_CANDIDATES_OUTPUT,
        handler=find_films_handler,
    ))

    register_tool(ToolDefinition(
        name="refine_films",
        version="v1",
        description=(
            "Adjust the ACTIVE movie search: 'older' -> year_to, 'something "
            "shorter' -> max_runtime_minutes, 'better rated' -> min_rating, "
            "extra mood/genre words add genres. Keeps earlier filters and "
            "returns fresh candidates."
        ),
        input_schema=REFINE_FILMS_INPUT,
        output_schema=_CANDIDATES_OUTPUT,
        handler=refine_films_handler,
    ))

    register_tool(ToolDefinition(
        name="film_providers",
        version="v1",
        description=(
            "Where a film streams in the UK ('is it on Netflix?'). playable "
            "tells you a known service carries it. When streaming is empty, "
            "say where it IS available (rent/buy) and offer an alternative "
            "candidate instead — never a dead end."
        ),
        input_schema=FILM_PROVIDERS_INPUT,
        output_schema=FILM_PROVIDERS_OUTPUT,
        handler=film_providers_handler,
    ))

    register_tool(ToolDefinition(
        name="play_film",
        version="v1",
        description=(
            "Open the chosen film's streaming service on a screen (default "
            "the bed screen): 'it's on Netflix — putting it on the bed "
            "screen.' Speak manual_step if present (one click on the title/"
            "profile may be needed, expected in v1). When it fails because "
            "no service carries it, relay the alternatives honestly."
        ),
        input_schema=PLAY_FILM_INPUT,
        output_schema=PLAY_FILM_OUTPUT,
        handler=play_film_handler,
    ))
