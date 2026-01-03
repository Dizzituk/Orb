# FILE: app/llm/weaver_stream.py
from __future__ import annotations

import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

from sqlalchemy.orm import Session

from app.llm.registry import RoutingTrace

# Direct imports (do NOT rely on app.specs package re-exports)
from app.specs import service as specs_service

from app.llm.weaver_stream_core import (
    WeaverContext,
    build_spec_from_dict,
    build_weaver_prompt,
    build_weaver_update_prompt,
    gather_weaver_context,
    gather_weaver_delta_context,
    parse_weaver_response,
)

# NOTE: we import memory service lazily inside the function where needed
# to reduce import-time coupling.


WEAVER_MODEL = os.getenv("WEAVER_MODEL", "gpt-5-mini")


def _sse(data: Dict[str, Any]) -> str:
    # Single "data:" line SSE frames (your frontend already expects this style)
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _is_weaver_spec_json(spec_json: Dict[str, Any]) -> bool:
    prov = spec_json.get("provenance", {}) or {}
    gm = str(prov.get("generator_model", "") or "").lower()
    # Weaver stamps: "weaver", "weaver-v1-...", "weaver-gpt-..."
    return ("weaver" in gm) or gm.startswith("gpt")  # keep permissive for early versions


def _get_last_consumed_message_id_from_spec(spec_json: Dict[str, Any]) -> int:
    prov = spec_json.get("provenance", {}) or {}
    ids = prov.get("source_message_ids") or []
    if not isinstance(ids, list) or not ids:
        return 0
    return max(_safe_int(i, 0) for i in ids)


def _merge_message_ids(prev_ids: List[int], delta_ids: List[int]) -> List[int]:
    s = set()
    for i in prev_ids or []:
        s.add(_safe_int(i, 0))
    for i in delta_ids or []:
        s.add(_safe_int(i, 0))
    s.discard(0)
    return sorted(s)


async def generate_weaver_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
    conversation_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Generate SSE stream for Weaver spec building.

    Fixes:
    - Incremental weaving: only weave delta messages since last weaver spec
    - Control-plane filtering handled inside weaver_stream_core (command triggers + weaver outputs)
    - Import stability: use app.specs.service directly (no __init__ re-exports required)
    """

    _ = trace  # present for routing trace, but we don't require it in this module

    yield _sse({"type": "status", "text": "🧵 Weaving spec from conversation..."})
    yield _sse({"type": "status", "text": "📋 Gathering conversation context..."})

    # 1) Try to load last WEAVER spec for this project
    previous_db_spec = None
    previous_spec_json: Optional[Dict[str, Any]] = None
    previous_spec_id: Optional[str] = None

    try:
        # You may have non-weaver specs in the same store; we pick the most recent WEAVER one.
        # We walk back a few recent specs if needed.
        recent = specs_service.list_specs(db, project_id, limit=10)  # type: ignore[attr-defined]
        for row in recent or []:
            sj = getattr(row, "content_json", None)
            if isinstance(sj, dict) and _is_weaver_spec_json(sj):
                previous_db_spec = row
                previous_spec_json = sj
                previous_spec_id = getattr(row, "spec_id", None)
                break
    except Exception:
        # If list_specs isn't available in your branch, fall back to get_latest_spec
        try:
            row = specs_service.get_latest_spec(db, project_id)
            sj = getattr(row, "content_json", None)
            if isinstance(sj, dict) and _is_weaver_spec_json(sj):
                previous_db_spec = row
                previous_spec_json = sj
                previous_spec_id = getattr(row, "spec_id", None)
        except Exception:
            previous_db_spec = None
            previous_spec_json = None
            previous_spec_id = None

    # 2) If we have a previous weaver spec, pull delta messages since it
    if previous_spec_json:
        since_id = _get_last_consumed_message_id_from_spec(previous_spec_json)
        delta_ctx = gather_weaver_delta_context(db, project_id, since_message_id=since_id)

        if len(delta_ctx.messages) == 0:
            # IMPORTANT: This is the behavior you want after restart + re-run with no new typing:
            # "Found 0 new messages" and re-show the last spec without regenerating.
            yield _sse(
                {
                    "type": "status",
                    "text": f"Found 0 new messages (0 tokens) since last weave. Re-using previous spec.",
                }
            )

            # Render previous spec
            try:
                # Prefer markdown conversion from stored JSON
                md = specs_service.spec_json_to_markdown(previous_spec_json)  # type: ignore[attr-defined]
            except Exception:
                # Fallback: if only SpecSchema markdown is available
                try:
                    schema = specs_service.get_spec_schema(previous_db_spec)  # type: ignore[arg-type]
                    md = specs_service.spec_to_markdown(schema)
                except Exception:
                    md = json.dumps(previous_spec_json, indent=2, ensure_ascii=False)

            yield _sse(
                {
                    "type": "weaver_result",
                    "mode": "reuse",
                    "spec_id": previous_spec_id,
                    "text": md,
                }
            )

            # Do NOT create a new spec record when there is no delta.
            return

        # Delta exists → UPDATE mode
        yield _sse(
            {
                "type": "status",
                "text": f"Found {len(delta_ctx.messages)} new messages ({delta_ctx.token_estimate} tokens) since last weave.",
            }
        )
        yield _sse({"type": "status", "text": "🤖 Updating spec structure..."})

        prev_core: Dict[str, Any] = dict(previous_spec_json)
        prev_weak: List[str] = []
        ws = previous_spec_json.get("weak_spots")
        if isinstance(ws, list):
            prev_weak = [str(x) for x in ws if str(x).strip()]

        prompt = build_weaver_update_prompt(prev_core, prev_weak, delta_ctx)

        # Call LLM streaming (local import keeps module import stable)
        from app.llm.stream_llm import stream_llm  # type: ignore

        response_text_parts: List[str] = []
        async for chunk in stream_llm(
            provider="openai",
            model=WEAVER_MODEL,
            messages=[{"role": "user", "content": prompt}],
        ):
            # Stream chunks through to UI (your stream_llm yields dict chunks)
            if isinstance(chunk, dict):
                t = chunk.get("text") or chunk.get("delta") or ""
            else:
                t = str(chunk)

            if t:
                response_text_parts.append(t)
                yield _sse({"type": "delta", "text": t})

        response_text = "".join(response_text_parts)
        spec_dict, err = parse_weaver_response(response_text)
        if not spec_dict:
            yield _sse({"type": "error", "text": f"Weaver produced invalid spec JSON: {err}"})
            return

        # Merge provenance IDs so next run advances correctly
        prev_ids = previous_spec_json.get("provenance", {}).get("source_message_ids") or []
        prev_ids_int = [_safe_int(i, 0) for i in prev_ids if _safe_int(i, 0) > 0]
        merged_ids = _merge_message_ids(prev_ids_int, delta_ctx.message_ids)

        # Build a combined context for provenance stamping
        combined_ctx = WeaverContext(
            messages=[],
            message_ids=merged_ids,
            token_estimate=delta_ctx.token_estimate,
            timestamp_start=delta_ctx.timestamp_start,
            timestamp_end=delta_ctx.timestamp_end,
            commit_hash=delta_ctx.commit_hash,
        )

        spec_schema = build_spec_from_dict(
            spec_dict,
            combined_ctx,
            project_id=project_id,
            conversation_id=conversation_id,
            generator_model=f"weaver-{WEAVER_MODEL}",
        )

        yield _sse({"type": "status", "text": "💾 Saving spec..."})

        # Save new spec, aliasing the previous
        new_spec_id = specs_service.create_spec(
            db,
            project_id=project_id,
            spec=spec_schema,
            status="draft",
            alias_of=previous_spec_id,
        )

        # Render for UI
        try:
            md = specs_service.spec_to_markdown(spec_schema)
        except Exception:
            md = json.dumps(spec_dict, indent=2, ensure_ascii=False)

        yield _sse(
            {
                "type": "weaver_result",
                "mode": "update",
                "spec_id": new_spec_id,
                "alias_of": previous_spec_id,
                "text": md,
            }
        )

        return

    # 3) No previous weaver spec → full weave from recent context
    ctx = gather_weaver_context(db, project_id)
    yield _sse(
        {
            "type": "status",
            "text": f"Found {len(ctx.messages)} messages ({ctx.token_estimate} tokens) in recent context.",
        }
    )
    yield _sse({"type": "status", "text": "🤖 Generating spec structure..."})

    prompt = build_weaver_prompt(ctx)

    from app.llm.stream_llm import stream_llm  # type: ignore

    response_text_parts = []
    async for chunk in stream_llm(
        provider="openai",
        model=WEAVER_MODEL,
        messages=[{"role": "user", "content": prompt}],
    ):
        if isinstance(chunk, dict):
            t = chunk.get("text") or chunk.get("delta") or ""
        else:
            t = str(chunk)

        if t:
            response_text_parts.append(t)
            yield _sse({"type": "delta", "text": t})

    response_text = "".join(response_text_parts)
    spec_dict, err = parse_weaver_response(response_text)
    if not spec_dict:
        yield _sse({"type": "error", "text": f"Weaver produced invalid spec JSON: {err}"})
        return

    spec_schema = build_spec_from_dict(
        spec_dict,
        ctx,
        project_id=project_id,
        conversation_id=conversation_id,
        generator_model=f"weaver-{WEAVER_MODEL}",
    )

    yield _sse({"type": "status", "text": "💾 Saving spec..."})
    spec_id = specs_service.create_spec(db, project_id=project_id, spec=spec_schema, status="draft")

    try:
        md = specs_service.spec_to_markdown(spec_schema)
    except Exception:
        md = json.dumps(spec_dict, indent=2, ensure_ascii=False)

    yield _sse(
        {
            "type": "weaver_result",
            "mode": "new",
            "spec_id": spec_id,
            "text": md,
        }
    )
