# FILE: app/llm/routing/image_chat_routing.py
# Purpose: Image-bearing chat turn router for the desktop /stream/chat endpoint.
# Called-by: app.llm.stream_router
# Depends-on: app.debug.gemini_tool_loop, app.llm.model_families, app.llm.routing.prompt_builders, app.memory.service
# Last-renovated: 2026-06-11
"""
Image-bearing chat turn router for the desktop /stream/chat endpoint.

When a request arrives with an image already uploaded to Gemini Files API
(file_upload_uri / file_upload_gemini_name on StreamRequest), this module
bypasses the normal MODEL_SELECT flow and routes the entire turn through
Gemini's multimodal + tool-calling loop. Text-only turns continue to use
the existing routing — GPT-mini stays the default conversational model.

Rationale (2026-05-24 design call):
    Routing image-bearing turns through Gemini end-to-end avoids the
    vision→text→GPT hop. The cross-provider hop loses fidelity, costs
    a second call, and prevents Gemini from acting on what it sees
    (file edits, tool invocations). For the work-dashboard use case
    Taz is building — upload screenshot → extract data → write to
    HTML — Gemini doing the whole job in one pass is the clean path.

Detection signal is simple and per-turn:
    image present on request  → Gemini multimodal + tools  (this module)
    image NOT present         → existing handle_chat_mode routing

No sticky session, no model carry-over. The durability lives in the
HTML/file artefacts Gemini writes during the image turn — subsequent
text-only turns just read those files via tool calls regardless of
which model the normal router picks.

v1.1 (2026-05-26): Added domain-classification preamble to the image-turn
    instructions. Gemini now decides upfront whether the request is a
    domain operation (lifestyle / nutrition / training → lifestyle tools),
    a document operation (work dashboards, reports → file tools), or just
    a description request (no tools). Existing file-tool guidance kept
    verbatim under its own heading — work-dashboard flow unchanged.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator, Any

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# Model for image+tools turns — resolved from the model registry
# (app.llm.model_families) so it lives in ONE place and is overridable via
# ASTRA_MODEL_ROLE_VISION_TOOLS in .env, not hardcoded here. Still a pinned
# concrete model by default (not a "-latest" alias): a silent model swap
# under a function-calling loop is exactly where tool calls break.
from app.llm.model_families import resolve as _resolve_model
IMAGE_TURN_MODEL = _resolve_model("role_vision_tools")


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def has_image_attachment(req: Any) -> bool:
    """Did the request carry an image uploaded to Gemini Files API?

    Defensive against stale fields — both URI and gemini_name being present
    is the strongest signal, but either alone is enough for routing.
    """
    return bool(
        getattr(req, "file_upload_uri", None)
        or getattr(req, "file_upload_gemini_name", None)
    )


# ---------------------------------------------------------------------------
# Gemini content part construction
# ---------------------------------------------------------------------------

def _build_file_data_part(uri: str, mime_type: str):
    """Build a Gemini Files API content part for the uploaded image."""
    import google.generativeai as genai
    return genai.protos.Part(
        file_data=genai.protos.FileData(
            file_uri=uri,
            mime_type=mime_type or "image/jpeg",
        )
    )


# ---------------------------------------------------------------------------
# System prompt addendum for image turns
# ---------------------------------------------------------------------------

_IMAGE_TURN_INSTRUCTIONS = r"""

## Image attached to this turn

You have two families of tools available: domain tools (lifestyle/nutrition/
training) and file tools (read/write files on the filesystem). Before
calling any tool, classify what the user is actually asking for.

### Classify the request first

Pick exactly one of these three lanes based on the user's words and the
image content:

  1. **Domain operation** — the user is logging or asking about food,
     drinks, supplements, weight, workouts, training, or wellbeing.
     Signals: "log this", "I'm eating this", "I just ate", "I had",
     "for breakfast/lunch/dinner", "snack", "I did a workout/surf/walk",
     "what have I eaten today", "what's my weight trend".
     Image content: a nutrition label, a packet, a plate of food, a
     receipt for food, ingredients of a meal/smoothie.
     → Use lifestyle tools: log_nutrition, log_workout,
       get_recent_nutrition, get_weight_trend, get_recent_workouts.
     → Do NOT write an HTML file or any document for this. The lifestyle
       database is the canonical home for this data; the lifestyle tab in
       the desktop UI reads directly from it.

  2. **Document / dashboard operation** — the user wants you to create,
     update, or read a document, report, dashboard, table, or other
     artefact on disk.
     Signals: "update my dashboard", "add this to my log document",
     "write a report on", "draft a Word doc", "save this somewhere".
     Image content: a work screenshot, a chart, a payslip, a delivery
     manifest, a meeting note.
     → Use file tools: list_dir, read_file, write_file, edit_file.
     → Follow the file-handling guidance below.

  3. **Describe only** — the user just wants you to look at the image
     and tell them what's in it, with no logging or document operation
     implied.
     Signals: "what's this", "can you read this", "what does this say",
     "explain this chart".
     → Don't call any tool. Describe the image in your reply.

When in genuine doubt, prefer asking a one-line clarifying question over
guessing. But "photo of food + 'log this'" is unambiguously lane 1 — go
straight to log_nutrition.

### Multi-photo batch flows (smoothies, batch cooking)

If the user is photographing multiple ingredients across several turns
and saying things like "this goes in the smoothie", "starting ingredients",
"that's all of them, log it" — accumulate the macros mentally across
turns and only call log_nutrition when the user signals they're done.
One log entry for the finished item (the smoothie / the batch portion),
not one per ingredient.

If they batch cook three meals' worth, ask how they want to log it
(one entry today, or split equally across the next three days) and
log accordingly with separate log_nutrition calls.

---

## File-tool guidance (lane 2 only)

The instructions below apply when you've decided this is a document /
dashboard operation. Do NOT apply them to lane 1 (lifestyle) — lifestyle
data goes through log_nutrition / log_workout, not write_file.

### Where the user keeps their files — read this carefully

The user's real working files live under **OneDrive\Documents**, not the
local Windows Documents folder. The local `C:\Users\dizzi\Documents`
contains only Windows defaults (My Music, My Pictures, My Videos) and is
NOT where the user organises their work.

Canonical paths:
  * Documents (work, finance, projects, etc.): C:/Users/dizzi/OneDrive/Documents/
  * Pictures and screenshots:                  C:/Users/dizzi/OneDrive/Pictures/
  * Desktop items:                             C:/Users/dizzi/OneDrive/Desktop/

NEVER write user-facing content to C:/Users/dizzi/Documents/ directly.
Always use the OneDrive-synced equivalent. The Drive indexer and file
watcher only see OneDrive paths, so anything outside OneDrive is invisible
to ASTRA on subsequent turns.

### Before you write — discover existing files

Conversation history can be incomplete (the user may have logged out, started
a new session, or be referring to a file built in an earlier project). Do not
assume "no mention in chat = file doesn't exist." Before creating a new file:

  1. Use list_dir on the relevant folder (e.g. C:/Users/dizzi/OneDrive/Documents/Work/)
     to see what's actually on disk.
  2. If a file with the obvious name already exists (dashboard.html, log.html,
     etc.), read it first with read_file, then update it with edit_file rather
     than overwriting via write_file.
  3. Only create a new file if list_dir confirms nothing matches.

This makes the filesystem the source of truth, so the system stays correct
even when conversation memory is missing.

### Structured data for future re-reads

When you write data into a file the user will refer to later, shape it so
a future LLM turn can parse values out without reading prose:

  * In HTML, attach data-attributes to the row/section that carries each
    value (data-date, data-mileage, data-fuel-cost, data-hours-worked).
  * For heavily structured datasets on a large page, write a JSON sidecar
    (e.g. dashboard.json) and have the HTML reference it; future turns
    can read the JSON directly.

### Honesty about changes

Do not silently rewrite a file. State what you changed: "I added today's
entry — 9-hour shift, 120 miles, £281.40 gross — and updated the weekly total."
If you create a new file because list_dir came back empty, say so:
"No existing dashboard found at <path>, so I created a new one."

### Don't over-create

If the user has only sent you an image with no instruction to log or update
anything, just describe the image and answer their question. Do not create
files speculatively.
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def stream_image_chat(
    req: Any,
    db: Session,
) -> AsyncGenerator[str, None]:
    """SSE generator for image-bearing chat turns.

    Yields data: <json>\\n\\n events matching the format the frontend
    already consumes from /stream/chat (type=metadata, type=token,
    type=tool_start, type=tool_result, type=done, type=error).
    """
    # Lazy imports to keep this module light on cold start and to avoid
    # circular import risk with chat_routing / prompt_builders.
    from app.memory.service import get_project
    from app.llm.routing.prompt_builders import (
        build_system_prompt,
        build_full_context,
        build_messages,
    )
    from app.debug.gemini_tool_loop import stream_gemini_tool_loop

    project = get_project(db, req.project_id)
    if not project:
        yield _sse({"type": "error", "error": f"Project {req.project_id} not found"})
        yield "data: [DONE]\n\n"
        return

    # ---- Build the same context blocks normal chat would use ----
    try:
        full_context = build_full_context(
            db=db,
            project_id=req.project_id,
            message=req.message,
            use_semantic_search=getattr(req, "use_semantic_search", True),
        )
        system_prompt = build_system_prompt(
            project=project,
            full_context=full_context,
            ui_context=getattr(req, "ui_context", None),
        )
    except Exception as e:
        logger.exception("[image_chat] context build failed: %s", e)
        system_prompt = f"Project: {project.name}."

    system_prompt += _IMAGE_TURN_INSTRUCTIONS

    # ---- Build conversation history ----
    try:
        messages = build_messages(
            message=req.message,
            project_id=req.project_id,
            db=db,
            include_history=getattr(req, "include_history", True),
            history_limit=getattr(req, "history_limit", 20),
        )
    except Exception as e:
        logger.exception("[image_chat] history build failed: %s", e)
        messages = [{"role": "user", "content": req.message}]

    # ---- Build content_parts: image FileData + user text ----
    file_uri = getattr(req, "file_upload_uri", None)
    file_mime = getattr(req, "file_upload_mime", None) or "image/jpeg"
    file_name = getattr(req, "file_upload_name", None) or "attached image"

    if not file_uri:
        # Shouldn't reach here (has_image_attachment guards) but fail loud.
        yield _sse({
            "type": "error",
            "error": "image_chat reached without file_upload_uri",
        })
        yield "data: [DONE]\n\n"
        return

    content_parts = [
        _build_file_data_part(file_uri, file_mime),
        f"[Image attached: {file_name}]\n{req.message}",
    ]

    logger.info(
        "[image_chat] Routing to Gemini multimodal+tools: "
        "project=%s, uri=%s, mime=%s, name=%s",
        req.project_id, file_uri, file_mime, file_name,
    )

    # ---- Emit metadata so the frontend shows the right model badge ----
    yield _sse({
        "type": "metadata",
        "provider": "google",
        "model": IMAGE_TURN_MODEL,
    })

    # ---- Stream Gemini tool loop, translating events to our SSE shape ----
    total_chars = 0
    try:
        async for event in stream_gemini_tool_loop(
            system_prompt=system_prompt,
            messages=messages,
            model_id=IMAGE_TURN_MODEL,
            content_parts=content_parts,
        ):
            etype = event.get("type")

            if etype == "final_text":
                content = event.get("content", "")
                total_chars += len(content)
                yield _sse({"type": "token", "content": content})

            elif etype == "thinking":
                # Surface as token too — frontend renders these inline today.
                content = event.get("content", "")
                yield _sse({"type": "token", "content": content})

            elif etype == "tool_start":
                yield _sse({
                    "type": "tool_start",
                    "tool": event.get("tool", ""),
                    "summary": event.get("summary", ""),
                })

            elif etype == "tool_result":
                yield _sse({
                    "type": "tool_result",
                    "tool": event.get("tool", ""),
                    "summary": event.get("summary", ""),
                })

            elif etype == "error":
                yield _sse({"type": "error", "error": event.get("error", "unknown")})
                break

    except Exception as e:
        logger.exception("[image_chat] Gemini tool loop raised: %s", e)
        yield _sse({"type": "error", "error": str(e)})

    yield _sse({
        "type": "done",
        "provider": "google",
        "model": IMAGE_TURN_MODEL,
        "total_length": total_chars,
    })
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Small helper
# ---------------------------------------------------------------------------

def _sse(payload: dict) -> str:
    """Format a single SSE event line."""
    return "data: " + json.dumps(payload) + "\n\n"
