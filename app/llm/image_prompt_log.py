# FILE: app/llm/image_prompt_log.py
# Purpose: Persistent log of synthesised image prompts.
# Called-by: app.llm.image_extractor, app.llm.image_router
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Persistent log of synthesised image prompts.

Stores prompts alongside generated image filenames so that follow-up
"recreate that image" / "make it darker" requests can thread the
previous prompt back into the synth's refinement path.

Storage: D:\\Orb\\output\\images\\.prompt_log.json
Format:  list of {timestamp, project_id, filename, prompt} entries
Capped at MAX_ENTRIES (rolling) to prevent unbounded growth.

v1.0 (2026-04-25): Initial implementation. Single-purpose sidecar
                    for the image refinement loop.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Resolve relative to D:\Orb so this works whether or not the process
# was launched from the project root.
_LOG_PATH = Path(
    os.getenv("ORB_IMAGE_PROMPT_LOG", "D:/Orb/output/images/.prompt_log.json")
)
_MAX_ENTRIES = 200


def _load() -> list[dict]:
    if not _LOG_PATH.exists():
        return []
    try:
        with _LOG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("[image_prompt_log] Read failed: %s", e)
        return []


def _save(entries: list[dict]) -> None:
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Trim to most recent MAX_ENTRIES before writing
        trimmed = entries[-_MAX_ENTRIES:]
        with _LOG_PATH.open("w", encoding="utf-8") as f:
            json.dump(trimmed, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.warning("[image_prompt_log] Write failed: %s", e)


def save_prompt(project_id: int, filename: str, synth_prompt: str) -> None:
    """Record a synthesised prompt for a generated image.

    Silently no-ops on empty prompt or write failure — this is a
    nice-to-have for refinement context, not a critical path.
    """
    if not synth_prompt or not filename:
        return
    entries = _load()
    entries.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project_id": int(project_id),
        "filename": filename,
        "prompt": synth_prompt,
    })
    _save(entries)
    logger.info(
        "[image_prompt_log] Saved prompt for %s (project=%d, total=%d)",
        filename, project_id, len(entries),
    )


def get_last_prompt_for_project(project_id: int) -> Optional[str]:
    """Return the most recent synth prompt for this project, if any.

    Used by image_router to thread previous_image_prompt into the
    synth when the user's request looks like a refinement.
    """
    entries = _load()
    for entry in reversed(entries):
        if entry.get("project_id") == int(project_id):
            prompt = entry.get("prompt")
            if prompt:
                return prompt
    return None


__all__ = ["save_prompt", "get_last_prompt_for_project"]
