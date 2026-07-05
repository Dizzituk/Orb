# FILE: app/content/distribution/posting_drivers/self_heal.py
# Purpose: Selector self-heal — relocate a broken step via the vision/architect model.
# Called-by: app.content.distribution.posting_drivers.meta_driver, tests.test_posting_self_heal
# Depends-on: app.llm.gemini_vision (ask_about_image, injectable), app.web_automation.bridge (injectable)
# Last-renovated: 2026-07-02
"""
Self-heal — when every candidate for a step is dead, this is the ONE
place a big model enters the posting flow (jobspec Job 5).

relocate(): screenshot + accessibility tree -> vision/architect model ->
a fresh CSS selector for the target. driver_runner prepends that
selector to the live map (so it becomes the new first candidate and is
persisted), retries the step once, and if it works the map has healed
itself for next time.

The model here READS the page to propose a selector; it never points a
cursor. Coordinates are never taken from vision. Model/tier come from
env (ASTRA_POSTING_HEAL_TIER), never a hardcoded model string. The
vision call is injectable so tests run with no model and no browser.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, List, Optional

logger = logging.getLogger(__name__)

# vision_fn(image_b64, goal_text, tree_text) -> model answer text
VisionFn = Callable[[str, str, str], Awaitable[str]]

_HEAL_PROMPT = (
    "You are repairing a broken web-automation selector for {surface}. "
    "The target element is: {goal}.\n\n"
    "Below is the page's accessibility tree (interactive elements with role, "
    "aria-label 'name', visible text and coordinates). Using it and the "
    "screenshot, return ONE robust CSS selector that uniquely targets that "
    "element.\n"
    "Rules: prefer [aria-label=...], role attributes, stable text-bearing "
    "attributes. NEVER use obfuscated/hashed class names (they change every "
    "deploy). If nothing matches, reply exactly NONE.\n"
    "Reply with ONLY the selector (or NONE) — no prose, no backticks.\n\n"
    "Accessibility tree:\n{tree}"
)


def _compact_tree(elements: List[dict], limit: int = 120) -> str:
    """Trim the accessibility tree to what the model needs to pick a selector."""
    rows = []
    for el in (elements or [])[:limit]:
        rows.append({
            "tag": el.get("tag"),
            "role": el.get("role"),
            "name": el.get("name"),
            "text": (el.get("text") or "")[:60],
        })
    return json.dumps(rows, ensure_ascii=False)


def _parse_selector(answer: str) -> Optional[str]:
    """Pull a single CSS selector out of the model's reply, or None."""
    if not answer:
        return None
    text = answer.strip()
    if text.upper().startswith("NONE") or text.upper() == "NONE":
        return None
    # First non-empty line, stripped of backticks/quotes/labels.
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    line = line.strip("`'\" ")
    line = re.sub(r"^(selector|css)\s*[:=]\s*", "", line, flags=re.I).strip("`'\" ")
    if not line or line.upper() == "NONE":
        return None
    # Sanity: a usable selector has a name/attr/id/class token, no sentence.
    if len(line) > 300 or line.count(" ") > 8:
        return None
    if not re.search(r"[\[\].#a-zA-Z]", line):
        return None
    return line


# Default surface wording, kept for backward compatibility with the
# original (meta-only) call sites and tests.
DEFAULT_SURFACE = "the Meta Business Suite composer"


async def _default_vision_fn(
    image_b64: str, goal_text: str, tree_text: str,
    surface: str = DEFAULT_SURFACE,
) -> str:
    """Real relocate brain: the configured vision model (tier from env)."""
    from app.llm.gemini_vision import ask_about_image

    tier = os.getenv("ASTRA_POSTING_HEAL_TIER", "complex")
    prompt = _HEAL_PROMPT.format(surface=surface, goal=goal_text, tree=tree_text[:8000])
    try:
        image_bytes = base64.b64decode(image_b64) if image_b64 else b""
    except Exception:
        image_bytes = b""
    if not image_bytes:
        # No screenshot — fall back to a tree-only text ask via the same model.
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: ask_about_image(
                image_source=b"", user_question=prompt, mime_type="image/png", tier=tier
            ),
        )
        return result.get("answer", "") if isinstance(result, dict) else ""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: ask_about_image(
            image_source=image_bytes, user_question=prompt, mime_type="image/png", tier=tier
        ),
    )
    return result.get("answer", "") if isinstance(result, dict) else ""


async def relocate(
    session_id: str,
    step_name: str,
    goal_text: str,
    dom_elements: List[dict],
    *,
    bridge: Any,
    vision_fn: Optional[VisionFn] = None,
    surface: str = DEFAULT_SURFACE,
) -> Optional[str]:
    """Return a fresh CSS selector for a broken step, or None.

    `surface` names the UI being repaired in the heal prompt (e.g.
    "the Coursera learner interface") so non-Meta drivers reuse this
    without lying to the model. The model proposes a CSS selector from
    the tree + screenshot; it never returns coordinates.
    """
    b64 = ""
    try:
        shot = await bridge.execute_action(
            session_id, "screenshot", {"full_page": False}, timeout_seconds=20.0
        )
        b64 = (shot.get("result") or {}).get("image_base64") or ""
    except Exception as e:
        logger.warning("[self_heal] screenshot failed (%s); relocating from tree only", e)
    tree = _compact_tree(dom_elements)
    try:
        if vision_fn is None:
            answer = await _default_vision_fn(b64, goal_text, tree, surface=surface)
        else:
            answer = await vision_fn(b64, goal_text, tree)
    except Exception as e:
        logger.warning("[self_heal] vision relocate raised: %s", e)
        return None
    css = _parse_selector(answer)
    logger.info("[self_heal] step=%s goal=%r -> %s", step_name, goal_text, css or "NONE")
    return css


async def dump_recon(
    session_id: str,
    platform: str,
    step_name: str,
    *,
    bridge: Any,
    audit_dir: str,
) -> str:
    """Save a full recon dump (tree + screenshot) when a step is unrecoverable."""
    Path(audit_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dom, shot_b64 = {}, ""
    try:
        r = await bridge.execute_action(session_id, "dom_snapshot", {}, timeout_seconds=20.0)
        dom = r.get("result") or {}
    except Exception:
        pass
    try:
        s = await bridge.execute_action(
            session_id, "screenshot", {"full_page": False}, timeout_seconds=20.0
        )
        shot_b64 = (s.get("result") or {}).get("image_base64") or ""
    except Exception:
        pass

    tree_path = Path(audit_dir) / f"heal-fail-{platform}-{step_name}-{stamp}.json"
    tree_path.write_text(json.dumps(dom, indent=2, ensure_ascii=False), encoding="utf-8")
    if shot_b64:
        try:
            (Path(audit_dir) / f"heal-fail-{platform}-{step_name}-{stamp}.png").write_bytes(
                base64.b64decode(shot_b64)
            )
        except Exception:
            pass
    logger.warning("[self_heal] unrecoverable step %s — recon dumped to %s", step_name, tree_path)
    return str(tree_path)
