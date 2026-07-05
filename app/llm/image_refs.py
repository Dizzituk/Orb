# FILE: app/llm/image_refs.py
# Purpose: Derek vision upgrade — image refs persist through chat history and get re-analysed at spec time.
# Called-by: app.llm.routing.chat_request_prep, app.llm.weaver_stream, app.pot_spec.grounded.spec_runner
# Depends-on: app.llm.stage_roles, app.llm.gemini_vision, app.pipeline_v2.ledger
# Last-renovated: 2026-07-04
"""
Pixels now travel (2026-07-04, Taz directive).

Before this module, a screenshot's only trace in the pipeline was whatever
the chat model happened to SAY about it — the image itself never reached
the Weaver or SpecGate. Now:

  1. chat persist appends a machine-readable marker to the stored message:
         [image_ref: <local_path> | <original name>]
     (uploads live durably in data/debug_uploads/).
  2. The Weaver extracts markers from the conversation deterministically
     and stores the paths in flow state (weaver_image_refs).
  3. SpecGate re-analyses the ORIGINAL images with the CHECKOUT_EYES vision
     tier at spec time — spec-focused, not chat-focused — and the evidence
     is appended to the job description AND recorded in the Decision Ledger.

Env:
    ASTRA_SPECGATE_IMAGE_ANALYSIS  default "1" (on — only fires when refs exist)
    ASTRA_SPECGATE_IMAGE_MAX       default 6 images per spec run
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

IMAGE_REF_RE = re.compile(r"\[image_ref:\s*([^|\]]+?)\s*(?:\|\s*([^\]]*?)\s*)?\]")

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")


def analysis_enabled() -> bool:
    return os.getenv("ASTRA_SPECGATE_IMAGE_ANALYSIS", "1").strip().lower() in ("1", "true", "yes")


def _max_images() -> int:
    try:
        return max(0, int(os.getenv("ASTRA_SPECGATE_IMAGE_MAX", "6")))
    except ValueError:
        return 6


def image_ref_marker(local_path: str, name: str = "") -> str:
    """The marker appended to a persisted chat message for an uploaded image."""
    return f"[image_ref: {local_path} | {name or os.path.basename(local_path)}]"


def extract_image_refs(messages: List[Dict[str, Any]]) -> List[str]:
    """Deterministically collect image-ref paths from conversation messages.

    Only EXISTING image files count (uploads can be pruned); deduped in
    order of appearance; capped at ASTRA_SPECGATE_IMAGE_MAX (newest kept —
    the most recent screenshots are usually the operative ones).
    """
    refs: List[str] = []
    seen = set()
    for msg in messages:
        for m in IMAGE_REF_RE.finditer(str(msg.get("content", "") or "")):
            path = m.group(1).strip()
            key = path.replace("\\", "/").lower()
            if key in seen:
                continue
            if not key.endswith(_IMAGE_EXTS):
                continue
            if not os.path.isfile(path):
                logger.debug("[image_refs] Skipping missing image: %s", path)
                continue
            seen.add(key)
            refs.append(path)
    cap = _max_images()
    return refs[-cap:] if cap else []


async def analyse_image_for_spec(path: str, goal: str) -> Optional[str]:
    """One spec-focused vision pass over the ORIGINAL image (CHECKOUT_EYES
    tier). None on any failure — evidence is additive, never blocking."""
    try:
        from app.llm.stage_roles import resolve_stage_role
        role = resolve_stage_role("CHECKOUT_EYES")
        prompt = (
            "You are gathering BUILD EVIDENCE from a user-supplied image for "
            f"this job: {goal[:300]}. Extract every build-relevant detail: "
            "visible text VERBATIM, UI elements and their layout, colours, "
            "data values, error messages, dimensions if inferable. Facts only, "
            "no advice. Be exhaustive but structured."
        )
        if role.provider in ("google", "gemini"):
            from app.llm.gemini_vision import ask_about_image
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: ask_about_image(path, prompt),
            )
            text = (result or {}).get("answer") or (result or {}).get("text") or ""
            if not text:
                text = str(result)[:1200] if result else ""
            try:
                from app.cost.cost_recorder import record_llm_cost
                usage = (result or {}).get("usage") or {}
                record_llm_cost(
                    provider="google", model=role.model,
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    stage="specgate_vision",
                )
            except Exception:
                pass
            return str(text)[:3000] if text else None
        logger.info("[image_refs] No vision path for provider %s — image %s skipped",
                    role.provider, path)
    except Exception as exc:
        logger.warning("[image_refs] Vision analysis failed for %s: %s", path, exc)
    return None


def _record_to_ledger(job_id: str, path: str, analysis: str) -> None:
    """Land the image evidence in the job's Decision Ledger (best effort)."""
    if not job_id:
        return
    try:
        from app.pot_spec.grounded._spec_runner_utils_11 import _get_job_dir_for_segmentation
        from app.pipeline_v2.ledger import load_or_create_ledger, ledger_append, save_ledger

        job_dir = _get_job_dir_for_segmentation(job_id)
        os.makedirs(job_dir, exist_ok=True)
        ledger = load_or_create_ledger(job_id, job_dir)
        ledger_append(
            ledger, entry_type="file_read", stage="spec_gate",
            relevant_to=["spec_gate", "agentic_builder"],
            summary=f"SpecGate vision re-analysed user image {os.path.basename(path)}",
            path=path, category="specgate_vision",
            description=analysis[:4000],
        )
        save_ledger(ledger, job_dir)
    except Exception as exc:
        logger.warning("[image_refs] ledger write skipped for %s: %s", path, exc)


async def build_image_evidence(
    image_refs: List[str],
    goal: str,
    job_id: str = "",
    chat_vision_context: str = "",
) -> str:
    """Analyse the user's original images and build the evidence block that
    gets appended to the job description. Also finally CONSUMES the
    chat-time vision context (which previously dead-ended in
    constraints_hint). Empty string when nothing usable."""
    parts: List[str] = []

    if analysis_enabled():
        for path in image_refs:
            analysis = await analyse_image_for_spec(path, goal)
            if analysis:
                parts.append(f"### Image: {os.path.basename(path)}\n{analysis}")
                _record_to_ledger(job_id, path, analysis)
                logger.info("[image_refs] Image evidence gathered: %s (%d chars)",
                            path, len(analysis))
    elif image_refs:
        logger.info("[image_refs] %d image ref(s) present but ASTRA_SPECGATE_IMAGE_ANALYSIS=0",
                    len(image_refs))

    if chat_vision_context:
        parts.append(
            "### Chat-time screenshot analysis (as discussed in conversation)\n"
            + chat_vision_context[:6000]
        )

    if not parts:
        return ""
    return (
        "\n\n## IMAGE EVIDENCE (re-analysed from the user's uploaded images — "
        "treat as ground truth requirements/context)\n\n" + "\n\n".join(parts)
    )


__all__ = [
    "IMAGE_REF_RE", "image_ref_marker", "extract_image_refs",
    "analyse_image_for_spec", "build_image_evidence", "analysis_enabled",
]
