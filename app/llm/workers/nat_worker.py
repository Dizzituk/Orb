# FILE: app/llm/workers/nat_worker.py
# Purpose: THE generic entry point for giving the local worker "Nat" a job.
# Called-by: app.memory.nat_jobs.* (now); the 26B orchestrator (later) — same door.
# Depends-on: app.llm.streaming.call_llm_text, app.llm.nat_tool_loop (tools path only)
# Last-renovated: 2026-06-19
"""
Generic worker interface for "Nat" (and, later, any local-fleet model).

Design rule (load-bearing): this module is the SINGLE door both the memory
jobs and the future 26B orchestrator use to give Nat work. It is deliberately
memory-agnostic — it routes by the NAT_PROVIDER / NAT_MODEL env strings through
the standard call_llm_text path. When the orchestrator arrives it becomes
another caller of run_nat_job with ZERO changes here, and swapping Nat's
backend (Ollama -> vLLM -> a 3090 box) is an env change, never a code change.

Failure-proof: any error/timeout returns an empty result (str "" for text
jobs, {} for tool jobs) and is logged at debug. A worker call must NEVER raise
into a caller — a memory job (or a chat reply) can always proceed without Nat.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def _nat_provider() -> str:
    return os.getenv("NAT_PROVIDER", "vllm_local").strip() or "vllm_local"


def _nat_model() -> Optional[str]:
    return (os.getenv("NAT_MODEL", "").strip() or None)


async def run_nat_job(
    system_prompt: str,
    user_prompt: str,
    *,
    tools: Optional[List[Dict]] = None,
    enable_thinking: bool = False,
    max_tokens: int = 512,
    timeout_seconds: int = 15,
) -> Union[str, Dict[str, Any]]:
    """Give Nat one bounded job and get its result.

    Args:
        system_prompt: the job framing (what to do, output shape).
        user_prompt:   the input payload.
        tools:         if provided, Nat runs the agentic tool-execution loop
                       (returns a dict) instead of plain text (returns a str).
        enable_thinking: turn on Nat's native step-by-step reasoning. Default
                       OFF — the memory jobs need millisecond, direct answers.
        max_tokens:    output cap.
        timeout_seconds: per-call timeout hint.

    Returns:
        str  for text jobs (the trimmed reply, or "" on any failure).
        dict for tool jobs: {"text","tool_calls","rounds","stopped"} (or {}).
    """
    provider = _nat_provider()
    model = _nat_model()

    try:
        if tools:
            from app.llm.nat_tool_loop import run_nat_tool_loop
            return await run_nat_tool_loop(
                system_prompt,
                user_prompt,
                tools=tools,
                enable_thinking=enable_thinking,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                provider=provider,
                model=model,
            )

        from app.llm.streaming import call_llm_text
        text = await call_llm_text(
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            enable_reasoning=enable_thinking,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
        return (text or "").strip()
    except Exception as exc:  # noqa: BLE001 — workers never raise into callers
        logger.debug("[nat_worker] run_nat_job failed (provider=%s model=%s): %s",
                     provider, model, exc)
        return {} if tools else ""


# ── Generic JSON coercion (small models wrap JSON in fences/prose) ────────
# Lives here (not in any memory module) because it is generic to ALL small-
# model callers, not memory-specific.

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def coerce_json(text: str) -> Optional[Any]:
    """Best-effort parse of a JSON value (array or object) out of a model
    reply that may be fenced or wrapped in prose. Returns the parsed value,
    or None if nothing parseable is found. Never raises."""
    if not text:
        return None
    s = text.strip()
    # 1) straight parse
    try:
        return json.loads(s)
    except Exception:
        pass
    # 2) fenced block
    m = _FENCE_RE.search(s)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    # 3) first balanced array or object substring
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = s.find(open_ch)
        end = s.rfind(close_ch)
        if 0 <= start < end:
            try:
                return json.loads(s[start:end + 1])
            except Exception:
                continue
    return None
