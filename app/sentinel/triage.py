# FILE: app/sentinel/triage.py
# Purpose: LLM interpretation of rule-flagged anomalies ONLY (never the raw stream) —
#          plain-English explanation + recommended_action + confidence, cheap/fast tier.
# Called-by: app.sentinel.collector
# Depends-on: app.llm.clients (async_call_*), app.llm.frontier_models
# Last-renovated: 2026-06-12
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_VALID_ACTIONS = ("ignore", "watch", "propose_block")

_SYSTEM_PROMPT = (
    "You are ASTRA's embedded network security analyst on Taz's Windows 11 PC. "
    "A deterministic rule flagged one network anomaly. Interpret it for a smart "
    "non-specialist: what is this connection most likely to be, and how worried "
    "should Taz be? Be concrete (name the vendor/service if the org/rDNS makes it "
    "obvious), never alarmist, and never recommend blocking unless the evidence "
    "genuinely points that way. Blocking is only ever PROPOSED — Taz decides.\n\n"
    "Reply with STRICT JSON only, no markdown fences:\n"
    '{"explanation": "<2-4 plain-English sentences>", '
    '"recommended_action": "ignore|watch|propose_block", '
    '"confidence": <0.0-1.0>}'
)


def _parse_response(content: str) -> Optional[Dict[str, Any]]:
    if not content:
        return None
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not data.get("explanation"):
        return None
    action = str(data.get("recommended_action") or "watch").strip().lower()
    if action not in _VALID_ACTIONS:
        action = "watch"
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence"))))
    except (TypeError, ValueError):
        confidence = None
    return {
        "explanation": str(data["explanation"]).strip(),
        "recommended_action": action,
        "confidence": confidence,
    }


async def triage_finding(finding: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Interpret one rule finding. Always returns a usable dict — on any LLM
    failure it falls back to the rule's own detail text with action 'watch'.
    Cost is tracked by the provider registry like every other call."""
    fallback = {
        "explanation": str(finding.get("detail") or finding.get("title") or ""),
        "recommended_action": "watch",
        "confidence": None,
    }
    provider = os.getenv("ASTRA_SENTINEL_TRIAGE_PROVIDER", "google").strip().lower()
    try:
        from app.llm.frontier_models import resolve_model_alias

        model = resolve_model_alias(
            os.getenv("ASTRA_SENTINEL_TRIAGE_MODEL", "google:frontier-flash")
        )
        payload = json.dumps(
            {
                "finding": {k: finding.get(k) for k in
                            ("rule_key", "severity", "title", "detail", "process", "remote")},
                "context": context,
            },
            ensure_ascii=False,
            default=str,
        )
        messages = [{"role": "user", "content": payload}]

        from app.llm import clients

        caller = {
            "google": clients.async_call_google,
            "anthropic": clients.async_call_anthropic,
            "openai": clients.async_call_openai,
        }.get(provider, clients.async_call_google)
        content, usage = await caller(_SYSTEM_PROMPT, messages, model=model)
        if usage.get("error"):
            logger.warning("[sentinel] triage LLM error: %s", usage.get("error"))
            return fallback
        parsed = _parse_response(content)
        if not parsed:
            logger.warning("[sentinel] triage response unparseable — using rule text")
            return fallback
        return parsed
    except Exception as exc:
        logger.warning("[sentinel] triage failed (%s) — using rule text", exc)
        return fallback
