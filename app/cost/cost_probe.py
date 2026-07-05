# FILE: app/cost/cost_probe.py
# Purpose: Derek phase 1 gate probe — one cheap labelled call per stage, verified against the ledger.
# Called-by: app.endpoints.cost_dashboard (POST /api/cost/probe), scripts/derek/cost_probe.py
# Depends-on: app.llm._streaming_utils_3, app.cost.cost_ledger, app.cost.cost_recorder
# Last-renovated: 2026-07-04
"""
Fires ONE tiny real API call per Derek stage label through the instrumented
streaming choke point (call_llm_text -> stream_llm -> provider -> done event
-> record_llm_cost), plus a no-API null-cost check, then scans the ledger
and reports PASS/FAIL per label. Total cost: a fraction of a cent.

Providers whose API key is absent from the process env are SKIPped cleanly —
run inside the booted backend (POST /api/cost/probe) where the encrypted
key store has been synced into os.environ.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

PROBE_JOB_ID = "derek-p1-probe"

# (stage label, provider, key env that must exist, cheap model)
PROBE_STAGES = [
    ("weaver",          "openai",    "OPENAI_API_KEY",    "gpt-5.4-mini"),
    ("spec_gate",       "openai",    "OPENAI_API_KEY",    "gpt-5.4-mini"),
    ("specgate_agent",  "openai",    "OPENAI_API_KEY",    "gpt-5.4-mini"),
    ("builder_main",    "openai",    "OPENAI_API_KEY",    "gpt-5.4-mini"),
    ("builder_worker",  "google",    "GEMINI_API_KEY",    "gemini-2.5-flash-lite"),
    ("checkout_judge",  "anthropic", "ANTHROPIC_API_KEY", "claude-haiku-4-5-20251001"),
    ("checkout_eyes",   "google",    "GEMINI_API_KEY",    "gemini-2.5-flash-lite"),
]


def _ledger_rows_for_probe() -> List[Dict[str, Any]]:
    from app.cost.cost_ledger import LEDGER_DIR, LEDGER_FILENAME
    path = Path(LEDGER_DIR) / LEDGER_FILENAME
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("job_id") == PROBE_JOB_ID:
                rows.append(r)
    return rows


async def run_probe() -> Dict[str, Any]:
    """Run the full probe. Returns a verdict dict (green, results, rows)."""
    from app.llm._streaming_utils_3 import call_llm_text
    from app.cost.cost_recorder import record_llm_cost

    results: Dict[str, str] = {}
    for stage, provider, key_env, model in PROBE_STAGES:
        keys = [key_env] + (["GOOGLE_API_KEY"] if key_env == "GEMINI_API_KEY" else [])
        if not any(os.getenv(k, "").strip() for k in keys):
            results[stage] = "SKIP (no key)"
            continue
        try:
            text = await call_llm_text(
                provider=provider,
                model=model,
                system_prompt="You are a test probe. Reply with exactly one word.",
                user_prompt="Say OK.",
                max_tokens=16,
                stage=stage,
                job_id=PROBE_JOB_ID,
            )
            results[stage] = f"call ok ({text[:20]!r})"
        except Exception as exc:
            results[stage] = f"CALL FAILED: {exc}"

    # Null-cost path needs no API: record an unknown model directly.
    record_llm_cost("local", "derek-probe-unknown-model", 123, 45,
                    stage="probe_null_cost", job_id=PROBE_JOB_ID)

    rows = _ledger_rows_for_probe()
    by_stage: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        by_stage[r["stage"]] = r  # last row per stage wins

    verdicts: Dict[str, str] = {}
    failures = 0
    for stage, provider, key_env, model in PROBE_STAGES:
        if results.get(stage, "").startswith("SKIP"):
            verdicts[stage] = "SKIP"
            continue
        r = by_stage.get(stage)
        if not r:
            verdicts[stage] = "FAIL: no ledger row"
            failures += 1
            continue
        cost = r.get("cost_usd")
        tokens_ok = (r.get("prompt_tokens") or 0) > 0
        if tokens_ok and (cost is None or cost > 0):
            verdicts[stage] = (
                f"PASS: model={r['model']} tokens={r['prompt_tokens']}+{r['completion_tokens']} cost={cost}"
            )
        else:
            verdicts[stage] = f"FAIL: row={r}"
            failures += 1

    nr = by_stage.get("probe_null_cost")
    if nr and nr.get("cost_usd") is None:
        verdicts["probe_null_cost"] = "PASS: unknown model recorded cost=null"
    else:
        verdicts["probe_null_cost"] = f"FAIL: {nr}"
        failures += 1

    return {
        "green": failures == 0,
        "failures": failures,
        "calls": results,
        "verdicts": verdicts,
        "ledger_rows": len(rows),
        "job_id": PROBE_JOB_ID,
    }


__all__ = ["run_probe", "PROBE_STAGES", "PROBE_JOB_ID"]
