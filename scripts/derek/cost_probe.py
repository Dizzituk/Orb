# FILE: scripts/derek/cost_probe.py
# Purpose: CLI wrapper for the Derek phase 1 cost probe (app/cost/cost_probe.py).
# Called-by: manual (D:\Orb\.venv\Scripts\python.exe scripts\derek\cost_probe.py)
# Depends-on: app.cost.cost_probe
# Last-renovated: 2026-07-04
"""
Offline runner for the phase-1 cost probe. NOTE: cloud API keys live in the
encrypted settings store and are only synced into the environment inside the
booted backend — from a bare shell every cloud stage SKIPs and only the
null-cost check runs. With the stack up, use the in-process endpoint instead:

    curl -X POST http://localhost:8000/api/cost/probe -H "Authorization: Bearer <token>"

(localhost-only; spends well under a cent across ~7 tiny calls).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


async def main() -> int:
    from app.cost.cost_probe import run_probe
    result = await run_probe()
    print("=== DEREK PHASE 1 COST PROBE ===")
    for stage, note in result["calls"].items():
        print(f"  {stage:<16} {note}")
    print(f"\nLedger rows for job_id={result['job_id']}: {result['ledger_rows']}")
    for stage, verdict in result["verdicts"].items():
        print(f"  {stage:<16} {verdict}")
    print(f"\n=== PROBE {'GREEN' if result['green'] else 'RED (' + str(result['failures']) + ' failures)'} ===")
    return 0 if result["green"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
