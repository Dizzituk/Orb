# FILE: scripts/derek/eyes_evidence_run.py
# Purpose: Derek phase 6 gate evidence — run the eyes on a deliberately-broken fixture, persist screenshots.
# Called-by: manual (D:\Orb\.venv\Scripts\python.exe scripts\derek\eyes_evidence_run.py)
# Depends-on: app.pipeline_v2.verifier_agent.eyes_judge
# Last-renovated: 2026-07-04
"""Persists eyes evidence (screenshots + crash capture) under docs/derek/evidence/."""
from __future__ import annotations

import asyncio
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).resolve().parents[2] / ".env")
os.environ["ASTRA_EYES_SHOTS"] = "2"

from app.pipeline_v2.verifier_agent import eyes_judge  # noqa: E402
from app.pipeline_v2.build_targets import BuildTargetProfile  # noqa: E402


async def _fake_desc(path, hint):
    return "(vision description deferred to live run - screenshot on disk)"


eyes_judge._describe_screenshot = _fake_desc

BROKEN = "print('booting toy app')\nraise RuntimeError('deliberately broken fixture')\n"


def main() -> int:
    root = pathlib.Path("docs/derek/evidence/broken-fixture-app")
    root.mkdir(parents=True, exist_ok=True)
    (root / "main.py").write_text(BROKEN, encoding="utf-8")
    profile = BuildTargetProfile(
        project_id="evidence-fixture", project_name="Evidence Fixture",
        project_root=str(root), language="python", build_system="pip",
        framework="cli", source_root="", package_name="",
        architecture_pattern="flat",
    )
    ev = asyncio.run(eyes_judge.run_eyes(profile, "a toy game", "docs/derek/evidence"))
    print(f"EYES EVIDENCE: launched={ev.launched} running={ev.still_running} "
          f"rc={ev.returncode} shots={len(ev.screenshots)}")
    if ev.stderr_tail:
        print("stderr tail:", ev.stderr_tail.splitlines()[-1])
    print("screenshots:", ev.screenshots)
    return 0 if (ev.launched and ev.returncode not in (None, 0) and ev.screenshots) else 1


if __name__ == "__main__":
    raise SystemExit(main())
