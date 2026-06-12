# FILE: app/lifestyle/scheduler.py
# Purpose: Lifestyle background scheduler — Jobs 4+5 of the memory roadmap (2026-06-10).
# Called-by: main
# Depends-on: app.lifestyle.habit_learner, app.lifestyle.nudges
# Last-renovated: 2026-06-11
"""
Lifestyle background scheduler — Jobs 4+5 of the memory roadmap (2026-06-10).

One asyncio loop, mirroring the decay_job scheduler pattern, driving:
  - Nudge evaluation at two checkpoints (default 13:30 and 19:15 local) —
    the "you've not eaten much / you're over, ease off" awareness.
  - Nightly habit learning (default 02:40 local) — staples, timing, shops
    into FoodPreference + Tier 2 mirror + hot-index bridge.

State (last run per task per day) persists to
data/lifestyle/scheduler_state.json so a restart never double-runs.

Env:
  ASTRA_LIFESTYLE_SCHEDULER_ENABLED  (default true)
  ASTRA_NUDGE_CHECK_TIMES            (default "13:30,19:15", local time)
  ASTRA_HABIT_LEARN_TIME             (default "02:40", local time)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_STATE_FILE = Path(__file__).resolve().parents[2] / "data" / "lifestyle" / "scheduler_state.json"
_POLL_SECONDS = 600  # check every 10 minutes


def _enabled() -> bool:
    return os.getenv("ASTRA_LIFESTYLE_SCHEDULER_ENABLED", "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _parse_times(raw: str) -> List[float]:
    out: List[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            hour_str, minute_str = part.split(":")
            out.append(int(hour_str) + int(minute_str) / 60.0)
        except Exception:
            logger.warning("[lifestyle_scheduler] bad time %r ignored", part)
    return out


def _load_state() -> Dict[str, str]:
    try:
        if _STATE_FILE.exists():
            return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_state(state: Dict[str, str]) -> None:
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.debug("[lifestyle_scheduler] state save failed: %s", exc)


class LifestyleScheduler:
    def __init__(self) -> None:
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._state = _load_state()

    def _due(self, task_key: str, at_hour: float, now: datetime) -> bool:
        """True if `now` is past today's slot and it hasn't run today."""
        today = now.strftime("%Y-%m-%d")
        slot_key = f"{task_key}@{at_hour:.2f}"
        if self._state.get(slot_key) == today:
            return False
        return (now.hour + now.minute / 60.0) >= at_hour

    def _mark(self, task_key: str, at_hour: float, now: datetime) -> None:
        self._state[f"{task_key}@{at_hour:.2f}"] = now.strftime("%Y-%m-%d")
        _save_state(self._state)

    async def _tick(self) -> None:
        now = datetime.now().astimezone()

        for slot in _parse_times(os.getenv("ASTRA_NUDGE_CHECK_TIMES", "13:30,19:15")):
            if self._due("nudges", slot, now):
                self._mark("nudges", slot, now)
                await asyncio.to_thread(self._run_nudges)

        for slot in _parse_times(os.getenv("ASTRA_HABIT_LEARN_TIME", "02:40")):
            if self._due("habits", slot, now):
                self._mark("habits", slot, now)
                await asyncio.to_thread(self._run_habits)

    @staticmethod
    def _run_nudges() -> None:
        try:
            from app.lifestyle.nudges import run_nudge_evaluation
            nudges = run_nudge_evaluation()
            logger.info("[lifestyle_scheduler] nudge check -> %d nudge(s)", len(nudges))
        except Exception as exc:
            logger.error("[lifestyle_scheduler] nudge run failed: %s", exc)

    @staticmethod
    def _run_habits() -> None:
        try:
            from app.lifestyle.habit_learner import run_habit_learning
            result = run_habit_learning()
            logger.info(
                "[lifestyle_scheduler] habit learning -> prefs=%s mirrored=%s indexed=%s",
                result.get("food_prefs"), result.get("tier2_mirrored"), result.get("hot_indexed"),
            )
        except Exception as exc:
            logger.error("[lifestyle_scheduler] habit learning failed: %s", exc)

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._tick()
                await asyncio.sleep(_POLL_SECONDS)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("[lifestyle_scheduler] loop error: %s", exc)
                await asyncio.sleep(300)


_scheduler: Optional[LifestyleScheduler] = None


def start_lifestyle_scheduler_background(loop: Optional[asyncio.AbstractEventLoop] = None) -> bool:
    """Start the scheduler on the given loop. Mirrors
    decay_job.start_decay_scheduler_background so root main.py's sync
    startup hook can call it without awaiting."""
    global _scheduler
    if not _enabled():
        logger.info("[lifestyle_scheduler] disabled via env")
        return False
    if _scheduler is None:
        _scheduler = LifestyleScheduler()
    if _scheduler._running:
        logger.warning("[lifestyle_scheduler] already running")
        return False
    if loop is None:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            logger.warning("[lifestyle_scheduler] no event loop")
            return False
    _scheduler._running = True
    _scheduler._task = loop.create_task(_scheduler._loop())
    logger.info("[lifestyle_scheduler] background task created (poll=%ss)", _POLL_SECONDS)
    return True
