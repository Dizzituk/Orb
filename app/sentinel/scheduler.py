# FILE: app/sentinel/scheduler.py
# Purpose: Sentinel background loop — collect every 30s, daily retention prune (30 days)
#          and baseline maintenance. Mirrors app/lifestyle/scheduler.py (asyncio pattern).
# Called-by: main (startup event)
# Depends-on: app.sentinel.collector, app.sentinel.baseline, app.sentinel.models, app.db
# Last-renovated: 2026-07-01
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_STATE_FILE = Path(__file__).resolve().parents[2] / "data" / "sentinel" / "scheduler_state.json"
_POLL_SECONDS = int(os.getenv("ASTRA_SENTINEL_POLL_SECONDS", "30"))
RETENTION_DAYS = 30          # connection events
ALERT_RETENTION_DAYS = 90    # acknowledged + suppressed alerts
_PRUNE_AT_HOUR = 3 + 25 / 60.0        # 03:25 local
_MAINTENANCE_AT_HOUR = 3 + 40 / 60.0  # 03:40 local


def _enabled() -> bool:
    return os.getenv("ASTRA_SENTINEL_ENABLED", "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _boot_prime_enabled() -> bool:
    """P2-1 kill-switch — off falls back to the old behaviour (first tick fires
    rules normally, so a restart's connection burst is diffed like anything else)."""
    return os.getenv("ASTRA_SENTINEL_BOOT_PRIME_ENABLED", "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


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
        logger.debug("[sentinel_scheduler] state save failed: %s", exc)


def _run_prune() -> None:
    """Rolling retention: connection events past 30 days, settled alerts past 90."""
    from app.db import SessionLocal
    from app.sentinel.models import SentinelAlert, SentinelConnection

    db = SessionLocal()
    try:
        conn_cutoff = datetime.utcnow() - timedelta(days=RETENTION_DAYS)
        removed_conns = (
            db.query(SentinelConnection)
            .filter(SentinelConnection.ts < conn_cutoff)
            .delete(synchronize_session=False)
        )
        alert_cutoff = datetime.utcnow() - timedelta(days=ALERT_RETENTION_DAYS)
        removed_alerts = (
            db.query(SentinelAlert)
            .filter(
                SentinelAlert.created_at < alert_cutoff,
                (SentinelAlert.acknowledged.is_(True)) | (SentinelAlert.suppressed.is_(True)),
            )
            .delete(synchronize_session=False)
        )
        db.commit()
        if removed_conns or removed_alerts:
            logger.info(
                "[sentinel_scheduler] prune: %s connection events, %s settled alerts",
                removed_conns, removed_alerts,
            )
    except Exception:
        logger.exception("[sentinel_scheduler] prune failed")
    finally:
        db.close()


def _run_maintenance() -> None:
    from app.db import SessionLocal
    from app.sentinel import baseline

    db = SessionLocal()
    try:
        baseline.maintenance(db)
    except Exception:
        logger.exception("[sentinel_scheduler] baseline maintenance failed")
    finally:
        db.close()


class SentinelScheduler:
    def __init__(self) -> None:
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._state = _load_state()
        self._primed = False

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
        from app.sentinel import collector

        await collector.collect_once()

        now = datetime.now().astimezone()
        if self._due("prune", _PRUNE_AT_HOUR, now):
            self._mark("prune", _PRUNE_AT_HOUR, now)
            await asyncio.to_thread(_run_prune)
        if self._due("maintenance", _MAINTENANCE_AT_HOUR, now):
            self._mark("maintenance", _MAINTENANCE_AT_HOUR, now)
            await asyncio.to_thread(_run_maintenance)

    async def _loop(self) -> None:
        if not self._primed:
            self._primed = True
            if _boot_prime_enabled():
                from app.sentinel import collector
                try:
                    await collector.prime_once()
                except Exception:
                    logger.exception("[sentinel_scheduler] boot prime failed")
        while self._running:
            try:
                await self._tick()
                await asyncio.sleep(_POLL_SECONDS)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("[sentinel_scheduler] loop error: %s", exc)
                await asyncio.sleep(max(60, _POLL_SECONDS))


_scheduler: Optional[SentinelScheduler] = None


def start_sentinel_scheduler_background(
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> bool:
    """Start the loop from main.py's sync startup hook (lifestyle pattern)."""
    global _scheduler
    if not _enabled():
        logger.info("[sentinel_scheduler] disabled via ASTRA_SENTINEL_ENABLED")
        return False
    if _scheduler is None:
        _scheduler = SentinelScheduler()
    if _scheduler._running:
        logger.warning("[sentinel_scheduler] already running")
        return False
    if loop is None:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            logger.warning("[sentinel_scheduler] no event loop")
            return False
    _scheduler._running = True
    _scheduler._task = loop.create_task(_scheduler._loop())
    logger.info("[sentinel_scheduler] background task created (poll=%ss)", _POLL_SECONDS)
    return True
