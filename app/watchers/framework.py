# FILE: app/watchers/framework.py
# Purpose: Generic watcher engine — observe -> record -> summarise -> alert; idle-task + chat-tool auto-wiring.
# Called-by: app.tools.registry (register_watcher_tools), app.idle.router (import registers idle task)
# Depends-on: app.watchers.models, app.idle.router, app.sentinel.alerts, app.providers.registry
# Last-renovated: 2026-07-01
"""
One pattern, many instances. A WatcherSpec declares its keys (regions, item
classes) and an async observe_key() that returns a Reading for one key.
The framework provides everything else:

  - a daily idle-ledger task ("watcher_observe", params={"watcher_id"}),
    resumable per key — a user message checkpoints between keys
  - append-only recording into the watcher_readings ledger, idempotent
    per (watcher, key, day)
  - a threshold alert hook that files a LOW-severity sentinel alert
    (active-JSON nudge pattern — surfaces in conversation, never interrupts)
  - an auto-registered chat tool per watcher ("how are the land prices
    looking") with an optional on-demand trend summary on the LOCAL model
    (execution_context="background_local"; never per-scrape)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Awaitable, Callable, Dict, List, Optional

from sqlalchemy.orm import Session

from app.idle.router import RecurringSpec, TaskContext, TaskOutcome, register_task_handler
from app.watchers.models import WatcherReading

logger = logging.getLogger(__name__)

WATCHER_OBSERVE_TASK = "watcher_observe"


@dataclass
class Reading:
    key: str
    value: Optional[float]
    source: str = ""
    note: str = ""


@dataclass
class WatcherSpec:
    watcher_id: str
    title: str
    unit: str  # e.g. "eur_per_m2", "gbp"
    keys: List[dict]  # per-key config: {"key": ..., "label": ..., + instance fields}
    observe_key: Callable[[dict], Awaitable[Reading]]
    tool_name: str
    tool_description: str
    cadence_hours: float = 24.0
    alert_pct: Optional[float] = None  # None -> ASTRA_WATCHER_ALERT_PCT
    currency_symbol: str = "€"


_WATCHERS: Dict[str, WatcherSpec] = {}


def _alert_pct(spec: WatcherSpec) -> float:
    if spec.alert_pct is not None:
        return spec.alert_pct
    try:
        return float(os.getenv("ASTRA_WATCHER_ALERT_PCT", "10"))
    except Exception:
        return 10.0


def register_watcher(spec: WatcherSpec) -> None:
    _WATCHERS[spec.watcher_id] = spec
    register_task_handler(
        WATCHER_OBSERVE_TASK,
        _observe_handler,
        recurring=RecurringSpec(
            task_type=WATCHER_OBSERVE_TASK,
            params={"watcher_id": spec.watcher_id},
            cadence_hours=spec.cadence_hours,
        ),
    )
    logger.info("[watchers] registered %s (%d keys, every %.0fh)", spec.watcher_id, len(spec.keys), spec.cadence_hours)


def get_watcher(watcher_id: str) -> Optional[WatcherSpec]:
    return _WATCHERS.get(watcher_id)


def list_watchers() -> List[WatcherSpec]:
    return list(_WATCHERS.values())


# ── record ──────────────────────────────────────────────────────────────────


def _today() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def record_reading(db: Session, watcher_id: str, unit: str, reading: Reading, day: Optional[str] = None) -> bool:
    """Append one row; idempotent per (watcher, key, day). True if appended."""
    day = day or _today()
    exists = (
        db.query(WatcherReading.id)
        .filter(
            WatcherReading.watcher_id == watcher_id,
            WatcherReading.key == reading.key,
            WatcherReading.observed_date == day,
        )
        .first()
    )
    if exists is not None:
        return False
    db.add(
        WatcherReading(
            watcher_id=watcher_id,
            observed_date=day,
            key=reading.key,
            value=reading.value,
            unit=unit,
            source=(reading.source or "")[:512],
            note=reading.note or "",
        )
    )
    db.commit()
    return True


def read_series(db: Session, watcher_id: str, days: int = 30) -> List[WatcherReading]:
    """Newest-last series for a watcher, bounded to the last `days` distinct dates."""
    rows = (
        db.query(WatcherReading)
        .filter(WatcherReading.watcher_id == watcher_id)
        .order_by(WatcherReading.observed_date.desc(), WatcherReading.key.asc())
        .limit(max(1, min(int(days), 366)) * max(1, len(_WATCHERS.get(watcher_id).keys) if _WATCHERS.get(watcher_id) else 8))
        .all()
    )
    return list(reversed(rows))


def previous_value(db: Session, watcher_id: str, key: str, before_day: str) -> Optional[float]:
    row = (
        db.query(WatcherReading)
        .filter(
            WatcherReading.watcher_id == watcher_id,
            WatcherReading.key == key,
            WatcherReading.observed_date < before_day,
            WatcherReading.value.isnot(None),
        )
        .order_by(WatcherReading.observed_date.desc())
        .first()
    )
    return row.value if row is not None else None


# ── alert hook ──────────────────────────────────────────────────────────────


def _emit_alert(db: Session, spec: WatcherSpec, key: str, prev: float, new: float, pct: float) -> None:
    """Low-severity nudge via the sentinel active-JSON pattern — non-interrupting."""
    try:
        from app.sentinel.alerts import create_alert, refresh_active_file

        direction = "up" if new > prev else "down"
        create_alert(
            db,
            severity="low",
            rule_key=f"watcher_{spec.watcher_id}_{key}",
            title=f"{spec.title}: {key} moved {direction} {abs(pct):.1f}%",
            explanation=(
                f"{key} went {prev:g} -> {new:g} {spec.unit} ({pct:+.1f}%), beyond the "
                f"{_alert_pct(spec):.0f}% watch threshold."
            ),
            recommended_action="watch",
        )
        refresh_active_file(db)
    except Exception as exc:
        logger.warning("[watchers] alert emit failed for %s/%s: %s", spec.watcher_id, key, exc)


def check_move_and_alert(db: Session, spec: WatcherSpec, reading: Reading, day: str) -> Optional[float]:
    """Returns the pct move if it crossed the threshold (and emits the alert)."""
    if reading.value is None:
        return None
    prev = previous_value(db, spec.watcher_id, reading.key, day)
    if prev is None or prev == 0:
        return None
    pct = (reading.value - prev) / prev * 100.0
    if abs(pct) >= _alert_pct(spec):
        _emit_alert(db, spec, reading.key, prev, reading.value, pct)
        return pct
    return None


# ── the idle task (resumable per key) ───────────────────────────────────────


async def _observe_handler(ctx: TaskContext) -> TaskOutcome:
    watcher_id = str(ctx.params.get("watcher_id") or "")
    spec = _WATCHERS.get(watcher_id)
    if spec is None:
        return TaskOutcome.failed(f"unknown watcher_id {watcher_id!r}")

    progress = ctx.load_progress()
    keys_done = list(progress.get("keys_done") or [])
    day = progress.get("day") or _today()
    recorded = int(progress.get("recorded") or 0)
    flagged = int(progress.get("flagged") or 0)
    failures = int(progress.get("failures") or 0)

    for key_cfg in spec.keys:
        key = key_cfg.get("key")
        if key in keys_done:
            continue
        if ctx.should_yield():
            ctx.save_progress(
                {"keys_done": keys_done, "day": day, "recorded": recorded, "flagged": flagged, "failures": failures}
            )
            return TaskOutcome.paused(f"checkpointed before key '{key}'")
        try:
            reading = await spec.observe_key(key_cfg)
        except Exception as exc:
            logger.warning("[watchers] observe %s/%s failed: %s", watcher_id, key, exc)
            reading = Reading(key=key, value=None, note=f"observe failed: {exc}")
            failures += 1
        if record_reading(ctx.db, watcher_id, spec.unit, reading, day=day):
            recorded += 1
            if check_move_and_alert(ctx.db, spec, reading, day) is not None:
                flagged += 1
        keys_done.append(key)
        ctx.save_progress(
            {"keys_done": keys_done, "day": day, "recorded": recorded, "flagged": flagged, "failures": failures}
        )

    summary = f"{spec.title}: {recorded} reading(s) recorded for {day}"
    if flagged:
        summary += f", {flagged} notable move(s) flagged"
    if failures:
        summary += f", {failures} key(s) failed to observe"
    return TaskOutcome.completed(summary=summary, coverage=f"{len(keys_done)} keys ({day})")


# ── chat tool per watcher ───────────────────────────────────────────────────


def _series_payload(db: Session, spec: WatcherSpec, days: int) -> dict:
    rows = read_series(db, spec.watcher_id, days=days)
    by_key: Dict[str, list] = {}
    for r in rows:
        by_key.setdefault(r.key, []).append(
            {"date": r.observed_date, "value": r.value, "source": (r.source or "")[:120], "note": (r.note or "")[:160]}
        )
    latest = {}
    for key, series in by_key.items():
        vals = [p for p in series if p["value"] is not None]
        if vals:
            first, last = vals[0], vals[-1]
            change = None
            if first["value"]:
                change = round((last["value"] - first["value"]) / first["value"] * 100.0, 1)
            latest[key] = {"value": last["value"], "date": last["date"], "change_pct_over_window": change}
        else:
            latest[key] = {"value": None, "date": None, "change_pct_over_window": None}
    return {"unit": spec.unit, "latest": latest, "series": by_key}


# Public alias — the reports renderer builds charts/tables from the same payload.
def series_payload(db: Session, spec: WatcherSpec, days: int = 30) -> dict:
    return _series_payload(db, spec, days)


async def _summarise_local(spec: WatcherSpec, payload: dict) -> str:
    """On-demand trend summary on the LOCAL model only. Empty string on any failure."""
    try:
        import json as _json

        from app.providers.registry import llm_call

        result = await llm_call(
            provider_id=None,
            model_id="",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"You are summarising tracked price data: {spec.title} ({spec.unit}).\n"
                        f"Data (per key: latest + daily series):\n{_json.dumps(payload, ensure_ascii=False)[:4000]}\n\n"
                        "In 2-4 plain sentences: current levels, direction of travel, anything notable. "
                        "No preamble, no markdown."
                    ),
                }
            ],
            max_tokens=220,
            temperature=0.3,
            execution_context="background_local",
            stage="watcher_summary",
        )
        return (result.content or "").strip() if result.is_success() else ""
    except Exception as exc:
        logger.debug("[watchers] local summary unavailable: %s", exc)
        return ""


def _make_tool_handler(spec: WatcherSpec):
    async def handler(input_data: dict, context=None) -> dict:
        from app.db import get_db_session

        days = 30
        try:
            days = max(1, min(int(input_data.get("days", 30)), 365))
        except Exception:
            pass
        db = get_db_session()
        try:
            payload = _series_payload(db, spec, days)
        finally:
            db.close()
        out = {"ok": True, "watcher": spec.watcher_id, "title": spec.title, "days": days, **payload}
        if bool(input_data.get("summarise", False)):
            summary = await _summarise_local(spec, payload)
            if summary:
                out["trend_summary"] = summary
        return out

    return handler


def register_watcher_tools() -> None:
    """Called once from registry._register_defaults at module load."""
    from app.tools.registry import ToolDefinition, register_tool

    _ensure_instances_loaded()
    for spec in list_watchers():
        register_tool(ToolDefinition(
            name=spec.tool_name,
            version="v1",
            description=spec.tool_description,
            input_schema={
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "minimum": 1, "maximum": 365, "default": 30,
                             "description": "How many days of history to return."},
                    "summarise": {"type": "boolean", "default": False,
                                  "description": "Add a short local-model trend summary."},
                },
                "required": [],
            },
            output_schema={"type": "object"},
            handler=_make_tool_handler(spec),
        ))


def _ensure_instances_loaded() -> None:
    try:
        import app.watchers.portugal_land  # noqa: F401
    except Exception as exc:
        logger.warning("[watchers] portugal_land instance failed to load: %s", exc)
    try:
        import app.watchers.hardware  # noqa: F401
    except Exception as exc:
        logger.warning("[watchers] hardware instance failed to load: %s", exc)


# Loading this module (idle router's ensure_builtin_tasks_registered does it at
# boot) registers the instances and, through them, their recurring idle tasks.
_ensure_instances_loaded()
