# FILE: app/idle/router.py
# Purpose: Task-type registry (handlers, fingerprints, recurring cadence) + run context contract.
# Called-by: app.idle.governor, app.idle.tasks, app.watchers.framework, app.llm.research_task
# Depends-on: app.idle.ledger
# Last-renovated: 2026-07-01
"""
Handlers are small resumable units: between units they check
ctx.should_yield() (True once the user sends a message), persist progress via
ctx.save_progress(), and return TaskOutcome.paused(). The governor resumes
them in the next idle window from ctx.load_progress().

Handler contract:
    async def handler(ctx: TaskContext) -> TaskOutcome
Fingerprint contract (optional, enables unchanged-inputs skip):
    def fingerprint_fn(params: dict) -> Optional[str]   # cheap, no LLM
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

from sqlalchemy.orm import Session

from app.idle import ledger
from app.idle.models import IdleTaskRecord

logger = logging.getLogger(__name__)


@dataclass
class TaskOutcome:
    status: str  # "completed" | "paused" | "failed"
    summary: str = ""
    coverage: str = ""
    error: str = ""

    @classmethod
    def completed(cls, summary: str = "", coverage: str = "") -> "TaskOutcome":
        return cls(status="completed", summary=summary, coverage=coverage)

    @classmethod
    def paused(cls, summary: str = "") -> "TaskOutcome":
        return cls(status="paused", summary=summary)

    @classmethod
    def failed(cls, error: str) -> "TaskOutcome":
        return cls(status="failed", error=error)


@dataclass
class TaskContext:
    db: Session
    record: IdleTaskRecord
    params: dict
    fingerprint: Optional[str]
    # True once user activity arrived after this run started — persist and yield.
    should_yield: Callable[[], bool]
    # Fresh short-lived sessions for work pushed to threads (never share ctx.db
    # across threads).
    session_factory: Callable[[], Session]

    def load_progress(self) -> dict:
        try:
            return json.loads(self.record.progress_json or "{}")
        except Exception:
            return {}

    def save_progress(self, progress: dict) -> None:
        ledger.save_progress(self.db, self.record, progress)


TaskHandler = Callable[[TaskContext], Awaitable[TaskOutcome]]
FingerprintFn = Callable[[dict], Optional[str]]


@dataclass
class RecurringSpec:
    task_type: str
    params: dict = field(default_factory=dict)
    cadence_hours: float = 24.0


_HANDLERS: Dict[str, TaskHandler] = {}
_FINGERPRINTS: Dict[str, FingerprintFn] = {}
_RECURRING: List[RecurringSpec] = []


def register_task_handler(
    task_type: str,
    handler: TaskHandler,
    fingerprint_fn: Optional[FingerprintFn] = None,
    recurring: Optional[RecurringSpec] = None,
) -> None:
    if task_type in _HANDLERS:
        logger.debug("[idle.router] handler for %s re-registered", task_type)
    _HANDLERS[task_type] = handler
    if fingerprint_fn is not None:
        _FINGERPRINTS[task_type] = fingerprint_fn
    if recurring is not None:
        _RECURRING[:] = [s for s in _RECURRING if s.task_type != recurring.task_type or s.params != recurring.params]
        _RECURRING.append(recurring)


def get_handler(task_type: str) -> Optional[TaskHandler]:
    return _HANDLERS.get(task_type)


def get_fingerprint_fn(task_type: str) -> Optional[FingerprintFn]:
    return _FINGERPRINTS.get(task_type)


def list_recurring() -> List[RecurringSpec]:
    return list(_RECURRING)


def ensure_builtin_tasks_registered() -> None:
    """Import the modules whose import side-effect registers idle tasks.
    Each import is independent — one broken subsystem must not take down
    the rest of the idle fleet."""
    try:
        import app.idle.tasks  # noqa: F401  (repo map)
    except Exception as exc:
        logger.warning("[idle.router] repo-map task unavailable: %s", exc)
    try:
        import app.watchers.framework  # noqa: F401  (watcher observations)
    except Exception as exc:
        logger.debug("[idle.router] watchers not registered: %s", exc)
    try:
        import app.llm.research_task  # noqa: F401  (deep research grind)
    except Exception as exc:
        logger.debug("[idle.router] research task not registered: %s", exc)
    # LANE E (2026-07-02): local-embeddings migration + visual ingest
    try:
        import app.idle.tasks_reembed  # noqa: F401  (reembed_batch + parity gate)
    except Exception as exc:
        logger.warning("[idle.router] reembed tasks unavailable: %s", exc)
    try:
        import app.idle.tasks_visual  # noqa: F401  (visual-embed queue drain)
    except Exception as exc:
        logger.warning("[idle.router] visual drain task unavailable: %s", exc)


def catch_up(db: Session) -> int:
    """Enqueue every recurring task that is due (no terminal run inside its
    cadence window). Called at boot and periodically while idle — this is how
    a nightly shutdown catches up the morning after."""
    queued = 0
    for spec in list_recurring():
        try:
            key = ledger.task_key(spec.task_type, spec.params)
            if ledger.is_due(db, key, spec.cadence_hours):
                if ledger.enqueue(db, spec.task_type, spec.params) is not None:
                    queued += 1
        except Exception as exc:
            logger.warning("[idle.router] catch_up failed for %s: %s", spec.task_type, exc)
    if queued:
        logger.info("[idle.router] catch-up queued %d due task(s)", queued)
    return queued
