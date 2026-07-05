# FILE: app/gpu/orchestrator.py
# Purpose: 4080 residency state machine — INTERACTIVE / BACKGROUND_INGEST / GAMING, park-never-kill converge.
# Called-by: app.gpu.router, app.idle.governor (activity + idle hooks), app.idle.tasks_visual, main.py (boot)
# Depends-on: app.gpu.actuators, app.gpu.nvsmi, app.embeddings.provider_router
# Last-renovated: 2026-07-02 (LANE E carryover: park/wake residency, idle-park, GAMING suspends drain)
"""
Single owner of what lives on the 4080 (LANE E + LANE-E-CARRYOVER.md).

CORE RULE: park, never kill. Components move between VRAM ("vram") and
system RAM ("ram"); processes stay alive. Cold starts happen once per model
per boot, never on an interactive path. RAM wake is 1-3s and that is the
interactive budget.

States and residency:
  INTERACTIVE        chatterbox vram · nat vram · text embedder vram
  BACKGROUND_INGEST  chatterbox RAM (its VRAM is what idle work inherits) ·
                     nat vram (idle jobs use it) · embedder vram ·
                     multimodal vram when local + queued work
  GAMING             everything RAM-parked + whisper unloaded; the idle
                     scheduler is fully suspended (governor checks
                     drain_suspended()); the 3090 is never touched.

Wake trigger is MESSAGE ARRIVAL: the governor's activity stamp calls
on_user_activity_signal(), which fires the Chatterbox wake in parallel with
LLM generation — TTS is VRAM-resident before the first reply sentence.
Park trigger is the governor's idle window (10 min, any user interaction
resets): on_idle_enter() moves to BACKGROUND_INGEST.

Desired state persists to data/gpu_state.json (atomic). Transitions are
idempotent — converge() probes each component and only acts on mismatch,
under a lock. Boot restores GAMING; an interrupted INGEST recovers to
INTERACTIVE (the drain tasks' ledger rows are their own checkpoint).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from app.gpu import actuators, nvsmi

logger = logging.getLogger(__name__)

INTERACTIVE = "INTERACTIVE"
BACKGROUND_INGEST = "BACKGROUND_INGEST"
GAMING = "GAMING"
_STATES = (INTERACTIVE, BACKGROUND_INGEST, GAMING)

VRAM = "vram"
RAM = "ram"

_STATE_FILE = Path("data") / "gpu_state.json"


def _text_embedder_is_local() -> bool:
    try:
        from app.embeddings.provider_router import text_write_spec
        return text_write_spec().provider == "local"
    except Exception:
        return False


def _multimodal_is_local() -> bool:
    try:
        from app.embeddings.provider_router import multimodal_write_spec
        return multimodal_write_spec().provider == "local"
    except Exception:
        return False


def _visual_work_pending() -> bool:
    try:
        from app.db import SessionLocal
        from app.embeddings.visual_queue import pending_count
        db = SessionLocal()
        try:
            return pending_count(db) > 0
        finally:
            db.close()
    except Exception:
        return False


class GpuOrchestrator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = INTERACTIVE
        self._reason = "boot default"
        self._updated_at = datetime.utcnow().isoformat()
        self._converging = False
        self._load_persisted()

    # ── persistence ──────────────────────────────────────────────

    def _load_persisted(self) -> None:
        try:
            if _STATE_FILE.exists():
                data = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
                persisted = data.get("desired_state")
                if persisted == GAMING:
                    # Restore GAMING (mid-game restarts must not reload TTS).
                    self._state = GAMING
                    self._reason = "restored from persisted state (boot)"
                elif persisted == BACKGROUND_INGEST:
                    # Crash mid-INGEST → INTERACTIVE; the drain task's queue
                    # rows recover via the idle ledger (running → paused).
                    self._state = INTERACTIVE
                    self._reason = "boot recovery from interrupted INGEST"
        except Exception as exc:
            logger.warning("[gpu] persisted state unreadable: %s", exc)

    def _persist(self) -> None:
        try:
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = _STATE_FILE.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({
                "desired_state": self._state,
                "reason": self._reason,
                "updated_at": self._updated_at,
            }, indent=2), encoding="utf-8")
            os.replace(tmp, _STATE_FILE)
        except Exception as exc:
            logger.warning("[gpu] state persist failed: %s", exc)

    # ── state accessors ──────────────────────────────────────────

    def current_state(self) -> str:
        return self._state

    def drain_suspended(self) -> bool:
        """CARRYOVER §3: GAMING suspends the idle scheduler ENTIRELY — the
        governor checks this before draining any ledger task."""
        return self._state == GAMING

    def desired_components(self) -> Dict[str, str]:
        """Desired residency per 4080 component in the current state."""
        state = self._state
        embed = VRAM if (_text_embedder_is_local() and state != GAMING) else RAM
        mm = RAM
        if (
            state == BACKGROUND_INGEST
            and _multimodal_is_local()
            and (_visual_work_pending() or actuators.mm_worker_running())
        ):
            mm = VRAM
        return {
            "chatterbox": VRAM if state == INTERACTIVE else RAM,
            "embed_text": embed,
            "nat": RAM if state == GAMING else VRAM,
            "multimodal_worker": mm,
        }

    # ── transitions (idempotent) ─────────────────────────────────

    def _set_state(self, state: str, reason: str) -> None:
        if state not in _STATES:
            raise ValueError(f"unknown GPU state {state!r}")
        with self._lock:
            changed = state != self._state
            self._state = state
            self._reason = reason
            self._updated_at = datetime.utcnow().isoformat()
            self._persist()
        if changed:
            logger.info("[gpu] state → %s (%s)", state, reason)

    def set_gaming(self, enabled: bool, reason: str = "manual toggle") -> Dict:
        self._set_state(GAMING if enabled else INTERACTIVE, reason)
        self.converge()
        return self.status(probe=False)

    def converge_async(self) -> None:
        """Converge on a worker thread — actuators do HTTP weight transfers
        and (fallback) WSL calls; never block the shared event loop."""
        threading.Thread(target=self.converge, daemon=True).start()

    def boot_bringup(self) -> None:
        """Boot residency (main.py runs this in a daemon thread). Ensures the
        permanently-resident text embedder actually comes up — STAGGERED behind
        Nat + Chatterbox and health-VERIFIED with retries — BEFORE the normal
        converge. Its cold-start otherwise raced Nat's load on the shared 4080
        and stranded every embedding call for ~10 min (2026-07-03 incident)."""
        if self._state != GAMING and _text_embedder_is_local():
            try:
                actuators.ensure_embed_server()
            except Exception as exc:
                logger.error("[gpu] embed bring-up failed: %s", exc)
        self.converge()

    def request_background_ingest(self, reason: str = "") -> bool:
        """Enter the idle/ingest state unless the user is gaming. Idempotent,
        non-blocking — callers poll readiness rather than waiting."""
        if self._state == GAMING:
            logger.info("[gpu] INGEST request refused (GAMING): %s", reason)
            return False
        if self._state != BACKGROUND_INGEST:
            self._set_state(BACKGROUND_INGEST, reason or "background ingest requested")
        self.converge_async()
        return True

    def notify_ingest_complete(self) -> None:
        if self._state == BACKGROUND_INGEST:
            self._set_state(INTERACTIVE, "ingest work drained")
            self.converge_async()

    def on_idle_enter(self) -> None:
        """Governor idle window opened (10 min, all surfaces). Park
        Chatterbox — its VRAM is what idle embedding/ingest work inherits.
        No-op during GAMING."""
        if self._state == INTERACTIVE:
            self.request_background_ingest("idle window (10 min inactivity)")

    def on_user_activity(self) -> None:
        """CARRYOVER §2: wake trigger is MESSAGE ARRIVAL. Fire the Chatterbox
        wake immediately (parallel with LLM generation — TTS must be resident
        before the first reply sentence), then converge the rest. Never
        touches GAMING. Cheap no-op in INTERACTIVE."""
        if self._state != BACKGROUND_INGEST:
            return
        threading.Thread(target=actuators.wake_chatterbox, daemon=True).start()
        self._set_state(INTERACTIVE, "user activity")
        self.converge_async()

    # ── converge ─────────────────────────────────────────────────

    def converge(self) -> Dict[str, str]:
        """Drive residency toward desired_components(). Probe-first, act on
        mismatch only, one converger at a time. Park/wake, never kill —
        hard stops only happen inside actuator fallbacks."""
        with self._lock:
            if self._converging:
                return {"skipped": "converge already running"}
            self._converging = True
        actions: Dict[str, str] = {}
        try:
            desired = self.desired_components()

            # Chatterbox — Electron owns the process; we own residency.
            if desired["chatterbox"] == VRAM:
                actions["chatterbox"] = (
                    "wake" if actuators.chatterbox_parked() is True
                    and actuators.wake_chatterbox() else "ok"
                )
            else:
                actions["chatterbox"] = (
                    "park" if actuators.chatterbox_parked() is False
                    and actuators.park_chatterbox() else "ok"
                )

            # Nat — vram everywhere except GAMING (idle jobs use it).
            if desired["nat"] == VRAM:
                actions["nat"] = "wake" if not actuators.nat_awake() and actuators.wake_nat() else "ok"
            else:
                actions["nat"] = "sleep" if actuators.nat_awake() and actuators.sleep_nat() else "ok"

            # Text embedder — permanently resident unless GAMING (or gemini).
            if desired["embed_text"] == VRAM:
                actions["embed_text"] = (
                    "wake" if not actuators.embed_server_awake()
                    and actuators.wake_embed_server() else "ok"
                )
            else:
                actions["embed_text"] = (
                    "sleep" if actuators.embed_server_awake()
                    and actuators.sleep_embed_server() else "ok"
                )

            # Multimodal — idle-window citizen only.
            if desired["multimodal_worker"] == VRAM:
                actions["multimodal_worker"] = (
                    "wake" if not actuators.mm_worker_awake()
                    and actuators.wake_mm_worker() else "ok"
                )
            else:
                actions["multimodal_worker"] = (
                    "sleep" if actuators.mm_worker_running() and actuators.mm_worker_awake()
                    and actuators.sleep_mm_worker() else "ok"
                )

            if self._state == GAMING:
                actuators.unload_whisper()
                actions["whisper"] = "unload"
        finally:
            self._converging = False
        logger.info("[gpu] converge(%s): %s", self._state, actions)
        return actions

    # ── status ───────────────────────────────────────────────────

    def status(self, probe: bool = True) -> Dict:
        desired = self.desired_components()
        components = {}
        if probe:
            cb_parked = actuators.chatterbox_parked()
            components = {
                "chatterbox": {
                    "desired": desired["chatterbox"],
                    "running": cb_parked is not None,
                    "parked": cb_parked,
                },
                "embed_text": {
                    "desired": desired["embed_text"],
                    "running": actuators.embed_server_running(),
                    "awake": actuators.embed_server_awake(),
                },
                "nat": {
                    "desired": desired["nat"],
                    "running": actuators.nat_running(),
                    "awake": actuators.nat_awake(),
                },
                "multimodal_worker": {
                    "desired": desired["multimodal_worker"],
                    "running": actuators.mm_worker_running(),
                    "awake": actuators.mm_worker_awake(),
                },
            }
        queue_depth = 0
        try:
            from app.db import SessionLocal
            from app.embeddings.visual_queue import pending_count
            db = SessionLocal()
            try:
                queue_depth = pending_count(db)
            finally:
                db.close()
        except Exception:
            pass
        return {
            "state": self._state,
            "reason": self._reason,
            "updated_at": self._updated_at,
            "components": components,
            "ingest_queue_depth": queue_depth,
            "gpus": nvsmi.query_gpus() if probe else [],
        }


_orchestrator: Optional[GpuOrchestrator] = None
_singleton_lock = threading.Lock()


def get_orchestrator() -> GpuOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        with _singleton_lock:
            if _orchestrator is None:
                _orchestrator = GpuOrchestrator()
    return _orchestrator


# ─── Module-level hooks (cheap, exception-proof) ─────────────────

def on_user_activity_signal() -> None:
    """Wired into the idle governor's activity stamp — must stay cheap."""
    try:
        orch = _orchestrator
        if orch is not None:
            orch.on_user_activity()
    except Exception as exc:
        logger.debug("[gpu] activity hook failed: %s", exc)


def on_idle_enter_signal() -> None:
    """Wired into the governor's idle-window entry."""
    try:
        orch = _orchestrator
        if orch is not None:
            orch.on_idle_enter()
    except Exception as exc:
        logger.debug("[gpu] idle hook failed: %s", exc)


def drain_suspended() -> bool:
    """Governor gate: True while GAMING — no ledger task may run."""
    try:
        orch = _orchestrator
        return orch is not None and orch.drain_suspended()
    except Exception:
        return False


def request_background_ingest(reason: str = "") -> bool:
    try:
        return get_orchestrator().request_background_ingest(reason)
    except Exception as exc:
        logger.warning("[gpu] ingest request failed: %s", exc)
        return False


def notify_ingest_complete() -> None:
    try:
        get_orchestrator().notify_ingest_complete()
    except Exception as exc:
        logger.debug("[gpu] ingest-complete notify failed: %s", exc)
