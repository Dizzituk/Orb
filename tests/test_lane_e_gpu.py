# FILE: tests/test_lane_e_gpu.py
# Purpose: LANE E + carryover — GPU orchestrator: park/wake residency, idle-park, GAMING suspends drain, boot recovery, interrupt cycles.
# Called-by: pytest
# Depends-on: app.gpu.*
# Last-renovated: 2026-07-02 (carryover: park-never-kill semantics)

import json
import time
from pathlib import Path

import pytest

from app.gpu import actuators, orchestrator
from app.gpu.orchestrator import (
    BACKGROUND_INGEST,
    GAMING,
    INTERACTIVE,
    RAM,
    VRAM,
    GpuOrchestrator,
)


class FakeActuators:
    """Park-never-kill fakes: every component tracks running (process alive)
    and awake (weights in VRAM). Kills never happen — only park/wake."""

    def __init__(self):
        self.running = {"chatterbox": True, "embed": True, "nat": True, "mm": False}
        self.awake = {"chatterbox": True, "embed": True, "nat": True, "mm": False}
        self.log = []

    def install(self, monkeypatch):
        m = monkeypatch
        # chatterbox: parked = running and not awake; None when down
        m.setattr(actuators, "chatterbox_running", lambda: self.running["chatterbox"])
        m.setattr(actuators, "chatterbox_parked",
                  lambda: (None if not self.running["chatterbox"]
                           else not self.awake["chatterbox"]))
        m.setattr(actuators, "park_chatterbox", lambda: self._set("chatterbox", awake=False, verb="park"))
        m.setattr(actuators, "wake_chatterbox", lambda: self._set("chatterbox", awake=True, verb="wake"))
        for name, key in (("nat", "nat"), ("embed_server", "embed"), ("mm_worker", "mm")):
            m.setattr(actuators, f"{key if key != 'embed' else 'embed_server'}_running"
                      if key != "mm" else "mm_worker_running",
                      (lambda k: lambda: self.running[k])(key))
            m.setattr(actuators, f"{'nat' if key == 'nat' else ('embed_server' if key == 'embed' else 'mm_worker')}_awake",
                      (lambda k: lambda: self.running[k] and self.awake[k])(key))
        m.setattr(actuators, "sleep_nat", lambda: self._set("nat", awake=False, verb="sleep"))
        m.setattr(actuators, "wake_nat", lambda: self._set("nat", awake=True, verb="wake", ensure_running=True))
        m.setattr(actuators, "sleep_embed_server", lambda: self._set("embed", awake=False, verb="sleep"))
        m.setattr(actuators, "wake_embed_server", lambda: self._set("embed", awake=True, verb="wake", ensure_running=True))
        m.setattr(actuators, "sleep_mm_worker", lambda: self._set("mm", awake=False, verb="sleep"))
        m.setattr(actuators, "wake_mm_worker", lambda: self._set("mm", awake=True, verb="wake", ensure_running=True))
        m.setattr(actuators, "unload_whisper", lambda: self.log.append("whisper:unload"))

    def _set(self, key, awake, verb, ensure_running=False):
        self.log.append(f"{key}:{verb}")
        if ensure_running:
            self.running[key] = True
        self.awake[key] = awake
        return True


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestrator, "_STATE_FILE", tmp_path / "gpu_state.json")
    monkeypatch.setenv("EMBEDDINGS_TEXT_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDINGS_MULTIMODAL_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDINGS_MULTIMODAL_MODEL", "jina-embeddings-v4")
    monkeypatch.setattr(orchestrator, "_visual_work_pending", lambda: True)
    fa = FakeActuators()
    fa.install(monkeypatch)
    return fa


def _fresh(tmp_path):
    return GpuOrchestrator()


def _wait_until(cond, seconds=3.0):
    deadline = time.time() + seconds
    while not cond() and time.time() < deadline:
        time.sleep(0.02)
    return cond()


# ── desired residency per state ──────────────────────────────────


def test_desired_residency_per_state(fakes, tmp_path):
    orch = _fresh(tmp_path)
    assert orch.desired_components() == {
        "chatterbox": VRAM, "embed_text": VRAM, "nat": VRAM,
        "multimodal_worker": RAM,
    }
    orch._set_state(BACKGROUND_INGEST, "test")
    d = orch.desired_components()
    assert d["chatterbox"] == RAM          # its VRAM is what idle work inherits
    assert d["embed_text"] == VRAM         # small text embedder stays resident
    assert d["nat"] == VRAM                # idle jobs use Nat
    assert d["multimodal_worker"] == VRAM  # local + queued work
    orch._set_state(GAMING, "test")
    d = orch.desired_components()
    assert d == {
        "chatterbox": RAM, "embed_text": RAM, "nat": RAM,
        "multimodal_worker": RAM,
    }


def test_mm_stays_parked_without_queued_work(fakes, tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "_visual_work_pending", lambda: False)
    orch = _fresh(tmp_path)
    orch._set_state(BACKGROUND_INGEST, "test")
    assert orch.desired_components()["multimodal_worker"] == RAM


def test_gemini_text_provider_means_embedder_parked(fakes, tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_TEXT_PROVIDER", "gemini")
    orch = _fresh(tmp_path)
    assert orch.desired_components()["embed_text"] == RAM


# ── park-never-kill converge ─────────────────────────────────────


def test_converge_is_idempotent(fakes, tmp_path):
    orch = _fresh(tmp_path)
    orch.converge()
    first = list(fakes.log)
    orch.converge()
    orch.converge()
    assert fakes.log == first  # already-converged reruns take no actions


def test_gaming_parks_everything_kills_nothing(fakes, tmp_path):
    orch = _fresh(tmp_path)
    orch.set_gaming(True)
    assert orch.current_state() == GAMING
    assert "nat:sleep" in fakes.log
    assert "embed:sleep" in fakes.log
    assert "chatterbox:park" in fakes.log
    assert "whisper:unload" in fakes.log
    # park-never-kill: every process is STILL RUNNING, just not in VRAM
    assert all(fakes.running[k] for k in ("chatterbox", "embed", "nat"))
    assert not any(fakes.awake[k] for k in ("chatterbox", "embed", "nat"))

    fakes.log.clear()
    orch.set_gaming(False)
    assert orch.current_state() == INTERACTIVE
    assert "chatterbox:wake" in fakes.log
    assert "nat:wake" in fakes.log and "embed:wake" in fakes.log


def test_ingest_refused_during_gaming(fakes, tmp_path):
    orch = _fresh(tmp_path)
    orch.set_gaming(True)
    assert orch.request_background_ingest("visual work") is False
    assert orch.current_state() == GAMING


def test_gaming_suspends_idle_drain(fakes, tmp_path, monkeypatch):
    orch = _fresh(tmp_path)
    monkeypatch.setattr(orchestrator, "_orchestrator", orch)
    assert orchestrator.drain_suspended() is False
    orch.set_gaming(True)
    assert orchestrator.drain_suspended() is True


@pytest.mark.asyncio
async def test_governor_run_pending_gated_by_gaming(fakes, tmp_path, monkeypatch):
    """CARRYOVER §3: the governor must not drain ANY ledger task in GAMING."""
    from app.idle.governor import IdleGovernor

    orch = _fresh(tmp_path)
    monkeypatch.setattr(orchestrator, "_orchestrator", orch)
    orch.set_gaming(True)
    monkeypatch.setenv("IDLE_MINUTES", "0")
    monkeypatch.setenv("ASTRA_IDLE_GOVERNOR_ENABLED", "true")
    boom = lambda: (_ for _ in ()).throw(AssertionError("session_factory touched"))
    gov = IdleGovernor(session_factory=boom)
    assert await gov.run_pending() == 0  # returns before touching the ledger


# ── idle-park + wake-on-message ──────────────────────────────────


def test_idle_enter_parks_chatterbox(fakes, tmp_path):
    orch = _fresh(tmp_path)
    orch.on_idle_enter()
    assert orch.current_state() == BACKGROUND_INGEST
    assert _wait_until(lambda: not fakes.awake["chatterbox"] and not orch._converging)
    assert "chatterbox:park" in fakes.log


def test_idle_enter_noop_during_gaming(fakes, tmp_path):
    orch = _fresh(tmp_path)
    orch.set_gaming(True)
    orch.on_idle_enter()
    assert orch.current_state() == GAMING


def test_user_activity_wakes_chatterbox_immediately(fakes, tmp_path):
    """CARRYOVER §2: wake fires on message arrival, racing LLM generation."""
    orch = _fresh(tmp_path)
    orch.on_idle_enter()
    assert _wait_until(lambda: not fakes.awake["chatterbox"] and not orch._converging)
    orch.on_user_activity()
    assert orch.current_state() == INTERACTIVE
    assert _wait_until(lambda: fakes.awake["chatterbox"] and not orch._converging)
    assert "chatterbox:wake" in fakes.log


def test_user_activity_never_touches_gaming(fakes, tmp_path):
    orch = _fresh(tmp_path)
    orch.set_gaming(True)
    orch.on_user_activity()
    assert orch.current_state() == GAMING


def test_module_hook_is_noop_without_singleton(monkeypatch):
    monkeypatch.setattr(orchestrator, "_orchestrator", None)
    orchestrator.on_user_activity_signal()
    orchestrator.on_idle_enter_signal()
    assert orchestrator.drain_suspended() is False


def test_governor_record_activity_fires_gpu_hook(fakes, tmp_path, monkeypatch):
    from app.idle.governor import IdleGovernor

    orch = _fresh(tmp_path)
    monkeypatch.setattr(orchestrator, "_orchestrator", orch)
    orch.on_idle_enter()
    assert _wait_until(lambda: not orch._converging)
    gov = IdleGovernor(session_factory=lambda: None)
    gov.record_activity()
    assert orch.current_state() == INTERACTIVE


# ── persistence + boot recovery ──────────────────────────────────


def test_gaming_survives_restart(fakes, tmp_path):
    orch = _fresh(tmp_path)
    orch.set_gaming(True)
    reborn = GpuOrchestrator()
    assert reborn.current_state() == GAMING


def test_interrupted_ingest_recovers_to_interactive(fakes, tmp_path):
    orch = _fresh(tmp_path)
    orch.request_background_ingest("crash test")
    assert orch.current_state() == BACKGROUND_INGEST
    reborn = GpuOrchestrator()
    assert reborn.current_state() == INTERACTIVE
    assert "boot recovery" in reborn._reason


def test_state_file_is_atomic_json(fakes, tmp_path):
    orch = _fresh(tmp_path)
    orch.set_gaming(True)
    data = json.loads((tmp_path / "gpu_state.json").read_text(encoding="utf-8"))
    assert data["desired_state"] == GAMING


# ── acceptance 4 (unit level): ten interrupt cycles, no leak ─────


def test_ten_interrupt_cycles_return_to_baseline(fakes, tmp_path):
    orch = _fresh(tmp_path)
    orch.converge()
    baseline_awake = dict(fakes.awake)
    for _ in range(10):
        assert orch.request_background_ingest("cycle") is True
        assert _wait_until(
            lambda: fakes.awake["mm"] and not fakes.awake["chatterbox"]
            and not orch._converging
        )
        orch.on_user_activity()
        assert orch.current_state() == INTERACTIVE
        # VRAM residency returns to baseline (mm parked again, voice back)
        assert _wait_until(
            lambda: fakes.awake == baseline_awake and not orch._converging
        )
        # park-never-kill invariant: processes NEVER stop running — the mm
        # worker stays alive (parked in RAM) after its first boot, and the
        # interactive residents never die at all.
        assert all(fakes.running[k] for k in ("chatterbox", "embed", "nat"))
    assert fakes.running["mm"] is True  # booted once, parked ever after


# ── Fix A/B regression (2026-07-03 collateral-kill incident) ─────


def test_wake_nat_never_cold_starts(monkeypatch):
    """wake_nat wakes a SLEEPING Nat but must NEVER cold-start a DOWN Nat:
    the cold start raced Electron's spawn and tripped 03_serve_nat.sh's
    (then-unscoped) cleanup that killed the :8004/:8005 embedders. A down Nat
    returns False; Electron's reconcile respawns the process."""
    calls = {"wsl": 0, "wake_up": 0}
    monkeypatch.setattr(actuators, "_wsl",
                        lambda *a, **k: calls.__setitem__("wsl", calls["wsl"] + 1) or True)

    monkeypatch.setattr(actuators, "nat_running", lambda: False)
    assert actuators.wake_nat() is False
    assert calls["wsl"] == 0  # no cold start

    monkeypatch.setattr(actuators, "nat_running", lambda: True)
    monkeypatch.setattr(actuators, "_vllm_sleeping", lambda base: True)
    monkeypatch.setattr(actuators, "_vllm_wake",
                        lambda base, label: calls.__setitem__("wake_up", 1) or True)
    assert actuators.wake_nat() is True
    assert calls["wake_up"] == 1 and calls["wsl"] == 0

    # start_nat was removed entirely — nothing can accidentally cold-start Nat.
    assert not hasattr(actuators, "start_nat")


def test_sleep_nat_fallback_is_port_scoped(monkeypatch):
    """The pre-sleep-mode GAMING fallback uses the PORT-SCOPED stop
    (stop_vllm_port.sh 8003), never the unscoped stop_nat.sh that also kills
    the :8004/:8005 embedders."""
    captured = {}
    monkeypatch.setattr(actuators, "nat_running", lambda: True)
    monkeypatch.setattr(actuators, "_vllm_sleeping", lambda base: None)   # no sleep endpoint
    monkeypatch.setattr(actuators, "_vllm_sleep", lambda base, label: False)  # sleep fails
    monkeypatch.setattr(actuators, "_wsl",
                        lambda cmd, **k: captured.__setitem__("cmd", cmd) or True)
    actuators.sleep_nat()
    assert "stop_vllm_port.sh" in captured["cmd"]
    assert "stop_nat.sh" not in captured["cmd"]


# ── embedder boot bring-up (2026-07-03 cold-start-race incident) ──


def test_ensure_embed_server_noop_when_healthy(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *a: None)
    starts = {"n": 0}
    monkeypatch.setattr(actuators, "_embed_truly_healthy", lambda: True)
    monkeypatch.setattr(actuators, "start_embed_server",
                        lambda: starts.__setitem__("n", starts["n"] + 1) or True)
    assert actuators.ensure_embed_server(stagger_wait=0, attempts=3, poll=0) is True
    assert starts["n"] == 0  # already healthy — never (re)started


def test_ensure_embed_server_starts_then_verifies(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *a: None)
    monkeypatch.setattr(actuators, "nat_running", lambda: True)
    monkeypatch.setattr(actuators, "chatterbox_running", lambda: True)
    calls = {"start": 0, "stop": 0}
    monkeypatch.setattr(actuators, "start_embed_server",
                        lambda: calls.__setitem__("start", calls["start"] + 1) or True)
    monkeypatch.setattr(actuators, "stop_embed_server",
                        lambda: calls.__setitem__("stop", calls["stop"] + 1) or True)
    # unhealthy at entry + attempt-1 top; healthy once started.
    seq = iter([False, False, True, True, True])
    monkeypatch.setattr(actuators, "_embed_truly_healthy", lambda: next(seq))
    assert actuators.ensure_embed_server(stagger_wait=0, attempts=3, poll=0) is True
    assert calls["start"] == 1 and calls["stop"] == 0  # single ensure-start, no restart


def test_ensure_embed_server_force_restarts_after_failed_first_attempt(monkeypatch):
    """The incident case: first start loses the VRAM race and stays dead — the
    retry must force a clean stop+restart (once Nat has settled) rather than
    leave it stranded."""
    monkeypatch.setattr("time.sleep", lambda *a: None)
    monkeypatch.setattr(actuators, "nat_running", lambda: True)
    monkeypatch.setattr(actuators, "chatterbox_running", lambda: True)
    calls = {"start": 0, "stop": 0}
    monkeypatch.setattr(actuators, "start_embed_server",
                        lambda: calls.__setitem__("start", calls["start"] + 1) or True)
    monkeypatch.setattr(actuators, "stop_embed_server",
                        lambda: calls.__setitem__("stop", calls["stop"] + 1) or True)
    # False for entry + attempt1 + attempt2 tops; True at attempt3 top.
    seq = iter([False, False, False, True])
    monkeypatch.setattr(actuators, "_embed_truly_healthy", lambda: next(seq))
    assert actuators.ensure_embed_server(stagger_wait=0, attempts=3, poll=0) is True
    assert calls["stop"] == 1        # retry forced a clean restart
    assert calls["start"] == 2       # attempt1 ensure-start + attempt2 restart


def test_boot_bringup_ensures_embedder_then_converges(fakes, tmp_path, monkeypatch):
    orch = _fresh(tmp_path)
    ensured = {"n": 0}
    monkeypatch.setattr(actuators, "ensure_embed_server",
                        lambda *a, **k: ensured.__setitem__("n", ensured["n"] + 1) or True)
    orch.boot_bringup()
    assert ensured["n"] == 1                 # embedder bring-up ran (local provider)
    assert orch.current_state() == INTERACTIVE


def test_boot_bringup_skips_embed_ensure_in_gaming(fakes, tmp_path, monkeypatch):
    orch = _fresh(tmp_path)
    orch.set_gaming(True)
    ensured = {"n": 0}
    monkeypatch.setattr(actuators, "ensure_embed_server",
                        lambda *a, **k: ensured.__setitem__("n", ensured["n"] + 1) or True)
    orch.boot_bringup()
    assert ensured["n"] == 0                 # GAMING: embedder stays parked


# ── router payload shapes ────────────────────────────────────────


def test_desired_endpoint_payload_shape(fakes, tmp_path, monkeypatch):
    orch = _fresh(tmp_path)
    monkeypatch.setattr(orchestrator, "_orchestrator", orch)
    from app.gpu.router import gpu_desired

    import asyncio
    payload = asyncio.run(gpu_desired())
    assert payload["state"] == INTERACTIVE
    assert payload["components"]["chatterbox"] == VRAM
