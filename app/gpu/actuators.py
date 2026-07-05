# FILE: app/gpu/actuators.py
# Purpose: Park/wake/probe for each 4080 resident — Chatterbox RAM-park, vLLM sleep mode, Whisper unload.
# Called-by: app.gpu.orchestrator
# Depends-on: scripts/embeddings + scripts/nat serve scripts, app.services.model_manager (lazy)
# Last-renovated: 2026-07-02 (LANE E carryover: park-never-kill)
"""
Per-component actuation (LANE E + carryover).

CORE RULE (LANE-E-CARRYOVER.md §1): park, never kill. Weights move
VRAM <-> system RAM (the 64GB wings); processes stay alive. Cold starts
(10-15s) are banned on interactive paths — RAM wake (1-3s) is the budget.
  - Chatterbox: POST :8002/park | /wake (added this lane; rides the
    server's own generation lock, auto-wakes on synth).
  - vLLM instances (:8003 Nat, :8004 text embed, :8005 multimodal):
    POST /sleep?level=1 | /wake_up (dev-mode endpoints; --enable-sleep-mode
    in the serve scripts). Level 1 = weights offloaded to CPU RAM.
Hard stops remain ONLY as fallbacks (sleep endpoint unavailable — e.g. a
server started before the flag landed) and are port-scoped; Electron owns
the app-quit teardown as before. Disk is touched once per model, at the
first boot of each server (start_* functions).
"""
from __future__ import annotations

import logging
import os
import subprocess

import httpx

logger = logging.getLogger(__name__)

_WSL_TIMEOUT = 20
_PROBE_TIMEOUT = httpx.Timeout(2.0, connect=1.0)
# Weight transfers move gigabytes — generous but bounded.
_PARK_TIMEOUT = httpx.Timeout(60.0, connect=2.0)

REPO_WSL_ROOT = "/mnt/d/Orb"

CHATTERBOX_BASE = "http://127.0.0.1:8002"
NAT_BASE = "http://127.0.0.1:8003"


def _embed_root() -> str:
    from app.embeddings.local_provider import text_base_url
    return text_base_url().rsplit("/v1", 1)[0]


def _mm_root() -> str:
    from app.embeddings.local_provider import multimodal_base_url
    return multimodal_base_url().rsplit("/v1", 1)[0]


def _probe_http(url: str) -> bool:
    try:
        return httpx.get(url, timeout=_PROBE_TIMEOUT).status_code == 200
    except Exception:
        return False


def _wsl(cmd: str, detach: bool = False) -> bool:
    """Run a command inside WSL as root (the vLLM instances live there).
    Detached server launches must end with '& sleep 3' — the wsl session
    tears its children down if the wrapper exits before setsid settles
    (verified live 2026-07-02)."""
    args = ["wsl", "-u", "root", "--", "bash", "-lc", cmd]
    try:
        if detach:
            subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return True
        out = subprocess.run(args, capture_output=True, text=True, timeout=_WSL_TIMEOUT)
        if out.returncode != 0:
            logger.warning("[gpu.actuators] wsl cmd failed (%s): %s",
                           cmd[:80], (out.stderr or out.stdout or "")[:200])
        return out.returncode == 0
    except Exception as exc:
        logger.warning("[gpu.actuators] wsl call failed: %s", exc)
        return False


# ─── vLLM sleep/wake primitives ──────────────────────────────────

def _vllm_sleeping(root: str):
    """True/False from /is_sleeping; None when the server is unreachable
    or predates --enable-sleep-mode."""
    try:
        r = httpx.get(f"{root}/is_sleeping", timeout=_PROBE_TIMEOUT)
        if r.status_code == 200:
            return bool(r.json().get("is_sleeping"))
    except Exception:
        pass
    return None


def _vllm_sleep(root: str, label: str) -> bool:
    try:
        r = httpx.post(f"{root}/sleep", params={"level": 1}, timeout=_PARK_TIMEOUT)
        ok = r.status_code == 200
        logger.info("[gpu.actuators] %s sleep(level=1): %s", label, "ok" if ok else r.status_code)
        return ok
    except Exception as exc:
        logger.warning("[gpu.actuators] %s sleep failed: %s", label, exc)
        return False


def _vllm_wake(root: str, label: str) -> bool:
    try:
        r = httpx.post(f"{root}/wake_up", timeout=_PARK_TIMEOUT)
        ok = r.status_code == 200
        logger.info("[gpu.actuators] %s wake_up: %s", label, "ok" if ok else r.status_code)
        return ok
    except Exception as exc:
        logger.warning("[gpu.actuators] %s wake failed: %s", label, exc)
        return False


# ─── Chatterbox (:8002) — RAM park/wake, Electron owns the process ──

def chatterbox_running() -> bool:
    """Process alive (parked or not)."""
    return _probe_http(f"{CHATTERBOX_BASE}/ping")


def chatterbox_parked():
    """True/False from /ping; None when the process is down."""
    try:
        r = httpx.get(f"{CHATTERBOX_BASE}/ping", timeout=_PROBE_TIMEOUT)
        if r.status_code == 200:
            return bool(r.json().get("parked"))
    except Exception:
        pass
    return None


def park_chatterbox() -> bool:
    """Weights -> system RAM (~3.7GB freed). Serialised with synthesis by
    the server's own generation lock; idempotent."""
    if chatterbox_parked() is not False:
        return True  # already parked, or process down (nothing to park)
    try:
        r = httpx.post(f"{CHATTERBOX_BASE}/park", timeout=_PARK_TIMEOUT)
        ok = r.status_code == 200
        logger.info("[gpu.actuators] chatterbox park: %s", "ok" if ok else r.status_code)
        return ok
    except Exception as exc:
        logger.warning("[gpu.actuators] chatterbox park failed: %s", exc)
        return False


def wake_chatterbox() -> bool:
    """Weights -> VRAM (1-3s). Fired on user-message arrival, racing LLM
    generation; the server also auto-wakes on any synth request."""
    parked = chatterbox_parked()
    if parked is None:
        return False  # process down — Electron's reconcile respawns it
    if parked is False:
        return True
    try:
        r = httpx.post(f"{CHATTERBOX_BASE}/wake", timeout=_PARK_TIMEOUT)
        ok = r.status_code == 200
        logger.info("[gpu.actuators] chatterbox wake: %s", "ok" if ok else r.status_code)
        return ok
    except Exception as exc:
        logger.warning("[gpu.actuators] chatterbox wake failed: %s", exc)
        return False


# ─── Text embedder vLLM (:8004) ──────────────────────────────────

def embed_server_running() -> bool:
    """Process serving (awake or asleep)."""
    return _probe_http(f"{_embed_root()}/health") or _probe_http(f"{_embed_root()}/v1/models")


def embed_server_awake() -> bool:
    sleeping = _vllm_sleeping(_embed_root())
    if sleeping is None:
        # Pre-sleep-mode server or down: fall back to plain reachability.
        from app.embeddings.local_provider import text_available
        return text_available()
    return not sleeping


def sleep_embed_server() -> bool:
    if not embed_server_running():
        return True
    if _vllm_sleeping(_embed_root()) is True:
        return True
    if _vllm_sleep(_embed_root(), "embed(:8004)"):
        return True
    # Fallback (server predates sleep mode): port-scoped hard stop.
    logger.warning("[gpu.actuators] embed sleep unavailable — falling back to stop")
    port = os.getenv("EMBED_PORT", "8004")
    return _wsl(f"bash {REPO_WSL_ROOT}/scripts/embeddings/stop_vllm_port.sh {port}")


def wake_embed_server() -> bool:
    """Wake from RAM; cold-start only if the process is gone entirely
    (first boot of the day — the once-only disk cost)."""
    if not embed_server_running():
        return start_embed_server()
    if _vllm_sleeping(_embed_root()) is True:
        return _vllm_wake(_embed_root(), "embed(:8004)")
    return True


def start_embed_server() -> bool:
    """Cold start via the serve script (disk -> VRAM; NOT an interactive-path
    operation). setsid + trailing sleep required (WSL teardown race)."""
    if embed_server_running():
        return True
    logger.info("[gpu.actuators] cold-starting text embed server (:8004)")
    return _wsl(
        f"setsid nohup bash {REPO_WSL_ROOT}/scripts/embeddings/serve_embed.sh "
        f">/root/embed-serve.log 2>&1 < /dev/null & sleep 3",
        detach=True,
    )


def stop_embed_server() -> bool:
    """Hard stop — fallback path only (see sleep_embed_server)."""
    if not embed_server_running():
        return True
    port = os.getenv("EMBED_PORT", "8004")
    return _wsl(f"bash {REPO_WSL_ROOT}/scripts/embeddings/stop_vllm_port.sh {port}")


def _embed_truly_healthy() -> bool:
    """A REAL embedding round-trips — api-server up AND engine alive. A bare
    /health can return 200 while the EngineCore is dead (OOM on KV cache),
    so probe with an actual embed."""
    if not embed_server_running():
        return False
    try:
        from app.embeddings import local_provider
        return bool(local_provider.generate_embedding(
            "healthcheck", task_type="RETRIEVAL_QUERY"))
    except Exception:
        return False


def ensure_embed_server(stagger_wait: int = 60, attempts: int = 3, poll: int = 60) -> bool:
    """Boot bring-up for the permanently-resident text embedder (:8004).

    2026-07-03 incident: the embedder's cold-start fired ~1s after Nat's, so
    both large models loaded onto the shared 4080 at once (worsened by
    sleep-mode's eager cumem reservation). The embedder lost the VRAM race,
    its first start failed silently, and — with no health check — every
    embedding call was stranded for ~10 min until the next converge restarted
    it cleanly. This staggers the start behind the other heavy co-loaders,
    then VERIFIES a real embedding round-trips, retrying a clean restart if
    not. Blocking; main.py runs it in a daemon thread."""
    import time
    if _embed_truly_healthy():
        return True
    # Stagger behind BOTH heavy 4080 co-loaders (Nat :8003 + Chatterbox :8002)
    # so the three don't peak-collide. Bounded — proceed anyway if they're slow.
    deadline = time.monotonic() + stagger_wait
    while time.monotonic() < deadline:
        if nat_running() and chatterbox_running():
            logger.info("[gpu.actuators] co-loaders up — bringing up embed server")
            break
        time.sleep(3)
    for attempt in range(1, attempts + 1):
        if _embed_truly_healthy():
            return True
        if attempt == 1:
            start_embed_server()                      # ensure-start
        else:
            stop_embed_server(); time.sleep(2); start_embed_server()  # clean restart
        end = time.monotonic() + poll
        while time.monotonic() < end:
            time.sleep(3)
            if _embed_truly_healthy():
                logger.info("[gpu.actuators] embed server healthy (attempt %d/%d)",
                            attempt, attempts)
                return True
        logger.warning("[gpu.actuators] embed not healthy after %ds (attempt %d/%d)",
                       poll, attempt, attempts)
    logger.error("[gpu.actuators] embed server failed to come up after %d attempts", attempts)
    return False


# ─── Nat vLLM (:8003) ────────────────────────────────────────────

def nat_running() -> bool:
    return _probe_http(f"{NAT_BASE}/health") or _probe_http(f"{NAT_BASE}/v1/models")


def nat_awake() -> bool:
    sleeping = _vllm_sleeping(NAT_BASE)
    if sleeping is None:
        return nat_running()
    return not sleeping


def sleep_nat() -> bool:
    """GAMING: park Nat's weights to RAM. Fallback for a pre-sleep-mode Nat
    is now PORT-SCOPED (stop_vllm_port.sh 8003), NOT stop_nat.sh — the latter
    is unscoped and would take the :8004/:8005 embedders down with it (the
    2026-07-03 collateral-kill incident). stop_nat.sh is reserved for
    Electron's whole-app-quit VRAM release only."""
    if not nat_running():
        return True
    if _vllm_sleeping(NAT_BASE) is True:
        return True
    if _vllm_sleep(NAT_BASE, "nat(:8003)"):
        return True
    logger.warning("[gpu.actuators] nat sleep unavailable — port-scoped stop")
    port = os.getenv("NAT_PORT", "8003")
    return _wsl(f"bash {REPO_WSL_ROOT}/scripts/embeddings/stop_vllm_port.sh {port}")


def wake_nat() -> bool:
    """Wake Nat's parked weights (sleep-mode only). The orchestrator NEVER
    cold-starts Nat: its PROCESS is Electron-owned (spawnNat at boot, respawn
    in the desktop reconcile loop), exactly like Chatterbox. Cold-starting it
    here raced Electron's spawn AND tripped 03_serve_nat.sh's cleanup, which
    (pre-fix) killed the embedders. If Nat's process is down, report it — the
    Electron reconcile brings it back."""
    if not nat_running():
        logger.info("[gpu.actuators] Nat process down — deferring cold start to Electron")
        return False
    if _vllm_sleeping(NAT_BASE) is True:
        return _vllm_wake(NAT_BASE, "nat(:8003)")
    return True


# ─── Multimodal vLLM (:8005) — idle-window citizen ──────────────

def mm_worker_running() -> bool:
    return _probe_http(f"{_mm_root()}/health") or _probe_http(f"{_mm_root()}/v1/models")


def mm_worker_awake() -> bool:
    sleeping = _vllm_sleeping(_mm_root())
    if sleeping is None:
        from app.embeddings.local_provider import multimodal_available
        return multimodal_available()
    return not sleeping


def wake_mm_worker() -> bool:
    """Wake from RAM if parked; first-ever use cold-starts (idle window
    only — never an interactive path)."""
    if not mm_worker_running():
        logger.info("[gpu.actuators] cold-starting multimodal embed server (:8005)")
        return _wsl(
            f"setsid nohup bash {REPO_WSL_ROOT}/scripts/embeddings/serve_multimodal.sh "
            f">/root/mm-serve.log 2>&1 < /dev/null & sleep 3",
            detach=True,
        )
    if _vllm_sleeping(_mm_root()) is True:
        return _vllm_wake(_mm_root(), "multimodal(:8005)")
    return True


def sleep_mm_worker() -> bool:
    if not mm_worker_running():
        return True
    if _vllm_sleeping(_mm_root()) is True:
        return True
    if _vllm_sleep(_mm_root(), "multimodal(:8005)"):
        return True
    logger.warning("[gpu.actuators] mm sleep unavailable — falling back to stop")
    port = os.getenv("EMBED_MM_PORT", "8005")
    return _wsl(f"bash {REPO_WSL_ROOT}/scripts/embeddings/stop_vllm_port.sh {port}")


# Back-compat aliases (older call sites / tests)
start_mm_worker = wake_mm_worker
stop_mm_worker = sleep_mm_worker


# ─── Whisper (in-process) — GAMING sweep ─────────────────────────

def unload_whisper() -> None:
    """Best-effort: free the transcription model's VRAM. (In-process torch
    model — model_manager reloads it from disk on next use; small.)"""
    try:
        from app.services.model_manager import get_model_manager
        get_model_manager().unload_model()
    except Exception as exc:
        logger.debug("[gpu.actuators] whisper unload skipped: %s", exc)
