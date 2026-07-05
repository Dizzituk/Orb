# FILE: app/gpu/__init__.py
# Purpose: 4080 VRAM orchestrator package (LANE E) — single owner of GPU residency.
# Called-by: app.router_registry, app.idle.governor (activity hook), app.idle.tasks_visual
# Depends-on: app.gpu.orchestrator
# Last-renovated: 2026-07-02 (LANE E)
"""
4080 VRAM orchestrator (LANE E, Task 4).

Three states, one owner:
  INTERACTIVE       Nat (:8003) + Chatterbox (:8002) + text embedder (:8004)
  BACKGROUND_INGEST Chatterbox unloaded; multimodal worker (:8005) loaded;
                    visual ingest / multimodal reembed drained
  GAMING            everything on the 4080 unloaded (manual toggle); the
                    3090 fleet (when installed) is untouched

Process ownership stays where it already lives: Electron owns Chatterbox
(main.js polls GET /gpu/desired and reconciles), the backend owns the WSL
vLLM instances and the multimodal worker. The RTX 3090 is reserved for the
26B fleet — nothing here may ever load onto it.
"""
from app.gpu.orchestrator import (  # noqa: F401
    get_orchestrator,
    notify_ingest_complete,
    on_user_activity_signal,
    request_background_ingest,
)
