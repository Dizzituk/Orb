# FILE: app/gpu/nvsmi.py
# Purpose: nvidia-smi query helper — VRAM map for the GPU state endpoint.
# Called-by: app.gpu.orchestrator, app.gpu.router
# Depends-on: nvidia-smi on PATH (stdlib only)
# Last-renovated: 2026-07-02 (LANE E)
"""Thin nvidia-smi wrapper. Per-process VRAM is [N/A] on Windows driver
builds, so the map is per-GPU totals plus the orchestrator's own per-service
health probes — that is still enough to verify 'VRAM returns to baseline'
per cycle (acceptance 4)."""
from __future__ import annotations

import logging
import subprocess
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_TIMEOUT = 5


def query_gpus() -> List[Dict]:
    """One dict per GPU: index, name, total/used/free MiB. [] on failure."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=_TIMEOUT,
        )
        if out.returncode != 0:
            return []
        gpus = []
        for line in (out.stdout or "").strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "total_mib": int(parts[2]),
                    "used_mib": int(parts[3]),
                    "free_mib": int(parts[4]),
                })
        return gpus
    except Exception as exc:
        logger.debug("[nvsmi] query failed: %s", exc)
        return []


def find_gpu(name_substring: str) -> Optional[Dict]:
    """First GPU whose name contains the substring (e.g. '4080')."""
    needle = (name_substring or "").lower()
    for gpu in query_gpus():
        if needle in gpu["name"].lower():
            return gpu
    return None
