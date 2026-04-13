# FILE: app/pipeline_v2/integration_runner.py
"""
Cross-Repo Integration Test Runner (Phase 3 Job 11)

Runs cross-repo smoke tests: start the backend target, exercise the
endpoints declared in segment contracts (Phase 1 Job 7), confirm they
respond. Designed for the common case where a multi-target build touches
both a FastAPI backend and an Android/frontend client that consumes it.

Scope boundary: this does NOT run Android instrumented tests (those need
an emulator or device). It DOES catch "endpoint was renamed, declared
wrong, returns 500" class of bugs — which is most cross-repo breakage.

v1.0 (2026-04-12): Phase 3 Job 11.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SmokeTestStatus(str, Enum):
    PASSED = "passed"
    ENDPOINT_FAILED = "endpoint_failed"
    BACKEND_START_FAILED = "backend_start_failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class EndpointCheck:
    endpoint: str
    status_code: Optional[int] = None
    ok: bool = False
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "status_code": self.status_code,
            "ok": self.ok,
            "detail": self.detail,
        }


@dataclass
class IntegrationReport:
    status: SmokeTestStatus = SmokeTestStatus.SKIPPED
    endpoint_checks: List[EndpointCheck] = field(default_factory=list)
    backend_target_id: Optional[str] = None
    detail: str = ""
    duration_sec: float = 0.0

    def is_passing(self) -> bool:
        return self.status == SmokeTestStatus.PASSED

    def summary(self) -> str:
        ok = sum(1 for c in self.endpoint_checks if c.ok)
        n = len(self.endpoint_checks)
        return (
            f"IntegrationReport(status={self.status.value}, "
            f"endpoints={ok}/{n}, backend={self.backend_target_id}, "
            f"duration={self.duration_sec:.1f}s)"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "endpoint_checks": [c.to_dict() for c in self.endpoint_checks],
            "backend_target_id": self.backend_target_id,
            "detail": self.detail,
            "duration_sec": round(self.duration_sec, 2),
        }


def _collect_cross_target_endpoints(manifest) -> List[Dict[str, str]]:
    """Walk the manifest's cross_target_edges (Phase 1 Job 6) and return a
    list of endpoint contracts where backend publishes and client consumes.

    Each entry: {"endpoint": "GET /path", "producer": seg_id, "consumer": seg_id}
    """
    if not manifest or not getattr(manifest, "segments", None):
        return []

    # Index producer endpoints by segment_id
    producer_endpoints: Dict[str, List[str]] = {}
    for seg in manifest.segments:
        if seg.exposes and seg.exposes.endpoint_paths:
            producer_endpoints[seg.segment_id] = list(seg.exposes.endpoint_paths)

    # For each cross-target edge, collect the producer's endpoints that the
    # consumer also declared in its consumes list.
    results: List[Dict[str, str]] = []
    segs_by_id = {s.segment_id: s for s in manifest.segments}

    for edge in getattr(manifest, "cross_target_edges", []):
        producer_id = edge["from"]
        consumer_id = edge["to"]
        prod_seg = segs_by_id.get(producer_id)
        cons_seg = segs_by_id.get(consumer_id)
        if not prod_seg or not cons_seg:
            continue
        if not prod_seg.exposes or not cons_seg.consumes:
            continue
        consumed_paths = set(cons_seg.consumes.endpoint_paths or [])
        for ep in prod_seg.exposes.endpoint_paths or []:
            if ep in consumed_paths:
                results.append({
                    "endpoint": ep,
                    "producer": producer_id,
                    "consumer": consumer_id,
                })
    return results


async def _wait_for_backend(base_url: str, timeout_sec: float = 20.0) -> bool:
    """Poll the backend's /health (or /) until it responds or we time out."""
    import httpx
    end_at = time.monotonic() + timeout_sec
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.monotonic() < end_at:
            for probe_path in ("/health", "/"):
                try:
                    resp = await client.get(base_url + probe_path)
                    if resp.status_code < 500:
                        return True
                except Exception:
                    pass
            await asyncio.sleep(0.5)
    return False


async def _smoke_check_endpoint(base_url: str, endpoint: str) -> EndpointCheck:
    """Parse "METHOD /path" and issue a smoke request. Replaces path params
    with placeholder values so we at least get a response shape, not a 404."""
    import httpx
    parts = endpoint.strip().split(None, 1)
    if len(parts) != 2:
        return EndpointCheck(endpoint=endpoint, ok=False, detail="malformed")
    method, path = parts[0].upper(), parts[1]
    # Replace path params with "1" placeholders
    import re
    resolved = re.sub(r"\{[^}]+\}", "1", path)
    check = EndpointCheck(endpoint=endpoint)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            req = getattr(client, method.lower(), None)
            if req is None:
                check.detail = f"unsupported method {method}"
                return check
            resp = await req(base_url + resolved)
            check.status_code = resp.status_code
            # Accept anything that isn't 404/5xx — 422 for missing body is fine
            check.ok = resp.status_code < 500 and resp.status_code != 404
            if not check.ok:
                body = resp.text[:200]
                check.detail = f"HTTP {resp.status_code}: {body}"
    except Exception as e:
        check.detail = f"request raised: {e}"
    return check


async def run_integration_smoke(manifest, backend_target_id: str = "astra-backend",
                                base_url: str = "http://127.0.0.1:8000") -> IntegrationReport:
    """Run a cross-repo smoke test against the already-running backend.

    Assumes the backend is already running locally (typical ASTRA dev state).
    Walks cross-target endpoints from the manifest and checks each responds.

    For a production flow this would spin up the backend itself; for now it
    relies on the dev server already being up, which matches how Taz uses it.
    """
    report = IntegrationReport(backend_target_id=backend_target_id)
    start = time.monotonic()

    try:
        endpoints = _collect_cross_target_endpoints(manifest)
        if not endpoints:
            report.status = SmokeTestStatus.SKIPPED
            report.detail = "no cross-target endpoint contracts to check"
            report.duration_sec = time.monotonic() - start
            logger.info("[integration] Job 11 skipped: %s", report.detail)
            return report

        backend_up = await _wait_for_backend(base_url, timeout_sec=10)
        if not backend_up:
            report.status = SmokeTestStatus.BACKEND_START_FAILED
            report.detail = f"backend at {base_url} did not respond within 10s"
            report.duration_sec = time.monotonic() - start
            logger.warning("[integration] Job 11 %s", report.detail)
            return report

        # Run endpoint checks in parallel
        check_tasks = [_smoke_check_endpoint(base_url, ep["endpoint"]) for ep in endpoints]
        checks = await asyncio.gather(*check_tasks, return_exceptions=False)
        report.endpoint_checks = list(checks)

        if all(c.ok for c in checks):
            report.status = SmokeTestStatus.PASSED
        else:
            report.status = SmokeTestStatus.ENDPOINT_FAILED
            report.detail = (
                f"{sum(1 for c in checks if not c.ok)} of {len(checks)} endpoints failed"
            )

        report.duration_sec = time.monotonic() - start
        logger.info("[integration] Job 11 %s", report.summary())
        return report

    except Exception as e:
        report.status = SmokeTestStatus.ERROR
        report.detail = f"runner raised: {e}"
        report.duration_sec = time.monotonic() - start
        logger.error("[integration] Job 11 ERROR: %s", e)
        return report