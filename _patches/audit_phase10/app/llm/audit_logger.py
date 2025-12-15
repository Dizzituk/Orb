# FILE: app/llm/audit_logger.py
""" 
Non-sensitive, structured audit logging + lightweight live telemetry.

Design goals
- Append-only JSONL (one line per event).
- Strict allowlist: NO prompts, NO outputs, NO keys/secrets, NO raw file names.
- Correlate everything by request_id (prefer JobEnvelope.job_id).
- Useful for debugging: routing decisions, model calls, tool calls, errors, timings.

Env
- ORB_AUDIT_ENABLED=1|0
- ORB_AUDIT_LOG_DIR=data/logs/audit
- ORB_AUDIT_MAX_RECENT=500

Notes
- This module is designed to be resilient to older/mismatched call-sites.
- Any unexpected fields are dropped by the allowlist sanitizer.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Helpers
# =============================================================================


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _truncate(s: Any, max_len: int) -> str:
    if s is None:
        return ""
    try:
        txt = str(s)
    except Exception:
        txt = repr(s)
    if len(txt) <= max_len:
        return txt
    return txt[: max_len - 1] + "…"


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        if isinstance(v, bool):
            return int(v)
        return int(v)
    except Exception:
        return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, bool):
            return float(int(v))
        return float(v)
    except Exception:
        return default


def _attachment_summary(attachments: Any) -> Dict[str, Any]:
    """Return a non-sensitive attachment summary.

    Accepts a list of AttachmentInfo-like objects or dicts.
    Only counts/types/sizes/extensions.
    """
    if not attachments:
        return {"count": 0, "total_bytes": 0, "by_kind": {}}

    by_kind: Dict[str, int] = {}
    total = 0
    count = 0

    def kind_for(ext: str, mime: str) -> str:
        e = (ext or "").lower()
        m = (mime or "").lower()
        if e in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".svg"}:
            return "image"
        if e in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv", ".flv"}:
            return "video"
        if e == ".pdf":
            return "pdf"
        if e in {".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"}:
            return "office"
        if e in {".txt", ".md"}:
            return "text"
        if e in {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".sql", ".sh", ".bash", ".ps1", ".html", ".css", ".json", ".yaml", ".yml", ".xml", ".toml"}:
            return "code"
        if m.startswith("image/"):
            return "image"
        if m.startswith("video/"):
            return "video"
        if m == "application/pdf":
            return "pdf"
        if m.startswith("text/"):
            return "text"
        return "other"

    for a in attachments or []:
        try:
            if isinstance(a, dict):
                ext = os.path.splitext(str(a.get("filename") or ""))[1]
                mime = str(a.get("mime_type") or "")
                size = _safe_int(a.get("size_bytes"), 0)
            else:
                # AttachmentInfo
                ext = os.path.splitext(str(getattr(a, "filename", "") or ""))[1]
                mime = str(getattr(a, "mime_type", "") or "")
                size = _safe_int(getattr(a, "size_bytes", 0), 0)

            k = kind_for(ext, mime)
            by_kind[k] = by_kind.get(k, 0) + 1
            total += max(0, size)
            count += 1
        except Exception:
            # If anything is weird, just count as unknown
            by_kind["other"] = by_kind.get("other", 0) + 1
            count += 1

    return {"count": count, "total_bytes": total, "by_kind": by_kind}


# =============================================================================
# Allowlist Sanitizer
# =============================================================================

_ALLOWED_TOP_LEVEL = {
    "ts",
    "event",
    "request_id",
    "session_id",
    "project_id",
    "lane",
    "job_type",
    "provider",
    "model",
    "ok",
    "latency_ms",
    "tokens",
    "cost_usd",
    "tool",
    "http",
    "attachments",
    "flags",
    "error",
    "warning",
    "note",
}


def _sanitize_event(ev: Dict[str, Any]) -> Dict[str, Any]:
    """Drop everything except the strict allowlist and hard-truncate strings."""
    out: Dict[str, Any] = {}

    for k, v in (ev or {}).items():
        if k not in _ALLOWED_TOP_LEVEL:
            continue

        if k in {"event", "request_id", "session_id", "lane", "job_type", "provider", "model"}:
            out[k] = _truncate(v, 120)
            continue

        if k == "project_id":
            out[k] = _safe_int(v, 0)
            continue

        if k in {"ok"}:
            out[k] = bool(v)
            continue

        if k in {"latency_ms"}:
            out[k] = _safe_int(v, 0)
            continue

        if k == "cost_usd":
            out[k] = _safe_float(v, 0.0)
            continue

        if k == "tokens" and isinstance(v, dict):
            out[k] = {
                "prompt": _safe_int(v.get("prompt"), 0),
                "completion": _safe_int(v.get("completion"), 0),
                "total": _safe_int(v.get("total"), 0),
            }
            continue

        if k == "tool" and isinstance(v, dict):
            out[k] = {
                "name": _truncate(v.get("name"), 80),
                "version": _truncate(v.get("version"), 40),
                "ok": bool(v.get("ok", True)),
            }
            if "elapsed_ms" in v:
                out[k]["elapsed_ms"] = _safe_int(v.get("elapsed_ms"), 0)
            # optional small ints
            if "result_bytes" in v:
                out[k]["result_bytes"] = _safe_int(v.get("result_bytes"), 0)
            if "result_count" in v:
                out[k]["result_count"] = _safe_int(v.get("result_count"), 0)
            continue

        if k == "http" and isinstance(v, dict):
            out[k] = {
                "host": _truncate(v.get("host"), 120),
                "status_code": _safe_int(v.get("status_code"), 0),
                "truncated": bool(v.get("truncated", False)),
            }
            continue

        if k == "attachments" and isinstance(v, dict):
            by_kind = v.get("by_kind") if isinstance(v.get("by_kind"), dict) else {}
            out[k] = {
                "count": _safe_int(v.get("count"), 0),
                "total_bytes": _safe_int(v.get("total_bytes"), 0),
                "by_kind": { _truncate(kk, 30): _safe_int(vv, 0) for kk, vv in list(by_kind.items())[:20] },
            }
            continue

        if k == "flags" and isinstance(v, dict):
            # flags are always boolean-ish
            out[k] = { _truncate(kk, 40): bool(vv) for kk, vv in list(v.items())[:40] }
            continue

        if k == "error" and isinstance(v, dict):
            out[k] = {
                "type": _truncate(v.get("type"), 80),
                "message": _truncate(v.get("message"), 400),
            }
            return out if out.get("event") == "ERROR" else out

        if k == "warning" and isinstance(v, dict):
            out[k] = {
                "type": _truncate(v.get("type"), 80),
                "message": _truncate(v.get("message"), 300),
            }
            continue

        if k == "note":
            out[k] = _truncate(v, 400)
            continue

        # fallback: best-effort JSON-safe
        try:
            json.dumps(v)
            out[k] = v
        except Exception:
            out[k] = _truncate(v, 200)

    # ensure timestamp exists
    if "ts" not in out:
        out["ts"] = _utc_iso()

    return out


# =============================================================================
# Telemetry (in-memory)
# =============================================================================


@dataclass
class TelemetrySnapshot:
    ok: bool
    window_s: int
    counts: Dict[str, int]
    by_provider: Dict[str, Dict[str, int]]
    avg_latency_ms: int
    in_memory_events: int


class _TelemetryStore:
    def __init__(self, max_recent: int = 500) -> None:
        self._lock = threading.Lock()
        self._recent: List[Dict[str, Any]] = []
        self._max_recent = max(50, int(max_recent))

        self._counts: Dict[str, int] = {}
        self._by_provider: Dict[str, Dict[str, int]] = {}
        self._lat_sum = 0
        self._lat_n = 0

    def record(self, ev: Dict[str, Any]) -> None:
        with self._lock:
            self._recent.append(ev)
            if len(self._recent) > self._max_recent:
                self._recent = self._recent[-self._max_recent :]

            et = str(ev.get("event") or "")
            self._counts[et] = self._counts.get(et, 0) + 1

            prov = str(ev.get("provider") or "")
            if prov:
                p = self._by_provider.setdefault(prov, {})
                p[et] = p.get(et, 0) + 1

            lat = _safe_int(ev.get("latency_ms"), 0)
            if lat > 0:
                self._lat_sum += lat
                self._lat_n += 1

    def snapshot(self) -> TelemetrySnapshot:
        with self._lock:
            avg = int(self._lat_sum / self._lat_n) if self._lat_n else 0
            return TelemetrySnapshot(
                ok=True,
                window_s=0,
                counts=dict(self._counts),
                by_provider={k: dict(v) for k, v in self._by_provider.items()},
                avg_latency_ms=avg,
                in_memory_events=len(self._recent),
            )

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        lim = max(1, min(500, int(limit)))
        with self._lock:
            return list(self._recent[-lim:])


# =============================================================================
# Public API
# =============================================================================


class AuditEventType(str, Enum):
    REQUEST_START = "REQUEST_START"
    ROUTING_DECISION = "ROUTING_DECISION"
    MODEL_CALL = "MODEL_CALL"
    TOOL_CALL = "TOOL_CALL"
    WARNING = "WARNING"
    ERROR = "ERROR"
    TRACE_END = "TRACE_END"


@dataclass
class RoutingTrace:
    """A per-request helper that emits events."""

    logger: "AuditLogger"
    request_id: str
    session_id: str
    project_id: int
    started_at_ms: int = field(default_factory=_now_ms)
    is_critical: bool = False
    sandbox_mode: bool = False

    def _emit(self, event_type: AuditEventType, payload: Dict[str, Any]) -> None:
        ev: Dict[str, Any] = {
            "ts": _utc_iso(),
            "event": event_type.value,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "flags": {
                "critical": bool(self.is_critical),
                "sandbox": bool(self.sandbox_mode),
            },
        }
        ev.update(payload or {})
        self.logger.emit(ev)

    # ---- convenience methods ----

    def log_request_start(
        self,
        *,
        job_type: Optional[str] = None,
        resolved_job_type: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        reason: Optional[str] = None,
        frontier_override: bool = False,
        file_map_injected: bool = False,
        attachments: Any = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "job_type": resolved_job_type or job_type or "",
            "provider": provider or "",
            "model": model or "",
            "attachments": _attachment_summary(attachments),
            "flags": {
                "critical": bool(self.is_critical),
                "sandbox": bool(self.sandbox_mode),
                "frontier_override": bool(frontier_override),
                "file_map_injected": bool(file_map_injected),
            },
        }
        if reason:
            payload["note"] = _truncate(reason, 240)
        self._emit(AuditEventType.REQUEST_START, payload)

    def log_routing_decision(
        self,
        *,
        job_type: str,
        provider: str,
        model: str,
        reason: str = "",
        frontier_override: bool = False,
        file_map_injected: bool = False,
    ) -> None:
        payload: Dict[str, Any] = {
            "job_type": job_type,
            "provider": provider,
            "model": model,
            "flags": {
                "frontier_override": bool(frontier_override),
                "file_map_injected": bool(file_map_injected),
            },
        }
        if reason:
            payload["note"] = _truncate(reason, 240)
        self._emit(AuditEventType.ROUTING_DECISION, payload)

    def log_model_call(self, *args, **kwargs) -> None:
        """Resilient model-call logger.

        Supports BOTH:
          - log_model_call(task_id, provider, model, role, input_tokens, output_tokens, duration_ms, success=True, error=None, cost_usd=None)
          - legacy/mismatched calls seen in router.py:
                log_model_call(task_id, provider, model, prompt_tokens, completion_tokens, cost_estimate)
        """

        # Parse positional usage
        lane = "primary"
        provider = ""
        model = ""
        role = ""
        prompt_tokens = 0
        completion_tokens = 0
        duration_ms = 0
        ok = True
        error_msg = ""
        cost = 0.0

        if len(args) >= 3:
            lane = str(args[0])
            provider = str(args[1])
            model = str(args[2])

        # Mismatched legacy: (lane, provider, model, prompt_toks, completion_toks, cost)
        if len(args) == 6 and isinstance(args[3], (int, float)) and isinstance(args[4], (int, float)):
            prompt_tokens = _safe_int(args[3], 0)
            completion_tokens = _safe_int(args[4], 0)
            cost = _safe_float(args[5], 0.0)
        else:
            # Canonical signature in kwargs/positional
            if len(args) >= 4 and isinstance(args[3], str):
                role = str(args[3])
            prompt_tokens = _safe_int(kwargs.get("input_tokens"), _safe_int(args[4], 0) if len(args) >= 5 else 0)
            completion_tokens = _safe_int(kwargs.get("output_tokens"), _safe_int(args[5], 0) if len(args) >= 6 else 0)
            duration_ms = _safe_int(kwargs.get("duration_ms"), _safe_int(args[6], 0) if len(args) >= 7 else 0)
            ok = bool(kwargs.get("success", True))
            err = kwargs.get("error")
            if err:
                error_msg = _truncate(err, 300)
            cost = _safe_float(kwargs.get("cost_usd"), 0.0)

        total = prompt_tokens + completion_tokens
        payload: Dict[str, Any] = {
            "lane": lane,
            "provider": provider,
            "model": model,
            "ok": ok,
            "latency_ms": duration_ms,
            "tokens": {"prompt": prompt_tokens, "completion": completion_tokens, "total": total},
            "cost_usd": cost,
        }
        if role:
            payload["note"] = f"role={_truncate(role, 40)}"
        if (not ok) or error_msg:
            payload["error"] = {"type": "MODEL_CALL", "message": error_msg or "model call failed"}
        self._emit(AuditEventType.MODEL_CALL, payload)

    def log_tool_call(
        self,
        *,
        tool_name: str,
        tool_version: str = "v1",
        ok: bool,
        latency_ms: int,
        result_bytes: int = 0,
        result_count: int = 0,
        http_host: str = "",
        http_status_code: int = 0,
        http_truncated: bool = False,
    ) -> None:
        payload: Dict[str, Any] = {
            "ok": bool(ok),
            "latency_ms": _safe_int(latency_ms, 0),
            "tool": {
                "name": tool_name,
                "version": tool_version,
                "ok": bool(ok),
                "result_bytes": _safe_int(result_bytes, 0),
                "result_count": _safe_int(result_count, 0),
            },
        }
        if http_host:
            payload["http"] = {
                "host": http_host,
                "status_code": _safe_int(http_status_code, 0),
                "truncated": bool(http_truncated),
            }
        self._emit(AuditEventType.TOOL_CALL, payload)

    def log_warning(self, *args) -> None:
        """Supports:
          - log_warning(task_id, warning_type, message)
          - log_warning(warning_type, message)
        """
        warning_type = "WARNING"
        message = ""
        lane = ""
        if len(args) == 2:
            warning_type = str(args[0])
            message = str(args[1])
        elif len(args) >= 3:
            lane = str(args[0])
            warning_type = str(args[1])
            message = str(args[2])

        payload: Dict[str, Any] = {
            "lane": lane or "",
            "warning": {"type": warning_type, "message": _truncate(message, 300)},
        }
        self._emit(AuditEventType.WARNING, payload)

    def log_error(self, *args, **kwargs) -> None:
        """Supports:
          - log_error(task_id, error_type, error_message)
          - log_error(error_type, error_message)
        """
        lane = ""
        error_type = "ERROR"
        message = ""
        if len(args) == 2:
            error_type = str(args[0])
            message = str(args[1])
        elif len(args) >= 3:
            lane = str(args[0])
            error_type = str(args[1])
            message = str(args[2])
        if kwargs.get("error_message"):
            message = str(kwargs.get("error_message"))
        payload: Dict[str, Any] = {
            "lane": lane or "",
            "error": {"type": error_type, "message": _truncate(message, 400)},
            "ok": False,
        }
        self._emit(AuditEventType.ERROR, payload)

    def finalize(self, *, success: bool = True, error_message: str = "") -> Dict[str, Any]:
        duration = _now_ms() - self.started_at_ms
        payload: Dict[str, Any] = {
            "ok": bool(success),
            "latency_ms": int(duration),
        }
        if (not success) and error_message:
            payload["error"] = {"type": "TRACE_END", "message": _truncate(error_message, 300)}
        self._emit(AuditEventType.TRACE_END, payload)
        return {"request_id": self.request_id, "ok": bool(success), "duration_ms": int(duration)}


class AuditLogger:
    def __init__(self) -> None:
        self.enabled = str(os.getenv("ORB_AUDIT_ENABLED", "1")).lower() in {"1", "true", "yes"}
        self.log_dir = os.getenv("ORB_AUDIT_LOG_DIR", "data/logs/audit")
        self.max_recent = int(os.getenv("ORB_AUDIT_MAX_RECENT", "500") or "500")

        self._lock = threading.Lock()
        self._telemetry = _TelemetryStore(max_recent=self.max_recent)

        if self.enabled:
            os.makedirs(self.log_dir, exist_ok=True)

    def _log_path(self) -> str:
        # daily file for minimal rollover
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"audit_{day}.jsonl")

    def emit(self, raw_event: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        ev = _sanitize_event(raw_event)

        # Always ensure request_id exists
        if not ev.get("request_id"):
            ev["request_id"] = "unknown"

        line = json.dumps(ev, ensure_ascii=False)

        with self._lock:
            # append-only write
            path = self._log_path()
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._telemetry.record(ev)

    def start_trace(
        self,
        session_id: str,
        project_id: int,
        user_text: Optional[str] = None,  # intentionally ignored
        is_critical: bool = False,
        sandbox_mode: bool = False,
        request_id: Optional[str] = None,
    ) -> RoutingTrace:
        rid = request_id or f"req-{_now_ms()}"
        sid = session_id or "session-unknown"
        pid = int(project_id or 0)

        trace = RoutingTrace(
            logger=self,
            request_id=str(rid),
            session_id=str(sid),
            project_id=pid,
            is_critical=bool(is_critical),
            sandbox_mode=bool(sandbox_mode),
        )

        # Emit a minimal start marker (no prompt).
        trace._emit(
            AuditEventType.REQUEST_START,
            {
                "job_type": "",
                "attachments": {"count": 0, "total_bytes": 0, "by_kind": {}},
                "flags": {"critical": bool(is_critical), "sandbox": bool(sandbox_mode)},
                "note": f"user_text_len={len(user_text or '')}",
            },
        )
        return trace

    def complete_trace(self, trace: RoutingTrace, *, success: bool = True, error_message: str = "") -> None:
        if not trace:
            return
        trace.finalize(success=success, error_message=error_message)

    def get_metrics(self) -> Dict[str, Any]:
        snap = self._telemetry.snapshot()
        return {
            "ok": snap.ok,
            "counts": snap.counts,
            "by_provider": snap.by_provider,
            "avg_latency_ms": snap.avg_latency_ms,
            "in_memory_events": snap.in_memory_events,
        }

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._telemetry.recent(limit=limit)


# =============================================================================
# Singleton
# =============================================================================


_AUDIT_SINGLETON: Optional[AuditLogger] = None


def get_audit_logger() -> Optional[AuditLogger]:
    global _AUDIT_SINGLETON
    if _AUDIT_SINGLETON is None:
        _AUDIT_SINGLETON = AuditLogger()
    return _AUDIT_SINGLETON
