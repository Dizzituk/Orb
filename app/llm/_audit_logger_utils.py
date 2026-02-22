from __future__ import annotations
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)


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

def get_audit_logger() -> Optional[AuditLogger]:
    global _AUDIT_SINGLETON
    if _AUDIT_SINGLETON is None:
        _AUDIT_SINGLETON = AuditLogger()
    return _AUDIT_SINGLETON
