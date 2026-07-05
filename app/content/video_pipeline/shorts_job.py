# FILE: app/content/video_pipeline/shorts_job.py
# Purpose: Minimal job-state container for the slim shorts pipeline.
# Called-by: app.content.video_pipeline.shorts_orchestrator, app.tools.social_posting_tools
# Depends-on: stdlib only
# Last-renovated: 2026-07-02
"""
ShortsJob — tiny per-run state for the shorts orchestrator.

Deliberately NOT PipelineJob: that one is welded to the longform
PipelineJobRequest (script_text/target_platform/scene plans). A short
has no scene plan, no style profile, no asset resolution — so it gets
its own 3-field-ish state container that serialises to
data/content/output/shorts/{ts}_{slug}/job.json.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SHORTS_OUTPUT_DIR = Path("data/content/output/shorts")


def slugify(text: str, limit: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "short").lower()).strip("-")
    return (s or "short")[:limit].strip("-")


@dataclass
class ShortsJob:
    topic: str
    notes: str = ""
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    stamp: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    status: str = "pending"          # pending|running|complete|error
    stage: str = ""
    # produced by the script stage
    script: str = ""
    caption: str = ""
    hashtags: List[str] = field(default_factory=list)
    title: str = ""
    # produced by later stages
    mp4_path: Optional[str] = None
    captioned_path: Optional[str] = None
    srt_path: Optional[str] = None
    delivered_filename: Optional[str] = None
    duration_s: float = 0.0
    cost_usd: float = 0.0
    output_id: Optional[str] = None  # ContentOutput id (pending -> published)
    permalink: Optional[str] = None
    error: Optional[str] = None
    events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return slugify(self.title or self.topic)

    @property
    def out_dir(self) -> Path:
        path = SHORTS_OUTPUT_DIR / f"{self.stamp}_{self.slug}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def record_event(self, stage: str, status: str, message: str = "", **data) -> Dict[str, Any]:
        self.stage = stage
        ev = {"job_id": self.job_id, "stage": stage, "status": status, "message": message, "data": data}
        self.events.append(ev)
        return ev

    def caption_with_tags(self) -> str:
        tags = " ".join(t if t.startswith("#") else f"#{t}" for t in self.hashtags)
        return f"{self.caption}\n\n{tags}".strip() if tags else (self.caption or "")

    def save(self) -> str:
        path = self.out_dir / "job.json"
        path.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
        return str(path)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
