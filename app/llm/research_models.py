# FILE: app/llm/research_models.py
# Purpose: SQLAlchemy model for persisted deep-research runs (the findings pack).
# Called-by: app.llm.research_task, app.llm.research_context, app.db (init_db import)
# Depends-on: app.db.Base
# Last-renovated: 2026-07-01
"""
One row per "Astra, do a deep dive into X". The row IS the working findings
document: claims+sources accumulate round by round, the evidence text grows
as pages are read, and the draft synthesis lands at the end. The idle-task
ledger tracks scheduling; this table owns the research content.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, String, Text

from app.db import Base

RESEARCH_QUEUED = "queued"
RESEARCH_RUNNING = "running"
RESEARCH_PAUSED = "paused"
RESEARCH_COMPLETED = "completed"
RESEARCH_FAILED = "failed"


class ResearchRun(Base):
    __tablename__ = "research_runs"

    id = Column(String(16), primary_key=True)  # short hex id, e.g. "a3f9c2d1e0b4"
    query = Column(Text, nullable=False)
    status = Column(String(20), nullable=False, default=RESEARCH_QUEUED, index=True)
    # Working findings doc: JSON [{claim, source_url, round}]
    findings_json = Column(Text, nullable=True)
    # De-duped sources: JSON [{title, url, credibility_label, source_type}]
    sources_json = Column(Text, nullable=True)
    # Accumulated evidence (page excerpts) — the raw working material.
    evidence_text = Column(Text, nullable=True)
    # Draft synthesis from the local model (the conversational layer speaks
    # its own synthesis from build_research_context, not this verbatim).
    synthesis = Column(Text, nullable=True)
    # JSON {rounds, queries, fetches, active_seconds}
    stats_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
