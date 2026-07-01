# FILE: app/llm/research_task.py
# Purpose: Ledger-run deep research — budgeted, lead-chasing, checkpointed grind on the local model.
# Called-by: app.idle.router (import registers task), app.tools.registry (register_research_tools)
# Depends-on: app.llm.deep_research_engine (phase functions), app.llm.research_models, app.idle.*
# Last-renovated: 2026-07-01
"""
"Astra, do a deep dive into X" queues a ResearchRun + an idle-ledger task.
When the desk goes quiet the governor grinds it: plan -> search -> read whole
pages -> extract claims into the findings doc -> chase leads -> gap-check ->
repeat, until the depth budget is spent or the gaps close. Every LLM call
rides execution_context="background_local" (registry refuses cloud); search
is Brave/DDG (whitelisted, non-LLM).

Depth budget (env): DEEP_RESEARCH_MAX_FETCHES (default 25) and
DEEP_RESEARCH_MAX_MINUTES of ACTIVE grind time (default 20) replace the
fixed pass count; DEEP_RESEARCH_MAX_ROUNDS_BACKGROUND (default 8) is a
safety rail. Follow-up leads are capped per round
(DEEP_RESEARCH_MAX_LEADS_PER_ROUND, default 3).

A user message checkpoints between rounds — state persists on the run row +
ledger progress and the next idle window resumes mid-research. Completion
raises a low-severity sentinel nudge ("findings pack ready").
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from app.idle.router import TaskContext, TaskOutcome, register_task_handler
from app.llm.research_models import (
    RESEARCH_COMPLETED,
    RESEARCH_FAILED,
    RESEARCH_PAUSED,
    RESEARCH_RUNNING,
    ResearchRun,
)

logger = logging.getLogger(__name__)

RESEARCH_TASK = "deep_research"

_EVIDENCE_CHAR_CAP = 200_000  # run-row working-doc bound; excerpts beyond this are dropped


def _max_fetches() -> int:
    try:
        return int(os.getenv("DEEP_RESEARCH_MAX_FETCHES", "25"))
    except Exception:
        return 25


def _max_active_seconds() -> float:
    try:
        return float(os.getenv("DEEP_RESEARCH_MAX_MINUTES", "20")) * 60.0
    except Exception:
        return 1200.0


def _max_leads_per_round() -> int:
    try:
        return int(os.getenv("DEEP_RESEARCH_MAX_LEADS_PER_ROUND", "3"))
    except Exception:
        return 3


def _max_rounds() -> int:
    try:
        return int(os.getenv("DEEP_RESEARCH_MAX_ROUNDS_BACKGROUND", "8"))
    except Exception:
        return 8


# ── queueing ────────────────────────────────────────────────────────────────


def create_research_run(db: Session, query: str) -> ResearchRun:
    """Persist the run and queue the grind on the idle ledger."""
    from app.idle import ledger

    run = ResearchRun(id=uuid.uuid4().hex[:12], query=query.strip())
    db.add(run)
    db.commit()
    ledger.enqueue(db, RESEARCH_TASK, {"research_id": run.id})
    logger.info("[research] queued deep research %s: %s", run.id, query[:120])
    return run


# ── local-model extraction (leads + claims) ─────────────────────────────────

_LEADS_SYSTEM = """You are a research lead-chaser. From the evidence, propose NEW
follow-up web search queries that chase concrete leads — named entities,
referenced documents, events, or numbers worth verifying. Rules:
- Only leads NOT already covered by the evidence
- Short queries (3-8 words)
- Return ONLY a JSON array of strings"""

_CLAIMS_SYSTEM = """You extract factual claims for a findings ledger. From the
evidence, list the most important factual claims relevant to the question.
Rules:
- Each claim one sentence, concrete and checkable
- Attach the source URL it came from
- At most 6 claims
- Return ONLY a JSON array: [{"claim": "...", "source_url": "..."}]"""


def _parse_json_array(raw: str) -> list:
    try:
        match = re.search(r"\[.*\]", raw or "", re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return parsed
    except Exception:
        pass
    return []


async def _extract_leads(question: str, evidence: str, cap: int) -> List[str]:
    from app.llm import deep_research_engine as engine

    raw = await engine._llm_call(
        f"Question: {question}\n\nEvidence:\n{evidence[:6000]}\n\nPropose follow-up queries.",
        _LEADS_SYSTEM, max_tokens=200, local_only=True,
    )
    leads = [str(q).strip() for q in _parse_json_array(raw) if str(q).strip()]
    return leads[: max(0, cap)]


async def _extract_claims(question: str, evidence: str) -> List[dict]:
    from app.llm import deep_research_engine as engine

    raw = await engine._llm_call(
        f"Question: {question}\n\nEvidence:\n{evidence[:6000]}\n\nExtract the claims.",
        _CLAIMS_SYSTEM, max_tokens=400, local_only=True,
    )
    out = []
    for item in _parse_json_array(raw):
        if isinstance(item, dict) and item.get("claim"):
            out.append({"claim": str(item["claim"])[:400], "source_url": str(item.get("source_url", ""))[:300]})
    return out


# ── the idle task ───────────────────────────────────────────────────────────


def _load_list(raw: Optional[str]) -> list:
    try:
        v = json.loads(raw or "[]")
        return v if isinstance(v, list) else []
    except Exception:
        return []


async def deep_research_handler(ctx: TaskContext) -> TaskOutcome:
    from app.llm import deep_research_engine as engine

    research_id = str(ctx.params.get("research_id") or "")
    run = ctx.db.query(ResearchRun).filter(ResearchRun.id == research_id).first()
    if run is None:
        return TaskOutcome.failed(f"research run {research_id!r} not found")

    progress = ctx.load_progress()
    round_num = int(progress.get("round_num") or 1)
    prev_gaps = str(progress.get("prev_gaps") or "")
    pending_leads = list(progress.get("pending_leads") or [])
    fetches_used = int(progress.get("fetches_used") or 0)
    active_seconds = float(progress.get("active_seconds") or 0.0)
    total_queries = int(progress.get("total_queries") or 0)

    findings = _load_list(run.findings_json)
    sources = _load_list(run.sources_json)
    seen_claims = {f.get("claim") for f in findings}
    seen_urls = {s.get("url") for s in sources}

    run.status = RESEARCH_RUNNING
    ctx.db.commit()

    def _persist(paused: bool) -> None:
        run.findings_json = json.dumps(findings, ensure_ascii=False)
        run.sources_json = json.dumps(sources, ensure_ascii=False)
        run.stats_json = json.dumps({
            "rounds": round_num - 1, "queries": total_queries,
            "fetches": fetches_used, "active_seconds": round(active_seconds, 1),
        })
        if paused:
            run.status = RESEARCH_PAUSED
        ctx.db.commit()
        ctx.save_progress({
            "round_num": round_num, "prev_gaps": prev_gaps, "pending_leads": pending_leads,
            "fetches_used": fetches_used, "active_seconds": active_seconds,
            "total_queries": total_queries,
        })

    try:
        while round_num <= _max_rounds():
            if fetches_used >= _max_fetches() or active_seconds >= _max_active_seconds():
                logger.info("[research] %s: depth budget spent (%d fetches, %.0fs active)",
                            research_id, fetches_used, active_seconds)
                break
            if ctx.should_yield():
                _persist(paused=True)
                return TaskOutcome.paused(f"checkpointed before round {round_num}")

            t0 = time.monotonic()

            planned = await engine._plan_queries(run.query, "", round_num, prev_gaps, local_only=True)
            queries: List[str] = []
            for q in pending_leads + planned:  # leads first — that's the digging
                if q and q not in queries:
                    queries.append(q)
            queries = queries[: engine.MAX_QUERIES_PER_ROUND]
            pending_leads = []
            total_queries += len(queries)

            fetch_budget = min(5, _max_fetches() - fetches_used)
            round_sources, evidence = await engine._search_and_collect(queries, fetch_budget)
            fetches_used += sum(1 for s in round_sources if s.get("fetched"))

            for s in round_sources:
                if s.get("url") and s["url"] not in seen_urls:
                    seen_urls.add(s["url"])
                    sources.append({
                        "title": s.get("title", ""), "url": s["url"],
                        "credibility_label": s.get("credibility_label", ""),
                        "source_type": s.get("source_type", ""),
                    })

            if evidence:
                existing = run.evidence_text or ""
                if len(existing) < _EVIDENCE_CHAR_CAP:
                    run.evidence_text = f"{existing}\n\n--- Round {round_num} ---\n{evidence}"[:_EVIDENCE_CHAR_CAP]

                for claim in await _extract_claims(run.query, evidence):
                    if claim["claim"] not in seen_claims:
                        seen_claims.add(claim["claim"])
                        findings.append({**claim, "round": round_num})

                pending_leads = await _extract_leads(run.query, evidence, _max_leads_per_round())

                gaps = await engine._analyse_gaps(run.query, run.evidence_text or evidence, local_only=True)
                active_seconds += time.monotonic() - t0
                round_num += 1
                if "NO_GAPS" in (gaps or "").upper() and not pending_leads:
                    _persist(paused=False)
                    break
                prev_gaps = gaps or prev_gaps
                _persist(paused=False)
            else:
                active_seconds += time.monotonic() - t0
                round_num += 1
                _persist(paused=False)
                if not queries:
                    break

        # Synthesis unit (checkpointable like any other unit).
        if ctx.should_yield():
            _persist(paused=True)
            return TaskOutcome.paused("checkpointed before synthesis")
        if run.evidence_text:
            t0 = time.monotonic()
            run.synthesis = await engine._synthesise(run.query, run.evidence_text[:40_000], len(sources), local_only=True)
            active_seconds += time.monotonic() - t0

        run.status = RESEARCH_COMPLETED
        run.completed_at = datetime.utcnow()
        _persist(paused=False)
        _flag_completion(ctx.db, run, len(findings), len(sources))

        summary = (
            f"deep research {run.id} \"{run.query[:80]}\": {len(findings)} findings from "
            f"{len(sources)} sources, {round_num - 1} round(s), {fetches_used} pages, "
            f"{active_seconds / 60:.1f} min active"
        )
        return TaskOutcome.completed(summary=summary, coverage=f"{total_queries} queries, {fetches_used} pages read")
    except Exception as exc:
        logger.exception("[research] %s crashed", research_id)
        run.status = RESEARCH_FAILED
        ctx.db.commit()
        return TaskOutcome.failed(str(exc))


def _flag_completion(db: Session, run: ResearchRun, findings_count: int, sources_count: int) -> None:
    """Completion flag via the sentinel nudge pattern — surfaces, never interrupts."""
    try:
        from app.sentinel.alerts import create_alert, refresh_active_file

        create_alert(
            db,
            severity="low",
            rule_key=f"research_ready_{run.id}",
            title=f"Deep research ready: {run.query[:80]}",
            explanation=(
                f"The background dig finished — {findings_count} findings from "
                f"{sources_count} sources. Ask me about it and I'll walk you through."
            ),
            recommended_action="watch",
        )
        refresh_active_file(db)
    except Exception as exc:
        logger.warning("[research] completion flag failed for %s: %s", run.id, exc)


register_task_handler(RESEARCH_TASK, deep_research_handler)


# ── chat tools ──────────────────────────────────────────────────────────────


async def start_deep_research_handler(input_data: dict, context=None) -> dict:
    from app.db import get_db_session

    query = str(input_data.get("query") or "").strip()
    if len(query) < 3:
        return {"ok": False, "error": "give me a real research question"}
    db = get_db_session()
    try:
        run = create_research_run(db, query)
        return {
            "ok": True,
            "research_id": run.id,
            "status": "queued",
            "detail": (
                "Queued. I'll grind on it in the background next time things go "
                "quiet, and flag you when the findings pack is ready."
            ),
        }
    finally:
        db.close()


async def get_research_findings_handler(input_data: dict, context=None) -> dict:
    from app.db import get_db_session
    from app.llm.research_context import build_research_context, latest_completed_research_id

    db = get_db_session()
    try:
        research_id = str(input_data.get("research_id") or "").strip()
        if not research_id:
            research_id = latest_completed_research_id(db) or ""
        if not research_id:
            return {"ok": False, "error": "no completed research runs yet"}
        run = db.query(ResearchRun).filter(ResearchRun.id == research_id).first()
        if run is None:
            return {"ok": False, "error": f"research {research_id!r} not found"}
        if run.status != RESEARCH_COMPLETED:
            return {
                "ok": True, "research_id": research_id, "status": run.status,
                "detail": "still grinding — the findings pack isn't ready yet",
                "stats": json.loads(run.stats_json or "{}"),
            }
        block = await build_research_context(research_id, db)
        return {"ok": True, "research_id": research_id, "status": run.status, "findings_block": block}
    finally:
        db.close()


def register_research_tools() -> None:
    """Called once from registry._register_defaults at module load."""
    from app.tools.registry import ToolDefinition, register_tool

    register_tool(ToolDefinition(
        name="start_deep_research",
        version="v1",
        description=(
            "Queue a deep background research dig on a topic. It runs while the "
            "user is idle (local model + web search, bounded by a depth budget) "
            "and flags when the findings pack is ready. Use when the user says "
            "'do a deep dive into X', 'research X properly', 'dig into X in the "
            "background'. NOT for quick lookups — use web_search for those."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The research question, self-contained."},
            },
            "required": ["query"],
        },
        output_schema={"type": "object"},
        handler=start_deep_research_handler,
    ))

    register_tool(ToolDefinition(
        name="get_research_findings",
        version="v1",
        description=(
            "Fetch the findings pack from a completed background deep-research "
            "run (claims with sources + draft synthesis + freshness). Use when "
            "the user asks what the research found, or after a research-ready "
            "flag. Synthesise the findings in your own voice, cite the sources, "
            "and mention when the research ran. Omit research_id for the most "
            "recent completed run."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "research_id": {"type": "string", "description": "Optional specific run id."},
            },
            "required": [],
        },
        output_schema={"type": "object"},
        handler=get_research_findings_handler,
    ))
