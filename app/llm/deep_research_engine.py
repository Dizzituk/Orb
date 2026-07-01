# FILE: app/llm/deep_research_engine.py
# Purpose: Deep Research Engine — Iterative multi-search research for user queries.
# Called-by: app.llm.deep_research_stream
# Depends-on: app.llm.web_search, app.providers.registry, app.tools.registry, app.tools.source_classifier
# Last-renovated: 2026-06-11
"""
Deep Research Engine — Iterative multi-search research for user queries.

Unlike the single-shot web search (web_search.py), this engine runs
an iterative loop:

    1. PLAN    — LLM generates 3-5 search queries from the user's question
    2. SEARCH  — Run all queries via Brave/DDG
    3. FETCH   — Fetch top pages for evidence
    4. TAG     — Classify source credibility
    5. GAP     — LLM identifies what's still missing
    6. REPEAT  — Generate follow-up queries and loop (max 3 rounds)
    7. SYNTH   — Final synthesis from all gathered evidence

Cost controls:
    - Max 3 research rounds
    - Max 5 queries per round (15 total)
    - Max 10 pages fetched total
    - Evidence budget: 12000 chars per round

Dependencies:
    - app.llm.web_search (search_and_answer, page fetching)
    - app.tools.source_classifier (credibility tagging)
    - app.tools.registry (web_search tool)

v2.1 (2026-02): Initial implementation — Capability #4.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =========================================================================
# Configuration
# =========================================================================

MAX_ROUNDS = int(os.getenv("DEEP_RESEARCH_MAX_ROUNDS", "3"))
MAX_QUERIES_PER_ROUND = 5
MAX_TOTAL_PAGES = 10
# v2.2 (2026-07-01): env-tunable evidence budget (was a bare constant).
EVIDENCE_BUDGET_PER_ROUND = int(os.getenv("DEEP_RESEARCH_EVIDENCE_PER_ROUND", "12000"))  # chars
SYNTHESIS_MAX_TOKENS = 1500
PLANNER_MAX_TOKENS = 500
GAP_ANALYSIS_MAX_TOKENS = 400


# =========================================================================
# Data Models
# =========================================================================

class DeepResearchRequest(BaseModel):
    """User request for deep research."""
    query: str = Field(..., min_length=3)
    max_rounds: int = Field(MAX_ROUNDS, ge=1, le=5)
    context: str = ""  # Optional additional context


@dataclass
class ResearchRound:
    """Results from a single research round."""
    round_number: int
    queries: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)  # title, url, snippet, credibility
    evidence_text: str = ""
    gap_analysis: str = ""
    pages_fetched: int = 0


@dataclass
class DeepResearchResult:
    """Complete deep research output."""
    query: str
    synthesis: str = ""
    rounds: list[ResearchRound] = field(default_factory=list)
    all_sources: list[dict] = field(default_factory=list)
    diversity_score: float = 0.0
    missing_perspectives: list[str] = field(default_factory=list)
    total_queries: int = 0
    total_pages_fetched: int = 0
    total_rounds: int = 0
    elapsed_seconds: float = 0.0
    error: str = ""


# =========================================================================
# LLM helper
# =========================================================================

async def _llm_call(prompt: str, system: str, max_tokens: int = 500,
                    local_only: bool = False) -> str:
    """Make a quick LLM call for planning/gap analysis.

    local_only=True (v2.2, 2026-07-01): route through the local lane with
    execution_context="background_local" — the idle-ledger research task
    grinds on the local model and the registry refuses any cloud provider.
    """
    try:
        from app.providers.registry import llm_call as registry_llm_call, is_provider_available

        if local_only:
            result = await registry_llm_call(
                provider_id=None,
                model_id="",  # resolves from LOCAL_LLM_DEFAULT_MODEL / NAT_MODEL
                messages=[{"role": "user", "content": prompt}],
                system_prompt=system,
                temperature=0.3,
                max_tokens=max_tokens,
                execution_context="background_local",
                stage="deep_research",
            )
            return result.content if hasattr(result, "content") else str(result or "")

        # Interactive path: cheapest available provider, model ids from env
        # only (v2.2: literal fallbacks removed — DEFAULT_MODEL is the last
        # resort; a provider with no configured model is skipped).
        for pid, model_env in [
            ("openai", "OPENAI_DEFAULT_MODEL"),
            ("anthropic", "ANTHROPIC_DEFAULT_MODEL"),
            ("google", "GEMINI_FRONTIER_MODEL_ID"),
        ]:
            try:
                if is_provider_available(pid):
                    model = os.getenv(model_env) or os.getenv("DEFAULT_MODEL") or ""
                    if not model:
                        logger.warning("[deep_research] no model configured for %s (%s/DEFAULT_MODEL); skipping", pid, model_env)
                        continue
                    result = await registry_llm_call(
                        provider_id=pid,
                        model_id=model,
                        messages=[{"role": "user", "content": prompt}],
                        system_prompt=system,
                        temperature=0.3,
                        max_tokens=max_tokens,
                        enable_web_search=False,
                    )
                    if isinstance(result, str):
                        return result
                    if hasattr(result, "content"):
                        return result.content
                    return str(result)
            except Exception as e:
                logger.warning("[deep_research] LLM call failed on %s: %s", pid, e)
                continue
    except ImportError:
        pass
    return ""


# =========================================================================
# Phase 1: Query Planning
# =========================================================================

PLANNER_SYSTEM = """You are a research query planner. Given a user's question,
generate 3-5 specific web search queries that would gather comprehensive
information to answer it thoroughly.

Rules:
- Each query should target a DIFFERENT aspect of the question
- Include at least one query for primary/official sources
- Include at least one query for recent news or developments
- Keep queries short (3-8 words each)
- Return ONLY a JSON array of strings, nothing else

Example: ["query one", "query two", "query three"]"""


async def _plan_queries(question: str, context: str = "", round_num: int = 1,
                        previous_gaps: str = "", local_only: bool = False) -> list[str]:
    """Generate search queries for a research round."""
    prompt_parts = [f"Research question: {question}"]
    if context:
        prompt_parts.append(f"Additional context: {context}")
    if round_num > 1 and previous_gaps:
        prompt_parts.append(f"Round {round_num}. Previous research found gaps:\n{previous_gaps}")
        prompt_parts.append("Generate queries to fill ONLY the gaps above.")

    raw = await _llm_call("\n".join(prompt_parts), PLANNER_SYSTEM, PLANNER_MAX_TOKENS, local_only=local_only)
    if not raw:
        return [question]  # Fallback: just use the original question

    # Parse JSON array
    try:
        import re
        # Extract JSON array from response
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            queries = json.loads(match.group(0))
            if isinstance(queries, list):
                return [str(q).strip() for q in queries if q][:MAX_QUERIES_PER_ROUND]
    except (json.JSONDecodeError, Exception):
        pass

    return [question]


# =========================================================================
# Phase 2+3: Search and Fetch
# =========================================================================

async def _search_and_collect(queries: list[str], pages_budget: int) -> tuple[list[dict], str]:
    """Run searches and fetch pages. Returns (sources, evidence_text)."""
    from app.tools.registry import execute_tool_async
    from app.llm.web_search import _fetch_source_text, _extract_relevant_excerpts

    # Credibility classifier (optional)
    try:
        from app.tools.source_classifier import classify_source, format_source_tag_for_evidence
        _classifier = True
    except ImportError:
        _classifier = False

    all_sources: list[dict] = []
    seen_urls: set[str] = set()

    # Run all queries
    for query in queries:
        try:
            resp = await execute_tool_async(
                "web_search", "v1",
                {"query": query, "max_results": 5},
                context=None,
            )
            if not resp.ok:
                continue
            results = (resp.result or {}).get("results", [])
            for r in results:
                url = str(r.get("url", ""))
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                source = {
                    "title": str(r.get("title", "")),
                    "url": url,
                    "snippet": str(r.get("snippet", "")),
                    "query": query,
                }
                if _classifier and url:
                    tag = classify_source(url)
                    source["credibility_tier"] = tag.tier
                    source["credibility_label"] = tag.tier_label
                    source["source_type"] = tag.source_type
                    source["bias_note"] = tag.bias_note or ""
                all_sources.append(source)
        except Exception as e:
            logger.warning("[deep_research] Search failed for '%s': %s", query, e)

    # Fetch top pages (prioritise higher credibility)
    sorted_sources = sorted(all_sources, key=lambda s: s.get("credibility_tier", 5))
    to_fetch = sorted_sources[:pages_budget]

    fetch_tasks = [_fetch_source_text(s["url"], None) for s in to_fetch]
    page_texts = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    # Build evidence
    evidence_parts = []
    for i, src in enumerate(to_fetch):
        cred_tag = ""
        if src.get("source_type", "unknown") != "unknown":
            cred_tag = f" [{src.get('credibility_label', '').upper()}] [{src.get('source_type', '')}]"
        line = f"[{i+1}]{cred_tag} {src['title']}\nURL: {src['url']}"
        if src["snippet"]:
            line += f"\nSnippet: {src['snippet']}"

        page_text = page_texts[i] if i < len(page_texts) and isinstance(page_texts[i], str) else ""
        if page_text:
            # Use the original query for relevant excerpts
            excerpt = _extract_relevant_excerpts(
                page_text, src.get("query", ""),
                max_chars=EVIDENCE_BUDGET_PER_ROUND // max(len(to_fetch), 1),
            )
            if excerpt:
                line += f"\nContent: {excerpt}"

        evidence_parts.append(line)
        src["fetched"] = bool(page_text)

    evidence_text = "\n\n".join(evidence_parts)
    return all_sources, evidence_text


# =========================================================================
# Phase 5: Gap Analysis
# =========================================================================

GAP_SYSTEM = """You are a research quality analyst. Given evidence gathered so far
and the original question, identify what information is still MISSING.

Rules:
- Be specific about what gaps exist
- Focus on factual gaps, not opinion gaps
- If the evidence is sufficient to answer the question well, say "NO_GAPS"
- Return a short paragraph (2-4 sentences) describing the gaps
- If no gaps, return exactly: NO_GAPS"""


async def _analyse_gaps(question: str, evidence: str, local_only: bool = False) -> str:
    """Identify what's still missing from the research."""
    prompt = (
        f"Question: {question}\n\n"
        f"Evidence gathered so far:\n{evidence[:6000]}\n\n"
        "What information is still missing to answer this question thoroughly?"
    )
    return await _llm_call(prompt, GAP_SYSTEM, GAP_ANALYSIS_MAX_TOKENS, local_only=local_only)


# =========================================================================
# Phase 7: Synthesis
# =========================================================================

SYNTHESIS_SYSTEM = """You are a research synthesiser. Using ONLY the evidence below,
write a comprehensive answer to the user's question.

Rules:
- Cite sources using [1], [2], etc.
- Prefer higher-credibility sources (AUTHORITATIVE > ESTABLISHED > COMMUNITY)
- Note conflicts between sources and which is more authoritative
- Be thorough but concise
- If evidence is insufficient, say what's missing
- Structure with clear paragraphs, not bullet points"""


async def _synthesise(question: str, all_evidence: str, source_count: int,
                      local_only: bool = False) -> str:
    """Generate final synthesis from all gathered evidence."""
    prompt = (
        f"Question: {question}\n\n"
        f"Research evidence ({source_count} sources across multiple searches):\n\n"
        f"{all_evidence}\n\n"
        "Provide a comprehensive, well-cited answer."
    )
    return await _llm_call(prompt, SYNTHESIS_SYSTEM, SYNTHESIS_MAX_TOKENS, local_only=local_only)


# =========================================================================
# Main Engine
# =========================================================================

async def run_deep_research(req: DeepResearchRequest) -> DeepResearchResult:
    """
    Execute the full deep research loop.

    Flow: PLAN → SEARCH → FETCH → TAG → GAP → (repeat) → SYNTHESISE
    """
    start = time.time()
    result = DeepResearchResult(query=req.query)
    all_evidence_parts: list[str] = []
    total_pages_fetched = 0
    previous_gaps = ""

    for round_num in range(1, req.max_rounds + 1):
        pages_budget = MAX_TOTAL_PAGES - total_pages_fetched
        if pages_budget <= 0:
            logger.info("[deep_research] Page budget exhausted after round %d", round_num - 1)
            break

        # 1. Plan queries
        queries = await _plan_queries(
            req.query, req.context, round_num, previous_gaps,
        )
        logger.info("[deep_research] Round %d: %d queries", round_num, len(queries))

        # 2+3. Search and fetch
        sources, evidence = await _search_and_collect(queries, min(pages_budget, 5))
        pages_this_round = sum(1 for s in sources if s.get("fetched"))
        total_pages_fetched += pages_this_round

        rnd = ResearchRound(
            round_number=round_num,
            queries=queries,
            sources=sources,
            evidence_text=evidence,
            pages_fetched=pages_this_round,
        )

        if evidence:
            all_evidence_parts.append(f"--- Round {round_num} ---\n{evidence}")

        # 5. Gap analysis (skip on last round)
        if round_num < req.max_rounds and evidence:
            combined = "\n\n".join(all_evidence_parts)
            gap_text = await _analyse_gaps(req.query, combined)
            rnd.gap_analysis = gap_text

            if "NO_GAPS" in gap_text.upper():
                logger.info("[deep_research] No gaps found after round %d, stopping", round_num)
                result.rounds.append(rnd)
                break

            previous_gaps = gap_text
        else:
            rnd.gap_analysis = "Final round — proceeding to synthesis."

        result.rounds.append(rnd)
        result.total_queries += len(queries)

    # 7. Synthesis
    combined_evidence = "\n\n".join(all_evidence_parts)
    all_sources = []
    for rnd in result.rounds:
        all_sources.extend(rnd.sources)

    if combined_evidence:
        result.synthesis = await _synthesise(req.query, combined_evidence, len(all_sources))

    # Diversity scoring
    try:
        from app.tools.source_classifier import classify_sources, get_diversity_summary
        tags = classify_sources([s["url"] for s in all_sources if s.get("url")])
        diversity = get_diversity_summary(tags)
        result.diversity_score = diversity.get("diversity_score", 0.0)
        result.missing_perspectives = diversity.get("missing_perspectives", [])
    except ImportError:
        pass

    result.all_sources = all_sources
    result.total_pages_fetched = total_pages_fetched
    result.total_rounds = len(result.rounds)
    result.elapsed_seconds = round(time.time() - start, 1)

    logger.info(
        "[deep_research] Complete: %d rounds, %d queries, %d pages, %.1fs",
        result.total_rounds, result.total_queries,
        result.total_pages_fetched, result.elapsed_seconds,
    )
    return result
