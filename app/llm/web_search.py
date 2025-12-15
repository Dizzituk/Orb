# FILE: app/llm/web_search.py
"""
Web search orchestration for Orb.

- Performs search via local tool: app.tools.registry -> web_search (DuckDuckGo Lite/HTML fallback).
- Fetches top sources via local http_fetch to provide real page text evidence.
- Asks an LLM to answer using ONLY those sources.
- Provider-side browsing is forbidden; this module never enables it.
"""

from __future__ import annotations

import asyncio
import html as _html
import inspect
import logging
import os
import re
from typing import Optional, Any

from pydantic import BaseModel, Field

from app.tools.registry import execute_tool_async
from app.providers.registry import llm_call as registry_llm_call, is_provider_available

logger = logging.getLogger(__name__)


class WebSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_results: int = Field(5, ge=1, le=10)


class WebSearchSource(BaseModel):
    title: str
    url: str
    snippet: str = ""


class WebSearchResponse(BaseModel):
    ok: bool
    query: str
    provider: str = ""
    answer: str = ""
    sources: list[WebSearchSource] = []
    error: str = ""


def _extract_content(llm_result: Any) -> str:
    if llm_result is None:
        return ""
    if isinstance(llm_result, str):
        return llm_result
    if isinstance(llm_result, tuple) and llm_result:
        return llm_result[0] if isinstance(llm_result[0], str) else str(llm_result[0])
    if isinstance(llm_result, dict):
        if "content" in llm_result and isinstance(llm_result["content"], str):
            return llm_result["content"]
        if "text" in llm_result and isinstance(llm_result["text"], str):
            return llm_result["text"]
    c = getattr(llm_result, "content", None)
    if isinstance(c, str):
        return c
    return str(llm_result)


def _pick_answer_provider() -> tuple[Optional[str], Optional[str]]:
    forced_provider = (os.getenv("WEB_SEARCH_PROVIDER_ID") or "").strip() or None
    forced_model = (os.getenv("WEB_SEARCH_MODEL_ID") or "").strip() or None
    if forced_provider:
        return forced_provider, forced_model

    for pid, default_model_env, default_model in [
        ("openai", "OPENAI_DEFAULT_MODEL", "gpt-4o-mini"),
        ("anthropic", "ANTHROPIC_DEFAULT_MODEL", "claude-3-5-sonnet-latest"),
        ("google", "GEMINI_FRONTIER_MODEL_ID", "gemini-3.0-pro-preview"),
    ]:
        try:
            if is_provider_available(pid):
                return pid, (os.getenv(default_model_env) or default_model)
        except Exception:
            continue
    return None, None


def _html_to_text(s: str) -> str:
    s = re.sub(r"<script[\s\S]*?</script>", " ", s, flags=re.I)
    s = re.sub(r"<style[\s\S]*?</style>", " ", s, flags=re.I)
    s = re.sub(r"<noscript[\s\S]*?</noscript>", " ", s, flags=re.I)
    s = re.sub(r"<svg[\s\S]*?</svg>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = _html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _keywords_from_query(q: str) -> list[str]:
    q = (q or "").lower()
    # crude tokenization; keep only useful-ish tokens
    toks = re.split(r"[^a-z0-9\-\_]+", q)
    toks = [t for t in toks if len(t) >= 3]
    # add some common pricing-related anchors
    extras = ["price", "pricing", "token", "tokens", "per", "1m", "million", "input", "output", "$", "usd"]
    merged = []
    seen = set()
    for t in toks + extras:
        if t and t not in seen:
            seen.add(t)
            merged.append(t)
    return merged[:20]


def _extract_relevant_excerpts(text: str, query: str, *, max_chars: int) -> str:
    """
    Return a compact excerpt that *actually contains* relevant bits (numbers/keywords),
    rather than just the first N characters.
    """
    if not text:
        return ""

    max_chars = max(500, min(8000, int(max_chars)))
    low = text.lower()
    kws = _keywords_from_query(query)

    # Find match positions
    positions: list[int] = []
    for kw in kws:
        if kw == "$":
            for m in re.finditer(r"\$", text):
                positions.append(m.start())
            continue
        for m in re.finditer(re.escape(kw), low):
            positions.append(m.start())

    # If nothing matched, return head (still capped)
    if not positions:
        return text[:max_chars]

    # Sort and de-dup positions
    positions = sorted(set(positions))

    # Build up to 4 windows around matches
    windows: list[str] = []
    budget = max_chars
    window_size = 700  # chars per window (before trimming)
    half = window_size // 2

    for pos in positions:
        if budget <= 200:
            break
        start = max(0, pos - half)
        end = min(len(text), pos + half)
        chunk = text[start:end].strip()
        if not chunk:
            continue

        # Avoid near-duplicate windows
        if windows and chunk[:120] in windows[-1]:
            continue

        # Trim to budget
        if len(chunk) > budget:
            chunk = chunk[:budget]
        windows.append(chunk)
        budget -= len(chunk) + 10

        if len(windows) >= 4:
            break

    # Join with separators so the LLM can see boundaries
    out = "\n...\n".join(windows)
    return out[:max_chars]


async def _fetch_source_text(url: str, context: Optional[dict]) -> str:
    try:
        resp = await execute_tool_async(
            "http_fetch",
            "v1",
            {
                "url": url,
                "method": "GET",
                "headers": {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-GB,en;q=0.9"},
                # bigger default so we don’t miss content further down the page
                "max_bytes": 1_000_000,
            },
            context=context,
        )
        if not resp.ok:
            return ""
        raw = str((resp.result or {}).get("text") or "")
        return _html_to_text(raw)
    except Exception:
        return ""


async def search_and_answer(req: WebSearchRequest, context: Optional[dict] = None) -> WebSearchResponse:
    try:
        # 1) Search
        tool_resp = await execute_tool_async(
            "web_search",
            "v1",
            {"query": req.query, "max_results": req.max_results},
            context=context,
        )
        if not tool_resp.ok:
            return WebSearchResponse(ok=False, query=req.query, error=tool_resp.error_message or "web_search tool failed")

        tool_result = tool_resp.result or {}
        provider = str(tool_result.get("provider") or "")
        results = tool_result.get("results") or []

        sources: list[WebSearchSource] = []
        for r in results[: req.max_results]:
            try:
                sources.append(
                    WebSearchSource(
                        title=str(r.get("title") or ""),
                        url=str(r.get("url") or ""),
                        snippet=str(r.get("snippet") or ""),
                    )
                )
            except Exception:
                continue

        if not sources:
            return WebSearchResponse(ok=True, query=req.query, provider=provider, answer="", sources=[])

        # 2) Fetch page text for MORE sources by default
        #    Key fix: don’t only fetch top 3, because the “numbers page” is often #4/#5.
        env_fetch = os.getenv("WEB_SEARCH_FETCH_SOURCES")
        if env_fetch is None or not env_fetch.strip():
            fetch_n = min(5, len(sources), req.max_results)
        else:
            fetch_n = int(env_fetch)
            fetch_n = max(0, min(10, fetch_n, len(sources)))

        fetched_texts: list[str] = []
        if fetch_n > 0:
            to_fetch = sources[:fetch_n]
            texts = await asyncio.gather(*[_fetch_source_text(s.url, context) for s in to_fetch], return_exceptions=False)
            fetched_texts = [t or "" for t in texts]

        # 3) Evidence: include RELEVANT excerpts, not just the top of the page
        cap = int(os.getenv("WEB_SEARCH_EVIDENCE_CHARS") or "7000")
        cap = max(500, min(8000, cap))

        evidence_lines: list[str] = []
        for i, s in enumerate(sources, start=1):
            line = f"[{i}] {s.title}\nURL: {s.url}"
            if s.snippet:
                line += f"\nSnippet: {s.snippet}"

            if i <= fetch_n:
                page_text = fetched_texts[i - 1] if i - 1 < len(fetched_texts) else ""
                excerpt = _extract_relevant_excerpts(page_text, req.query, max_chars=cap) if page_text else ""
                if excerpt:
                    line += f"\nPageText: {excerpt}"

            evidence_lines.append(line)

        evidence = "\n\n".join(evidence_lines)

        # 4) Ask an LLM to answer (optional)
        answer_provider, answer_model = _pick_answer_provider()
        if not answer_provider:
            return WebSearchResponse(ok=True, query=req.query, provider=provider, answer="", sources=sources)

        prompt = (
            "Answer the user's question using ONLY the sources below. "
            "If the sources don't contain the answer, say you can't find it in the sources. "
            "Be direct. Include citations like [1], [2].\n\n"
            f"Question: {req.query}\n\n"
            f"Sources:\n{evidence}\n"
        )

        try:
            sig = inspect.signature(registry_llm_call)
        except Exception:
            sig = None

        kwargs = {
            "provider_id": answer_provider,
            "model_id": answer_model,
            "messages": [{"role": "user", "content": prompt}],
            "system_prompt": None,
            "temperature": 0.2,
            "max_tokens": 700,
            "enable_web_search": False,  # forbid provider-side browsing
            "context": context,
        }
        if sig:
            kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        llm_res = await registry_llm_call(**kwargs)  # type: ignore[arg-type]
        answer = _extract_content(llm_res).strip()

        return WebSearchResponse(ok=True, query=req.query, provider=provider, answer=answer, sources=sources)

    except Exception as e:
        logger.exception("[web_search] search_and_answer failed: %s", e)
        return WebSearchResponse(ok=False, query=req.query, error=str(e))
