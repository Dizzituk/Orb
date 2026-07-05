# FILE: app/tools/registry.py
# Purpose: Local tool registry for Orb.
# Called-by: app.debug.executors.user_files, app.investments.news_service, app.llm.deep_research_engine, app.llm.web_search (+6 more)
# Depends-on: app.db, app.llm.audit_logger, app.settings.service, app.tools.executor (+4 more)
# Last-renovated: 2026-06-12
"""
Local tool registry for Orb.

This registry is the ONLY place that "internet access" is allowed, via local tools (http_fetch/web_search),
not via provider-side browsing.

Public API used by the rest of the app:
  - execute_tool_async(tool_name, tool_version, input_data, context=None) -> ToolResponse
  - list_tools()
"""

from __future__ import annotations

import html as _html
import json
import logging
import os
import re
import urllib.parse
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from app.tools.executor import ToolExecutor, ToolResponse
from app.tools.schemas import TOOL_SCHEMAS

logger = logging.getLogger(__name__)


def _emit_tool_audit(tool_name: str, tool_version: str, input_data: dict, resp, context: Optional[dict]) -> None:
    """Emit a non-sensitive tool_call event.

    Rules:
    - Do NOT log raw input args (query text, full URLs, etc.).
    - For http_fetch, log host + status_code only.
    - For web_search, log result_count only.
    """
    try:
        from urllib.parse import urlparse
        from app.llm.audit_logger import get_audit_logger, AuditEventType

        audit = get_audit_logger()
        if not audit or not audit.enabled:
            return

        # request_id best-effort (context may include job_envelope)
        request_id = None
        job_env = None
        if isinstance(context, dict):
            request_id = context.get('request_id')
            job_env = context.get('job_envelope')
        if job_env is not None:
            try:
                if isinstance(job_env, dict):
                    request_id = request_id or job_env.get('job_id')
                else:
                    request_id = request_id or getattr(job_env, 'job_id', None)
            except Exception:
                pass

        request_id = str(request_id or f"tool-{int(__import__('time').time()*1000)}")

        host = ''
        status_code = None
        result_count = None

        if tool_name == 'http_fetch':
            url = str((input_data or {}).get('url') or '')
            try:
                host = (urlparse(url).hostname or '')
            except Exception:
                host = ''
            try:
                if resp and getattr(resp, 'result', None):
                    status_code = (resp.result or {}).get('status_code')
            except Exception:
                status_code = None

        if tool_name == 'web_search':
            try:
                if resp and getattr(resp, 'result', None):
                    results = (resp.result or {}).get('results') or []
                    result_count = len(results)
            except Exception:
                result_count = None

        audit.emit({
            'ts': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
            'request_id': request_id,
            'event': AuditEventType.TOOL_CALL.value,
            'tool': {
                'name': tool_name,
                'version': tool_version,
                'ok': bool(getattr(resp, 'ok', False)),
                'elapsed_ms': int(getattr(resp, 'elapsed_ms', 0) or 0),
                **({'result_count': int(result_count)} if result_count is not None else {}),
            },
            'http': {'host': host, 'status_code': status_code} if host or status_code is not None else None,
            'error': {'message': getattr(resp, 'error_message', '')} if not getattr(resp, 'ok', False) else None,
        })
    except Exception:
        # Never let audit break tool execution
        return


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    version: str
    description: str
    input_schema: dict
    output_schema: dict
    handler: Callable[[dict, Optional[dict]], Awaitable[dict]]


_TOOL_DEFS: dict[tuple[str, str], ToolDefinition] = {}
_EXECUTOR: Optional[ToolExecutor] = None


def get_tool_executor() -> ToolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ToolExecutor()
    return _EXECUTOR


def register_tool(tool: ToolDefinition) -> None:
    _TOOL_DEFS[(tool.name, tool.version)] = tool


def get_tool(tool_name: str, tool_version: str = "v1") -> ToolDefinition:
    key = (tool_name, tool_version)
    if key not in _TOOL_DEFS:
        raise KeyError(f"tool not registered: {tool_name}:{tool_version}")
    return _TOOL_DEFS[key]


def list_tools() -> list[dict]:
    out: list[dict] = []
    for (name, version), td in sorted(_TOOL_DEFS.items()):
        out.append({"name": name, "version": version, "description": td.description})
    return out


def list_tool_definitions() -> list[ToolDefinition]:
    """All registered ToolDefinitions (read-only view for adapters, e.g.
    app.tools.chat_bridge exposing registry tools to the chat tool loop)."""
    return [_TOOL_DEFS[key] for key in sorted(_TOOL_DEFS.keys())]


# -------------------------
# Handlers
# -------------------------

async def http_fetch_handler(input_data: dict, context: Optional[dict]) -> dict:
    url = str(input_data.get("url") or "").strip()
    method = str(input_data.get("method") or "GET").strip()
    headers = input_data.get("headers") or {}
    max_bytes = input_data.get("max_bytes") or None
    return await get_tool_executor().http_fetch(url=url, method=method, headers=headers, max_bytes=max_bytes)


def _strip_tags(s: str) -> str:
    s = re.sub(r"<script[\s\S]*?</script>", " ", s, flags=re.I)
    s = re.sub(r"<style[\s\S]*?</style>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = _html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _unwrap_ddg_redirect(u: str) -> str:
    try:
        parsed = urllib.parse.urlparse(u)
        if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
            qs = urllib.parse.parse_qs(parsed.query)
            uddg = qs.get("uddg", [None])[0]
            if uddg:
                return urllib.parse.unquote(uddg)
    except Exception:
        pass
    return u


def _is_ad_url(u: str) -> bool:
    lu = u.lower()
    return (
        "duckduckgo.com/y.js" in lu
        or "ad_domain=" in lu
        or "bing.com/aclick" in lu
        or "doubleclick.net" in lu
        or "googleadservices" in lu
    )


def _parse_ddg_lite(page: str, max_results: int) -> list[dict]:
    # DuckDuckGo Lite patterns
    link_pattern = re.compile(r'<a[^>]+class="result-link"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.I | re.S)
    snip_pattern = re.compile(r'<td[^>]+class="result-snippet"[^>]*>(.*?)</td>', re.I | re.S)

    links = link_pattern.findall(page)
    snippets = snip_pattern.findall(page)

    results: list[dict] = []
    for i, (href, title_html) in enumerate(links):
        if len(results) >= max_results:
            break
        url = _unwrap_ddg_redirect(_html.unescape(href))
        if not url or _is_ad_url(url):
            continue
        title = _strip_tags(title_html)
        snippet = _strip_tags(snippets[i]) if i < len(snippets) else ""
        if not title:
            continue
        results.append({"title": title, "url": url, "snippet": snippet})
    return results


def _parse_ddg_html(page: str, max_results: int) -> list[dict]:
    link_pattern = re.compile(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        re.I | re.S,
    )
    snip_pattern = re.compile(
        r'<(?:a|div)[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>',
        re.I | re.S,
    )

    links = link_pattern.findall(page)
    snippets = snip_pattern.findall(page)

    results: list[dict] = []
    for i, (href, title_html) in enumerate(links):
        if len(results) >= max_results:
            break
        url = _unwrap_ddg_redirect(_html.unescape(href))
        if not url or _is_ad_url(url):
            continue
        title = _strip_tags(title_html)
        snippet = _strip_tags(snippets[i]) if i < len(snippets) else ""
        if not title:
            continue
        results.append({"title": title, "url": url, "snippet": snippet})
    return results


# -------------------------
# Brave Search
# -------------------------

def _get_brave_api_key() -> Optional[str]:
    """Return Brave Search API key — checks DB first, then env."""
    try:
        from app.settings.service import get_key_value
        from app.db import SessionLocal
        db = SessionLocal()
        try:
            val = get_key_value(db, "brave_search")
            if val:
                return val.strip()
        finally:
            db.close()
    except Exception:
        pass
    # Fallback to env
    key = (os.getenv("BRAVE_SEARCH_API_KEY") or "").strip()
    return key if key else None


def _parse_brave_results(data: dict, max_results: int) -> list[dict]:
    """Parse Brave Search API JSON into our standard result format."""
    results: list[dict] = []
    web = data.get("web") or {}
    for item in (web.get("results") or [])[:max_results]:
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("description") or "").strip()
        if not title or not url:
            continue
        results.append({"title": title, "url": url, "snippet": snippet})
    return results


async def _brave_search(query: str, max_results: int, api_key: str) -> Optional[list[dict]]:
    """Query Brave Search API. Returns results list or None on failure."""
    try:
        q = urllib.parse.quote_plus(query)
        url = f"https://api.search.brave.com/res/v1/web/search?q={q}&count={max_results}"
        resp = await get_tool_executor().http_fetch(
            url=url,
            method="GET",
            headers={
                "X-Subscription-Token": api_key,
                "Accept": "application/json",
            },
            max_bytes=500_000,
        )
        if not resp.get("ok"):
            logger.warning("[web_search] Brave API request failed: status=%s", resp.get("status_code"))
            return None

        text = str(resp.get("text") or "")
        if not text:
            return None

        data = json.loads(text)
        results = _parse_brave_results(data, max_results)
        return results if results else None
    except Exception as e:
        logger.warning("[web_search] Brave search failed: %s", e)
        return None


# -------------------------
# DuckDuckGo (fallback)
# -------------------------

async def _ddg_search(query: str, max_results: int) -> tuple[str, list[dict]]:
    """DuckDuckGo Lite + HTML fallback. Returns (provider, results)."""
    q = urllib.parse.quote_plus(query)

    # 1) Lite first (less ad noise)
    lite_url = f"https://lite.duckduckgo.com/lite/?q={q}"
    lite = await get_tool_executor().http_fetch(
        url=lite_url,
        method="GET",
        headers={"User-Agent": "Mozilla/5.0"},
        max_bytes=200_000,
    )
    results: list[dict] = []
    provider = "duckduckgo_lite"
    if lite.get("ok"):
        results = _parse_ddg_lite(str(lite.get("text") or ""), max_results)

    # 2) HTML fallback if Lite produced nothing (format change / block)
    if not results:
        html_url = f"https://duckduckgo.com/html/?q={q}"
        page = await get_tool_executor().http_fetch(
            url=html_url,
            method="GET",
            headers={"User-Agent": "Mozilla/5.0"},
            max_bytes=250_000,
        )
        provider = "duckduckgo_html"
        if page.get("ok"):
            results = _parse_ddg_html(str(page.get("text") or ""), max_results)

    return provider, results


async def web_search_handler(input_data: dict, context: Optional[dict]) -> dict:
    query = str(input_data.get("query") or "").strip()
    max_results = int(input_data.get("max_results") or 5)
    max_results = max(1, min(10, max_results))

    if not query:
        return {"query": query, "provider": "none", "results": []}

    # 1) Brave Search (primary — if API key configured)
    brave_key = _get_brave_api_key()
    if brave_key:
        results = await _brave_search(query, max_results, brave_key)
        if results:
            logger.info("[web_search] Brave returned %d results for: %s", len(results), query[:80])
            return {"query": query, "provider": "brave", "results": results}
        logger.info("[web_search] Brave failed or empty, falling back to DuckDuckGo")

    # 2) DuckDuckGo fallback
    provider, results = await _ddg_search(query, max_results)
    return {"query": query, "provider": provider, "results": results}


async def weather_handler(input_data: dict, context: Optional[dict]) -> dict:
    # Placeholder: keep tool registered for future; return a friendly error for now.
    location = str(input_data.get("location") or "").strip()
    return {"ok": False, "location": location, "forecast": {}, "error": "weather tool not implemented yet"}


# -------------------------
# Capability gating seam (2026-06-12)
# -------------------------
# Future family profiles will need per-session tool gating (a restricted
# client must not reach every tool). This is the single dispatch chokepoint,
# so the seam lives here: a policy callable consulted before every execution.
# Default = allow-all; nothing changes for the single-user system. A future
# profile layer swaps the policy via set_tool_policy() — no dispatch rewrite.

ToolPolicy = Callable[[str, str, Optional[dict]], bool]


def _allow_all_policy(tool_name: str, tool_version: str, context: Optional[dict]) -> bool:
    return True


_TOOL_POLICY: ToolPolicy = _allow_all_policy


def set_tool_policy(policy: Optional[ToolPolicy]) -> None:
    """Install a capability policy (None restores allow-all)."""
    global _TOOL_POLICY
    _TOOL_POLICY = policy or _allow_all_policy


def get_tool_policy() -> ToolPolicy:
    return _TOOL_POLICY


# -------------------------
# Execution API
# -------------------------

async def execute_tool_async(tool_name: str, tool_version: str, input_data: dict, context: Optional[dict] = None) -> ToolResponse:
    tool = get_tool(tool_name, tool_version)

    try:
        allowed = _TOOL_POLICY(tool_name, tool_version, context)
    except Exception as policy_exc:
        logger.warning("[tools] policy check failed open for %s: %s", tool_name, policy_exc)
        allowed = True
    if not allowed:
        logger.info("[tools] policy DENIED %s:%s", tool_name, tool_version)
        return ToolResponse(
            ok=False,
            tool_name=tool_name,
            tool_version=tool_version,
            error_message="tool blocked by capability policy for this session",
        )

    resp = await get_tool_executor().run_tool(
        tool_name=tool.name,
        tool_version=tool.version,
        input_data=input_data or {},
        input_schema=tool.input_schema,
        output_schema=tool.output_schema,
        handler=tool.handler,
        context=context,
    )

    _emit_tool_audit(tool_name, tool_version, input_data or {}, resp, context)
    return resp


def execute_tool(tool_name: str, tool_version: str, input_data: dict, context: Optional[dict] = None) -> ToolResponse:
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(execute_tool_async(tool_name, tool_version, input_data, context=context))
    raise RuntimeError("execute_tool() called inside an event loop; use execute_tool_async()")


# -------------------------
# Register defaults
# -------------------------

def _register_defaults() -> None:
    if _TOOL_DEFS:
        return

    register_tool(
        ToolDefinition(
            name="http_fetch",
            version="v1",
            description="Fetch a URL over HTTP(S) with safety checks and a response-size cap.",
            input_schema=TOOL_SCHEMAS["http_fetch"]["input"],
            output_schema=TOOL_SCHEMAS["http_fetch"]["output"],
            handler=http_fetch_handler,
        )
    )

    register_tool(
        ToolDefinition(
            name="web_search",
            version="v1",
            description="Search the public web (Brave Search primary, DuckDuckGo fallback) and return structured results.",
            input_schema=TOOL_SCHEMAS["web_search"]["input"],
            output_schema=TOOL_SCHEMAS["web_search"]["output"],
            handler=web_search_handler,
        )
    )

    register_tool(
        ToolDefinition(
            name="weather",
            version="v1",
            description="Weather lookup (stub).",
            input_schema=TOOL_SCHEMAS["weather"]["input"],
            output_schema=TOOL_SCHEMAS["weather"]["output"],
            handler=weather_handler,
        )
    )

    # Lifestyle / health coaching tools — added 2026-05-25.
    # Five tools that let any chat read recent nutrition / weight / workouts
    # and log new nutrition + workout entries. See app/tools/lifestyle_tools.py
    # for the handlers; schemas live in app/tools/schemas.py.
    try:
        from app.tools.lifestyle_tools import register_lifestyle_tools
        register_lifestyle_tools()
    except Exception as exc:
        # Never let a tool-registration failure stop the rest of the
        # registry from coming up — log and carry on.
        logger.exception("[registry] lifestyle tools registration failed: %s", exc)

    # Reminders (2026-07-01): create_reminder/list_reminders/cancel_reminder —
    # one-shot reminders that fire on desktop + phone alike. Handlers in
    # app/tools/reminder_tools.py; schemas in app/tools/schemas.py.
    try:
        from app.tools.reminder_tools_registration import register_reminder_tools
        register_reminder_tools()
    except Exception as exc:
        logger.exception("[registry] reminder tools registration failed: %s", exc)

    # Strength / gym set-logging tools - added 2026-05-31.
    # log_exercise_set + get_exercise_history for per-exercise progressive
    # overload. Handlers in app/tools/strength_tools.py; schemas in
    # app/tools/schemas.py; service in app/lifestyle/strength.py.
    try:
        from app.tools.strength_tools import register_strength_tools
        register_strength_tools()
    except Exception as exc:
        logger.exception("[registry] strength tools registration failed: %s", exc)

    # Work-day ledger tools - added 2026-05-30 (v8.0 finance rebuild).
    # Lets any chat (and AstraBridge) start/finish a work day, reconcile a
    # forgotten day, and set personal mileage - written straight to the DB.
    # See app/tools/finance_tools.py (self-contained: schemas + handlers).
    try:
        from app.tools.finance_tools import register_finance_tools
        register_finance_tools()
    except Exception as exc:
        logger.exception("[registry] finance tools registration failed: %s", exc)

    # ASTRA Sentinel security tools - added 2026-06-12 (Phase 1).
    # Network status / recent connections / alerts / explain-alert /
    # request_block_remote_address (PROPOSES only - ask-first is absolute,
    # never executes) / trust_process. Self-contained in app/sentinel/tools.py.
    try:
        from app.sentinel.tools import register_sentinel_tools
        register_sentinel_tools()
    except Exception as exc:
        logger.exception("[registry] sentinel tools registration failed: %s", exc)

    # Vehicle tools (2026-06-12): calibrate the van's virtual odometer, set
    # maintenance dates, log fitted components, read status, classify trips.
    # Self-contained in app/tools/vehicle_tools.py (OBD2 van module).
    try:
        from app.tools.vehicle_tools import register_vehicle_tools
        register_vehicle_tools()
    except Exception as exc:
        logger.exception("[registry] vehicle tools registration failed: %s", exc)

    # Display tools (2026-06-12): named-display media windows ("put it on
    # the bed screen"). Self-contained in app/tools/display_tools.py.
    try:
        from app.tools.display_tools import register_display_tools
        register_display_tools()
    except Exception as exc:
        logger.exception("[registry] display tools registration failed: %s", exc)

    # Podcast tools (2026-06-12): discovery (PodcastIndex/iTunes), RSS
    # subscriptions, desktop playback. Self-contained in podcast_tools.py.
    try:
        from app.tools.podcast_tools import register_podcast_tools
        register_podcast_tools()
    except Exception as exc:
        logger.exception("[registry] podcast tools registration failed: %s", exc)

    # Email tools (2026-06-12): provider-agnostic read/search + the
    # draft-confirm-send loop (never auto-send). app/email owns providers
    # (gmail | proton adapter over app/email_service).
    try:
        from app.tools.email_tools import register_email_tools
        register_email_tools()
    except Exception as exc:
        logger.exception("[registry] email tools registration failed: %s", exc)

    # Stream/media tools (2026-06-12): SoundCloud/Mixcloud desktop playback
    # via display windows + YouTube voice search. app/media owns the flows.
    try:
        from app.tools.stream_tools import register_stream_tools
        register_stream_tools()
    except Exception as exc:
        logger.exception("[registry] stream tools registration failed: %s", exc)

    # Movie-night tools (2026-06-12): TMDB narrowing + GB providers + bed-
    # screen dispatch. app/media/movie_night owns the flow.
    try:
        from app.tools.movie_tools import register_movie_tools
        register_movie_tools()
    except Exception as exc:
        logger.exception("[registry] movie tools registration failed: %s", exc)

    # Document/editor tools (2026-06-12): read + edit whatever is open in
    # the desktop Univer editor pane. app/documents owns the action channel.
    try:
        from app.tools.document_tools import register_document_tools
        register_document_tools()
    except Exception as exc:
        logger.exception("[registry] document tools registration failed: %s", exc)

    # Idle-agents tools (2026-07-01): background task log ("what have you
    # been working on"). app/idle owns the governor + ledger.
    try:
        from app.idle.tools_registration import register_idle_tools
        register_idle_tools()
    except Exception as exc:
        logger.exception("[registry] idle tools registration failed: %s", exc)

    # Watcher tools (2026-07-01): auto-registered per watcher instance
    # (Portugal land prices, hardware prices). app/watchers owns the fleet.
    try:
        from app.watchers.framework import register_watcher_tools
        register_watcher_tools()
    except Exception as exc:
        logger.exception("[registry] watcher tools registration failed: %s", exc)

    # Report tools (2026-07-01): render watcher ledgers to styled HTML on
    # the desktop reports window / as a Bridge document artifact.
    try:
        from app.reports.tools_registration import register_report_tools
        register_report_tools()
    except Exception as exc:
        logger.exception("[registry] report tools registration failed: %s", exc)

    # Deep-research tools (2026-07-01): queue a background dig on the idle
    # ledger + fetch the findings pack. app/llm/research_task owns the grind.
    try:
        from app.llm.research_task import register_research_tools
        register_research_tools()
    except Exception as exc:
        logger.exception("[registry] research tools registration failed: %s", exc)

    # Social posting + shorts tools (2026-07-02): post_image/post_reel to the
    # Meta Business Suite composer (FB+IG one login) + create_short (async
    # captioned 9:16 render). Driver in app/content/distribution/posting_drivers;
    # shorts pipeline in app/content/video_pipeline/shorts_*.
    try:
        from app.tools.social_posting_tools import register_social_posting_tools
        register_social_posting_tools()
    except Exception as exc:
        logger.exception("[registry] social posting tools registration failed: %s", exc)


_register_defaults()
