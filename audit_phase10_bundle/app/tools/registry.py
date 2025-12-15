# FILE: app/tools/registry.py
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
import logging
import re
import urllib.parse
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from app.tools.executor import ToolExecutor, ToolResponse
from app.tools.schemas import TOOL_SCHEMAS

logger = logging.getLogger(__name__)


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


async def web_search_handler(input_data: dict, context: Optional[dict]) -> dict:
    query = str(input_data.get("query") or "").strip()
    max_results = int(input_data.get("max_results") or 5)
    max_results = max(1, min(10, max_results))

    if not query:
        return {"query": query, "provider": "duckduckgo_lite", "results": []}

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

    return {"query": query, "provider": provider, "results": results}


async def weather_handler(input_data: dict, context: Optional[dict]) -> dict:
    # Placeholder: keep tool registered for future; return a friendly error for now.
    location = str(input_data.get("location") or "").strip()
    return {"ok": False, "location": location, "forecast": {}, "error": "weather tool not implemented yet"}


# -------------------------
# Execution API
# -------------------------

async def execute_tool_async(tool_name: str, tool_version: str, input_data: dict, context: Optional[dict] = None) -> ToolResponse:
    tool = get_tool(tool_name, tool_version)
    return await get_tool_executor().run_tool(
        tool_name=tool.name,
        tool_version=tool.version,
        input_data=input_data or {},
        input_schema=tool.input_schema,
        output_schema=tool.output_schema,
        handler=tool.handler,
        context=context,
    )


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
            description="Search the public web (DuckDuckGo Lite + HTML fallback) and return structured results.",
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


_register_defaults()
