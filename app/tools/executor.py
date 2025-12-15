# FILE: app/tools/executor.py
"""
Tool execution + lightweight schema validation for Orb local tools.

Goals:
- No external jsonschema dependency.
- Central place for HTTP safety checks, timeouts, and response-size caps.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = float(os.getenv("ORB_TOOL_HTTP_TIMEOUT_S") or "15")
_DEFAULT_MAX_BYTES = int(os.getenv("ORB_TOOL_HTTP_MAX_BYTES") or "200000")  # 200 KB default cap


@dataclass
class ToolResponse:
    ok: bool
    tool_name: str
    tool_version: str
    result: Optional[dict] = None
    error_message: str = ""
    elapsed_ms: int = 0


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool))


def _validate_required(schema: dict, data: dict) -> list[str]:
    errs: list[str] = []
    required = schema.get("required") or []
    for key in required:
        if key not in data:
            errs.append(f"missing required field: {key}")
    return errs


def _validate_basic(schema: dict, data: Any, path: str = "") -> list[str]:
    """
    Minimal schema validator (enough to catch the common mistakes Orb makes).
    Supports:
      - type: object/array/string/integer/number/boolean
      - required (objects)
      - properties (objects)
      - items (arrays)
      - minLength/maxLength (strings)
      - minimum/maximum (numbers/integers)
    """
    errs: list[str] = []
    t = schema.get("type")

    def p(msg: str) -> str:
        return f"{path}{msg}" if path else msg

    if t == "object":
        if not isinstance(data, dict):
            return [p(f"expected object, got {type(data).__name__}")]
        errs += _validate_required(schema, data)
        props = schema.get("properties") or {}
        for k, v in data.items():
            if k in props:
                errs += _validate_basic(props[k], v, path=f"{path}{k}.")
        return errs

    if t == "array":
        if not isinstance(data, list):
            return [p(f"expected array, got {type(data).__name__}")]
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(data):
                errs += _validate_basic(item_schema, item, path=f"{path}{i}.")
        return errs

    if t == "string":
        if not isinstance(data, str):
            return [p(f"expected string, got {type(data).__name__}")]
        min_len = schema.get("minLength")
        max_len = schema.get("maxLength")
        if min_len is not None and len(data) < int(min_len):
            errs.append(p(f"string shorter than minLength {min_len}"))
        if max_len is not None and len(data) > int(max_len):
            errs.append(p(f"string longer than maxLength {max_len}"))
        return errs

    if t == "integer":
        if not _is_int(data):
            return [p(f"expected integer, got {type(data).__name__}")]
        mn = schema.get("minimum")
        mx = schema.get("maximum")
        if mn is not None and data < int(mn):
            errs.append(p(f"integer less than minimum {mn}"))
        if mx is not None and data > int(mx):
            errs.append(p(f"integer greater than maximum {mx}"))
        return errs

    if t == "number":
        if not _is_number(data):
            return [p(f"expected number, got {type(data).__name__}")]
        mn = schema.get("minimum")
        mx = schema.get("maximum")
        if mn is not None and data < float(mn):
            errs.append(p(f"number less than minimum {mn}"))
        if mx is not None and data > float(mx):
            errs.append(p(f"number greater than maximum {mx}"))
        return errs

    if t == "boolean":
        if not isinstance(data, bool):
            return [p(f"expected boolean, got {type(data).__name__}")]
        return errs

    # unknown schema types are treated as "no validation"
    return errs


class ToolExecutor:
    """
    Executes registered tools with:
      - schema validation
      - timeouts
      - safe HTTP fetch primitive
    """

    def __init__(self) -> None:
        # Optional allow/deny lists for http_fetch (comma-separated domains)
        self.http_allowlist = {d.strip().lower() for d in (os.getenv("ORB_HTTP_ALLOWLIST") or "").split(",") if d.strip()}
        self.http_denylist = {d.strip().lower() for d in (os.getenv("ORB_HTTP_DENYLIST") or "").split(",") if d.strip()}

    def _domain_allowed(self, url: str) -> bool:
        try:
            u = urlparse(url)
            host = (u.hostname or "").lower()
            if not host:
                return False
            if self.http_denylist and any(host == d or host.endswith("." + d) for d in self.http_denylist):
                return False
            if self.http_allowlist:
                return any(host == d or host.endswith("." + d) for d in self.http_allowlist)
            return True
        except Exception:
            return False

    async def http_fetch(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[dict[str, str]] = None,
        max_bytes: Optional[int] = None,
    ) -> dict:
        # Safety: scheme + domain
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return {
                "ok": False,
                "url": url,
                "status_code": 0,
                "final_url": "",
                "headers": {},
                "text": "",
                "truncated": False,
                "error": f"blocked scheme: {parsed.scheme}",
            }
        if not self._domain_allowed(url):
            return {
                "ok": False,
                "url": url,
                "status_code": 0,
                "final_url": "",
                "headers": {},
                "text": "",
                "truncated": False,
                "error": "blocked domain",
            }

        method_u = (method or "GET").upper().strip()
        if method_u not in ("GET", "HEAD"):
            # Keep this tight for now
            return {
                "ok": False,
                "url": url,
                "status_code": 0,
                "final_url": "",
                "headers": {},
                "text": "",
                "truncated": False,
                "error": f"blocked method: {method_u}",
            }

        cap = int(max_bytes or _DEFAULT_MAX_BYTES)
        cap = max(1, min(5_000_000, cap))

        hdrs = {"User-Agent": "Mozilla/5.0 (Orb; +local-tools)"}
        if headers:
            for k, v in headers.items():
                if isinstance(k, str) and isinstance(v, str):
                    hdrs[k] = v

        try:
            timeout = httpx.Timeout(_DEFAULT_TIMEOUT_S)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                resp = await client.request(method_u, url, headers=hdrs)
                raw = resp.content or b""
                truncated = False
                if len(raw) > cap:
                    raw = raw[:cap]
                    truncated = True

                # Best-effort decode
                try:
                    text = raw.decode(resp.encoding or "utf-8", errors="replace")
                except Exception:
                    text = raw.decode("utf-8", errors="replace")

                return {
                    "ok": True,
                    "url": url,
                    "status_code": int(resp.status_code),
                    "final_url": str(resp.url),
                    "headers": {k: v for k, v in resp.headers.items()},
                    "text": text,
                    "truncated": truncated,
                    "error": "",
                }
        except Exception as e:
            return {
                "ok": False,
                "url": url,
                "status_code": 0,
                "final_url": "",
                "headers": {},
                "text": "",
                "truncated": False,
                "error": str(e),
            }

    async def run_tool(
        self,
        tool_name: str,
        tool_version: str,
        input_data: dict,
        input_schema: dict,
        output_schema: dict,
        handler: Callable[[dict, Optional[dict]], Awaitable[dict]],
        *,
        context: Optional[dict] = None,
        timeout_s: Optional[float] = None,
    ) -> ToolResponse:
        started = time.perf_counter()

        # Validate input
        errs = _validate_basic(input_schema, input_data, path="")
        if errs:
            elapsed = int((time.perf_counter() - started) * 1000)
            return ToolResponse(
                ok=False,
                tool_name=tool_name,
                tool_version=tool_version,
                result=None,
                error_message="; ".join(errs),
                elapsed_ms=elapsed,
            )

        # Execute with timeout
        try:
            coro = handler(input_data, context)
            out = await asyncio.wait_for(coro, timeout=timeout_s or _DEFAULT_TIMEOUT_S)
        except asyncio.TimeoutError:
            elapsed = int((time.perf_counter() - started) * 1000)
            return ToolResponse(
                ok=False,
                tool_name=tool_name,
                tool_version=tool_version,
                result=None,
                error_message="tool timeout",
                elapsed_ms=elapsed,
            )
        except Exception as e:
            logger.exception("[tools] %s:%s handler error: %s", tool_name, tool_version, e)
            elapsed = int((time.perf_counter() - started) * 1000)
            return ToolResponse(
                ok=False,
                tool_name=tool_name,
                tool_version=tool_version,
                result=None,
                error_message=str(e),
                elapsed_ms=elapsed,
            )

        # Validate output (best-effort; don't fail hard if tool returns extras)
        out_errs = _validate_basic(output_schema, out, path="")
        if out_errs:
            logger.warning("[tools] %s:%s output schema mismatch: %s", tool_name, tool_version, "; ".join(out_errs))

        elapsed = int((time.perf_counter() - started) * 1000)
        return ToolResponse(ok=True, tool_name=tool_name, tool_version=tool_version, result=out, error_message="", elapsed_ms=elapsed)
