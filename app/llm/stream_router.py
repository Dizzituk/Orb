# FILE: app/llm/stream_router.py
r"""
Streaming endpoints for real-time LLM responses.
Uses Server-Sent Events (SSE).

LOCAL TOOL TRIGGER - CREATE ARCHITECTURE MAP
- Runs mapper script locally (host) against sandbox controller API
- Builds an evidence pack (tree + mapper artifacts + selected code files)
- Calls an LLM to produce a deeply detailed architecture map
- Saves output markdown as ARCH_MAP_FULL_V{N}_*.md
- Also ingests the map into Memory as a DocumentContent so Orb can answer questions about it later

NOTE ABOUT OPENAI MODELS:
- Anything routed through /v1/chat/completions MUST use a chat-capable model ID (e.g. *-chat-latest).
- The error you saw for gpt-5.2-pro is expected if this code path uses chat-completions.
"""

import os
import re
import json
import uuid
import sys
import logging
import asyncio
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth
from app.auth.middleware import AuthResult
from app.memory import service as memory_service, schemas as memory_schemas
from app.llm.schemas import JobType, RoutingConfig, LLMTask

from .streaming import stream_llm, get_available_streaming_provider

# Audit/telemetry
from app.llm.audit_logger import get_audit_logger, RoutingTrace

# Critique pipeline imports
from app.llm.router import (
    run_high_stakes_with_critique,
    synthesize_envelope_from_task,
    is_high_stakes_job,
    is_opus_model,
)

router = APIRouter(prefix="/stream", tags=["streaming"])
logger = logging.getLogger(__name__)

# Default models
DEFAULT_MODELS = {
    # NOTE: stream_router.py reads OPENAI_DEFAULT_MODEL for DEFAULT_MODELS["openai"]
    "openai": os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5-mini"),
    "anthropic": os.getenv("ANTHROPIC_SONNET_MODEL", "claude-sonnet-4-5-20250929"),
    "anthropic_opus": os.getenv("ANTHROPIC_OPUS_MODEL", "claude-opus-4-5-20251101"),
    "gemini": os.getenv("GEMINI_DEFAULT_MODEL", "gemini-2.0-flash"),
}

# =============================================================================
# LOCAL TOOL: ZOBIE MAP (host-side mapper -> sandbox controller)
# =============================================================================

_ZOBIE_TRIGGER_SET = {"zobie map", "zombie map", "zobie_map", "/zobie_map", "/zombie_map"}

ZOBIE_CONTROLLER_URL = os.getenv("ORB_ZOBIE_CONTROLLER_URL", "http://192.168.250.2:8765")
ZOBIE_MAPPER_SCRIPT = os.getenv("ORB_ZOBIE_MAPPER_SCRIPT", r"D:\tools\zobie_mapper\zobie_map.py")
ZOBIE_MAPPER_OUT_DIR = os.getenv("ORB_ZOBIE_MAPPER_OUT_DIR", r"D:\tools\zobie_mapper\out")
ZOBIE_MAPPER_TIMEOUT_SEC = int(os.getenv("ORB_ZOBIE_MAPPER_TIMEOUT_SEC", "300"))

# OPTIONAL: extra mapper args (lets you run your “full” mode without changing code)
# Example:
# ORB_ZOBIE_MAPPER_ARGS=200000 false 1500000 full
ZOBIE_MAPPER_ARGS_RAW = os.getenv("ORB_ZOBIE_MAPPER_ARGS", "").strip()
ZOBIE_MAPPER_ARGS = ZOBIE_MAPPER_ARGS_RAW.split() if ZOBIE_MAPPER_ARGS_RAW else []

# =============================================================================
# LOCAL TOOL: CREATE ARCHITECTURE MAP (mapper -> evidence pack -> GPT -> file)
# =============================================================================

_ARCHMAP_TRIGGER_SET = {
    "create architecture map",
    "arch map",
    "architecture map",
    "/arch_map",
    "/architecture_map",
    "/create_architecture_map",
}

# Model/provider for this tool only (does NOT affect normal chat routing)
ARCHMAP_PROVIDER = os.getenv("ORB_ZOBIE_ARCHMAP_PROVIDER", "openai")

# IMPORTANT: must be chat-capable if your OpenAI streaming uses chat-completions
ARCHMAP_MODEL = os.getenv(
    "ORB_ZOBIE_ARCHMAP_MODEL",
    os.getenv("OPENAI_MODEL_HEAVY_TEXT", os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5-mini")),
)

# Optional: path on HOST to a style template (your big map)
ARCHMAP_TEMPLATE_PATH = os.getenv("ORB_ZOBIE_ARCHMAP_TEMPLATE_PATH", "")

# Evidence limits (deterministic + safe)
ARCHMAP_MAX_FILES = int(os.getenv("ORB_ZOBIE_ARCHMAP_MAX_FILES", "200000"))
ARCHMAP_MAX_CODE_FILES = int(os.getenv("ORB_ZOBIE_ARCHMAP_MAX_CODE_FILES", "120"))
ARCHMAP_MAX_CHARS_PER_FILE = int(os.getenv("ORB_ZOBIE_ARCHMAP_MAX_CHARS_PER_FILE", "18000"))
ARCHMAP_CONTROLLER_TIMEOUT_SEC = int(os.getenv("ORB_ZOBIE_ARCHMAP_CONTROLLER_TIMEOUT_SEC", "30"))
ARCHMAP_MAX_ARTIFACT_CHARS = int(os.getenv("ORB_ZOBIE_ARCHMAP_MAX_ARTIFACT_CHARS", "200000"))

# Ingest the final map into Memory so other windows can query it
ARCHMAP_INGEST_TO_MEMORY = os.getenv("ORB_ZOBIE_ARCHMAP_INGEST_TO_MEMORY", "1") == "1"
ARCHMAP_DOC_RAW_MAX_CHARS = int(os.getenv("ORB_ZOBIE_ARCHMAP_DOC_RAW_MAX_CHARS", "400000"))

# Do NOT pull secrets (controller allows it, so tool must refuse)
_DENY_FILE_PATTERNS = [
    r"(^|/)\.env($|/)",
    r"\.pem$",
    r"\.key$",
    r"\.pfx$",
    r"\.p12$",
    r"\.p8$",
    r"(^|/)\.git($|/)",
    r"(^|/)id_rsa($|/)",
    r"(^|/)known_hosts($|/)",
    r"secrets?",
    r"credentials?",
    r"token",
    r"apikey",
    r"api_key",
]


def _is_zobie_map_trigger(msg: str) -> bool:
    s = (msg or "").strip().lower()
    return s in _ZOBIE_TRIGGER_SET


def _is_archmap_trigger(msg: str) -> bool:
    s = (msg or "").strip().lower()
    return s in _ARCHMAP_TRIGGER_SET


def _chunk_text(s: str, chunk_size: int = 80) -> List[str]:
    if not s:
        return []
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]


def _is_denied_repo_path(p: str) -> bool:
    p2 = (p or "").replace("\\", "/").lower()
    return any(re.search(pat, p2) for pat in _DENY_FILE_PATTERNS)


def _safe_read_text(path: str, max_bytes: int = 2_000_000) -> str:
    try:
        with open(path, "rb") as f:
            b = f.read(max_bytes)
        return b.decode("utf-8", errors="replace")
    except Exception as e:
        return f"<<failed to read {path}: {e}>>"


def _cap_text(label: str, text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n<<truncated {label}: {len(text)} chars total>>\n"


def _line_number(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines, start=1):
        out.append(f"{i:05d}| {line}")
    return "\n".join(out)


def _extract_mapper_stamp(tree_path: str) -> str:
    # expects REPO_TREE_YYYY-MM-DD_HHMM.txt
    base = os.path.basename(tree_path or "")
    m = re.search(r"REPO_TREE_(\d{4}-\d{2}-\d{2}_\d{4})", base)
    return m.group(1) if m else ""


def _next_archmap_version(out_dir: str) -> int:
    # looks for ARCH_MAP_FULL_V{N}_*.md
    try:
        existing = []
        for name in os.listdir(out_dir):
            m = re.match(r"ARCH_MAP_FULL_V(\d+)_", name)
            if m:
                existing.append(int(m.group(1)))
        return (max(existing) + 1) if existing else 1
    except Exception:
        return 1


def _controller_http_json(url: str):
    """
    stdlib-only JSON fetch.
    """
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError

    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=ARCHMAP_CONTROLLER_TIMEOUT_SEC) as r:
            raw = r.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from {url}") from e
    except URLError as e:
        raise RuntimeError(f"Network error calling {url}: {e}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Bad JSON from {url}: {e}") from e


def _controller_fetch_repo_tree(max_files: int) -> List[str]:
    tree_url = f"{ZOBIE_CONTROLLER_URL}/repo/tree?include_hashes=false&max_files={int(max_files)}"
    tree = _controller_http_json(tree_url)
    return [x["path"] for x in tree if isinstance(x, dict) and "path" in x]


def _controller_fetch_file_content(repo_path: str) -> str:
    """
    Fetch repo file content from sandbox controller.
    Applies deny rules and per-file size cap.
    """
    from urllib.parse import quote

    if _is_denied_repo_path(repo_path):
        return "<<skipped: denied path>>"

    data = _controller_http_json(f"{ZOBIE_CONTROLLER_URL}/repo/file?path={quote(repo_path)}")
    content = data.get("content", "") or ""

    if len(content) > ARCHMAP_MAX_CHARS_PER_FILE:
        content = content[:ARCHMAP_MAX_CHARS_PER_FILE] + "\n\n<<truncated>>\n"

    # Line-numbering helps the LLM reference “where” something is.
    return _line_number(content)


def _select_archmap_code_files(all_paths: List[str]) -> List[str]:
    """
    Deterministic selection of "evidence" files that drive architecture maps.
    This does NOT guess contents; it just selects paths by patterns.
    """
    wanted: List[str] = []
    path_set = set(all_paths)

    explicit = [
        "main.py",
        "app/main.py",
        "app/router.py",
        "app/llm/router.py",
        "app/llm/stream_router.py",
        "app/llm/streaming.py",
        "app/llm/web_search_router.py",
        "app/llm/telemetry_router.py",
        "app/auth/router.py",
        "app/memory/router.py",
        "app/memory/service.py",
        "app/memory/models.py",
        "app/embeddings/router.py",
        "orb-desktop/main.js",
        "orb-desktop/package.json",
        "sandbox_controller/main.py",
    ]
    for p in explicit:
        if p in path_set and not _is_denied_repo_path(p):
            wanted.append(p)

    # Prefer routers + entrypoints
    for p in all_paths:
        if _is_denied_repo_path(p):
            continue
        low = p.lower()
        if low.endswith("router.py") or low.endswith("main.py") or low.endswith("stream_router.py"):
            wanted.append(p)

    # Config/docs
    for p in all_paths:
        if _is_denied_repo_path(p):
            continue
        low = p.lower()
        if low.endswith(("pyproject.toml", "requirements.txt", "package.json", "vite.config.ts", "vite.config.js", "readme.md")):
            wanted.append(p)

    # De-dupe preserve order
    seen = set()
    out = []
    for p in wanted:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)

    return out[:ARCHMAP_MAX_CODE_FILES]


def _ingest_archmap_to_memory(
    db: Session,
    project_id: int,
    file_path: str,
    provider: str,
    model: str,
    raw_map: str,
):
    """
    Store the generated map as:
    - a File record (points to the saved markdown)
    - a DocumentContent record (so _build_document_context can inject it)
    This is what allows “another window” to quiz Orb about the map.
    """
    try:
        filename = os.path.basename(file_path)
        file_rec = memory_service.create_file(
            db,
            memory_schemas.FileCreate(
                project_id=project_id,
                path=file_path,
                original_name=filename,
                file_type="generated/architecture_map",
                description=f"Generated architecture map saved at {file_path}",
            ),
        )

        trimmed = raw_map[:ARCHMAP_DOC_RAW_MAX_CHARS]
        if len(raw_map) > ARCHMAP_DOC_RAW_MAX_CHARS:
            trimmed += "\n\n<<trimmed for DB storage>>\n"

        memory_service.create_document_content(
            db,
            memory_schemas.DocumentContentCreate(
                project_id=project_id,
                file_id=file_rec.id,
                filename=filename,
                doc_type="architecture_map",
                raw_text=trimmed,
                summary=None,
                structured_data=None,
            ),
        )

        # Also leave a small assistant message breadcrumb (lightweight)
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id,
                role="assistant",
                content=f"[archmap] Saved + ingested: {file_path}",
                provider=provider,
                model=model,
            ),
        )
    except Exception as e:
        logger.exception("[archmap] ingest failed: %s", e)


def _parse_reasoning_tags(raw: str) -> tuple:
    thinking_match = re.search(r"<THINKING>([\s\S]*?)</THINKING>", raw, re.IGNORECASE)
    answer_match = re.search(r"<ANSWER>([\s\S]*?)</ANSWER>", raw, re.IGNORECASE)

    if thinking_match and answer_match:
        reasoning = thinking_match.group(1).strip()
        answer = answer_match.group(1).strip()
        return answer, reasoning

    cleaned = re.sub(r"</?THINKING[^>]*>", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?ANSWER[^>]*>", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()

    return cleaned if cleaned else raw, ""


def _make_session_id(auth: AuthResult) -> str:
    for attr in ("session_id", "sid", "session", "session_token"):
        try:
            v = getattr(auth, attr, None)
            if v:
                return str(v)
        except Exception:
            pass
    try:
        user = getattr(auth, "user", None)
        if isinstance(user, dict):
            return str(user.get("id") or user.get("email") or user.get("username") or "")
    except Exception:
        pass
    return f"legacy-{uuid.uuid4()}"


def _coerce_int(v) -> int:
    try:
        if v is None:
            return 0
        return int(v)
    except Exception:
        return 0


def _extract_usage_tokens(usage_obj) -> tuple[int, int]:
    if usage_obj is None:
        return (0, 0)
    if isinstance(usage_obj, dict):
        pt = usage_obj.get("prompt_tokens") or usage_obj.get("input_tokens") or usage_obj.get("prompt")
        ct = usage_obj.get("completion_tokens") or usage_obj.get("output_tokens") or usage_obj.get("completion")
        return (_coerce_int(pt), _coerce_int(ct))
    pt = getattr(usage_obj, "prompt_tokens", None) or getattr(usage_obj, "input_tokens", None)
    ct = getattr(usage_obj, "completion_tokens", None) or getattr(usage_obj, "output_tokens", None)
    return (_coerce_int(pt), _coerce_int(ct))


async def generate_local_zobie_map_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
):
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    trace_finished = False

    provider = "local"
    model = "zobie_mapper"

    os.makedirs(ZOBIE_MAPPER_OUT_DIR, exist_ok=True)

    header = (
        "Running ZOBIE MAP (local tool)\n"
        f"- Controller: {ZOBIE_CONTROLLER_URL}\n"
        f"- Script: {ZOBIE_MAPPER_SCRIPT}\n"
        f"- Out dir: {ZOBIE_MAPPER_OUT_DIR}\n"
        f"- Extra args: {ZOBIE_MAPPER_ARGS_RAW or '(none)'}\n\n"
    )
    for ch in _chunk_text(header, 120):
        yield f"data: {json.dumps({'type': 'token', 'content': ch})}\n\n"

    cmd = [sys.executable, ZOBIE_MAPPER_SCRIPT, ZOBIE_CONTROLLER_URL, ZOBIE_MAPPER_OUT_DIR] + ZOBIE_MAPPER_ARGS

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.getcwd(),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=ZOBIE_MAPPER_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            raise RuntimeError(f"ZOBIE MAP timed out after {ZOBIE_MAPPER_TIMEOUT_SEC}s")

        rc = int(proc.returncode or 0)
        stdout = (stdout_b or b"").decode("utf-8", errors="replace")
        stderr = (stderr_b or b"").decode("utf-8", errors="replace")

        duration_ms = max(0, int(loop.time() * 1000) - started_ms)

        if rc != 0:
            err_text = (
                "ZOBIE MAP failed\n"
                f"- returncode: {rc}\n"
                f"- stderr:\n{stderr.strip() or '(empty)'}\n"
            )
            if trace and not trace_finished:
                trace.log_model_call("primary", provider, model, "primary", 0, 0, duration_ms, success=False, error=err_text[:500])
                trace.finalize(success=False, error_message=err_text[:500])
                trace_finished = True

            memory_service.create_message(db, memory_schemas.MessageCreate(project_id=project_id, role="user", content=message, provider="local"))
            memory_service.create_message(db, memory_schemas.MessageCreate(project_id=project_id, role="assistant", content=err_text, provider=provider, model=model))

            for ch in _chunk_text(err_text, 120):
                yield f"data: {json.dumps({'type': 'token', 'content': ch})}\n\n"
            yield f"data: {json.dumps({'type': 'error', 'error': 'ZOBIE MAP failed'})}\n\n"
            return

        wrote_lines = []
        capture = False
        for line in stdout.splitlines():
            if line.strip().lower() == "wrote:":
                capture = True
                continue
            if capture and line.strip():
                wrote_lines.append(line.strip())

        summary = "ZOBIE MAP complete.\n\nOutputs:\n"
        summary += "\n".join(f"- {p}" for p in wrote_lines) + "\n" if wrote_lines else f"- (Check {ZOBIE_MAPPER_OUT_DIR})\n"

        memory_service.create_message(db, memory_schemas.MessageCreate(project_id=project_id, role="user", content=message, provider="local"))
        memory_service.create_message(db, memory_schemas.MessageCreate(project_id=project_id, role="assistant", content=summary, provider=provider, model=model))

        if trace and not trace_finished:
            trace.log_model_call("primary", provider, model, "primary", 0, 0, duration_ms, success=True, error=None)
            trace.finalize(success=True)
            trace_finished = True

        for ch in _chunk_text(summary, 120):
            yield f"data: {json.dumps({'type': 'token', 'content': ch})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'provider': provider, 'model': model, 'total_length': len(summary)})}\n\n"
        return

    except asyncio.CancelledError:
        if trace and not trace_finished:
            trace.log_warning("STREAM", "client_disconnect")
            trace.finalize(success=False, error_message="client_disconnect")
            trace_finished = True
        raise
    except Exception as e:
        duration_ms = max(0, int(loop.time() * 1000) - started_ms)
        logger.exception("[stream] Local ZOBIE MAP failed: %s", e)

        if trace and not trace_finished:
            trace.log_model_call("primary", provider, model, "primary", 0, 0, duration_ms, success=False, error=str(e))
            trace.finalize(success=False, error_message=str(e))
            trace_finished = True

        err_text = f"ZOBIE MAP error: {e}"
        try:
            memory_service.create_message(db, memory_schemas.MessageCreate(project_id=project_id, role="user", content=message, provider="local"))
            memory_service.create_message(db, memory_schemas.MessageCreate(project_id=project_id, role="assistant", content=err_text, provider=provider, model=model))
        except Exception:
            pass

        for ch in _chunk_text(err_text, 120):
            yield f"data: {json.dumps({'type': 'token', 'content': ch})}\n\n"
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        return


async def generate_local_architecture_map_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
):
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    trace_finished = False

    provider = ARCHMAP_PROVIDER
    model = ARCHMAP_MODEL

    # Hard guard: if you use OpenAI chat-completions, a non-chat model will fail exactly like you saw.
    if provider == "openai" and ("-pro" in model and "chat" not in model):
        raise RuntimeError(
            f"OpenAI model '{model}' is not chat-capable for this endpoint. "
            f"Set ORB_ZOBIE_ARCHMAP_MODEL to a chat model (e.g. gpt-5.2-chat-latest)."
        )

    os.makedirs(ZOBIE_MAPPER_OUT_DIR, exist_ok=True)

    header = (
        "Running CREATE ARCHITECTURE MAP (local tool)\n\n"
        f"Controller: {ZOBIE_CONTROLLER_URL}\n"
        f"Mapper: {ZOBIE_MAPPER_SCRIPT}\n"
        f"Out dir: {ZOBIE_MAPPER_OUT_DIR}\n"
        f"LLM: {provider}/{model}\n"
        f"Extra mapper args: {ZOBIE_MAPPER_ARGS_RAW or '(none)'}\n\n"
    )
    for ch in _chunk_text(header, 120):
        yield f"data: {json.dumps({'type': 'token', 'content': ch})}\n\n"

    try:
        # 1) Run mapper subprocess
        yield f"data: {json.dumps({'type': 'token', 'content': 'Step 1/4: Running mapper...\\n'})}\n\n"
        cmd = [sys.executable, ZOBIE_MAPPER_SCRIPT, ZOBIE_CONTROLLER_URL, ZOBIE_MAPPER_OUT_DIR] + ZOBIE_MAPPER_ARGS

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.getcwd(),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=ZOBIE_MAPPER_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            raise RuntimeError(f"Mapper timed out after {ZOBIE_MAPPER_TIMEOUT_SEC}s")

        rc = int(proc.returncode or 0)
        stdout = (stdout_b or b"").decode("utf-8", errors="replace")
        stderr = (stderr_b or b"").decode("utf-8", errors="replace")
        if rc != 0:
            raise RuntimeError(f"Mapper failed (rc={rc}): {stderr.strip() or '(empty)'}")

        wrote_lines: List[str] = []
        capture = False
        for line in stdout.splitlines():
            if line.strip().lower() == "wrote:":
                capture = True
                continue
            if capture and line.strip():
                wrote_lines.append(line.strip())

        tree_path = next((p for p in wrote_lines if "REPO_TREE_" in p and p.lower().endswith(".txt")), "")
        arch_path = next((p for p in wrote_lines if "ARCH_MAP_" in p and p.lower().endswith(".md")), "")
        index_path = next((p for p in wrote_lines if "INDEX_" in p and p.lower().endswith(".json")), "")

        if not tree_path:
            raise RuntimeError("Mapper outputs missing: REPO_TREE_*.txt not found in mapper stdout")
        if not arch_path:
            raise RuntimeError("Mapper outputs missing: ARCH_MAP_*.md not found in mapper stdout")
        if not index_path:
            raise RuntimeError("Mapper outputs missing: INDEX_*.json not found in mapper stdout")

        stamp = _extract_mapper_stamp(tree_path)

        # 2) Build evidence pack
        yield f"data: {json.dumps({'type': 'token', 'content': 'Step 2/4: Building evidence pack...\\n'})}\n\n"

        repo_tree_txt = _cap_text("REPO_TREE", _safe_read_text(tree_path), ARCHMAP_MAX_ARTIFACT_CHARS)
        arch_map_md = _cap_text("ARCH_MAP", _safe_read_text(arch_path), ARCHMAP_MAX_ARTIFACT_CHARS)
        index_raw = _cap_text("INDEX", _safe_read_text(index_path), ARCHMAP_MAX_ARTIFACT_CHARS)

        # Include extra “full mode” artifacts if mapper produced them (same stamp)
        extra_artifacts = {}
        if stamp:
            candidates = {
                "TREE_JSON": os.path.join(ZOBIE_MAPPER_OUT_DIR, f"TREE_{stamp}.json"),
                "SYMBOL_INDEX": os.path.join(ZOBIE_MAPPER_OUT_DIR, f"SYMBOL_INDEX_{stamp}.json"),
                "IMPORT_GRAPH": os.path.join(ZOBIE_MAPPER_OUT_DIR, f"IMPORT_GRAPH_{stamp}.json"),
                "ENV_MAP": os.path.join(ZOBIE_MAPPER_OUT_DIR, f"ENV_MAP_{stamp}.json"),
                "ROUTE_MAP": os.path.join(ZOBIE_MAPPER_OUT_DIR, f"ROUTE_MAP_{stamp}.json"),
                "CALLGRAPH_EDGES": os.path.join(ZOBIE_MAPPER_OUT_DIR, f"CALLGRAPH_EDGES_{stamp}.json"),
                "EXCERPTS": os.path.join(ZOBIE_MAPPER_OUT_DIR, f"EXCERPTS_{stamp}.jsonl"),
                "STATS": os.path.join(ZOBIE_MAPPER_OUT_DIR, f"STATS_{stamp}.json"),
                "EVIDENCE_MANIFEST": os.path.join(ZOBIE_MAPPER_OUT_DIR, f"EVIDENCE_MANIFEST_{stamp}.json"),
            }
            for k, p in candidates.items():
                if os.path.exists(p):
                    extra_artifacts[k] = _cap_text(k, _safe_read_text(p, max_bytes=3_000_000), ARCHMAP_MAX_ARTIFACT_CHARS)

        # Fetch full repo tree + select code files from controller
        all_paths = _controller_fetch_repo_tree(max_files=ARCHMAP_MAX_FILES)
        selected_files = _select_archmap_code_files(all_paths)

        code_blocks: List[str] = []
        for p in selected_files:
            content = _controller_fetch_file_content(p)
            code_blocks.append(f"=== FILE: {p} ===\n{content}\n")

        template_txt = ""
        if ARCHMAP_TEMPLATE_PATH and os.path.exists(ARCHMAP_TEMPLATE_PATH):
            template_txt = _safe_read_text(ARCHMAP_TEMPLATE_PATH, max_bytes=1_500_000)

        # Versioning (starts at V1 and increments)
        version_n = _next_archmap_version(ZOBIE_MAPPER_OUT_DIR)

        # 3) Call LLM and stream output
        yield f"data: {json.dumps({'type': 'token', 'content': 'Step 3/4: Generating architecture map with LLM...\\n\\n'})}\n\n"

        system_prompt = (
            "You are Orb's repository architecture mapper.\n"
            "Hard rules:\n"
            "- Use ONLY the evidence provided.\n"
            "- If a detail is not explicitly supported, write: 'Unknown (not in provided data)'.\n"
            "- Prefer explicit paths, symbol/route tables, env var tables, import graph, and line references.\n"
            "- Output ONE Markdown document.\n"
            f"- Title the document: 'Orb Architecture Map v{version_n}'.\n"
        )
        if template_txt:
            system_prompt += "\nSTYLE TEMPLATE (follow its tone/organization, but DO NOT copy any 'v31' wording):\n" + template_txt + "\nEND TEMPLATE\n"

        extras_block = ""
        if extra_artifacts:
            parts = []
            for k in sorted(extra_artifacts.keys()):
                parts.append(f"---- {k} ----\n{extra_artifacts[k]}\n")
            extras_block = "\n".join(parts)

        user_prompt = (
            f"Create a deeply detailed repository architecture map (v{version_n}).\n\n"
            "EVIDENCE PACK:\n\n"
            "---- REPO_TREE (mapper output) ----\n"
            f"{repo_tree_txt}\n\n"
            "---- ARCH_MAP (mapper output) ----\n"
            f"{arch_map_md}\n\n"
            "---- INDEX (mapper output JSON) ----\n"
            f"{index_raw}\n\n"
            + (f"{extras_block}\n\n" if extras_block else "")
            + "---- SELECTED CODE FILES (from sandbox controller; line-numbered) ----\n"
            + f"{''.join(code_blocks)}\n\n"
            "REQUIREMENTS:\n"
            "- No gaps in the explanation of *how the repo works* based on evidence.\n"
            "- Include explicit tables/indices where evidence supports it:\n"
            "  1) Repo layout + entrypoints\n"
            "  2) API route map (method/path -> handler file -> line range if available)\n"
            "  3) Module/symbol index (file -> functions/classes -> line numbers)\n"
            "  4) Env var map (var -> where read -> purpose if evidenced)\n"
            "  5) Startup/run flow (process boundaries: Electron/Backend/Controller)\n"
            "  6) Data/encryption flows (only what is evidenced)\n"
            "- Add an 'Evidence References' section listing exact FILE paths used.\n"
            "- Do not invent anything.\n"
        )

        accumulated = ""
        final_usage = None

        async for event in stream_llm(
            provider=provider,
            model=model,
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
        ):
            if isinstance(event, str):
                event = {"type": "token", "content": event}
            if not isinstance(event, dict):
                event = {"type": "token", "content": str(event)}

            event_type = event.get("type", "token")

            if event_type == "token":
                content = event.get("content") or event.get("text") or ""
                if content:
                    accumulated += content
                    yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

            elif event_type in ("metadata", "usage"):
                u = event.get("usage") or event.get("data")
                if isinstance(u, dict):
                    final_usage = u

            elif event_type == "error":
                error_msg = event.get("message") or event.get("error") or "Unknown LLM error"
                raise RuntimeError(error_msg)

            elif event_type == "done":
                u = event.get("usage")
                if isinstance(u, dict):
                    final_usage = u
                break

        # 4) Save + ingest
        yield f"data: {json.dumps({'type': 'token', 'content': '\\n\\nStep 4/4: Saving output...\\n'})}\n\n"

        stamp2 = datetime.now().strftime("%Y-%m-%d_%H%M")
        out_full = os.path.join(ZOBIE_MAPPER_OUT_DIR, f"ARCH_MAP_FULL_V{version_n}_{stamp2}.md")
        with open(out_full, "w", encoding="utf-8") as f:
            f.write(accumulated)

        duration_ms = max(0, int(loop.time() * 1000) - started_ms)

        # Persist user message + completion breadcrumb
        memory_service.create_message(db, memory_schemas.MessageCreate(project_id=project_id, role="user", content=message, provider="local"))
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id,
                role="assistant",
                content=f"CREATE ARCHITECTURE MAP complete.\n\nSaved:\n- {out_full}\n",
                provider=provider,
                model=model,
            ),
        )

        # Ingest into DocumentContent so other windows can query it (via _build_document_context)
        if ARCHMAP_INGEST_TO_MEMORY:
            _ingest_archmap_to_memory(db=db, project_id=project_id, file_path=out_full, provider=provider, model=model, raw_map=accumulated)

        # Audit trace
        if trace and not trace_finished:
            prompt_tokens, completion_tokens = _extract_usage_tokens(final_usage)
            trace.log_model_call("primary", provider, model, "primary", prompt_tokens, completion_tokens, duration_ms, success=True, error=None)
            trace.finalize(success=True)
            trace_finished = True

        yield f"data: {json.dumps({'type': 'token', 'content': f'Saved: {out_full}\\n'})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'provider': provider, 'model': model, 'total_length': len(accumulated)})}\n\n"
        return

    except asyncio.CancelledError:
        if trace and not trace_finished:
            trace.log_warning("STREAM", "client_disconnect")
            trace.finalize(success=False, error_message="client_disconnect")
            trace_finished = True
        raise
    except Exception as e:
        duration_ms = max(0, int(loop.time() * 1000) - started_ms)
        logger.exception("[stream] Local CREATE ARCHITECTURE MAP failed: %s", e)

        if trace and not trace_finished:
            trace.log_model_call("primary", provider, model, "primary", 0, 0, duration_ms, success=False, error=str(e))
            trace.finalize(success=False, error_message=str(e))
            trace_finished = True

        err_text = f"CREATE ARCHITECTURE MAP error: {e}"
        try:
            memory_service.create_message(db, memory_schemas.MessageCreate(project_id=project_id, role="user", content=message, provider="local"))
            memory_service.create_message(db, memory_schemas.MessageCreate(project_id=project_id, role="assistant", content=err_text, provider=provider, model=model))
        except Exception:
            pass

        for ch in _chunk_text(err_text, 120):
            yield f"data: {json.dumps({'type': 'token', 'content': ch})}\n\n"
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        return


def _build_context_block(db: Session, project_id: int) -> str:
    sections = []
    notes = memory_service.list_notes(db, project_id)[:10]
    if notes:
        notes_text = "\n".join(f"- [{n.id}] {n.title}: {n.content[:200]}..." for n in notes)
        sections.append(f"PROJECT NOTES:\n{notes_text}")

    tasks = memory_service.list_tasks(db, project_id, status="pending")[:10]
    if tasks:
        tasks_text = "\n".join(f"- {t.title}" for t in tasks)
        sections.append(f"PENDING TASKS:\n{tasks_text}")

    return "\n\n".join(sections) if sections else ""


def _build_document_context(db: Session, project_id: int) -> str:
    try:
        from app.memory.models import DocumentContent

        recent_docs = (
            db.query(DocumentContent)
            .filter(DocumentContent.project_id == project_id)
            .order_by(DocumentContent.created_at.desc())
            .limit(5)
            .all()
        )

        if not recent_docs:
            return ""

        context_parts = []
        for doc in recent_docs:
            summary = doc.summary[:500] if doc.summary else ""
            raw_preview = doc.raw_text[:1000] if doc.raw_text else ""
            if summary or raw_preview:
                context_parts.append(f"[{doc.filename}]:\nSummary: {summary}\nContent: {raw_preview}...")

        return "\n\n".join(context_parts)
    except Exception as e:
        print(f"[stream_router] Error building document context: {e}")
        return ""


def _get_semantic_context(db: Session, project_id: int, query: str) -> str:
    try:
        from app.embeddings import service as embeddings_service

        results = embeddings_service.search(db=db, project_id=project_id, query=query, top_k=5)
        if not results:
            return ""

        context_parts = ["=== RELEVANT CONTEXT (semantic search) ==="]
        for result in results:
            content_preview = result.content[:200] + "..." if len(result.content) > 200 else result.content
            context_parts.append(f"\n[Score: {result.similarity:.3f}] {content_preview}")

        return "\n".join(context_parts)
    except Exception as e:
        print(f"[stream_router] Semantic search failed: {e}")
        return ""


def _classify_job_type(message: str, requested_type: str) -> JobType:
    if requested_type and requested_type != "casual_chat":
        try:
            return JobType(requested_type)
        except ValueError:
            pass

    msg_lower = message.lower()
    print(f"[stream_router] Classifying message (first 200 chars): {repr(message[:200])}")

    security_keywords = [
        "security review",
        "security audit",
        "security assessment",
        "penetration test",
        "pentest",
        "threat model",
        "threat modeling",
        "vulnerability",
        "vulnerabilities",
        "vulnerability assessment",
        "exploit",
        "attack vector",
        "attack surface",
        "sql injection",
        "xss",
        "csrf",
        "authentication bypass",
        "privilege escalation",
        "session fixation",
        "session hijacking",
        "security analysis",
        "security check",
        "encryption review",
        "key management",
        "secrets management",
        "authentication security",
        "authorization security",
        "security hardening",
        "security posture",
    ]

    arch_keywords = [
        "architect",
        "architecture",
        "design a system",
        "system design",
        "microservice",
        "micro-service",
        "infrastructure",
        "infra",
        "scalab",
        "database schema",
        "db schema",
        "api design",
        "high-level design",
        "hld",
        "distributed system",
        "design pattern",
        "tech stack",
    ]

    review_keywords = [
        "review this",
        "review my",
        "code review",
        "check this code",
        "find bugs",
        "audit this",
        "critique",
        "what's wrong with",
    ]

    code_keywords = [
        "write a function",
        "write code",
        "implement",
        "debug",
        "fix this code",
        "refactor",
        "def ",
        "function ",
        "```",
    ]

    language_keywords = [
        "python",
        "javascript",
        "typescript",
        "java",
        "c++",
        "rust",
        "react",
        "vue",
        "fastapi",
        "django",
    ]

    if any(kw in msg_lower for kw in security_keywords):
        print("[stream_router] Classified: SECURITY_REVIEW (explicit security keyword)")
        return JobType.SECURITY_REVIEW

    if any(kw in msg_lower for kw in arch_keywords):
        print("[stream_router] Classified: ARCHITECTURE_DESIGN")
        return JobType.ARCHITECTURE_DESIGN

    if any(kw in msg_lower for kw in review_keywords):
        print("[stream_router] Classified: CODE_REVIEW")
        return JobType.CODE_REVIEW

    is_code_related = any(kw in msg_lower for kw in code_keywords) or any(kw in msg_lower for kw in language_keywords)
    if is_code_related:
        complex_indicators = ["complex", "full file", "entire file", "production"]
        if any(x in msg_lower for x in complex_indicators):
            print("[stream_router] Classified: COMPLEX_CODE_CHANGE")
            return JobType.COMPLEX_CODE_CHANGE
        print("[stream_router] Classified: SIMPLE_CODE_CHANGE")
        return JobType.SIMPLE_CODE_CHANGE

    print("[stream_router] Classified: CASUAL_CHAT (default)")
    return JobType.CASUAL_CHAT


def _select_provider_for_job_type(job_type: JobType) -> tuple:
    if job_type in RoutingConfig.GPT_ONLY_JOBS:
        return ("openai", DEFAULT_MODELS["openai"])

    if job_type in RoutingConfig.HIGH_STAKES_JOBS:
        print(f"[stream_router] High-stakes job '{job_type.value}' → Opus")
        return ("anthropic", DEFAULT_MODELS["anthropic_opus"])

    if job_type in RoutingConfig.CLAUDE_PRIMARY_JOBS:
        return ("anthropic", DEFAULT_MODELS["anthropic"])

    if job_type == JobType.DEEP_RESEARCH:
        return ("gemini", DEFAULT_MODELS["gemini"])

    provider_key = os.getenv("ORB_DEFAULT_PROVIDER", "anthropic")
    return (provider_key, DEFAULT_MODELS.get(provider_key, DEFAULT_MODELS["openai"]))


class StreamChatRequest(BaseModel):
    project_id: int
    message: str
    provider: Optional[str] = None
    model: Optional[str] = None
    job_type: Optional[str] = None
    requested_type: Optional[str] = None
    include_history: bool = True
    history_limit: int = 20
    use_semantic_search: bool = False
    enable_reasoning: bool = False


async def generate_high_stakes_critique_stream(
    project_id: int,
    message: str,
    provider: str,
    model: str,
    system_prompt: str,
    messages: List[dict],
    full_context: str,
    job_type_str: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
    enable_reasoning: bool = False,
):
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    trace_finished = False

    try:
        task = LLMTask(
            project_id=project_id,
            user_message=message,
            system_prompt=system_prompt,
            messages=messages,
            full_context=full_context,
            job_type=job_type_str,
            enable_reasoning=enable_reasoning,
        )

        result = await run_high_stakes_with_critique(
            task=task,
            provider_id=provider,
            model_id=model,
            envelope=synthesize_envelope_from_task(task),
            job_type_str=job_type_str,
        )

        usage_obj = getattr(result, "usage", None)
        if usage_obj is None and isinstance(result, dict):
            usage_obj = result.get("usage")
        hs_prompt_tokens, hs_completion_tokens = _extract_usage_tokens(usage_obj)
        duration_ms = max(0, int(loop.time() * 1000) - started_ms)

        error_message = getattr(result, "error_message", None)
        content = getattr(result, "content", None)
        if isinstance(result, dict):
            error_message = error_message or result.get("error_message") or result.get("error")
            content = content if content is not None else result.get("content")

        if error_message:
            if trace and not trace_finished:
                trace.log_model_call("primary", provider, model, "primary", hs_prompt_tokens, hs_completion_tokens, duration_ms, success=False, error=str(error_message))
                trace.finalize(success=False, error_message=str(error_message))
                trace_finished = True
            yield f"data: {json.dumps({'type': 'error', 'error': str(error_message)})}\n\n"
            return

        content = content or ""
        final_answer, reasoning = _parse_reasoning_tags(content)

        if trace and not trace_finished:
            trace.log_model_call("primary", provider, model, "primary", hs_prompt_tokens, hs_completion_tokens, duration_ms, success=True)

        chunk_size = 50
        for i in range(0, len(final_answer), chunk_size):
            chunk = final_answer[i : i + chunk_size]
            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            await asyncio.sleep(0.01)

        if enable_reasoning and reasoning:
            yield f"data: {json.dumps({'type': 'reasoning', 'content': reasoning})}\n\n"

        memory_service.create_message(db, memory_schemas.MessageCreate(project_id=project_id, role="user", content=message, provider="local"))
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id,
                role="assistant",
                content=final_answer,
                provider=provider,
                model=model,
                reasoning=reasoning if reasoning else None,
            ),
        )

        if trace and not trace_finished:
            trace.finalize(success=True)
            trace_finished = True

        yield f"data: {json.dumps({'type': 'done', 'provider': provider, 'model': model, 'total_length': len(final_answer)})}\n\n"

    except asyncio.CancelledError:
        if trace and not trace_finished:
            trace.log_warning("STREAM", "client_disconnect")
            trace.finalize(success=False, error_message="client_disconnect")
            trace_finished = True
        raise
    except Exception as e:
        logger.exception("[stream] High-stakes stream failed: %s", e)
        if trace and not trace_finished:
            trace.log_model_call("primary", provider, model, "primary", 0, 0, 0, success=False, error=str(e))
            trace.finalize(success=False, error_message=str(e))
            trace_finished = True
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        return


async def generate_sse_stream(
    project_id: int,
    message: str,
    provider: str,
    model: str,
    system_prompt: str,
    messages: List[dict],
    db: Session,
    trace: Optional[RoutingTrace] = None,
    enable_reasoning: bool = False,
):
    accumulated = ""
    reasoning_content = ""
    current_provider = provider
    current_model = model
    final_usage = None

    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    trace_finished = False

    try:
        async for event in stream_llm(provider=provider, model=model, messages=messages, system_prompt=system_prompt):
            if isinstance(event, str):
                event = {"type": "token", "content": event}
            if not isinstance(event, dict):
                event = {"type": "token", "content": str(event)}

            event_type = event.get("type", "token")

            if event_type == "token":
                content = event.get("content") or event.get("text") or ""
                if content:
                    accumulated += content
                    yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

            elif event_type == "reasoning":
                content = event.get("content") or ""
                if content:
                    reasoning_content += content
                    if enable_reasoning:
                        yield f"data: {json.dumps({'type': 'reasoning', 'content': content})}\n\n"

            elif event_type == "metadata":
                maybe_provider = event.get("provider")
                maybe_model = event.get("model")
                if maybe_provider:
                    current_provider = str(maybe_provider)
                if maybe_model:
                    current_model = str(maybe_model)
                u = event.get("usage")
                if isinstance(u, dict):
                    final_usage = u
                if os.getenv("ORB_ROUTER_DEBUG") == "1":
                    yield f"data: {json.dumps({'type': 'metadata', 'provider': current_provider, 'model': current_model})}\n\n"

            elif event_type == "usage":
                u = event.get("usage") or event.get("data")
                if isinstance(u, dict):
                    final_usage = u

            elif event_type == "error":
                error_msg = event.get("message") or event.get("error") or "Unknown error"
                logger.error(f"[stream] Provider error: {error_msg}")
                duration_ms = max(0, int(loop.time() * 1000) - started_ms)
                if trace and not trace_finished:
                    trace.log_model_call("primary", current_provider, current_model, "primary", 0, 0, duration_ms, success=False, error=str(error_msg))
                    trace.finalize(success=False, error_message=str(error_msg))
                    trace_finished = True
                yield f"data: {json.dumps({'type': 'error', 'error': str(error_msg)})}\n\n"
                return

            elif event_type == "done":
                maybe_provider = event.get("provider")
                maybe_model = event.get("model")
                if maybe_provider:
                    current_provider = str(maybe_provider)
                if maybe_model:
                    current_model = str(maybe_model)
                u = event.get("usage")
                if isinstance(u, dict):
                    final_usage = u
                break

            else:
                logger.warning(f"[stream] Unknown event type '{event_type}' from provider: {event}")

    except asyncio.CancelledError:
        if trace and not trace_finished:
            trace.log_warning("STREAM", "client_disconnect")
            trace.finalize(success=False, error_message="client_disconnect")
            trace_finished = True
        raise
    except Exception as e:
        logger.exception("[stream] Stream failed: %s", e)
        duration_ms = max(0, int(loop.time() * 1000) - started_ms)
        if trace and not trace_finished:
            trace.log_model_call("primary", current_provider, current_model, "primary", 0, 0, duration_ms, success=False, error=str(e))
            trace.finalize(success=False, error_message=str(e))
            trace_finished = True
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        return

    duration_ms = max(0, int(loop.time() * 1000) - started_ms)

    answer_content, extracted_reasoning = _parse_reasoning_tags(accumulated)
    if extracted_reasoning:
        reasoning_content = extracted_reasoning

    memory_service.create_message(db, memory_schemas.MessageCreate(project_id=project_id, role="user", content=message, provider="local"))
    memory_service.create_message(
        db,
        memory_schemas.MessageCreate(
            project_id=project_id,
            role="assistant",
            content=answer_content,
            provider=current_provider,
            model=current_model,
            reasoning=reasoning_content or None,
        ),
    )

    if trace and not trace_finished:
        prompt_tokens, completion_tokens = _extract_usage_tokens(final_usage)
        trace.log_model_call("primary", current_provider, current_model, "primary", prompt_tokens, completion_tokens, duration_ms, success=True)
        trace.finalize(success=True)
        trace_finished = True

    yield f"data: {json.dumps({'type': 'done', 'provider': current_provider, 'model': current_model, 'total_length': len(answer_content)})}\n\n"


@router.post("/chat")
async def stream_chat(req: StreamChatRequest, db: Session = Depends(get_db), auth: AuthResult = Depends(require_auth)):
    project = memory_service.get_project(db, req.project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {req.project_id} not found")

    audit = get_audit_logger()
    trace: Optional[RoutingTrace] = None
    request_id = str(uuid.uuid4())
    if audit:
        trace = audit.start_trace(session_id=_make_session_id(auth), project_id=req.project_id, user_text=req.message, request_id=request_id)

    if _is_archmap_trigger(req.message):
        routing_reason = "Local tool trigger: CREATE ARCHITECTURE MAP"
        if trace:
            trace.log_request_start(job_type=req.job_type or "", resolved_job_type="local.architecture_map", provider=ARCHMAP_PROVIDER, model=ARCHMAP_MODEL, reason=routing_reason, frontier_override=False, file_map_injected=False, attachments=None)
            trace.log_routing_decision(job_type="local.architecture_map", provider=ARCHMAP_PROVIDER, model=ARCHMAP_MODEL, reason=routing_reason, frontier_override=False, file_map_injected=False)

        return StreamingResponse(
            generate_local_architecture_map_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if _is_zobie_map_trigger(req.message):
        routing_reason = "Local tool trigger: ZOBIE MAP"
        if trace:
            trace.log_request_start(job_type=req.job_type or "", resolved_job_type="local.zobie_map", provider="local", model="zobie_mapper", reason=routing_reason, frontier_override=False, file_map_injected=False, attachments=None)
            trace.log_routing_decision(job_type="local.zobie_map", provider="local", model="zobie_mapper", reason=routing_reason, frontier_override=False, file_map_injected=False)

        return StreamingResponse(
            generate_local_zobie_map_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    context_block = _build_context_block(db, req.project_id)
    semantic_context = _get_semantic_context(db, req.project_id, req.message) if req.use_semantic_search else ""
    doc_context = _build_document_context(db, req.project_id)

    full_context = ""
    if context_block:
        full_context += context_block + "\n\n"
    if semantic_context:
        full_context += semantic_context + "\n\n"
    if doc_context:
        full_context += "=== UPLOADED DOCUMENTS ===\n" + doc_context

    job_type = _classify_job_type(req.message, req.job_type or "")
    job_type_value = job_type.value

    if req.provider and req.model:
        provider = req.provider
        model = req.model
        routing_reason = "Explicit provider+model from request"
    elif req.provider:
        provider = req.provider
        model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])
        routing_reason = "Explicit provider from request (default model)"
    else:
        provider, model = _select_provider_for_job_type(job_type)
        routing_reason = f"Job-type routing: {job_type_value} -> {provider}/{model}"

    available = get_available_streaming_provider()
    if not available:
        if trace:
            trace.log_error("STREAM", "no_provider_available")
            trace.finalize(success=False, error_message="No LLM provider available")
        raise HTTPException(status_code=503, detail="No LLM provider available")

    from .streaming import get_available_streaming_providers
    providers_available = get_available_streaming_providers()
    if not providers_available.get(provider, False):
        provider = available
        model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])
        routing_reason = f"{routing_reason} | fallback_to={provider}/{model}"

    if trace:
        trace.log_request_start(job_type=req.job_type or "", resolved_job_type=job_type_value, provider=provider, model=model, reason=routing_reason, frontier_override=False, file_map_injected=False, attachments=None)
        trace.log_routing_decision(job_type=job_type_value, provider=provider, model=model, reason=routing_reason, frontier_override=False, file_map_injected=False)

    messages: List[dict] = []
    if req.include_history:
        history = memory_service.list_messages(db, req.project_id, limit=req.history_limit)
        messages = [{"role": msg.role, "content": msg.content} for msg in history]
    messages.append({"role": "user", "content": req.message})

    system_prompt = f"Project: {project.name}."
    if project.description:
        system_prompt += f" {project.description}"

    if full_context:
        system_prompt += f"""

You have access to the following context from this project:

{full_context}

Use this context to answer the user's questions. If asked about people, documents,
or information that appears in the context above, use that information to respond.
Do NOT claim you don't have information if it's present in the context."""

    if provider == "anthropic" and is_opus_model(model) and is_high_stakes_job(job_type_value):
        return StreamingResponse(
            generate_high_stakes_critique_stream(
                project_id=req.project_id,
                message=req.message,
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                messages=messages,
                full_context=full_context,
                job_type_str=job_type_value,
                db=db,
                trace=trace,
                enable_reasoning=req.enable_reasoning,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        generate_sse_stream(
            project_id=req.project_id,
            message=req.message,
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            db=db,
            trace=trace,
            enable_reasoning=req.enable_reasoning,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
