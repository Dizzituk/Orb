# FILE: app/llm/local_tools/archmap_helpers.py
"""Helpers and configuration for the CREATE ARCHITECTURE MAP local tool.

Extracted from app.llm.stream_router to make the streaming router thinner and
easier to reason about. Behaviour is unchanged; this module simply centralises
ARCHMAP configuration and helper functions.
"""

import os
import re
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session
from app.memory import service as memory_service, schemas as memory_schemas

logger = logging.getLogger(__name__)

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

# Evidence tuning (hard-capped inside code to avoid context overflow)
ARCHMAP_MAX_FILES = int(os.getenv("ORB_ZOBIE_ARCHMAP_MAX_FILES", "200000"))
ARCHMAP_MAX_CODE_FILES = int(os.getenv("ORB_ZOBIE_ARCHMAP_MAX_CODE_FILES", "120"))
ARCHMAP_MAX_CHARS_PER_FILE = int(os.getenv("ORB_ZOBIE_ARCHMAP_MAX_CHARS_PER_FILE", "18000"))
ARCHMAP_MAX_ARTIFACT_CHARS = int(os.getenv("ORB_ZOBIE_ARCHMAP_MAX_ARTIFACT_CHARS", "200000"))

# Controller base URL (sandbox controller)
ARCHMAP_CONTROLLER_BASE_URL = os.getenv("ORB_ZOBIE_CONTROLLER_URL", "http://192.168.250.2:8765")

# Controller call timeout (artifact fetch)
ARCHMAP_CONTROLLER_TIMEOUT_SEC = int(os.getenv("ORB_ZOBIE_ARCHMAP_CONTROLLER_TIMEOUT_SEC", "30"))

# Output behavior (Option 2: write files, only brief UI output)
ARCHMAP_STREAM_TOKENS_TO_CLIENT = os.getenv("ORB_ZOBIE_ARCHMAP_STREAM_TOKENS_TO_CLIENT", "0") == "1"

# Sectioned generation (prevents context_length_exceeded)
ARCHMAP_SECTION_MODE = os.getenv("ORB_ZOBIE_ARCHMAP_SECTION_MODE", "1") == "1"
ARCHMAP_SECTION_MAX_SCANNED_FILES = int(os.getenv("ORB_ZOBIE_ARCHMAP_SECTION_MAX_SCANNED_FILES", "60"))
ARCHMAP_SECTION_MAX_CHARS = int(os.getenv("ORB_ZOBIE_ARCHMAP_SECTION_MAX_CHARS", "120000"))

# OpenAI direct-call settings for this tool (bypasses max_tokens drift)
ARCHMAP_OPENAI_URL = os.getenv("ORB_ZOBIE_OPENAI_URL", "https://api.openai.com/v1/chat/completions")
ARCHMAP_OPENAI_TIMEOUT_SEC = int(os.getenv("ORB_ZOBIE_OPENAI_TIMEOUT_SEC", "180"))
ARCHMAP_OPENAI_MAX_COMPLETION_TOKENS = int(os.getenv("ORB_ZOBIE_OPENAI_MAX_COMPLETION_TOKENS", "8000"))
ARCHMAP_OPENAI_TEMPERATURE = float(os.getenv("ORB_ZOBIE_OPENAI_TEMPERATURE", "1"))

# Ingest into memory so other windows can query it
ARCHMAP_INGEST_TO_MEMORY = os.getenv("ORB_ZOBIE_ARCHMAP_INGEST_TO_MEMORY", "1") == "1"
ARCHMAP_DOC_RAW_MAX_CHARS = int(os.getenv("ORB_ZOBIE_ARCHMAP_DOC_RAW_MAX_CHARS", "60000"))

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
        existing: List[int] = []
        for name in os.listdir(out_dir):
            m = re.match(r"ARCH_MAP_FULL_V(\d+)_", name)
            if m:
                existing.append(int(m.group(1)))
        return (max(existing) + 1) if existing else 1
    except Exception:
        return 1


def _controller_http_json(url: str) -> Dict[str, Any]:
    """stdlib-only JSON fetch."""
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


def _controller_fetch_file_content(repo_path: str) -> str:
    """Fetch repo file content from sandbox controller, with deny rules and size cap."""
    from urllib.parse import quote

    if _is_denied_repo_path(repo_path):
        return "<<skipped: denied path>>"

    url = f"{ARCHMAP_CONTROLLER_BASE_URL}/repo/file?path={quote(repo_path)}"
    data = _controller_http_json(url)
    content = data.get("content", "") or ""

    if len(content) > ARCHMAP_MAX_CHARS_PER_FILE:
        content = content[:ARCHMAP_MAX_CHARS_PER_FILE] + "\n\n<<truncated>>\n"

    return _line_number(content)


def _select_archmap_code_files(all_paths: List[str]) -> List[str]:
    """Deterministic selection of evidence files for architecture maps."""
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
        if low.endswith(
            (
                "pyproject.toml",
                "requirements.txt",
                "package.json",
                "vite.config.ts",
                "vite.config.js",
                "readme.md",
            )
        ):
            wanted.append(p)

    # De-dupe preserve order
    seen = set()
    out: List[str] = []
    for p in wanted:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)

    hard_cap = min(max(10, ARCHMAP_MAX_CODE_FILES), 160)
    return out[:hard_cap]


def _ingest_generated_markdown_to_memory(
    db: Session,
    project_id: int,
    file_path: str,
    provider: str,
    model: str,
    raw_text: str,
    doc_type: str,
    description: str,
) -> None:
    """Store generated markdown as File + DocumentContent, plus a breadcrumb message."""
    try:
        filename = os.path.basename(file_path)
        file_rec = memory_service.create_file(
            db,
            memory_schemas.FileCreate(
                project_id=project_id,
                path=file_path,
                original_name=filename,
                file_type=f"generated/{doc_type}",
                description=description,
            ),
        )

        trimmed = raw_text[:ARCHMAP_DOC_RAW_MAX_CHARS]
        if len(raw_text) > ARCHMAP_DOC_RAW_MAX_CHARS:
            trimmed += "\n\n<<trimmed for DB storage>>\n"

        memory_service.create_document_content(
            db,
            memory_schemas.DocumentContentCreate(
                project_id=project_id,
                file_id=file_rec.id,
                filename=filename,
                doc_type=doc_type,
                raw_text=trimmed,
                summary=None,
                structured_data=None,
            ),
        )

        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id,
                role="assistant",
                content=f"[{doc_type}] Saved + ingested: {file_path}",
                provider=provider,
                model=model,
            ),
        )
    except Exception as e:
        logger.exception("[archmap] ingest failed: %s", e)


def _load_index_json(index_path: str) -> Dict[str, Any]:
    try:
        raw = _safe_read_text(index_path, max_bytes=5_000_000)
        return json.loads(raw)
    except Exception:
        return {}


def _fmt_scanned_file(sf: Dict[str, Any], max_items: int = 50) -> str:
    """Compact markdown representation of one scanned file record from INDEX_*.json."""
    p = sf.get("path", "")
    out: List[str] = []
    out.append(f"### {p}")
    out.append(f"- language: `{sf.get('language','')}`")
    out.append(f"- bytes: `{sf.get('bytes','')}`")

    sigs = sf.get("signals") or []
    if sigs:
        out.append("- signals:")
        for s in sigs[:12]:
            out.append(f"  - {s}")

    envs = sf.get("env_vars") or []
    if envs:
        out.append("- env vars referenced:")
        out.append("  - " + ", ".join(envs[:50]))

    routes = sf.get("routes") or []
    if routes:
        out.append("- FastAPI routes (decorators):")
        for r in routes[:max_items]:
            out.append(
                f"  - L{r.get('line','?')}: `{r.get('decorator_target','?')}.{r.get('method','?')}` `{r.get('path','?')}`"
            )

    inc = sf.get("include_router_hits") or []
    if inc:
        out.append("- include_router hits:")
        for h in inc[:20]:
            out.append(f"  - L{h.get('line','?')}: {str(h.get('text',''))[:240]}")

    syms = sf.get("symbols") or []
    if syms:
        out.append("- symbols:")
        for s in syms[:max_items]:
            out.append(f"  - L{s.get('line','?')}: `{s.get('kind','?')}` `{s.get('name','?')}`")

    kl = sf.get("key_lines") or []
    if kl:
        out.append("- key lines:")
        for k in kl[:20]:
            out.append(f"  - L{k.get('line','?')}: {str(k.get('text',''))[:260]}")

    out.append("")
    return "\n".join(out)


def _section_plan() -> List[Dict[str, Any]]:
    """Section plan tuned for Orb/Zobie layout."""
    return [
        {
            "name": "System overview & process boundaries",
            "prefixes": ["Orb-backend/", "orb-desktop/", "sandbox_controller/"],
            "limit": 30,
        },
        {
            "name": "Sandbox controller (repo scanner API)",
            "prefixes": ["sandbox_controller/"],
            "limit": 20,
        },
        {
            "name": "Backend entrypoints, FastAPI app wiring, route topology",
            "prefixes": ["Orb-backend/"],
            "contains": ["/main.py", "/router.py", "app/router.py", "app/main.py"],
            "limit": 50,
        },
        {
            "name": "LLM subsystem: routing, streaming, critique pipeline, providers",
            "prefixes": ["Orb-backend/"],
            "contains": ["/app/llm/"],
            "limit": 60,
        },
        {
            "name": "Memory + embeddings: DB models, document ingestion, semantic search",
            "prefixes": ["Orb-backend/"],
            "contains": ["/app/memory/", "/app/embeddings/"],
            "limit": 60,
        },
        {
            "name": "Auth + security + encryption evidence",
            "prefixes": ["Orb-backend/"],
            "contains": ["/app/auth/", "encrypt", "Fernet", "keytar", "Credential"],
            "limit": 60,
        },
        {
            "name": "Desktop (Electron/Vite): boot flow, backend spawn, IPC boundaries",
            "prefixes": ["orb-desktop/"],
            "limit": 40,
        },
    ]


def _pick_scanned_files_for_section(index_data: Dict[str, Any], sec: Dict[str, Any]) -> List[Dict[str, Any]]:
    scanned = index_data.get("scanned_files") or []
    if not isinstance(scanned, list):
        return []

    prefixes = sec.get("prefixes") or []
    contains = sec.get("contains") or []
    limit = int(sec.get("limit") or ARCHMAP_SECTION_MAX_SCANNED_FILES)

    def _match_path(p: str) -> bool:
        if not p:
            return False
        if prefixes and not any(p.startswith(pref) for pref in prefixes):
            return False
        if contains:
            low = p.lower()
            if not any(tok.lower() in low for tok in contains):
                return False
        return True

    matches: List[Dict[str, Any]] = []
    for sf in scanned:
        try:
            p = sf.get("path", "")
            if _match_path(p):
                matches.append(sf)
        except Exception:
            continue

    if not matches and prefixes:
        for sf in scanned:
            p = (sf or {}).get("path", "")
            if any(p.startswith(pref) for pref in prefixes):
                matches.append(sf)

    return matches[:max(5, min(limit, 120))]


def _openai_chat_completion_nonstream(
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_sec: int,
    max_completion_tokens: int,
    temperature: float,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Direct OpenAI call for the archmap tool only.

    Robust to models that reject non-default temperature values.
    Returns (content, usage_dict_or_None)
    """
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_ORB") or ""
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set (required for ORB_ZOBIE_ARCHMAP_PROVIDER=openai).")

    base_payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_completion_tokens": int(max_completion_tokens),
        "stream": False,
    }

    want_temp: Optional[float]
    try:
        t = float(temperature)
        want_temp = t if abs(t - 1.0) > 1e-9 else None
    except Exception:
        want_temp = None

    def _do(payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            ARCHMAP_OPENAI_URL,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(req, timeout=timeout_sec) as r:
                raw = r.read().decode("utf-8", errors="replace")
                return json.loads(raw)
        except HTTPError as e:
            try:
                msg = e.read().decode("utf-8", errors="replace")
            except Exception:
                msg = ""
            raise RuntimeError(f"OpenAI HTTP {e.code}: {msg or str(e)}") from e
        except URLError as e:
            raise RuntimeError(f"OpenAI network error: {e}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"OpenAI bad JSON: {e}") from e

    payload1 = dict(base_payload)
    if want_temp is not None:
        payload1["temperature"] = want_temp

    try:
        data = _do(payload1)
    except RuntimeError as e:
        msg = str(e)
        if ("\"param\": \"temperature\"" in msg) or ("param" in msg and "temperature" in msg and "unsupported" in msg):
            payload2 = dict(base_payload)
            data = _do(payload2)
        else:
            raise

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenAI returned no choices: {data}")

    msg_obj = choices[0].get("message") or {}
    content = msg_obj.get("content") or ""
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
    return content, usage
