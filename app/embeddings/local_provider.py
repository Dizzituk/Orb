# FILE: app/embeddings/local_provider.py
# Purpose: Local embedding provider — OpenAI-compatible /v1/embeddings (text vLLM :8004, multimodal worker :8005).
# Called-by: app.embeddings.provider_router
# Depends-on: app.embeddings.gemini_provider (TaskType + limits only)
# Last-renovated: 2026-07-02 (LANE E)
"""
Local embedding provider (LANE E).

Mirrors gemini_provider's public surface against OpenAI-compatible
/v1/embeddings endpoints:

  - Text: Qwen3-Embedding-0.6B on a second vLLM instance (WSL2, port 8004,
    ~1.2GB fp16) — permanently resident on the 4080 in the INTERACTIVE gap.
    Launch: scripts/embeddings/serve_embed.sh (see its README for gotchas).
  - Image/multimodal: the load-on-demand multimodal worker (port 8005),
    resident only during BACKGROUND_INGEST (app/gpu/orchestrator.py owns it).

Qwen3 embedders are instruction-aware: query-side task types get the
instruction prefix, documents are encoded raw. TaskType (imported from
gemini_provider) stays the cross-provider interface so callers never change.

Failure contract matches gemini_provider: return None (or [None] per item)
with a loud log — never raise into callers. A dead :8004 therefore degrades
exactly like a Gemini outage does today (content picked up later by
reindex_unembedded / the idle retry paths).
"""
from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import httpx

from app.embeddings.gemini_provider import (
    MAX_TEXT_CHARS,
    TaskType,
    _guess_image_mime,
)

logger = logging.getLogger(__name__)

# ─── Defaults (env-overridable, read at call time) ───────────────

DEFAULT_TEXT_MODEL = "qwen3-embedding-0.6b"
DEFAULT_TEXT_BASE_URL = "http://127.0.0.1:8004/v1"
DEFAULT_MULTIMODAL_BASE_URL = "http://127.0.0.1:8005/v1"

# Qwen3-Embedding official asymmetric-retrieval instruction. Queries are
# wrapped as "Instruct: <instruction>\nQuery: <text>"; documents stay raw.
DEFAULT_QUERY_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

# Task types describing the QUERY side of asymmetric retrieval.
QUERY_TASK_TYPES = {
    TaskType.RETRIEVAL_QUERY.value,
    TaskType.CODE_RETRIEVAL_QUERY.value,
    TaskType.QUESTION_ANSWERING.value,
}


def text_base_url() -> str:
    v = (os.getenv("EMBEDDINGS_TEXT_BASE_URL") or "").strip()
    return (v or DEFAULT_TEXT_BASE_URL).rstrip("/")


def text_model() -> str:
    v = (os.getenv("EMBEDDINGS_TEXT_MODEL") or "").strip()
    return v or DEFAULT_TEXT_MODEL


def multimodal_base_url() -> str:
    v = (os.getenv("EMBEDDINGS_MULTIMODAL_BASE_URL") or "").strip()
    return (v or DEFAULT_MULTIMODAL_BASE_URL).rstrip("/")


def multimodal_model() -> str:
    return (os.getenv("EMBEDDINGS_MULTIMODAL_MODEL") or "").strip()


def _timeout() -> httpx.Timeout:
    try:
        total = float(os.getenv("EMBEDDINGS_TIMEOUT_SECONDS", "30"))
    except Exception:
        total = 30.0
    return httpx.Timeout(total, connect=2.0)


# ─── HTTP client (cached, resettable like gemini_provider) ───────

_client: Optional[httpx.Client] = None


def _http() -> httpx.Client:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.Client(timeout=_timeout())
    return _client


def reset_client() -> None:
    """Reset the cached HTTP client (testing / endpoint change)."""
    global _client
    try:
        if _client is not None and not _client.is_closed:
            _client.close()
    except Exception:
        pass
    _client = None


# ─── Text embedding ─────────────────────────────────────────────

def _local_max_chars() -> int:
    """The local server runs max-model-len 4096 (VRAM budget — see
    scripts/embeddings/serve_embed.sh); an over-length item 400s and comes
    back None. Truncate to a safe char budget (~3.5k tokens) BEFORE sending;
    gemini keeps the full MAX_TEXT_CHARS."""
    try:
        v = int(os.getenv("EMBEDDINGS_TEXT_MAX_CHARS", "14000"))
    except Exception:
        v = 14000
    return min(v, MAX_TEXT_CHARS)


def _prepare_text(text: str, task: str) -> str:
    """Truncate + apply the Qwen3 query instruction for query-side tasks."""
    limit = _local_max_chars()
    if len(text) > limit:
        text = text[:limit]
    if task in QUERY_TASK_TYPES:
        instruction = (
            os.getenv("EMBEDDINGS_TEXT_QUERY_INSTRUCTION") or ""
        ).strip() or DEFAULT_QUERY_INSTRUCTION
        return f"Instruct: {instruction}\nQuery: {text}"
    return text


def _post_embeddings(
    base_url: str,
    model: str,
    inputs: list,
) -> Optional[List[Optional[List[float]]]]:
    """POST an OpenAI-compatible /embeddings request. None on transport/HTTP
    failure; per-item None for any item the server skipped."""
    try:
        resp = _http().post(
            f"{base_url}/embeddings",
            json={"model": model, "input": inputs},
        )
        if resp.status_code != 200:
            logger.error(
                "[local_embed] HTTP %s from %s/embeddings: %s",
                resp.status_code, base_url, (resp.text or "")[:200],
            )
            return None
        data = resp.json().get("data") or []
        out: List[Optional[List[float]]] = [None] * len(inputs)
        for item in data:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(out):
                vec = item.get("embedding") or None
                out[idx] = list(vec) if vec else None
        return out
    except Exception as e:
        logger.error("[local_embed] embeddings call failed (%s): %s", base_url, e)
        return None


def generate_embedding(
    text: str,
    *,
    task_type: Union[TaskType, str] = TaskType.RETRIEVAL_DOCUMENT,
) -> Optional[List[float]]:
    """Embed one text via the local vLLM server. None on failure."""
    if not text or not text.strip():
        return None
    task = task_type.value if isinstance(task_type, TaskType) else str(task_type)
    result = _post_embeddings(text_base_url(), text_model(), [_prepare_text(text, task)])
    return result[0] if result else None


def generate_embeddings_batch(
    texts: List[str],
    *,
    task_type: Union[TaskType, str] = TaskType.RETRIEVAL_DOCUMENT,
) -> List[Optional[List[float]]]:
    """Batch-embed texts in one request. Same-length list; None per failure."""
    if not texts:
        return []
    task = task_type.value if isinstance(task_type, TaskType) else str(task_type)
    prepared = [_prepare_text(t or "", task) for t in texts]
    result = _post_embeddings(text_base_url(), text_model(), prepared)
    return result if result is not None else [None] * len(texts)


def text_available(timeout_seconds: float = 2.0) -> bool:
    """Cheap probe: does the local text embedding server answer /models?"""
    try:
        r = httpx.get(
            f"{text_base_url()}/models",
            timeout=httpx.Timeout(timeout_seconds, connect=1.0),
        )
        return r.status_code == 200
    except Exception:
        return False


# ─── Multimodal embedding (load-on-demand worker, :8005) ────────

def multimodal_available(timeout_seconds: float = 2.0) -> bool:
    """True when the multimodal worker is resident and answering. It is only
    loaded during BACKGROUND_INGEST — callers queue work when this is False."""
    try:
        r = httpx.get(
            f"{multimodal_base_url()}/models",
            timeout=httpx.Timeout(timeout_seconds, connect=1.0),
        )
        return r.status_code == 200
    except Exception:
        return False


def generate_multimodal_embedding(
    text: Optional[str] = None,
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    mime_type: str = "image/jpeg",
    *,
    task_type: Union[TaskType, str] = TaskType.RETRIEVAL_DOCUMENT,
) -> Optional[List[float]]:
    """Embed text and/or image via the load-on-demand multimodal vLLM
    instance (:8005, scripts/embeddings/serve_multimodal.sh).

    Text-only inputs (the video asset library today) ride the standard
    OpenAI /v1/embeddings shape. Image inputs use vLLM's multimodal
    embedding encoding (chat-style content parts with a base64 data URL) —
    exercised by the bench (scripts/embeddings/bench_multimodal.py) and
    treated as experimental until the model pick lands.
    """
    if image_path:
        p = Path(image_path)
        if not p.exists():
            logger.error("[local_embed] image not found: %s", image_path)
            return None
        image_bytes = p.read_bytes()
        mime_type = _guess_image_mime(p.suffix) or mime_type

    clean_text = (text or "")[:MAX_TEXT_CHARS].strip()
    if not image_bytes and not clean_text:
        logger.error("[local_embed] no content provided for multimodal embedding")
        return None

    base = multimodal_base_url()
    model = multimodal_model()
    try:
        if not image_bytes:
            result = _post_embeddings(base, model, [clean_text])
            return result[0] if result else None

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        content = [{
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64}"},
        }]
        if clean_text:
            content.insert(0, {"type": "text", "text": clean_text})
        resp = _http().post(
            f"{base}/embeddings",
            json={"model": model, "messages": [{"role": "user", "content": content}]},
        )
        if resp.status_code != 200:
            logger.error(
                "[local_embed] multimodal HTTP %s: %s",
                resp.status_code, (resp.text or "")[:200],
            )
            return None
        data = resp.json().get("data") or []
        vec = (data[0].get("embedding") if data else None) or None
        return list(vec) if vec else None
    except Exception as e:
        logger.error("[local_embed] multimodal call failed: %s", e)
        return None


def generate_image_embedding(
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    mime_type: str = "image/jpeg",
    *,
    task_type: Union[TaskType, str] = TaskType.RETRIEVAL_DOCUMENT,
) -> Optional[List[float]]:
    """Embed an image (path or bytes) via the multimodal worker."""
    return generate_multimodal_embedding(
        image_path=image_path,
        image_bytes=image_bytes,
        mime_type=mime_type,
        task_type=task_type,
    )
