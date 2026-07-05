# FILE: app/embeddings/provider_router.py
# Purpose: Role-based embedding provider routing (gemini vs local) + model-space specs for search.
# Called-by: app.embeddings.service, app.llm.clients, app.self_model.fragments, app.content.video_pipeline.asset_library (+idle tasks)
# Depends-on: app.embeddings.gemini_provider, app.embeddings.local_provider
# Last-renovated: 2026-07-02 (LANE E)
"""
Embedding provider router (LANE E).

Two roles ("collections"):
  - text:       notes, messages, docs, code chunks, memory semantic,
                self-model fragments, experience patterns.
  - multimodal: visual content (video assets, drive images, screenshots).

Env (read at call time; a .env change applies at the next backend restart):
  EMBEDDINGS_TEXT_PROVIDER          gemini | local     (default gemini)
  EMBEDDINGS_TEXT_MODEL             stamp/served label (default per provider)
  EMBEDDINGS_TEXT_DIMS              int (default 1536 gemini / 1024 local)
  EMBEDDINGS_TEXT_QUERY_CUTOVER     0|1 — flip to 1 ONLY after the parity
                                    report (reports/LANE-E-PARITY-REPORT.md)
                                    is signed off. Gates query-time local
                                    embedding AND the reembed_batch drain.
  EMBEDDINGS_MULTIMODAL_PROVIDER    gemini | local     (default gemini until
                                    the 50-item bench picks a local model)
  EMBEDDINGS_MULTIMODAL_MODEL/_DIMS label + dims for the local pick

Semantics:
  - WRITE-time embeds use the role's provider immediately. Every stored row
    is stamped (embedding_model + dims), so both populations coexist and
    rollback is just flipping the provider env back.
  - QUERY-time text embeds stay on gemini until the cutover flag is set.
    With cutover on, search dual-reads: each stored model-space is scored
    with a query vector from ITS OWN model (app/embeddings/service.py),
    merged by per-space normalised score, until the legacy space empties —
    then the gemini path drops out naturally (zero API calls).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Union

from app.embeddings import gemini_provider, local_provider
from app.embeddings.gemini_provider import (
    EMBEDDING_DIMENSIONS,
    GEMINI_EMBEDDING_MODEL,
    TaskType,
)

# The stamp legacy rows carry (backfilled by app/embeddings/migrations.py);
# reembed_batch walks rows with this label.
LEGACY_TEXT_LABEL = GEMINI_EMBEDDING_MODEL

DEFAULT_LOCAL_TEXT_DIMS = 1024  # Qwen3-Embedding-0.6B native (MRL-truncatable)


@dataclass(frozen=True)
class ProviderSpec:
    provider: str  # "gemini" | "local"
    label: str     # embedding_model stamp for rows produced by this spec
    dims: int      # expected dimensionality (stamped dims = actual len())


def _env(name: str, default: str = "") -> str:
    v = (os.getenv(name) or "").strip()
    return v or default


def _int_env(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except Exception:
        return default


def is_query_task(task_type: Union[TaskType, str]) -> bool:
    task = task_type.value if isinstance(task_type, TaskType) else str(task_type)
    return task in local_provider.QUERY_TASK_TYPES


# ─── Role specs ──────────────────────────────────────────────────

_GEMINI_SPEC = ProviderSpec("gemini", GEMINI_EMBEDDING_MODEL, EMBEDDING_DIMENSIONS)


def text_write_spec() -> ProviderSpec:
    """Provider that embeds NEW text content (stamped, effective immediately)."""
    if _env("EMBEDDINGS_TEXT_PROVIDER", "gemini").lower() == "local":
        return ProviderSpec(
            "local",
            _env("EMBEDDINGS_TEXT_MODEL", local_provider.DEFAULT_TEXT_MODEL),
            _int_env("EMBEDDINGS_TEXT_DIMS", DEFAULT_LOCAL_TEXT_DIMS),
        )
    return _GEMINI_SPEC


def query_cutover_enabled() -> bool:
    return _env("EMBEDDINGS_TEXT_QUERY_CUTOVER", "0").lower() in {"1", "true", "yes"}


def text_query_spec() -> ProviderSpec:
    """Provider that embeds text QUERIES. Local only after the parity gate."""
    write = text_write_spec()
    if write.provider == "local" and query_cutover_enabled():
        return write
    return _GEMINI_SPEC


def text_query_spaces() -> List[ProviderSpec]:
    """Model spaces to score at search time, primary first.

    Dual-read: while the primary is local, the legacy gemini space is also
    listed; app/embeddings/service.py drops it once its row count is zero.
    """
    primary = text_query_spec()
    if primary.provider == "local" and primary.label != LEGACY_TEXT_LABEL:
        return [primary, _GEMINI_SPEC]
    return [primary]


def multimodal_write_spec() -> ProviderSpec:
    """Provider for the visual collection. Defaults to gemini until the
    multimodal bench (scripts/embeddings/bench_multimodal.py) picks a local
    model and Taz flips EMBEDDINGS_MULTIMODAL_PROVIDER=local."""
    if _env("EMBEDDINGS_MULTIMODAL_PROVIDER", "gemini").lower() == "local":
        return ProviderSpec(
            "local",
            _env("EMBEDDINGS_MULTIMODAL_MODEL", "jina-embeddings-v4"),
            _int_env("EMBEDDINGS_MULTIMODAL_DIMS", 1024),
        )
    return _GEMINI_SPEC


def multimodal_ready() -> bool:
    """Can the multimodal role embed RIGHT NOW? gemini: always (API call).
    local: only while the worker is resident (BACKGROUND_INGEST). Callers
    enqueue into the visual queue when this is False."""
    spec = multimodal_write_spec()
    if spec.provider == "gemini":
        return True
    return local_provider.multimodal_available()


# ─── Embedding entrypoints ───────────────────────────────────────

def embed_text(
    text: str,
    *,
    task_type: Union[TaskType, str] = TaskType.RETRIEVAL_DOCUMENT,
    purpose: Optional[str] = None,
    spec: Optional[ProviderSpec] = None,
) -> Optional[List[float]]:
    """Embed text in the right model space.

    purpose: "write" | "query" | None (None → inferred from task_type).
    spec: pin a specific space (dual-read search, parity job, reembed).
    """
    sp = spec or _spec_for(task_type, purpose)
    if sp.provider == "local":
        return local_provider.generate_embedding(text, task_type=task_type)
    return gemini_provider.generate_embedding(text, task_type=task_type)


def embed_text_batch(
    texts: List[str],
    *,
    task_type: Union[TaskType, str] = TaskType.RETRIEVAL_DOCUMENT,
    purpose: Optional[str] = None,
    spec: Optional[ProviderSpec] = None,
) -> List[Optional[List[float]]]:
    sp = spec or _spec_for(task_type, purpose)
    if sp.provider == "local":
        return local_provider.generate_embeddings_batch(texts, task_type=task_type)
    return gemini_provider.generate_embeddings_batch(texts, task_type=task_type)


def _spec_for(task_type: Union[TaskType, str], purpose: Optional[str]) -> ProviderSpec:
    if purpose == "query" or (purpose is None and is_query_task(task_type)):
        return text_query_spec()
    return text_write_spec()


def embed_multimodal(
    text: Optional[str] = None,
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    mime_type: str = "image/jpeg",
    *,
    task_type: Union[TaskType, str] = TaskType.RETRIEVAL_DOCUMENT,
) -> Optional[List[float]]:
    """Embed visual (or text-for-visual-space) content in the multimodal
    role's space. Returns None when the local worker is not resident —
    callers queue the item (app/embeddings/visual_queue.py) by design."""
    spec = multimodal_write_spec()
    if spec.provider == "local":
        return local_provider.generate_multimodal_embedding(
            text=text, image_path=image_path, image_bytes=image_bytes,
            mime_type=mime_type, task_type=task_type,
        )
    return gemini_provider.generate_multimodal_embedding(
        text=text, image_path=image_path, image_bytes=image_bytes,
        mime_type=mime_type, task_type=task_type,
    )


# ─── Drop-in provider-surface aliases ─────────────────────────
# app/embeddings/__init__.py re-exports these under the historical names so
# existing callers transparently ride the router.

generate_multimodal_embedding = embed_multimodal


def generate_image_embedding(
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    mime_type: str = "image/jpeg",
    *,
    task_type: Union[TaskType, str] = TaskType.RETRIEVAL_DOCUMENT,
) -> Optional[List[float]]:
    """Embed an image in the multimodal role's space (router-selected)."""
    return embed_multimodal(
        image_path=image_path, image_bytes=image_bytes,
        mime_type=mime_type, task_type=task_type,
    )


def generate_embeddings_batch(
    texts: List[str],
    *,
    task_type: Union[TaskType, str] = TaskType.RETRIEVAL_DOCUMENT,
) -> List[Optional[List[float]]]:
    """Batch text embedding under the historical provider name."""
    return embed_text_batch(texts, task_type=task_type)
