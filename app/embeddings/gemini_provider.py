# FILE: app/embeddings/gemini_provider.py
"""
Gemini Embedding 2 provider.

Replaces OpenAI text-embedding-3-small with Google's
gemini-embedding-2-preview — the first natively multimodal
embedding model.

Key differences from the old OpenAI provider:
  - Supports text, images, video, audio, PDFs in one vector space
  - Task-type hints optimise vectors for retrieval vs classification
  - Matryoshka dimensions (768 / 1536 / 3072) — we use 1536 for
    backward compatibility with existing DB schema
  - Uses google-genai SDK (not the older google-generativeai)

Pricing: ~$0.20 per 1M text tokens (comparable to OpenAI small).
"""
from __future__ import annotations

import base64
import logging
import os
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────

GEMINI_EMBEDDING_MODEL = "gemini-embedding-2-preview"
EMBEDDING_DIMENSIONS = 1536  # Matryoshka: keep at 1536 for DB compat

# Max input tokens for the model
MAX_INPUT_TOKENS = 8192

# Max text chars (rough 4-char-per-token estimate → 32K chars)
MAX_TEXT_CHARS = 30_000


class TaskType(str, Enum):
    """Gemini embedding task types.

    Using the right task type optimises vector geometry for
    asymmetric search (short query → long document).
    """
    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
    CODE_RETRIEVAL_QUERY = "CODE_RETRIEVAL_QUERY"
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    CLASSIFICATION = "CLASSIFICATION"
    CLUSTERING = "CLUSTERING"
    QUESTION_ANSWERING = "QUESTION_ANSWERING"
    FACT_VERIFICATION = "FACT_VERIFICATION"


# ─── Client singleton ───────────────────────────────────────────

_client = None


def _get_client():
    """Lazy-init the google-genai client."""
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY (or GEMINI_API_KEY) not set — cannot generate Gemini embeddings"
        )

    from google import genai
    _client = genai.Client(api_key=api_key)
    return _client


# ─── Text embedding ─────────────────────────────────────────────

def generate_embedding(
    text: str,
    *,
    task_type: Union[TaskType, str] = TaskType.RETRIEVAL_DOCUMENT,
) -> Optional[List[float]]:
    """
    Generate an embedding vector for text using Gemini Embedding 2.

    Args:
        text: The text to embed.
        task_type: Hint for optimising the vector. Use
                   RETRIEVAL_DOCUMENT when indexing content,
                   RETRIEVAL_QUERY when searching.

    Returns:
        List of floats (1536-d) or None on failure.
    """
    if not text or not text.strip():
        return None

    # Truncate to stay within token budget
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]

    task = task_type.value if isinstance(task_type, TaskType) else task_type

    try:
        client = _get_client()
        from google.genai import types

        result = client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task,
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )
        return list(result.embeddings[0].values)

    except Exception as e:
        logger.error("[gemini_embed] Text embedding failed: %s", e)
        return None


def generate_embeddings_batch(
    texts: List[str],
    *,
    task_type: Union[TaskType, str] = TaskType.RETRIEVAL_DOCUMENT,
) -> List[Optional[List[float]]]:
    """
    Batch-embed multiple texts in a single API call.

    Returns a list the same length as `texts`. Each element is
    either a 1536-d float list or None if that item failed.
    """
    if not texts:
        return []

    task = task_type.value if isinstance(task_type, TaskType) else task_type

    # Truncate each text
    truncated = [t[:MAX_TEXT_CHARS] if len(t) > MAX_TEXT_CHARS else t for t in texts]

    try:
        client = _get_client()
        from google.genai import types

        result = client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=truncated,
            config=types.EmbedContentConfig(
                task_type=task,
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )

        return [
            list(emb.values) if emb else None
            for emb in result.embeddings
        ]

    except Exception as e:
        logger.error("[gemini_embed] Batch text embedding failed: %s", e)
        return [None] * len(texts)


# ─── Multimodal embedding ───────────────────────────────────────

def generate_image_embedding(
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    mime_type: str = "image/jpeg",
    *,
    task_type: Union[TaskType, str] = TaskType.RETRIEVAL_DOCUMENT,
) -> Optional[List[float]]:
    """
    Embed an image using Gemini Embedding 2.

    Provide either image_path OR image_bytes (not both).
    Supported formats: PNG, JPEG.

    Returns:
        List of floats (1536-d) or None on failure.
    """
    try:
        client = _get_client()
        from google.genai import types

        if image_path:
            path = Path(image_path)
            if not path.exists():
                logger.error("[gemini_embed] Image not found: %s", image_path)
                return None
            raw = path.read_bytes()
            mime_type = _guess_image_mime(path.suffix) or mime_type
        elif image_bytes:
            raw = image_bytes
        else:
            logger.error("[gemini_embed] No image data provided")
            return None

        b64 = base64.b64encode(raw).decode("utf-8")
        task = task_type.value if isinstance(task_type, TaskType) else task_type

        content = types.Content(
            parts=[types.Part(inline_data=types.Blob(mime_type=mime_type, data=b64))]
        )

        result = client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=content,
            config=types.EmbedContentConfig(
                task_type=task,
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )
        return list(result.embeddings[0].values)

    except Exception as e:
        logger.error("[gemini_embed] Image embedding failed: %s", e)
        return None


def generate_multimodal_embedding(
    text: Optional[str] = None,
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    mime_type: str = "image/jpeg",
    *,
    task_type: Union[TaskType, str] = TaskType.RETRIEVAL_DOCUMENT,
) -> Optional[List[float]]:
    """
    Embed interleaved text + image as a single vector.

    Gemini Embedding 2 natively handles mixed-modality inputs,
    producing one embedding that captures the combined meaning.

    Returns:
        List of floats (1536-d) or None on failure.
    """
    try:
        client = _get_client()
        from google.genai import types

        parts = []

        if text and text.strip():
            truncated = text[:MAX_TEXT_CHARS] if len(text) > MAX_TEXT_CHARS else text
            parts.append(types.Part(text=truncated))

        if image_path:
            path = Path(image_path)
            if path.exists():
                raw = path.read_bytes()
                mt = _guess_image_mime(path.suffix) or mime_type
                b64 = base64.b64encode(raw).decode("utf-8")
                parts.append(types.Part(inline_data=types.Blob(mime_type=mt, data=b64)))
        elif image_bytes:
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            parts.append(types.Part(inline_data=types.Blob(mime_type=mime_type, data=b64)))

        if not parts:
            logger.error("[gemini_embed] No content provided for multimodal embedding")
            return None

        task = task_type.value if isinstance(task_type, TaskType) else task_type

        content = types.Content(parts=parts)
        result = client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=content,
            config=types.EmbedContentConfig(
                task_type=task,
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )
        return list(result.embeddings[0].values)

    except Exception as e:
        logger.error("[gemini_embed] Multimodal embedding failed: %s", e)
        return None


# ─── Helpers ─────────────────────────────────────────────────────

def _guess_image_mime(suffix: str) -> Optional[str]:
    """Map file extension to MIME type."""
    mapping = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mapping.get(suffix.lower())


def reset_client():
    """Reset the cached client (for testing or key rotation)."""
    global _client
    _client = None
