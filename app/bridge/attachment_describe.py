# FILE: app/bridge/attachment_describe.py
"""
Bridge attachment description layer.

Single responsibility: given a list of upload IDs from POST /bridge/upload,
resolve each to a file on disk, route by file type, and return a prompt-ready
block of text the chat-and-speak handler can inject as context for the LLM.

Routing:
  - Image (jpeg/png/gif/webp/heic/heif)  -> Gemini Vision via ask_about_image
  - Anything else (PDF, DOCX, XLSX, PPTX,
    text formats, source code, configs)  -> file_analyzer.extract_text
  - Truly unsupported binaries           -> graceful "no extractor" note

Why text-only for non-images: per design discussion 2026-05-24, documents
(insurance docs, contracts, spreadsheets, etc.) are read for their text
content. Embedded images in PDFs are decorative or self-describing; running
vision over them adds cost and latency for marginal benefit. If a PDF is
purely scanned and yields no text, we surface that explicitly so the LLM
can tell the user instead of hallucinating.

Why a separate module:
  - Keeps router.py focused on HTTP plumbing
  - Vision call (ask_about_image) lives in app/llm/gemini_vision.py
  - Text extraction lives in app/llm/file_analyzer.py (battle-tested,
    used by the Drive content indexer)
  - This module just glues them to bridge-specific concerns: multi-id loop,
    ledger lookup, type dispatch, size capping, prompt formatting

v2.0 (2026-05-24): renamed from vision_describe.py. Added text extraction
for documents via file_analyzer.extract_text. 20k char cap per attachment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from app.bridge.uploads_store import UploadRecord, find_by_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Mimes we run vision on. Anything else routes to text extraction.
_VISION_MIMES = {
    "image/jpeg", "image/jpg", "image/png", "image/gif",
    "image/webp", "image/heic", "image/heif",
}

# Per-attachment ceiling on extracted text. Past this we truncate and
# append a marker — keeps prompt tokens bounded so a fat PDF doesn't
# blow the LLM context. Bump if you find real docs getting clipped.
_MAX_EXTRACTED_CHARS = 20_000

# Fallback prompt when the user caption is empty or too short to be a real
# question (e.g. "this", "yo"). Vision still needs SOMETHING to anchor on.
_DEFAULT_VISION_QUESTION = (
    "Describe this image in detail. If it contains text, numbers, dates, "
    "amounts, or any structured information, transcribe them accurately."
)

# Friendly type labels by extension. Used in the prompt block header so
# the LLM sees "Word document" rather than a UUID filename.
_FRIENDLY_LABELS = {
    ".pdf":  "PDF document",
    ".docx": "Word document",
    ".doc":  "Word document",
    ".xlsx": "Excel spreadsheet",
    ".xls":  "Excel spreadsheet",
    ".pptx": "PowerPoint presentation",
    ".html": "HTML document",
    ".htm":  "HTML document",
    ".md":   "Markdown document",
    ".txt":  "text document",
    ".csv":  "CSV data file",
    ".json": "JSON document",
    ".xml":  "XML document",
    ".log":  "log file",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def describe_attachments(
    attachment_ids: Optional[Sequence[str]],
    user_message: str,
) -> str:
    """Return a prompt-injectable block describing all resolvable attachments.

    Empty/None ids -> empty string (caller can short-circuit).
    Unresolvable id -> noted so the LLM knows something was sent but
    couldn't be read, rather than silently dropping it.

    Output is ready to prepend to the user's message. Blocks are separated
    by blank lines so multiple attachments stay visually distinct in the
    LLM's view of the prompt.
    """
    if not attachment_ids:
        return ""

    # Use the user's caption as the vision question if it's substantive;
    # otherwise fall back to the generic description prompt.
    question = (user_message or "").strip()
    if len(question) < 4:
        question = _DEFAULT_VISION_QUESTION

    records = _resolve_ids(attachment_ids)
    if not records:
        return "[Attachment(s) were sent but could not be resolved on the server.]"

    blocks: List[str] = []
    for idx, rec in enumerate(records, start=1):
        block = _describe_single(rec, question, idx, total=len(records))
        if block:
            blocks.append(block)

    if not blocks:
        return ""

    # Two newlines between blocks — clearer visual separation than one.
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _describe_single(
    rec: UploadRecord,
    question: str,
    idx: int,
    total: int,
) -> str:
    """Dispatch a single upload to the right describer."""
    mime = (rec.content_type or "").split(";", 1)[0].strip().lower()

    if mime in _VISION_MIMES:
        return _describe_image(rec, question, idx, total, mime=mime)

    # Everything else -> text extraction.
    return _describe_document(rec, idx, total)


# ---------------------------------------------------------------------------
# Image branch (Gemini Vision)
# ---------------------------------------------------------------------------

def _describe_image(
    rec: UploadRecord,
    question: str,
    idx: int,
    total: int,
    mime: str,
) -> str:
    label = "Image attached" if total == 1 else f"Image {idx} of {total} attached"

    try:
        from app.llm.gemini_vision import ask_about_image  # lazy import
    except Exception as e:
        logger.error("[attachment_describe] gemini_vision import failed: %s", e)
        return f"[{label}: vision module unavailable]"

    try:
        result = ask_about_image(
            image_source=rec.path,
            user_question=question,
            mime_type=mime,
            tier="default",
        )
    except Exception as e:
        logger.exception("[attachment_describe] ask_about_image raised: %s", e)
        return f"[{label}: vision call failed ({e})]"

    answer = (result or {}).get("answer", "").strip()
    if not answer:
        err = (result or {}).get("error", "no description returned")
        return f"[{label}: {err}]"

    return f"[{label}: {answer}]"


# ---------------------------------------------------------------------------
# Document branch (text extraction)
# ---------------------------------------------------------------------------

def _describe_document(rec: UploadRecord, idx: int, total: int) -> str:
    """Extract text from any non-image attachment and frame it for the LLM."""
    ext = Path(rec.path).suffix.lower()
    type_label = _FRIENDLY_LABELS.get(ext, f"file ({ext or 'no extension'})")
    prefix = "Document attached" if total == 1 else f"Document {idx} of {total} attached"

    try:
        from app.llm.file_analyzer import extract_text  # lazy import
    except Exception as e:
        logger.error("[attachment_describe] file_analyzer import failed: %s", e)
        return f"[{prefix} ({type_label}): text extraction module unavailable]"

    try:
        text, err = extract_text(file_path=rec.path)
    except Exception as e:
        logger.exception("[attachment_describe] extract_text raised: %s", e)
        return f"[{prefix} ({type_label}): extraction failed ({e})]"

    if err and not text:
        # No extractor for this type, or hard failure on a known type.
        # Acknowledge the file exists so the LLM can tell the user we
        # received it but can't read it.
        return (
            f"[{prefix} ({type_label}): could not be read — {err}. "
            f"File saved at {rec.path}]"
        )

    text = (text or "").strip()
    if not text:
        # Extraction succeeded mechanically but produced nothing useful.
        # Most common cause: a scanned-image PDF with no embedded text.
        return (
            f"[{prefix} ({type_label}): the document appears to be "
            f"scanned or image-based and contains no readable text]"
        )

    truncated = False
    if len(text) > _MAX_EXTRACTED_CHARS:
        text = text[:_MAX_EXTRACTED_CHARS]
        truncated = True

    header = f"[{prefix} — {type_label} ({len(text)} characters"
    header += ", truncated" if truncated else ""
    header += ")]"

    end_marker = "--- END DOCUMENT (truncated) ---" if truncated else "--- END DOCUMENT ---"

    return f"{header}\n--- BEGIN DOCUMENT ---\n{text}\n{end_marker}"


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _resolve_ids(ids: Iterable[str]) -> List[UploadRecord]:
    out: List[UploadRecord] = []
    for raw_id in ids:
        rec = find_by_id((raw_id or "").strip())
        if rec is None:
            logger.warning("[attachment_describe] upload id not in ledger: %r", raw_id)
            continue
        out.append(rec)
    return out

# ---------------------------------------------------------------------------
# Augmentation helper used by both bridge_chat and bridge_chat_and_speak
# ---------------------------------------------------------------------------

def augment_user_message(req, history_messages):
    """If the request carries attachment_ids, describe them and return the
    LLM-input version of the user message. Also patches history_messages[-1]
    so the model's view of the conversation matches the augmented prompt.

    Returns req.message unchanged when there are no attachments, or when
    the description layer produced no usable output (e.g. all ids missing
    from the ledger). The original req.message is never mutated — only the
    return value is augmented, and the history list is patched in place so
    callers can decide whether to use the augmented form.
    """
    ids = getattr(req, "attachment_ids", None) or []
    logger.info(
        "[bridge_augment] entry: attachment_ids_count=%d ids=%s",
        len(ids),
        [i[:8] for i in ids][:5],
    )
    if not ids:
        return req.message

    desc = describe_attachments(ids, req.message)
    if not desc:
        logger.info("[bridge_augment] describe_attachments returned empty — using raw message")
        return req.message

    augmented = f"{desc}\n\n{req.message}".strip()
    if history_messages and history_messages[-1].get("role") == "user":
        history_messages[-1]["content"] = augmented
    logger.info(
        "[bridge_augment] exit: desc_chars=%d augmented_chars=%d patched_history=%s",
        len(desc), len(augmented),
        bool(history_messages and history_messages[-1].get("role") == "user"),
    )
    return augmented
