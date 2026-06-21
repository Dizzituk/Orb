# FILE: app/llm/routing/chat_request_prep.py
# Purpose: Request->message prep helpers for chat routing (split from chat_routing.py).
# Called-by: app.llm.routing.chat_routing
# Depends-on: app.llm.routing.prompt_builders, app.memory (+ lazy app.llm.file_analyzer)
# Last-renovated: 2026-06-21
from __future__ import annotations
from typing import Optional, Any
from sqlalchemy.orm import Session
from app.memory import service as memory_service
from .prompt_builders import build_messages


def _resolve_message_with_documents(req: Any) -> str:
    """Resolve req.message by inlining any attached document content.

    v7.1 (2026-04-09): When the frontend converts large pastes into structured
    documents, req.message is just '[Documents attached]'. This helper inlines
    the actual document content so it gets persisted to DB and is available
    to the Weaver and other downstream consumers that read conversation history.
    """
    user_content = req.message
    docs = getattr(req, 'documents', None)
    if docs and isinstance(docs, list) and len(docs) > 0:
        doc_blocks = []
        for doc in docs:
            label = doc.get('label', 'Document')
            body = doc.get('content', '')
            idx = doc.get('index', '')
            doc_blocks.append(
                f"--- Document {idx}: {label} ---\n{body}\n--- End Document {idx} ---"
            )
        doc_text = '\n\n'.join(doc_blocks)
        if user_content in ('[Documents attached]', ''):
            user_content = doc_text
        else:
            user_content = user_content + '\n\n' + doc_text
    return user_content


def _persist_user_message(req: Any, db: Session) -> Optional[int]:
    """Persist the user's message before any routing decisions.

    v2026-06-14 (live-sync): returns the new message id so handle_chat_mode can
    hand it to generate_sse_stream, which then SKIPS its own (historically
    redundant) user-row write. Returns None on failure — the stream then falls
    back to persisting the user row itself, preserving the old behaviour.
    """
    try:
        from app.memory import schemas as _mem_schemas
        msg = memory_service.create_message(
            db,
            _mem_schemas.MessageCreate(
                project_id=req.project_id,
                role="user",
                # v7.1 (2026-04-09): Persist document content, not placeholder.
                # Large pastes arrive as req.documents but req.message is just
                # '[Documents attached]'. Persisting the placeholder means the
                # Weaver never sees the pasted content when it reads history.
                content=_resolve_message_with_documents(req),
                provider="system",
            ),
        )
        return msg.id
    except Exception as e:
        print(f"[CHAT_MODE] Failed to persist user message: {e}")
        return None


def _process_file_uploads(req: Any, full_context: str) -> tuple[list, bool, bool, str]:
    """Handle file_upload_* fields on the request.

    Returns (synthetic_attachments, is_image_upload, is_video_upload, updated_full_context).
    """
    _file_local_path = getattr(req, "file_upload_local_path", None)
    _file_name = getattr(req, "file_upload_name", None)
    _file_mime = getattr(req, "file_upload_mime", None) or ""
    _file_gemini_name = getattr(req, "file_upload_gemini_name", None)
    _file_gemini_uri = getattr(req, "file_upload_uri", None)
    is_image_upload = _file_mime.startswith("image/")
    is_video_upload = _file_mime.startswith("video/")
    synthetic_attachments: list = []

    if not (_file_local_path and _file_name):
        return synthetic_attachments, is_image_upload, is_video_upload, full_context

    if is_image_upload or is_video_upload:
        synthetic_attachments.append({
            "path": _file_local_path,
            "name": _file_name,
            "mime": _file_mime,
            "gemini_name": _file_gemini_name,
            "gemini_uri": _file_gemini_uri,
        })
        print(f"[CHAT_MODE] Image/video upload detected: {_file_name} ({_file_mime})")
        print(f"[CHAT_MODE]   -> Gemini URI: {_file_gemini_uri}")
        print(f"[CHAT_MODE]   -> Gemini name: {_file_gemini_name}")
        print(f"[CHAT_MODE]   -> Local path: {_file_local_path}")
    else:
        try:
            from app.llm.file_analyzer import extract_text as _extract_text
            _file_text, _file_err = _extract_text(file_path=_file_local_path, filename=_file_name)
            if _file_text:
                _preview = _file_text[:50000]
                _upload_block = (
                    f"=== UPLOADED FILE: {_file_name} ===\n"
                    f"{_preview}\n"
                    f"=== END FILE ===\n\n"
                )
                full_context = _upload_block + (full_context or "")
                print(f"[CHAT_MODE] Injected uploaded file content: {_file_name} ({len(_file_text)} chars)")
            elif _file_err:
                print(f"[CHAT_MODE] File extraction failed for {_file_name}: {_file_err}")
        except Exception as _fex:
            print(f"[CHAT_MODE] File injection error: {_fex}")

    return synthetic_attachments, is_image_upload, is_video_upload, full_context


def _build_chat_messages(req: Any, db: Session) -> list:
    """Build message list — panel history if available, else DB history."""
    panel_hist = getattr(req, 'panel_history', None)
    if panel_hist and isinstance(panel_hist, list) and len(panel_hist) > 0:
        messages = []
        for entry in panel_hist[-20:]:
            role = entry.get('role', 'user')
            content = entry.get('content', '')
            if role in ('user', 'assistant') and content:
                messages.append({'role': role, 'content': content})
        # v7.0 (2026-04-08): Inject pasted document content into user message.
        # The frontend converts large pastes (>3000 chars) into structured
        # document attachments. The content arrives in req.documents but was
        # never added to the LLM messages — the model only saw the placeholder
        # '[Documents attached]' and had to use file-search tools to find content.
        user_content = req.message
        _docs = getattr(req, 'documents', None)
        if _docs and isinstance(_docs, list) and len(_docs) > 0:
            _doc_blocks = []
            for _doc in _docs:
                _label = _doc.get('label', 'Document')
                _body = _doc.get('content', '')
                _idx = _doc.get('index', '')
                _doc_blocks.append(
                    f"--- Document {_idx}: {_label} ---\n{_body}\n--- End Document {_idx} ---"
                )
            _doc_text = '\n\n'.join(_doc_blocks)
            if user_content in ('[Documents attached]', ''):
                user_content = _doc_text
            else:
                user_content = user_content + '\n\n' + _doc_text
            print(
                f"[CHAT_MODE] Injected {len(_docs)} document(s) into user message "
                f"({len(_doc_text)} chars)"
            )
        messages.append({'role': 'user', 'content': user_content})
        print(f"[CHAT_MODE] Using panel history: {len(messages)-1} prior messages + current")
        return messages

    # v7.0: Also inject documents in the DB-history path
    _fb_message = req.message
    _fb_docs = getattr(req, 'documents', None)
    if _fb_docs and isinstance(_fb_docs, list) and len(_fb_docs) > 0:
        _fb_blocks = []
        for _d in _fb_docs:
            _fb_blocks.append(
                f"--- Document {_d.get('index', '')}: {_d.get('label', 'Document')} ---\n"
                f"{_d.get('content', '')}\n--- End Document {_d.get('index', '')} ---"
            )
        _fb_text = '\n\n'.join(_fb_blocks)
        _fb_message = _fb_text if _fb_message in ('[Documents attached]', '') else _fb_message + '\n\n' + _fb_text
        print(f"[CHAT_MODE] Injected {len(_fb_docs)} doc(s) into DB-history message ({len(_fb_text)} chars)")
    return build_messages(
        message=_fb_message,
        project_id=req.project_id,
        db=db,
        include_history=req.include_history,
        history_limit=req.history_limit,
    )


def _inject_image_into_messages(
    messages: list,
    file_local_path: str | None,
    file_mime: str,
) -> list:
    """Inject base64 image into the last user message for Gemini vision."""
    if not file_local_path:
        return messages
    try:
        import base64 as _b64
        with open(file_local_path, "rb") as _img_f:
            _img_bytes = _img_f.read()
        _img_b64 = _b64.b64encode(_img_bytes).decode("utf-8")
        for _i in range(len(messages) - 1, -1, -1):
            if messages[_i].get("role") == "user":
                _text_content = messages[_i].get("content", "")
                messages[_i]["content"] = [
                    {"type": "text", "text": _text_content},
                    {"type": "image", "mime_type": file_mime, "data": _img_b64},
                ]
                print(f"[CHAT_MODE] Injected image into user message [{_i}]: {file_mime}, {len(_img_bytes)} bytes")
                break
    except Exception as _img_err:
        print(f"[CHAT_MODE] Failed to inject image: {_img_err}")
    return messages
