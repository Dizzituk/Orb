# FILE: app/tools/email_tools.py
# Purpose: Email chat tools — read/search by voice, draft-confirm-send loop
#          (never auto-send), provider status + Gmail OAuth setup.
# Called-by: app.tools.registry (_register_defaults)
# Depends-on: app.email (provider factory), app.email.gmail_auth
# Last-renovated: 2026-06-12
"""
Email chat tools (v1, 2026-06-12).

  check_email           "anything new?" — unread senders + subjects
  read_email_thread     "read the one from Dave" — plain-text body for TTS
  search_email          "find the email about the invoice"
  draft_email           dictated reply/new mail -> DRAFT ONLY, read back
  send_email_draft      sends ONLY with confirm=true after spoken confirmation
  email_provider_status which provider, connected?, setup hints
  setup_email_auth      starts the Gmail OAuth flow in the PC browser

SAFETY RAILS (non-negotiable, also stated in the tool descriptions):
  - Sending always goes draft -> read back -> explicit user confirmation ->
    send_email_draft(confirm=true). There is no auto-send path.
  - Declining leaves the draft in place.
  - Never guess an address: if the user names a person without an address,
    ask. (Contact resolution is conversational, not tool-side.)

Provider-agnostic: handlers call app.email.get_provider() — swapping
gmail/proton/fake via config touches nothing here (the seam the spec wants
proven). Provider calls are sync; handlers wrap them in asyncio.to_thread.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Optional

logger = logging.getLogger(__name__)

SPOKEN_BODY_LIMIT = 700  # chars of body the model should read aloud before offering more


def _summaries_payload(summaries) -> list:
    return [asdict(s) for s in summaries]


async def _provider():
    from app.email import get_provider
    return await asyncio.to_thread(get_provider)


# ────────────────────────────────────────────────────────────────────────
# Handlers
# ────────────────────────────────────────────────────────────────────────

async def check_email_handler(input_data: dict, context: Optional[dict]) -> dict:
    try:
        provider = await _provider()
        label = "UNREAD" if input_data.get("unread_only", True) else None
        threads = await asyncio.to_thread(
            provider.list_threads, input_data.get("query"), label,
            int(input_data.get("max", 5)),
        )
        return {"ok": True, "provider": provider.name,
                "threads": _summaries_payload(threads)}
    except Exception as exc:
        logger.exception("[email_tools] check_email failed")
        return {"ok": False, "error": str(exc)}


async def read_email_thread_handler(input_data: dict, context: Optional[dict]) -> dict:
    thread_id = (input_data.get("thread_id") or "").strip()
    sender = (input_data.get("from_sender") or "").strip()
    if not thread_id and not sender:
        return {"ok": False, "error": "thread_id or from_sender is required"}
    try:
        provider = await _provider()
        if not thread_id:
            matches = await asyncio.to_thread(
                provider.list_threads, f"from:{sender}", None, 3
            )
            if not matches:
                return {"ok": False, "error": f"nothing recent from {sender}"}
            thread_id = matches[0].id
        thread = await asyncio.to_thread(provider.read_thread, thread_id)
        newest = thread.messages[-1] if thread.messages else None
        body = newest.body if newest else ""
        truncated = len(body) > SPOKEN_BODY_LIMIT
        return {
            "ok": True,
            "thread_id": thread.id,
            "subject": thread.subject,
            "sender": newest.sender if newest else "",
            "date": newest.date if newest else "",
            "spoken_preview": body[:SPOKEN_BODY_LIMIT],
            "truncated": truncated,
            "full_body": body,
            "message_count": len(thread.messages),
        }
    except Exception as exc:
        logger.exception("[email_tools] read_email_thread failed")
        return {"ok": False, "error": str(exc)}


async def search_email_handler(input_data: dict, context: Optional[dict]) -> dict:
    query = (input_data.get("query") or "").strip()
    if not query:
        return {"ok": False, "error": "query is required"}
    try:
        provider = await _provider()
        threads = await asyncio.to_thread(
            provider.search, query, int(input_data.get("max", 5))
        )
        return {"ok": True, "threads": _summaries_payload(threads)}
    except Exception as exc:
        logger.exception("[email_tools] search_email failed")
        return {"ok": False, "error": str(exc)}


async def draft_email_handler(input_data: dict, context: Optional[dict]) -> dict:
    to = (input_data.get("to") or "").strip()
    body = (input_data.get("body") or "").strip()
    subject = (input_data.get("subject") or "").strip()
    reply_to_thread = input_data.get("reply_to_thread")
    if not body:
        return {"ok": False, "error": "body is required"}
    if not reply_to_thread and not to:
        return {"ok": False, "error": "to (an actual email address) is required for a "
                                      "new email — never guess an address, ask the user"}
    if not reply_to_thread and not subject:
        return {"ok": False, "error": "subject is required for a new email"}
    try:
        provider = await _provider()
        draft = await asyncio.to_thread(
            provider.create_draft, to, subject, body, reply_to_thread
        )
        if not draft.get("ok"):
            return draft
        return {
            **draft,
            "next_step": "Read the draft back to the user. Send ONLY after an "
                         "explicit yes, via send_email_draft with confirm=true.",
        }
    except Exception as exc:
        logger.exception("[email_tools] draft_email failed")
        return {"ok": False, "error": str(exc)}


async def send_email_draft_handler(input_data: dict, context: Optional[dict]) -> dict:
    draft_id = (input_data.get("draft_id") or "").strip()
    if not draft_id:
        return {"ok": False, "error": "draft_id is required"}
    if input_data.get("confirm") is not True:
        return {
            "ok": False,
            "error": "not sent — confirm=true is required after the user has heard "
                     "the draft and explicitly said to send it. The draft is intact.",
        }
    try:
        provider = await _provider()
        return await asyncio.to_thread(provider.send_draft, draft_id)
    except Exception as exc:
        logger.exception("[email_tools] send_email_draft failed")
        return {"ok": False, "error": str(exc)}


async def email_provider_status_handler(input_data: dict, context: Optional[dict]) -> dict:
    try:
        from app.email import configured_provider_name
        name = configured_provider_name()
        out: dict = {"ok": True, "provider": name}
        if name == "gmail":
            from app.email import gmail_auth
            out["status"] = await asyncio.to_thread(gmail_auth.check_auth_status)
        elif name == "proton":
            from app.email_service import imap_service
            out["status"] = await asyncio.to_thread(imap_service.check_connection)
        return out
    except Exception as exc:
        logger.exception("[email_tools] provider status failed")
        return {"ok": False, "error": str(exc)}


async def setup_email_auth_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Gmail only: open the OAuth consent screen on the PC."""
    try:
        from app.email import gmail_auth
        return await asyncio.to_thread(gmail_auth.start_auth_flow)
    except Exception as exc:
        logger.exception("[email_tools] setup_email_auth failed")
        return {"ok": False, "error": str(exc)}


# ────────────────────────────────────────────────────────────────────────
# Schemas (inline — self-contained file)
# ────────────────────────────────────────────────────────────────────────

_OK_ERR = {"ok": {"type": "boolean"}, "error": {"type": "string"}}

_THREADS_OUTPUT = {
    "type": "object",
    "required": ["ok"],
    "properties": {
        **_OK_ERR,
        "provider": {"type": ["string", "null"]},
        "threads": {"type": ["array", "null"], "items": {"type": "object"}},
    },
}

CHECK_EMAIL_INPUT = {
    "type": "object",
    "properties": {
        "unread_only": {"type": "boolean", "description": "Default true"},
        "query": {"type": ["string", "null"],
                  "description": "Optional provider search syntax filter"},
        "max": {"type": "integer", "description": "Default 5"},
    },
}

READ_THREAD_INPUT = {
    "type": "object",
    "properties": {
        "thread_id": {"type": ["string", "null"],
                      "description": "Exact thread id from a previous listing"},
        "from_sender": {"type": ["string", "null"],
                        "description": "Or: most recent thread from this sender "
                                       "('read the one from Dave')"},
    },
}
READ_THREAD_OUTPUT = {
    "type": "object",
    "required": ["ok"],
    "properties": {
        **_OK_ERR,
        "thread_id": {"type": ["string", "null"]},
        "subject": {"type": ["string", "null"]},
        "sender": {"type": ["string", "null"]},
        "date": {"type": ["string", "null"]},
        "spoken_preview": {"type": ["string", "null"]},
        "truncated": {"type": ["boolean", "null"]},
        "full_body": {"type": ["string", "null"]},
        "message_count": {"type": ["integer", "null"]},
    },
}

SEARCH_EMAIL_INPUT = {
    "type": "object",
    "required": ["query"],
    "properties": {
        "query": {"type": "string"},
        "max": {"type": "integer"},
    },
}

DRAFT_EMAIL_INPUT = {
    "type": "object",
    "required": ["body"],
    "properties": {
        "to": {"type": ["string", "null"],
               "description": "Recipient email ADDRESS. Required for new mail; "
                              "optional for replies (defaults to the thread's "
                              "reply-to). NEVER invent an address — ask."},
        "subject": {"type": ["string", "null"],
                    "description": "Required for new mail; replies default to Re: …"},
        "body": {"type": "string", "description": "The dictated message body"},
        "reply_to_thread": {"type": ["string", "null"],
                            "description": "Thread id when this is a reply"},
    },
}
DRAFT_EMAIL_OUTPUT = {
    "type": "object",
    "required": ["ok"],
    "properties": {
        **_OK_ERR,
        "draft_id": {"type": ["string", "null"]},
        "to": {"type": ["string", "null"]},
        "subject": {"type": ["string", "null"]},
        "body": {"type": ["string", "null"]},
        "next_step": {"type": ["string", "null"]},
    },
}

SEND_DRAFT_INPUT = {
    "type": "object",
    "required": ["draft_id", "confirm"],
    "properties": {
        "draft_id": {"type": "string"},
        "confirm": {"type": "boolean",
                    "description": "MUST be true, and only after the user has "
                                   "heard the draft read back and explicitly "
                                   "confirmed sending. No exceptions."},
    },
}

EMPTY_INPUT = {"type": "object", "properties": {}}

_STATUS_OUTPUT = {
    "type": "object",
    "required": ["ok"],
    "properties": {
        **_OK_ERR,
        "provider": {"type": ["string", "null"]},
        "status": {"type": ["object", "null"]},
        "started": {"type": ["boolean", "null"]},
        "message": {"type": ["string", "null"]},
    },
}


# ────────────────────────────────────────────────────────────────────────
# Registration
# ────────────────────────────────────────────────────────────────────────

def register_email_tools() -> None:
    """Register the email tools with the global tool registry.
    Called once from registry._register_defaults at module load."""
    from app.tools.registry import ToolDefinition, register_tool

    register_tool(ToolDefinition(
        name="check_email",
        version="v1",
        description=(
            "List recent email threads — default unread only ('anything "
            "new?'). Speak the top few as sender + subject in one or two "
            "short sentences; offer to read any of them."
        ),
        input_schema=CHECK_EMAIL_INPUT,
        output_schema=_THREADS_OUTPUT,
        handler=check_email_handler,
    ))

    register_tool(ToolDefinition(
        name="read_email_thread",
        version="v1",
        description=(
            "Read an email aloud — by thread_id from a listing, or "
            "from_sender for 'read the one from Dave'. Speak spoken_preview; "
            "if truncated is true, offer 'shall I continue?' (the rest is in "
            "full_body). Bodies are already HTML-stripped and de-signatured."
        ),
        input_schema=READ_THREAD_INPUT,
        output_schema=READ_THREAD_OUTPUT,
        handler=read_email_thread_handler,
    ))

    register_tool(ToolDefinition(
        name="search_email",
        version="v1",
        description="Search mail by free text ('find the email about the invoice').",
        input_schema=SEARCH_EMAIL_INPUT,
        output_schema=_THREADS_OUTPUT,
        handler=search_email_handler,
    ))

    register_tool(ToolDefinition(
        name="draft_email",
        version="v1",
        description=(
            "Create a DRAFT (never sends). Use for 'reply saying…' (pass "
            "reply_to_thread) and 'send an email to…' (needs a real address "
            "— if the user only gave a name, ask for the address, never "
            "guess). Then READ THE DRAFT BACK and ask whether to send; only "
            "an explicit yes leads to send_email_draft."
        ),
        input_schema=DRAFT_EMAIL_INPUT,
        output_schema=DRAFT_EMAIL_OUTPUT,
        handler=draft_email_handler,
    ))

    register_tool(ToolDefinition(
        name="send_email_draft",
        version="v1",
        description=(
            "Send a previously drafted email. ONLY call with confirm=true "
            "after the user has heard the draft and explicitly confirmed. "
            "If they decline or hesitate, do NOT call this — the draft "
            "stays saved."
        ),
        input_schema=SEND_DRAFT_INPUT,
        output_schema=_STATUS_OUTPUT,
        handler=send_email_draft_handler,
    ))

    register_tool(ToolDefinition(
        name="email_provider_status",
        version="v1",
        description=(
            "Which email provider is configured (gmail/proton) and whether "
            "it's connected. Use when email tools fail, to explain what "
            "setup is missing."
        ),
        input_schema=EMPTY_INPUT,
        output_schema=_STATUS_OUTPUT,
        handler=email_provider_status_handler,
    ))

    register_tool(ToolDefinition(
        name="setup_email_auth",
        version="v1",
        description=(
            "Start the Gmail OAuth sign-in (opens the consent screen in the "
            "PC browser, callback on port 8091). Use when the user asks to "
            "connect Gmail or email_provider_status says not authenticated."
        ),
        input_schema=EMPTY_INPUT,
        output_schema=_STATUS_OUTPUT,
        handler=setup_email_auth_handler,
    ))
