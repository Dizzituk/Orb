# FILE: tests/test_email_adapter.py
# Purpose: Email adapter tests — the provider seam (fake provider via config,
#          intents unchanged), draft-confirm-send rails, body hygiene, factory.
# Called-by: pytest
# Depends-on: app.email, app.tools.email_tools
# Last-renovated: 2026-06-12
from __future__ import annotations

from typing import List, Optional

import pytest

import app.email as email_factory
from app.email.provider import (
    EmailProvider,
    Message,
    Thread,
    ThreadSummary,
    clean_body,
    strip_html,
    trim_signature,
)
from app.tools import email_tools


# ── a complete fake provider: proves the seam ──────────────────────────────

class FakeProvider(EmailProvider):
    """In-memory mailbox implementing the full contract."""

    name = "fake"

    def __init__(self):
        self.threads = {
            "t1": Thread(id="t1", subject="Van MOT booking", messages=[
                Message(id="m1", sender="Dave <dave@garage.example>",
                        to="taz@example.com", date="2026-06-12T09:00:00",
                        body="Morning — van's booked for Tuesday 8am. Dave"),
            ]),
            "t2": Thread(id="t2", subject="Invoice 441", messages=[
                Message(id="m2", sender="accounts@supplier.example",
                        to="taz@example.com", date="2026-06-11T15:00:00",
                        body="Please find invoice 441 attached."),
            ]),
        }
        self.unread = {"t1"}
        self.drafts = {}
        self.sent = []
        self._draft_seq = 0

    def list_threads(self, query=None, label=None, max_results=10) -> List[ThreadSummary]:
        out = []
        for tid, thread in self.threads.items():
            if label == "UNREAD" and tid not in self.unread:
                continue
            if query and query.lower().replace("from:", "") not in (
                    thread.messages[0].sender.lower() + thread.subject.lower()):
                continue
            newest = thread.messages[-1]
            out.append(ThreadSummary(
                id=tid, sender=newest.sender, subject=thread.subject,
                snippet=newest.body[:80], unread=tid in self.unread,
                date=newest.date, tags=sorted({"UNREAD"} if tid in self.unread else set()),
            ))
        return out[:max_results]

    def read_thread(self, thread_id: str) -> Thread:
        return self.threads[thread_id]

    def create_draft(self, to, subject, body, reply_to_thread=None) -> dict:
        self._draft_seq += 1
        draft_id = f"d{self._draft_seq}"
        if reply_to_thread:
            thread = self.threads[reply_to_thread]
            to = to or thread.messages[-1].sender
            subject = subject or f"Re: {thread.subject}"
        self.drafts[draft_id] = {"to": to, "subject": subject, "body": body,
                                 "reply_to_thread": reply_to_thread}
        return {"ok": True, "draft_id": draft_id, "to": to,
                "subject": subject, "body": body}

    def send_draft(self, draft_id: str) -> dict:
        draft = self.drafts.pop(draft_id, None)
        if not draft:
            return {"ok": False, "error": "no such draft"}
        self.sent.append(draft)
        return {"ok": True, "id": f"sent-{len(self.sent)}"}

    def send(self, to, subject, body, reply_to_thread=None) -> dict:
        self.sent.append({"to": to, "subject": subject, "body": body})
        return {"ok": True, "id": f"sent-{len(self.sent)}"}


@pytest.fixture()
def fake_mailbox(monkeypatch):
    fake = FakeProvider()
    email_factory.register_provider("fake", lambda: fake)
    monkeypatch.setenv("ASTRA_EMAIL_PROVIDER", "fake")
    yield fake
    email_factory._PROVIDER_FACTORIES.pop("fake", None)
    email_factory._instances.pop("fake", None)


# ── the seam: every intent works against the fake, unchanged ──────────────

@pytest.mark.asyncio
async def test_check_email_speaks_unread(fake_mailbox):
    out = await email_tools.check_email_handler({}, None)
    assert out["ok"] and out["provider"] == "fake"
    assert len(out["threads"]) == 1
    assert out["threads"][0]["subject"] == "Van MOT booking"
    assert out["threads"][0]["unread"] is True


@pytest.mark.asyncio
async def test_read_the_one_from_dave(fake_mailbox):
    out = await email_tools.read_email_thread_handler({"from_sender": "dave"}, None)
    assert out["ok"] and "Tuesday 8am" in out["spoken_preview"]
    assert out["truncated"] is False


@pytest.mark.asyncio
async def test_full_voice_loop_draft_confirm_send(fake_mailbox):
    drafted = await email_tools.draft_email_handler(
        {"reply_to_thread": "t1", "body": "Perfect, see you Tuesday."}, None)
    assert drafted["ok"] and drafted["draft_id"]
    assert drafted["to"].startswith("Dave")          # reply-to resolved
    assert drafted["subject"] == "Re: Van MOT booking"
    assert "explicit yes" in drafted["next_step"]
    assert fake_mailbox.sent == []                   # drafting sent NOTHING

    sent = await email_tools.send_email_draft_handler(
        {"draft_id": drafted["draft_id"], "confirm": True}, None)
    assert sent["ok"]
    assert len(fake_mailbox.sent) == 1
    assert fake_mailbox.sent[0]["body"] == "Perfect, see you Tuesday."


@pytest.mark.asyncio
async def test_declining_leaves_draft_unsent(fake_mailbox):
    drafted = await email_tools.draft_email_handler(
        {"to": "dave@garage.example", "subject": "Hi", "body": "Hello"}, None)
    refused = await email_tools.send_email_draft_handler(
        {"draft_id": drafted["draft_id"], "confirm": False}, None)
    assert refused["ok"] is False and "not sent" in refused["error"]
    assert fake_mailbox.sent == []
    assert drafted["draft_id"] in fake_mailbox.drafts  # draft intact


@pytest.mark.asyncio
async def test_new_email_requires_real_address_and_subject(fake_mailbox):
    no_addr = await email_tools.draft_email_handler({"body": "hi"}, None)
    assert no_addr["ok"] is False and "address" in no_addr["error"]
    no_subj = await email_tools.draft_email_handler(
        {"to": "x@y.example", "body": "hi"}, None)
    assert no_subj["ok"] is False and "subject" in no_subj["error"]


@pytest.mark.asyncio
async def test_search_email(fake_mailbox):
    out = await email_tools.search_email_handler({"query": "invoice"}, None)
    assert out["ok"] and out["threads"][0]["subject"] == "Invoice 441"


# ── body hygiene ───────────────────────────────────────────────────────────

def test_strip_html_basics():
    html = "<div><p>Hello <b>world</b></p><br><script>evil()</script>x &amp; y</div>"
    text = strip_html(html)
    assert "Hello world" in text and "evil" not in text and "x & y" in text


def test_trim_signature():
    body = "See you then.\n-- \nDave Smith\nGarage Ltd\nLegal footer"
    assert trim_signature(body) == "See you then."
    body2 = "On my way\nSent from my iPhone"
    assert trim_signature(body2) == "On my way"


def test_clean_body_pipeline():
    out = clean_body("<p>Hi</p><p>Done.</p><p>-- </p><p>Sig</p>", is_html=True)
    assert out.startswith("Hi") and "Sig" not in out


# ── factory config ─────────────────────────────────────────────────────────

def test_factory_defaults_to_gmail(monkeypatch):
    monkeypatch.delenv("ASTRA_EMAIL_PROVIDER", raising=False)
    monkeypatch.setattr(email_factory, "configured_provider_name", lambda: "gmail")
    provider = email_factory.get_provider()
    assert provider.name == "gmail"


def test_factory_rejects_unknown(monkeypatch):
    with pytest.raises(ValueError, match="unknown email provider"):
        email_factory.get_provider("pigeon-post")


# ── gmail mime construction (no network) ───────────────────────────────────

def test_gmail_reply_mime_has_threading_headers():
    from app.email.gmail_provider import GmailProvider
    provider = GmailProvider()
    msg = provider._build_mime(
        "dave@garage.example", "Re: Van", "body text",
        reply_headers={"message_id": "<abc@mail>", "references": "<root@mail>"},
    )
    assert msg["In-Reply-To"] == "<abc@mail>"
    assert "<root@mail>" in msg["References"] and "<abc@mail>" in msg["References"]
    assert msg["To"] == "dave@garage.example"
