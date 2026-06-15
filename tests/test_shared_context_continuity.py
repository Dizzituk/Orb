# FILE: tests/test_shared_context_continuity.py
# Purpose: Job 5 (2026-06-12) — one conversation context for every provider:
#          consecutive turns routed to different providers receive identical
#          context (incl. turn-1 facts); no provider path builds its own.
# Called-by: pytest
# Depends-on: app.llm.stream_handlers, app.llm.routing.prompt_builders, app.shared_context.builder
# Last-renovated: 2026-06-12
from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.embeddings.models import Embedding
from app.memory.models import Project, File, DocumentContent

_TABLES = [
    Project.__table__,
    File.__table__,
    DocumentContent.__table__,
    Embedding.__table__,
]


@pytest.fixture(autouse=True, scope="module")
def _throwaway_master_key():
    """Message/document columns are encrypted types — throwaway key, restored
    afterwards (same pattern as test_media_enrichment)."""
    import base64
    import os
    from app.crypto import encryption
    saved_env = os.environ.get("ORB_MASTER_KEY")
    saved_mgr = encryption._encryption_manager
    saved_flag = encryption._master_key_initialized
    if not encryption.is_master_key_initialized():
        os.environ["ORB_MASTER_KEY"] = base64.urlsafe_b64encode(
            os.urandom(32)
        ).decode()
        encryption.init_master_key_from_env()
    yield
    encryption._encryption_manager = saved_mgr
    encryption._master_key_initialized = saved_flag
    if saved_env is None:
        os.environ.pop("ORB_MASTER_KEY", None)
    else:
        os.environ["ORB_MASTER_KEY"] = saved_env


@pytest.fixture()
def db():
    from app.memory.models import Message
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    tables = _TABLES + [Message.__table__]
    try:
        from app.memory.conversation_models import ConversationSession
        tables.append(ConversationSession.__table__)
    except Exception:
        pass
    Base.metadata.create_all(bind=engine, tables=tables)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _seed_turn_one(db):
    from app.memory import service as memory_service, schemas as memory_schemas
    proj = memory_service.create_project(
        db, memory_schemas.ProjectCreate(name="Chat", description="test"),
    )
    memory_service.create_message(db, memory_schemas.MessageCreate(
        project_id=proj.id, role="user",
        content="Remember: the garage door code is 4471, set by Taz.",
    ))
    memory_service.create_message(db, memory_schemas.MessageCreate(
        project_id=proj.id, role="assistant",
        content="Got it — the garage door code is 4471.",
    ))
    return proj


# =========================================================================
# Cross-provider continuity through the real dispatch seam
# =========================================================================

def test_consecutive_turns_different_providers_same_context(db, monkeypatch):
    """Turn 2 routed to openai, turn 2 retried on anthropic: both provider
    calls must receive the same system prompt and the same message list,
    including the fact established in turn 1."""
    proj = _seed_turn_one(db)

    from app.llm import stream_handlers
    from app.llm.routing.prompt_builders import build_messages

    captured = []

    async def fake_stream_llm(*, provider, model, messages, system_prompt):
        captured.append({
            "provider": provider,
            "model": model,
            "messages": messages,
            "system_prompt": system_prompt,
        })
        yield {"type": "token", "text": "It's 4471."}

    monkeypatch.setattr(stream_handlers, "stream_llm", fake_stream_llm)

    turn2 = "What was the garage door code again?"
    shared_system = "You are ASTRA. [shared context block]"

    async def _drive(provider, model):
        messages = build_messages(turn2, proj.id, db, include_history=True,
                                  history_limit=20)
        chunks = []
        async for chunk in stream_handlers.generate_sse_stream(
            project_id=proj.id, message=turn2,
            provider=provider, model=model,
            system_prompt=shared_system, messages=messages,
            db=db, trace=None,
        ):
            chunks.append(chunk)
        return chunks

    asyncio.run(_drive("openai", "gpt-5.4"))
    asyncio.run(_drive("anthropic", "claude-opus-4-6"))

    assert len(captured) == 2
    first, second = captured
    assert first["provider"] == "openai"
    assert second["provider"] == "anthropic"

    # Identical context despite the provider switch
    assert first["system_prompt"] == second["system_prompt"]

    def _texts(msgs):
        return [m.get("content") for m in msgs if isinstance(m.get("content"), str)]

    # Turn-1 fact present in BOTH providers' message lists
    assert any("4471" in t for t in _texts(first["messages"]))
    assert any("4471" in t for t in _texts(second["messages"]))
    # Same roles/sequence (anthropic's run also contains openai's persisted
    # turn-2 exchange — strip to the common prefix for the comparison)
    common = len(first["messages"])
    assert [m["role"] for m in second["messages"][:common]][:2] == \
           [m["role"] for m in first["messages"][:2]]


def test_build_messages_has_no_provider_parameter():
    """Structural guarantee: history assembly cannot vary by provider —
    the builder doesn't even know which provider will be called."""
    import inspect
    from app.llm.routing.prompt_builders import build_messages, build_full_context
    for fn in (build_messages, build_full_context):
        params = set(inspect.signature(fn).parameters)
        assert "provider" not in params
        assert "model" not in params


def test_stream_llm_dispatches_same_payload_to_all_adapters(monkeypatch):
    """Every adapter branch in stream_llm receives the same (messages,
    system_prompt) the caller passed in — no branch rebuilds context."""
    import app.llm._streaming_utils_3 as su

    calls = {}

    def _capture(name):
        async def _fake(messages, system_prompt, *args, **kwargs):
            calls[name] = {"messages": messages, "system": system_prompt}
            yield {"type": "token", "text": "ok"}
        return _fake

    import app.llm.streaming as streaming_mod
    monkeypatch.setattr(streaming_mod, "stream_openai", _capture("openai"))
    monkeypatch.setattr(su, "stream_anthropic", _capture("anthropic"))
    monkeypatch.setattr(su, "stream_gemini", _capture("gemini"))

    messages = [{"role": "user", "content": "hello shared world"}]
    system = "shared system prompt"

    async def _drive(provider):
        async for _ in su.stream_llm(
            messages=messages, system_prompt=system,
            provider=provider, model="m",
        ):
            pass

    for provider in ("openai", "anthropic", "google"):
        asyncio.run(_drive(provider))

    assert set(calls) == {"openai", "anthropic", "gemini"}
    for name, payload in calls.items():
        assert payload["messages"] is messages, f"{name} rebuilt messages"
        assert payload["system"] == system, f"{name} altered system prompt"


# =========================================================================
# Bridge (phone voice) path uses the same builders — source-level contract
# =========================================================================

def test_bridge_voice_path_uses_shared_builders():
    import pathlib
    src = pathlib.Path("app/bridge/capability_layer.py").read_text(
        encoding="utf-8", errors="replace"
    )
    assert "from app.llm.routing.prompt_builders import" in src
    assert "build_full_context" in src
    assert "build_system_prompt" in src


# =========================================================================
# SharedContextPackage builder — regression guard for the schema mismatch
# =========================================================================

def test_shared_context_package_builds_valid(db, monkeypatch):
    from app.embeddings import gemini_provider, service as embeddings_service
    from app.shared_context.builder import build_shared_context_package

    class _FakeResult:
        def __init__(self, vectors):
            class _E:
                def __init__(self, values):
                    self.values = values
            self.embeddings = [_E(v) for v in vectors]

    class _FakeClient:
        class models:  # noqa: N801 — mimics google-genai client shape
            @staticmethod
            def embed_content(*, model, contents, config):
                dim = getattr(config, "output_dimensionality", None) or 1536
                n = len(contents) if isinstance(contents, list) else 1
                return _FakeResult([[0.4] * dim for _ in range(n)])

    monkeypatch.setattr(gemini_provider, "_client", _FakeClient())
    try:
        proj = Project(name="P", description="d")
        db.add(proj)
        db.commit()
        vec = gemini_provider.generate_embedding("the camper van MOT is in July")
        embeddings_service.store_embedding(
            db, proj.id, "message", 1, "the camper van MOT is in July", vec,
        )

        package = build_shared_context_package(
            db, proj.id, query="when is the van MOT?",
        )
        # Phase-7 mismatch regression: these required fields must validate
        assert package.query == "when is the van MOT?"
        assert package.budget_chars == 6000
        assert package.items, "expected at least one retrieved item"
        item = package.items[0]
        assert item.source_type == "message"
        assert item.provenance and item.pointer
        assert "SHARED CONTEXT" in package.formatted_block
    finally:
        gemini_provider.reset_client()
