# FILE: tests/test_shorts_pipeline.py
# Purpose: shorts script/caption/orchestrator/delivery (Jobs 6, 7, 8).
# Called-by: pytest
# Depends-on: app.content.video_pipeline.{shorts_script,caption_align,shorts_orchestrator,shorts_delivery,shorts_job}
# Last-renovated: 2026-07-02
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.content.video_pipeline import (
    caption_align, shorts_delivery, shorts_orchestrator, shorts_script,
)
from app.content.video_pipeline import shorts_job as shorts_job_mod
from app.content.video_pipeline.shorts_job import ShortsJob, slugify


# ── shorts_script ────────────────────────────────────────────────────

def _mk_llm(script_words, title="Test Title", caption="cap", tags=None):
    async def llm(messages, system, model, provider):
        return json.dumps({
            "script": " ".join(["word"] * script_words),
            "title": title,
            "caption": caption,
            "hashtags": tags if tags is not None else ["ai", "shorts"],
        })
    return llm


@pytest.mark.asyncio
async def test_generate_script_parses_and_keeps_short():
    out = await shorts_script.generate_script("AI news", llm=_mk_llm(40))
    assert out["word_count"] == 40 and out["title"] == "Test Title"
    assert out["hashtags"] == ["ai", "shorts"]


@pytest.mark.asyncio
async def test_generate_script_enforces_word_cap():
    # Model ignores the limit every time -> hard-trim to WORD_CAP.
    out = await shorts_script.generate_script("Rambly topic", llm=_mk_llm(200))
    assert out["word_count"] <= shorts_script.WORD_CAP


@pytest.mark.asyncio
async def test_generate_script_fallback_on_garbage():
    async def junk_llm(messages, system, model, provider):
        return "not json at all"
    out = await shorts_script.generate_script("Topic", llm=junk_llm)
    assert out["script"] and out["title"]  # never returns empty


def test_provider_inference():
    assert shorts_script._provider_for("gpt-5.5") == "openai"
    assert shorts_script._provider_for("claude-opus-4-6") == "anthropic"
    assert shorts_script._provider_for("gemini-3.5-flash") == "google"
    assert shorts_script._provider_for("nat-local-3b") is None


# ── caption_align (pure grouping) ────────────────────────────────────

def test_group_words_2_to_4_per_caption():
    words = [{"word": f"w{i}", "start": i * 0.5, "end": i * 0.5 + 0.4} for i in range(10)]
    caps = caption_align._group_words(words, min_n=2, max_n=4)
    assert all(1 <= len(c.text.split()) <= 4 for c in caps)
    # timing is monotonic and non-degenerate
    for c in caps:
        assert c.end_seconds > c.start_seconds
    assert caps[0].start_seconds == 0.0


def test_fallback_from_script_even_split():
    caps = caption_align._fallback_from_script("one two three four five six", 6.0)
    assert caps and caps[-1].end_seconds <= 6.01
    assert caption_align._fallback_from_script("", 5.0) == []


def test_shorts_caption_style_is_9x16_safe():
    s = caption_align.SHORTS_CAPTION_STYLE
    assert s["bold"] and s["alignment"] == 2
    assert s["margin_v"] >= 200  # clear of platform UI overlays


# ── shorts_job ───────────────────────────────────────────────────────

def test_slugify_and_caption_tags():
    assert slugify("Hello, World! AI & Shorts") == "hello-world-ai-shorts"
    job = ShortsJob(topic="t", caption="Nice clip", hashtags=["ai", "#news"])
    ct = job.caption_with_tags()
    assert "#ai" in ct and "#news" in ct and "Nice clip" in ct


# ── shorts_orchestrator ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_short_job_full_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(shorts_job_mod, "SHORTS_OUTPUT_DIR", tmp_path / "shorts")
    mp4 = tmp_path / "render.mp4"
    mp4.write_bytes(b"\x00")

    captured = {}

    async def fake_heygen(*, text, segment_id, aspect_ratio):
        captured["aspect_ratio"] = aspect_ratio
        return {"file_path": str(mp4), "duration_s": 30.0, "cost_usd": 0.1}

    def fake_caption(video_path, out_dir, *, slug, script_text, duration_s):
        return {"burned_path": video_path, "srt_path": f"{out_dir}/x.srt", "caption_count": 6}

    async def fake_deliver(job, *, project_id, autopublish, post_fn):
        captured["autopublish"] = autopublish
        job.output_id = "out-123"
        return {"ok": True}

    job = shorts_orchestrator.create_short_job("AI safety", "punchy")
    out = await shorts_orchestrator.run_short_job(
        job, project_id=7, autopublish=False,
        llm=_mk_llm(50), heygen_fn=fake_heygen,
        caption_fn=fake_caption, deliver_fn=fake_deliver,
    )
    assert out.status == "complete"
    assert captured["aspect_ratio"] == "9:16"       # AC: 9:16 render
    assert out.mp4_path == str(mp4) and out.captioned_path == str(mp4)
    assert out.output_id == "out-123"
    stages = [e["stage"] for e in out.events]
    assert "script" in stages and "render" in stages and "captions" in stages and "deliver" in stages


@pytest.mark.asyncio
async def test_run_short_job_error_path(tmp_path, monkeypatch):
    monkeypatch.setattr(shorts_job_mod, "SHORTS_OUTPUT_DIR", tmp_path / "shorts")

    async def boom_heygen(*, text, segment_id, aspect_ratio):
        raise RuntimeError("heygen down")

    job = shorts_orchestrator.create_short_job("topic")
    out = await shorts_orchestrator.run_short_job(
        job, llm=_mk_llm(30), heygen_fn=boom_heygen,
    )
    assert out.status == "error" and "heygen down" in out.error


def test_autopublish_env_flag(monkeypatch):
    monkeypatch.delenv("ASTRA_SHORTS_AUTOPUBLISH", raising=False)
    assert shorts_orchestrator.autopublish_enabled() is False
    monkeypatch.setenv("ASTRA_SHORTS_AUTOPUBLISH", "true")
    assert shorts_orchestrator.autopublish_enabled() is True


# ── shorts_delivery ──────────────────────────────────────────────────

@pytest.fixture
def content_db():
    # Create ONLY the content tables — the shared Base carries a
    # messages->conversation_sessions FK (the known fresh-DB bootstrap gap,
    # see CLAUDE.md) that a full create_all can't resolve in isolation.
    from app.content.models import (
        ContentSeries, ContentTopic, ContentPiece, ContentOutput,
    )
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                        poolclass=StaticPool)
    tables = [t.__table__ for t in (ContentSeries, ContentTopic, ContentPiece, ContentOutput)]
    tables[0].metadata.create_all(eng, tables=tables)
    Session = sessionmaker(bind=eng)
    db = Session()
    yield db
    db.close()


def test_latest_pending_and_mark_published(content_db):
    from app.content.models import ContentPiece, ContentOutput
    piece = ContentPiece(title="p", content_category="short", status="review")
    content_db.add(piece)
    content_db.commit()
    content_db.refresh(piece)

    older = ContentOutput(piece_id=piece.id, output_format="instagram_reel",
                          platform="instagram", primary_asset_path="/a.mp4")
    newer = ContentOutput(piece_id=piece.id, output_format="instagram_reel",
                          platform="instagram", primary_asset_path="/b.mp4")
    content_db.add(older)
    content_db.commit()
    content_db.add(newer)
    content_db.commit()

    latest = shorts_delivery.get_latest_pending_short(content_db)
    assert latest.primary_asset_path == "/b.mp4"

    assert shorts_delivery.mark_short_published(content_db, latest.id, "https://x/p/1") is True
    content_db.refresh(latest)
    assert latest.published_at is not None and latest.platform_post_id == "https://x/p/1"
    # now the older one is the only pending
    assert shorts_delivery.get_latest_pending_short(content_db).primary_asset_path == "/a.mp4"


@pytest.mark.asyncio
async def test_deliver_short_review_hold(tmp_path, monkeypatch):
    monkeypatch.setenv("SHORTS_DELIVERY_DIR", str(tmp_path / "delivery"))
    monkeypatch.setattr(shorts_job_mod, "SHORTS_OUTPUT_DIR", tmp_path / "shorts")

    mp4 = tmp_path / "final.mp4"
    mp4.write_bytes(b"\x00\x01")

    class _Out:
        id = "out-9"

    posted = {}

    class _DummyDB:
        def close(self):
            pass

    monkeypatch.setattr("app.db.SessionLocal", lambda: _DummyDB())
    monkeypatch.setattr(shorts_delivery, "_create_records", lambda db, job, asset: _Out())
    monkeypatch.setattr(shorts_delivery, "_resolve_delivery_project", lambda db, pid: object())

    def cap_msg(db, project, text):
        posted["text"] = text
    monkeypatch.setattr(shorts_delivery, "_post_message", cap_msg)

    job = ShortsJob(topic="t", title="My Short", caption="hi", hashtags=["ai"])
    job.captioned_path = str(mp4)
    out = await shorts_delivery.deliver_short(job, project_id=3, autopublish=False)

    assert out["ok"] and out["autopublished"] is False and out["output_id"] == "out-9"
    assert "Short ready: My Short" in posted["text"]
    assert "[ASTRA_ARTIFACT:video:" in posted["text"]
    assert job.output_id == "out-9" and job.delivered_filename


@pytest.mark.asyncio
async def test_deliver_short_autopublish(tmp_path, monkeypatch):
    monkeypatch.setenv("SHORTS_DELIVERY_DIR", str(tmp_path / "delivery"))
    monkeypatch.setattr(shorts_job_mod, "SHORTS_OUTPUT_DIR", tmp_path / "shorts")
    mp4 = tmp_path / "final.mp4"
    mp4.write_bytes(b"\x00\x01")

    class _Out:
        id = "out-42"

    marks = {}

    class _DummyDB:
        def close(self):
            pass

    monkeypatch.setattr("app.db.SessionLocal", lambda: _DummyDB())
    monkeypatch.setattr(shorts_delivery, "_create_records", lambda db, job, asset: _Out())
    monkeypatch.setattr(shorts_delivery, "_resolve_delivery_project", lambda db, pid: object())
    monkeypatch.setattr(shorts_delivery, "_post_message", lambda db, p, t: None)
    monkeypatch.setattr(shorts_delivery, "mark_short_published",
                        lambda db, oid, link: marks.update({"id": oid, "link": link}) or True)

    from app.content.distribution.posting_drivers.results import PostResult

    async def fake_post(path, caption):
        return PostResult(ok=True, platform="meta_business", permalink="https://insta/p/9")

    job = ShortsJob(topic="t", title="Auto Short", caption="hi")
    job.captioned_path = str(mp4)
    out = await shorts_delivery.deliver_short(job, autopublish=True, post_fn=fake_post)

    assert out["ok"] and out["autopublished"] is True
    assert out["permalink"] == "https://insta/p/9"
    assert marks["id"] == "out-42" and marks["link"] == "https://insta/p/9"
