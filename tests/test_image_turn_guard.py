# FILE: tests/test_image_turn_guard.py
# Purpose: Prove the per-turn image guard bills exactly one paid generation
#          for a retried-but-identical image request (J2 cost defense).
# Called-by: pytest
# Depends-on: app.llm.image_turn_guard, asyncio
# Last-renovated: 2026-06-16
"""
Pure tests (no real image generation). A stub generator increments a
counter; we assert the guard runs it EXACTLY ONCE across repeated calls
with the same key, and that distinct keys each get their own generation.
"""
from __future__ import annotations

import asyncio

import pytest

from app.llm import image_turn_guard as guard


@pytest.fixture(autouse=True)
def _clean_cache():
    guard.reset_for_tests()
    yield
    guard.reset_for_tests()


def _make_generator():
    """Return (generator, counter_box). Each call increments the counter and
    returns a unique-per-call artifact so we can prove cache identity."""
    box = {"count": 0}

    async def gen():
        box["count"] += 1
        # tiny await so concurrent callers genuinely overlap on the lock
        await asyncio.sleep(0.01)
        return {"filename": f"img_{box['count']}.png", "provider": "stub"}

    return gen, box


def test_three_identical_deliveries_generate_once():
    """3 sequential deliveries of the same turn -> 1 generation, same artifact."""
    async def run():
        gen, box = _make_generator()
        results = []
        for _ in range(3):
            r = await guard.run_guarded_image(
                project_id=42, message="draw a cat",
                generator=gen, source="test",
            )
            results.append(r)
        return box["count"], results

    count, results = asyncio.run(run())
    assert count == 1, f"generator should run exactly once, ran {count}"
    assert all(r == results[0] for r in results), "all callers get same artifact"
    assert results[0]["filename"] == "img_1.png"


def test_concurrent_retries_join_one_generation():
    """3 concurrent deliveries (the real retry-storm shape) -> 1 generation."""
    async def run():
        gen, box = _make_generator()
        results = await asyncio.gather(*[
            guard.run_guarded_image(
                project_id=7, message="make a sunset",
                generator=gen, source="test",
            )
            for _ in range(3)
        ])
        return box["count"], results

    count, results = asyncio.run(run())
    assert count == 1, f"concurrent retries must share one generation, ran {count}"
    assert all(r == results[0] for r in results)


def test_two_different_keys_generate_twice():
    """Distinct turns (different messages) each get their own generation."""
    async def run():
        gen, box = _make_generator()
        a = await guard.run_guarded_image(
            project_id=1, message="a red car", generator=gen, source="test",
        )
        b = await guard.run_guarded_image(
            project_id=1, message="a blue boat", generator=gen, source="test",
        )
        return box["count"], a, b

    count, a, b = asyncio.run(run())
    assert count == 2, f"two distinct turns -> two generations, got {count}"
    assert a["filename"] != b["filename"]


def test_client_request_id_dedupes_across_different_messages():
    """Same client_request_id collides even if the message text drifts —
    mirrors a phone retry that re-sends with the same X-Idempotency-Key."""
    async def run():
        gen, box = _make_generator()
        a = await guard.run_guarded_image(
            project_id=1, message="draw a dog",
            generator=gen, client_request_id="abc-123", source="test",
        )
        b = await guard.run_guarded_image(
            project_id=1, message="draw a dog please",  # slightly different text
            generator=gen, client_request_id="abc-123", source="test",
        )
        return box["count"], a, b

    count, a, b = asyncio.run(run())
    assert count == 1, f"same client id -> one generation, got {count}"
    assert a == b


def test_normalized_message_collides():
    """Casing/whitespace variants of the same turn share one generation."""
    async def run():
        gen, box = _make_generator()
        await guard.run_guarded_image(
            project_id=3, message="Draw   A Cat", generator=gen, source="test",
        )
        await guard.run_guarded_image(
            project_id=3, message="draw a cat", generator=gen, source="test",
        )
        return box["count"]

    count = asyncio.run(run())
    assert count == 1, f"normalized variants must collide, got {count}"


def test_failed_generation_is_not_cached():
    """A None (failed) result is NOT cached, so an honest retry can try again."""
    async def run():
        box = {"count": 0}

        async def gen():
            box["count"] += 1
            # first call fails, second succeeds
            if box["count"] == 1:
                return None
            return {"filename": "ok.png", "provider": "stub"}

        first = await guard.run_guarded_image(
            project_id=9, message="a guarded prompt", generator=gen, source="test",
        )
        second = await guard.run_guarded_image(
            project_id=9, message="a guarded prompt", generator=gen, source="test",
        )
        return box["count"], first, second

    count, first, second = asyncio.run(run())
    assert first is None
    assert second == {"filename": "ok.png", "provider": "stub"}
    assert count == 2, "failure must not be cached; retry re-runs the generator"


def test_kill_switch_passes_through(monkeypatch):
    """With the guard disabled, every call hits the generator (no dedupe)."""
    monkeypatch.setenv("ASTRA_IMAGE_TURN_GUARD", "0")

    async def run():
        gen, box = _make_generator()
        for _ in range(3):
            await guard.run_guarded_image(
                project_id=5, message="same prompt", generator=gen, source="test",
            )
        return box["count"]

    count = asyncio.run(run())
    assert count == 3, f"kill-switch should disable dedupe, got {count}"
