# FILE: tests/test_image_refs.py
# Purpose: Derek vision upgrade — image refs persist, extract, and re-analyse into spec evidence + ledger.
# Called-by: pytest
# Depends-on: app.llm.image_refs, app.llm._weaver_stream_utils_15
# Last-renovated: 2026-07-04
"""Pixels-travel tests: markers round-trip through chat history, get extracted
deterministically (dedup/cap/missing-file guards), and re-analyse into a spec
evidence block that lands in the Decision Ledger; plus the raised vision cap."""

import asyncio
import json

import pytest

from app.llm import image_refs
from app.llm.image_refs import (
    build_image_evidence,
    extract_image_refs,
    image_ref_marker,
)


def _img(tmp_path, name="shot.png"):
    p = tmp_path / name
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)  # plausible PNG header
    return str(p)


def test_marker_round_trips_through_message_content(tmp_path):
    path = _img(tmp_path)
    # This is exactly what chat persistence appends to the stored message
    content = "Astra, build me a dashboard like this\n\n" + image_ref_marker(path, "shot.png")
    refs = extract_image_refs([{"role": "user", "content": content}])
    assert refs == [path]


def test_extraction_dedups_and_skips_missing(tmp_path):
    real = _img(tmp_path, "real.png")
    msgs = [
        {"role": "user", "content": image_ref_marker(real, "real.png")},
        {"role": "user", "content": image_ref_marker(real, "real.png")},          # dup
        {"role": "user", "content": "[image_ref: C:/gone/missing.png | missing.png]"},  # not on disk
        {"role": "user", "content": "[image_ref: C:/x/notes.txt | notes.txt]"},   # not an image
    ]
    assert extract_image_refs(msgs) == [real]


def test_extraction_caps_newest(monkeypatch, tmp_path):
    monkeypatch.setenv("ASTRA_SPECGATE_IMAGE_MAX", "2")
    paths = [_img(tmp_path, f"s{i}.png") for i in range(4)]
    msgs = [{"role": "user", "content": image_ref_marker(p)} for p in paths]
    # newest 2 kept
    assert extract_image_refs(msgs) == paths[-2:]


def test_weaver_vision_cap_raised(monkeypatch):
    from app.llm._weaver_stream_utils_15 import _extract_vision_context
    from app.llm import _weaver_stream_utils_10 as u10
    monkeypatch.setattr(u10, "_is_vision_context", lambda c: True)
    monkeypatch.setenv("ASTRA_WEAVER_VISION_CONTEXT_CHARS", "4000")
    big = "X" * 5000
    out = _extract_vision_context([{"role": "assistant", "content": big}])
    assert len(out) == 4000            # raised from the old hardcoded 1000
    assert len(out) > 1000


def test_build_evidence_reanalyses_and_records_ledger(monkeypatch, tmp_path):
    path = _img(tmp_path, "screen.png")

    async def fake_analyse(p, goal):
        assert p == path
        return "VERBATIM TEXT: 'Score: 9000'. Layout: grid 10x20. Colour: neon."

    monkeypatch.setattr(image_refs, "analyse_image_for_spec", fake_analyse)

    import app.pot_spec.grounded._spec_runner_utils_11 as u11
    monkeypatch.setattr(u11, "_get_job_dir_for_segmentation", lambda jid: str(tmp_path / jid))

    block = asyncio.run(build_image_evidence(
        image_refs=[path], goal="build a tetris clone", job_id="vis-job-1",
    ))
    assert "IMAGE EVIDENCE" in block
    assert "Score: 9000" in block
    assert "screen.png" in block

    # Landed in the Decision Ledger as spec-gate vision evidence
    ledger_file = tmp_path / "vis-job-1" / "decision_ledger.json"
    assert ledger_file.exists()
    entries = json.loads(ledger_file.read_text(encoding="utf-8"))["entries"]
    assert any(
        e.get("category") == "specgate_vision" and "screen.png" in (e.get("path") or "")
        for e in entries
    )


def test_build_evidence_consumes_chat_vision_context(monkeypatch, tmp_path):
    # No image refs, but chat-time vision context still gets carried through
    block = asyncio.run(build_image_evidence(
        image_refs=[], goal="x", job_id="",
        chat_vision_context="The screenshot showed a red login button top-right.",
    ))
    assert "red login button" in block
    assert "Chat-time screenshot analysis" in block


def test_build_evidence_empty_when_nothing(monkeypatch):
    assert asyncio.run(build_image_evidence(image_refs=[], goal="x", job_id="")) == ""


def test_disabled_flag_skips_reanalysis(monkeypatch, tmp_path):
    monkeypatch.setenv("ASTRA_SPECGATE_IMAGE_ANALYSIS", "0")
    path = _img(tmp_path)
    called = {"n": 0}

    async def spy(p, g):
        called["n"] += 1
        return "should not run"

    monkeypatch.setattr(image_refs, "analyse_image_for_spec", spy)
    block = asyncio.run(build_image_evidence(image_refs=[path], goal="x", job_id=""))
    assert called["n"] == 0
    assert block == ""   # analysis off + no chat context = nothing
