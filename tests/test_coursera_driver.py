# FILE: tests/test_coursera_driver.py
# Purpose: Amendment A Job 10 — Coursera study driver (resume / read / next),
#          quiz hard-stop, verbatim health relays, zero vision targeting.
# Called-by: pytest
# Depends-on: app.content.distribution.posting_drivers.coursera_driver (+ selector map)
# Last-renovated: 2026-07-02
"""
Acceptance criteria under test (Amendment A v2):
  AC12 — resume lands on the current lesson and reports its title;
         read returns the full transcript; next advances and reports;
         a quiz is announced and NOT attempted.
  AC13 — expired login / closed desktop produce the existing actionable
         relay messages, verbatim.
  AC14 — the driver contains zero vision-targeting calls; the selector
         map holds no coordinate candidates and every candidate is cited.

All offline: a page-state FakeBridge scripts the Coursera surfaces.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from app.content.distribution.posting_drivers import coursera_driver as drv
from app.content.distribution.posting_drivers.driver_runner import load_selector_map
from app.web_automation.coursera_reader import (
    DESKTOP_OFFLINE_MESSAGE,
    LOGIN_NEEDED_MESSAGE,
)

MY_LEARNING = "https://www.coursera.org/my-learning"
COURSE_HOME = "https://www.coursera.org/learn/economics-of-ai/home/module/2"
LECTURE = "https://www.coursera.org/learn/economics-of-ai/lecture/abc12/intro-video"
READING = "https://www.coursera.org/learn/economics-of-ai/supplement/rd34/key-terms"
QUIZ = "https://www.coursera.org/learn/economics-of-ai/exam/qz9/module-quiz"


class Page:
    """One fake page: which selectors exist, what text the tree shows,
    what extract_text returns, and where clicks lead."""

    def __init__(self, selectors=(), elements=(), extracts=None, goto=None, on_click_add=None):
        self.selectors = set(selectors)
        self.elements = list(elements)
        self.extracts = dict(extracts or {})
        self.goto = dict(goto or {})              # clicked-text/selector -> url
        self.on_click_add = dict(on_click_add or {})  # clicked-key -> (selectors, extracts)


def el(text, role="", x=100, y=100, w=200, h=48, href=""):
    e = {"tag": "div", "role": role, "text": text, "x": x, "y": y, "w": w, "h": h}
    if href:
        e["href"] = href
    return e


class FakeCourseraBridge:
    def __init__(self, pages: dict, start: str):
        self.pages = pages
        self.url = start
        self.calls = []

    def page(self) -> Page:
        return self.pages[self.url]

    async def ensure_session_open(self, sid, timeout_seconds=20.0):
        return {"ok": True, "result": {"current_url": self.url, "view_was_fresh": False}}

    def _click_key(self, payload):
        if payload.get("selector"):
            return payload["selector"]
        if payload.get("name"):
            # role/name click (click.js v4, driver_runner 2026-07-03):
            # match by role attr/tag + name/text substring — mirrors the
            # live executor's in-page resolution.
            want_name = str(payload["name"]).strip().lower()
            want_role = str(payload.get("role") or "").strip().lower()
            for e in self.page().elements:
                role = str(e.get("role") or "").lower()
                tag = str(e.get("tag") or "").lower()
                if want_role and role != want_role and tag != want_role:
                    continue
                hay = f"{e.get('text') or ''} {e.get('name') or ''}".lower()
                if want_name in hay:
                    return e.get("text", "")
            return ""
        # coordinate click: reverse-map to the element whose centre matches
        for e in self.page().elements:
            cx = int(e.get("x", 0)) + int(e.get("w", 0)) // 2
            cy = int(e.get("y", 0)) + int(e.get("h", 0)) // 2
            if cx == payload.get("x") and cy == payload.get("y"):
                return e.get("text", "")
        return ""

    async def execute_action(self, sid, action_type, payload=None, timeout_seconds=None):
        payload = payload or {}
        self.calls.append((action_type, payload, self.url))
        page = self.page()
        if action_type == "current_state":
            return {"ok": True, "result": {"url": self.url, "title": "t"}}
        if action_type == "wait_for":
            if payload.get("selector"):
                matched = payload["selector"] in page.selectors
            else:
                hay = " ".join(e.get("text", "") for e in page.elements)
                matched = payload.get("text", "") in hay
            return {"ok": True, "result": {"matched": matched, "timeout": not matched, "waited_ms": 1}}
        if action_type == "dom_snapshot":
            return {"ok": True, "result": {"elements": page.elements}}
        if action_type == "extract_text":
            text = page.extracts.get(payload.get("selector", ""))
            return {"ok": True, "result": {"matches": [text] if text else []}}
        if action_type == "click":
            key = self._click_key(payload)
            if key in page.on_click_add:
                sels, extracts = page.on_click_add[key]
                page.selectors.update(sels)
                page.extracts.update(extracts)
            if key in page.goto:
                self.url = page.goto[key]
            return {"ok": True, "result": {"changed": True}}
        if action_type == "navigate":
            target = payload.get("url", "")
            if target in self.pages:
                self.url = target
            return {"ok": True, "result": {"current_url": self.url}}
        return {"ok": True, "result": {}}


def _lecture_page(**kw):
    # The Next control is reachable by its css candidate (the map's first
    # choice), so clicks resolve by selector; the Transcript tab has no
    # live css yet and resolves text->fresh-snapshot coords instead —
    # both selector-map paths get exercised.
    return Page(
        selectors={"h1", "button[aria-label='Next Item']"},
        elements=[el("Transcript", role="tab", x=50, y=600, w=120, h=40),
                  el("Next Item", role="button", x=800, y=700, w=100, h=40)],
        extracts={"h1": "Intro Video"},
        on_click_add={"Transcript": ({"div.rc-Transcript"},
                                     {"div.rc-Transcript": "Welcome to the course. This is the full transcript."})},
        **kw,
    )


def _pages():
    return {
        MY_LEARNING: Page(
            elements=[el("My Learning"), el("Go To Course", role="button", x=200, y=300, w=180, h=48)],
            goto={"Go To Course": LECTURE},
        ),
        LECTURE: _lecture_page(goto={"button[aria-label='Next Item']": READING}),
        READING: Page(
            selectors={"h1", "div.rc-CML", "button[aria-label='Next Item']"},
            elements=[el("Next Item", role="button", x=800, y=700, w=100, h=40)],
            extracts={"h1": "Key Terms", "div.rc-CML": "Reading body: definitions of the key terms."},
            goto={"button[aria-label='Next Item']": QUIZ},
        ),
        QUIZ: Page(selectors={"h1"}, elements=[el("Start Assignment", role="button")],
                   extracts={"h1": "Module Quiz"}),
        COURSE_HOME: Page(
            selectors={"h1"},
            elements=[el("Resume", role="button", x=300, y=200, w=160, h=48)],
            goto={"Resume": LECTURE},
        ),
    }


async def _ok_health(ref):
    return {"ok": True, "state": "ok", "message": "", "current_url": MY_LEARNING}


def _kw(bridge, **extra):
    base = dict(
        bridge=bridge, session_id="cs1", health_fn=_ok_health,
        pace_range=(0, 0), persist_heal=False, poll_s=0, nav_timeout_s=0.2,
    )
    base.update(extra)
    return base


def _no_vision_calls(bridge):
    kinds = {a for a, _, _ in bridge.calls}
    assert "screenshot" not in kinds, "driver took a screenshot (vision path)"
    assert not any(p.get("snap_to_button") for _, p, _ in bridge.calls)


# ── URL classification ───────────────────────────────────────────────

def test_item_type_from_url():
    assert drv.item_type_from_url(LECTURE) == "video"
    assert drv.item_type_from_url(READING) == "reading"
    assert drv.item_type_from_url(QUIZ) == "quiz"
    assert drv.item_type_from_url(COURSE_HOME) == "course_home"
    assert drv.item_type_from_url(MY_LEARNING) == ""
    assert drv.item_type_from_url("https://www.coursera.org/learn/x/assignment-submission/1/a") == "assignment"


# ── AC12 · resume ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resume_lands_on_lesson_and_reports():
    bridge = FakeCourseraBridge(_pages(), MY_LEARNING)
    res = await drv.resume_course(**_kw(bridge))
    assert res["ok"] is True and res["state"] == "ok"
    assert res["item_title"] == "Intro Video"
    assert res["item_type"] == "video"
    assert res["course"] == "Economics Of Ai"
    _no_vision_calls(bridge)


@pytest.mark.asyncio
async def test_resume_hops_through_course_home():
    pages = _pages()
    pages[MY_LEARNING].goto["Go To Course"] = COURSE_HOME
    bridge = FakeCourseraBridge(pages, MY_LEARNING)
    res = await drv.resume_course(**_kw(bridge))
    assert res["ok"] is True and res["item_title"] == "Intro Video"
    _no_vision_calls(bridge)


@pytest.mark.asyncio
async def test_resume_named_course_uses_dashboard_link():
    pages = _pages()
    pages[MY_LEARNING].elements.append(
        el("The Economics of AI", role="", href="/learn/economics-of-ai/home/module/2")
    )
    pages[MY_LEARNING].goto.pop("Go To Course")
    bridge = FakeCourseraBridge(pages, MY_LEARNING)
    res = await drv.resume_course(course="economics", **_kw(bridge))
    assert res["ok"] is True and res["item_type"] == "video"


# ── AC13 · verbatim relays ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_logged_out_relayed_verbatim():
    async def health(ref):
        return {"ok": False, "state": "logged_out",
                "message": LOGIN_NEEDED_MESSAGE, "current_url": "https://www.coursera.org/"}

    bridge = FakeCourseraBridge(_pages(), MY_LEARNING)
    res = await drv.resume_course(**_kw(bridge, health_fn=health))
    assert res["state"] == "logged_out"
    assert res["message"] == LOGIN_NEEDED_MESSAGE  # verbatim, no rewording
    assert res["ok"] is False


@pytest.mark.asyncio
async def test_desktop_offline_relayed_verbatim():
    async def health(ref):
        return {"ok": False, "state": "desktop_offline",
                "message": DESKTOP_OFFLINE_MESSAGE, "current_url": ""}

    bridge = FakeCourseraBridge(_pages(), MY_LEARNING)
    res = await drv.read_transcript(**_kw(bridge, health_fn=health))
    assert res["state"] == "desktop_offline"
    assert res["message"] == DESKTOP_OFFLINE_MESSAGE


@pytest.mark.asyncio
async def test_mid_lesson_expiry_caught_by_markers():
    pages = _pages()
    pages[LECTURE].elements.append(el("Join for Free", role="button"))
    bridge = FakeCourseraBridge(pages, LECTURE)
    res = await drv.read_transcript(**_kw(bridge))
    assert res["state"] == "logged_out"
    assert res["message"] == LOGIN_NEEDED_MESSAGE


# ── AC12 · read transcript / reading ─────────────────────────────────

@pytest.mark.asyncio
async def test_read_transcript_opens_panel_and_returns_full_text():
    bridge = FakeCourseraBridge(_pages(), LECTURE)
    res = await drv.read_transcript(**_kw(bridge))
    assert res["ok"] is True and res["item_type"] == "video"
    assert res["transcript_text"] == "Welcome to the course. This is the full transcript."
    assert res["item_title"] == "Intro Video"
    # The Transcript toggle was clicked because the panel started closed.
    assert any(a == "click" for a, _, u in bridge.calls if u == LECTURE)
    _no_vision_calls(bridge)


@pytest.mark.asyncio
async def test_read_reading_item_extracts_article_body():
    bridge = FakeCourseraBridge(_pages(), READING)
    res = await drv.read_transcript(**_kw(bridge))
    assert res["ok"] is True and res["item_type"] == "reading"
    assert "definitions of the key terms" in res["transcript_text"]
    _no_vision_calls(bridge)


@pytest.mark.asyncio
async def test_read_transcript_resumes_first_when_not_on_lesson():
    bridge = FakeCourseraBridge(_pages(), MY_LEARNING)
    res = await drv.read_transcript(**_kw(bridge))
    assert res["ok"] is True
    assert res["transcript_text"].startswith("Welcome to the course.")


@pytest.mark.asyncio
async def test_transcript_truncated_at_cap():
    pages = _pages()
    long_text = "word " * 8000  # ~40k chars
    pages[LECTURE].on_click_add["Transcript"] = (
        {"div.rc-Transcript"}, {"div.rc-Transcript": long_text},
    )
    bridge = FakeCourseraBridge(pages, LECTURE)
    res = await drv.read_transcript(**_kw(bridge))
    assert res["truncated"] is True
    assert len(res["transcript_text"]) <= drv.TRANSCRIPT_CHAR_CAP + 40
    assert res["transcript_text"].endswith("[transcript truncated]")


# ── AC12 · next item + quiz hard-stop ────────────────────────────────

@pytest.mark.asyncio
async def test_next_item_advances_and_reports():
    bridge = FakeCourseraBridge(_pages(), LECTURE)
    res = await drv.next_item(**_kw(bridge))
    assert res["ok"] is True
    assert res["item_type"] == "reading"
    assert res["item_title"] == "Key Terms"
    _no_vision_calls(bridge)


@pytest.mark.asyncio
async def test_next_item_announces_quiz_and_stops():
    bridge = FakeCourseraBridge(_pages(), READING)
    res = await drv.next_item(**_kw(bridge))
    assert res["ok"] is True
    assert res["item_type"] == "quiz"
    assert res["message"] == drv.QUIZ_STOP_MESSAGE
    # Nothing was clicked or extracted ON the quiz page — landing only.
    on_quiz = [(a, p) for a, p, u in bridge.calls if u == QUIZ]
    assert all(a in ("current_state",) for a, _ in on_quiz), on_quiz
    _no_vision_calls(bridge)


@pytest.mark.asyncio
async def test_next_item_no_advance_reports_without_blind_retry():
    pages = _pages()
    pages[LECTURE].goto = {}  # Next click goes nowhere
    bridge = FakeCourseraBridge(pages, LECTURE)
    res = await drv.next_item(**_kw(bridge))
    assert res["ok"] is False
    assert "didn't move on" in res["message"] or "did not advance" in res["error"]
    clicks = [(a, p) for a, p, _ in bridge.calls if a == "click"]
    assert len(clicks) == 1, f"blind retry detected: {clicks}"


@pytest.mark.asyncio
async def test_resume_stops_on_quiz_landing():
    pages = _pages()
    pages[MY_LEARNING].goto["Go To Course"] = QUIZ
    bridge = FakeCourseraBridge(pages, MY_LEARNING)
    res = await drv.resume_course(**_kw(bridge))
    assert res["ok"] is True and res["item_type"] == "quiz"
    assert res["message"] == drv.QUIZ_STOP_MESSAGE
    on_quiz = [(a, p) for a, p, u in bridge.calls if u == QUIZ]
    assert all(a in ("current_state",) for a, _ in on_quiz)


# ── AC14 · zero vision targeting; map hygiene ────────────────────────

def test_driver_source_has_no_vision_targeting():
    src = inspect.getsource(drv)
    assert "vision_check" not in src
    assert "snap_to_button" not in src
    assert '"screenshot"' not in src and "'screenshot'" not in src


def test_selector_map_cited_and_coordinate_free():
    smap = load_selector_map("coursera")
    assert smap["landing_url"].endswith("/my-learning")
    for step, cands in smap["steps"].items():
        assert cands, f"{step} empty"
        for c in cands:
            assert "cite" in c and "verified" in c, f"{step}: uncited candidate {c}"
            assert "x" not in c and "y" not in c, f"{step}: coordinate candidate {c}"
            assert c.get("css") or c.get("text"), f"{step}: candidate has no locator {c}"


def test_recon_importable_and_nondestructive_by_design():
    from app.content.distribution.posting_drivers import coursera_recon
    src = inspect.getsource(coursera_recon)
    # The quiz state is captured from its landing page only.
    assert "LANDING ONLY" in src
    assert callable(coursera_recon.run_coursera_recon)


# ── tool-surface wiring ──────────────────────────────────────────────

def test_chat_layer_exposes_study_tools():
    from app.debug.web_tool_definitions import get_web_tools
    names = {t["name"] for t in get_web_tools()}
    assert {"coursera_resume", "coursera_read_lesson", "coursera_next_item"} <= names


@pytest.mark.asyncio
async def test_study_handler_unknown_session_is_clean_error(monkeypatch):
    from app.web_automation import coursera_study_tools as study
    from app.web_automation import tool_handlers

    monkeypatch.setattr(tool_handlers, "_resolve_session", lambda ref: None)
    out = await study.coursera_resume_handler({"session": "nope"}, None)
    assert out["ok"] is False and "no session matches" in out["error"]
