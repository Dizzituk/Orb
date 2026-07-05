# FILE: app/content/distribution/posting_drivers/coursera_driver.py
# Purpose: Coursera study driver — resume course / read transcript / next item (selector-map composite).
# Called-by: app.web_automation.coursera_study_tools, app.content.distribution.posting_drivers.coursera_recon, tests.test_coursera_driver
# Depends-on: .driver_runner, .self_heal, app.web_automation.coursera_reader, app.web_automation.bridge (injectable)
# Last-renovated: 2026-07-02
"""
Coursera study driver (Amendment A Job 10) — the second consumer of the
shared driver_runner, powering hands-free study from the phone:
"resume my course" / "read me the lesson" / "next lesson".

Design rules:
  * Zero vision targeting. Navigation is selector-map candidates
    (role/aria/stable-text) gated by wait_for; text is read with
    extract_text (plain DOM). The only vision in the Coursera surface
    stays web_coursera_progress's completion-tick read. Self-heal may
    consult the vision model to PROPOSE a css selector for a dead step
    (shared layer, same as the meta driver) — it never yields
    coordinates.
  * Login safety: resume_course (and any call that has to leave the
    current page) runs coursera_reader._login_state first and relays
    its logged_out / desktop_offline messages VERBATIM. When a call
    finds the view already parked on a lesson item, it does a
    no-navigation marker check instead — a full _login_state would
    navigate to My Learning and yank the browser off the lesson
    mid-study-loop; an expired session still gets caught because the
    public page carries the LOGGED_OUT_MARKERS.
  * Quiz safety: item types come from the URL path. quiz / exam /
    assignment items are ANNOUNCED and never interacted with.

Everything network-shaped is injectable (bridge, session_id, health_fn,
vision_fn) so the whole flow is unit-testable with a fake bridge.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urljoin, urlsplit

from app.content.distribution.posting_drivers import self_heal
from app.content.distribution.posting_drivers.driver_runner import StepRunner

logger = logging.getLogger(__name__)

PLATFORM = "coursera"
MAP_NAME = "coursera"
SURFACE = "the Coursera learner interface"

# URL path segment (after /learn/<slug>/) -> item type.
SEGMENT_TYPES = {
    "lecture": "video",
    "supplement": "reading",
    "quiz": "quiz",
    "exam": "quiz",
    "assessment": "quiz",
    "assignment-submission": "assignment",
    "programming": "assignment",
    "peer": "assignment",
    "gradedLti": "assignment",
    "ungradedLti": "assignment",
    "ungradedWidget": "activity",
    "discussionPrompt": "activity",
}
# Item types the driver refuses to drive into / act on.
BLOCKED_TYPES = {"quiz", "assignment"}

QUIZ_STOP_MESSAGE = (
    "This item is a quiz/assessment — I've stopped on its landing page and "
    "won't attempt it. Quizzes are yours to take; tell me when you're done "
    "and I'll carry on with the next lesson."
)

# Keep transcripts inside the chat tool-result budget (30k chars/tool).
TRANSCRIPT_CHAR_CAP = 24000

# HealFn-compatible wrapper is built per-call so bridge/vision_fn inject.
HealthFn = Callable[[str], Any]


# ── URL classification ───────────────────────────────────────────────

def _path_parts(url: str) -> List[str]:
    try:
        return [p for p in urlsplit(url or "").path.split("/") if p]
    except (ValueError, TypeError):
        return []


def item_type_from_url(url: str) -> str:
    """'video' / 'reading' / 'quiz' / 'assignment' / 'activity' /
    'course_home' / '' (not on a course page)."""
    parts = _path_parts(url)
    if len(parts) < 2 or parts[0] != "learn":
        return ""
    if len(parts) < 3 or parts[2] in ("home", "welcome", ""):
        return "course_home"
    return SEGMENT_TYPES.get(parts[2], "activity")


def _is_item(url: str) -> bool:
    return item_type_from_url(url) not in ("", "course_home")


def _course_from_url(url: str) -> str:
    parts = _path_parts(url)
    if len(parts) >= 2 and parts[0] == "learn":
        return parts[1].replace("-", " ").title()
    return ""


# ── plumbing (mirrors meta_driver) ───────────────────────────────────

def _default_bridge():
    from app.web_automation import bridge
    return bridge


def _resolve_session_id(platform: str) -> Optional[str]:
    from app.db import SessionLocal
    from app.web_automation import session_registry
    db = SessionLocal()
    try:
        s = session_registry.get_session_by_platform(db, platform)
        return s.id if s else None
    finally:
        db.close()


async def _default_health(ref: str) -> dict:
    from app.web_automation.coursera_reader import _login_state
    return await _login_state(ref)


def _offline(error: str) -> bool:
    from app.web_automation.action_queue import DESKTOP_OFFLINE_PREFIX
    return (error or "").startswith(DESKTOP_OFFLINE_PREFIX)


def _result(state: str, **kw) -> Dict[str, Any]:
    base = {
        "ok": state == "ok",
        "state": state,
        "course": "",
        "item_title": "",
        "item_type": "",
        "current_url": "",
        "message": "",
        "error": "",
    }
    base.update(kw)
    return base


def _relay_health(health: dict) -> Dict[str, Any]:
    """Pass coursera_reader's health verdict through untouched (message
    verbatim), reshaped to the study-tool result contract."""
    return _result(
        health.get("state", "error"),
        message=health.get("message", ""),
        current_url=health.get("current_url", ""),
        error=health.get("error", ""),
    )


class _Driver:
    """One study action against one session. Thin state bundle so the
    three public functions share plumbing without re-passing kwargs."""

    def __init__(
        self,
        *,
        bridge: Any = None,
        session_id: Optional[str] = None,
        health_fn: Optional[HealthFn] = None,
        vision_fn=None,
        pace_range: tuple = (0.8, 2.5),
        persist_heal: bool = True,
        selector_map: Optional[dict] = None,
        poll_s: float = 0.5,
        nav_timeout_s: float = 12.0,
    ):
        self.bridge = bridge or _default_bridge()
        self.sid = session_id or _resolve_session_id(PLATFORM)
        self.health_fn = health_fn or _default_health
        self.poll_s = poll_s
        self.nav_timeout_s = nav_timeout_s

        async def heal(session_id_, step, goal, els):
            return await self_heal.relocate(
                session_id_, step, goal, els,
                bridge=self.bridge, vision_fn=vision_fn, surface=SURFACE,
            )

        self.runner = (
            StepRunner(
                self.sid, PLATFORM, MAP_NAME,
                bridge=self.bridge, heal=heal, pace_range=pace_range,
                persist_heal=persist_heal, selector_map=selector_map,
            )
            if self.sid
            else None
        )

    # ── primitives ───────────────────────────────────────────────────

    async def url(self) -> str:
        r = await self.bridge.execute_action(
            self.sid, "current_state", {}, timeout_seconds=8.0
        )
        if not r.get("ok") and _offline(r.get("error") or ""):
            raise _Offline(r.get("error") or "")
        return ((r.get("result") or {}).get("url") or "")

    async def poll_url(self, done, timeout_s: Optional[float] = None) -> str:
        """Poll current_state until done(url) or timeout. Returns last URL."""
        if timeout_s is None:
            timeout_s = self.nav_timeout_s
        deadline = asyncio.get_event_loop().time() + timeout_s
        last = ""
        while True:
            last = await self.url()
            if done(last):
                return last
            if asyncio.get_event_loop().time() >= deadline:
                return last
            await asyncio.sleep(self.poll_s)

    async def snapshot_texts(self) -> str:
        r = await self.bridge.execute_action(
            self.sid, "dom_snapshot", {}, timeout_seconds=20.0
        )
        els = ((r.get("result") or {}).get("elements")) or []
        return " | ".join(str(e.get("text") or "") for e in els if e.get("text")).lower()

    async def extract_first(self, step: str, limit: int = 3) -> str:
        """extract_text over the step's css candidates; first non-empty wins."""
        for cand in self.runner.candidates(step):
            css = cand.get("css")
            if not css:
                continue
            r = await self.bridge.execute_action(
                self.sid, "extract_text", {"selector": css, "limit": limit},
                timeout_seconds=15.0,
            )
            if not r.get("ok"):
                if _offline(r.get("error") or ""):
                    raise _Offline(r.get("error") or "")
                continue
            matches = [m for m in ((r.get("result") or {}).get("matches") or []) if str(m).strip()]
            if matches:
                return "\n".join(str(m).strip() for m in matches)
        return ""

    async def title(self) -> str:
        text = await self.extract_first("item_title", limit=1)
        return text.splitlines()[0].strip() if text else ""

    async def logged_out_markers_present(self) -> bool:
        from app.web_automation.coursera_reader import LOGGED_OUT_MARKERS
        texts = await self.snapshot_texts()
        return any(m in texts for m in LOGGED_OUT_MARKERS)

    # ── composite hops ───────────────────────────────────────────────

    async def resume_from_dashboard(self, course: Optional[str]) -> Dict[str, Any]:
        """From a health-verified My Learning page, land on the current
        lesson item. Returns a partial result dict on failure, {} on success."""
        from app.web_automation.coursera_reader import MY_LEARNING_URL

        if course:
            target = await self._find_course_link(course)
            if not target:
                return _result(
                    "error",
                    error=f"no course matching '{course}' found on My Learning",
                    message=f"I couldn't find a course called '{course}' on your My Learning page.",
                )
            nav = await self.bridge.execute_action(
                self.sid, "navigate", {"url": target}, timeout_seconds=25.0
            )
            if not nav.get("ok") and _offline(nav.get("error") or ""):
                raise _Offline(nav.get("error") or "")
        else:
            r = await self.runner.act(
                "resume_entry", "click", verify=False, timeout_ms=10000,
                goal="the Resume / Go To Course button on the My Learning dashboard",
            )
            if not r["ok"]:
                return _result(
                    "error",
                    error=r.get("error") or "could not find a resume button on My Learning",
                    message="I couldn't find a Resume / Go To Course button on your My Learning page.",
                )

        await self.poll_url(lambda u: "/learn/" in u)

        # Course home is one hop short of the item — click its Resume.
        for _ in range(2):
            cur = await self.url()
            if _is_item(cur):
                return {}
            if item_type_from_url(cur) != "course_home":
                break
            if await self.runner.gate("course_home_resume", timeout_ms=6000):
                await self.runner.act(
                    "course_home_resume", "click", verify=False, timeout_ms=6000,
                    goal="the Resume/Continue button on the course home page",
                )
                await self.poll_url(_is_item)
            else:
                # No resume CTA — jump to the first lesson item link.
                target = await self._first_item_link()
                if not target:
                    break
                await self.bridge.execute_action(
                    self.sid, "navigate", {"url": target}, timeout_seconds=25.0
                )
                await self.poll_url(_is_item)

        cur = await self.url()
        if _is_item(cur):
            return {}
        return _result(
            "error",
            current_url=cur,
            error=f"resume did not reach a lesson item (stopped at {cur or MY_LEARNING_URL})",
            message="I couldn't get from the dashboard to a lesson item — the page layout may have changed. A recon dump will help me heal.",
        )

    async def _find_course_link(self, course: str) -> Optional[str]:
        needle = course.strip().lower()
        r = await self.bridge.execute_action(
            self.sid, "dom_snapshot", {}, timeout_seconds=20.0
        )
        best = None
        for el in ((r.get("result") or {}).get("elements")) or []:
            href = el.get("href") or ""
            hay = f"{el.get('text') or ''} {el.get('name') or ''}".lower()
            if "/learn/" in href and needle in hay:
                best = href
                break
        if not best:
            return None
        from app.web_automation.coursera_reader import MY_LEARNING_URL
        return urljoin(MY_LEARNING_URL, best)

    async def _first_item_link(self) -> Optional[str]:
        r = await self.bridge.execute_action(
            self.sid, "dom_snapshot", {}, timeout_seconds=20.0
        )
        for el in ((r.get("result") or {}).get("elements")) or []:
            href = el.get("href") or ""
            if "/lecture/" in href or "/supplement/" in href:
                from app.web_automation.coursera_reader import MY_LEARNING_URL
                return urljoin(MY_LEARNING_URL, href)
        return None

    async def describe_item(self, url: str) -> Dict[str, Any]:
        """ok-result for the item the view is sitting on (title read,
        quiz announced)."""
        itype = item_type_from_url(url)
        if itype in BLOCKED_TYPES:
            return _result(
                "ok",
                course=_course_from_url(url),
                item_type=itype,
                current_url=url,
                message=QUIZ_STOP_MESSAGE,
            )
        await self.runner.gate("lesson_ready", timeout_ms=10000)
        title = await self.title()
        return _result(
            "ok",
            course=_course_from_url(url),
            item_title=title,
            item_type=itype,
            current_url=url,
            message=f"On '{title or 'the current item'}' ({itype}).",
        )


class _Offline(Exception):
    def __init__(self, error: str):
        super().__init__(error)
        self.error = error


def _offline_result(err: str) -> Dict[str, Any]:
    from app.web_automation.coursera_reader import DESKTOP_OFFLINE_MESSAGE
    return _result("desktop_offline", message=DESKTOP_OFFLINE_MESSAGE, error=err)


def _no_session_result() -> Dict[str, Any]:
    return _result(
        "error",
        error=f"no '{PLATFORM}' web session registered",
        message="No Coursera browser session is registered on the desktop.",
    )


async def _show_view(d: _Driver) -> None:
    """Best-effort: put the session's browser view ON SCREEN (Electron
    show_view executor, 2026-07-03) so the user can watch the study
    drive and so vision checks capture a composited page — hidden views
    screenshot empty (16:00 incident: the login vision check ran blind).
    Never blocks the flow: older Electron builds answer 'unknown action
    type' and we carry on off-screen exactly as before."""
    try:
        await d.bridge.execute_action(d.sid, "show_view", {}, timeout_seconds=6.0)
    except Exception:
        pass


async def _gated_entry(d: _Driver, course: Optional[str]) -> Dict[str, Any]:
    """Full health check (navigates to My Learning) + resume hop.
    Returns {} when the view is on a lesson item, else the failure result."""
    health = await d.health_fn(PLATFORM)
    # Accept "ok" and "unknown" — if the health check couldn't confirm
    # login state but found no explicit logout markers, proceed rather
    # than looping.  The visual fallback in _login_state (2026-07-03)
    # should almost never let "unknown" through, but this is defense-
    # in-depth against verification loops.
    if health.get("state") not in ("ok", "unknown"):
        return _relay_health(health)
    fail = await d.resume_from_dashboard(course)
    return fail or {}


# ── public API ───────────────────────────────────────────────────────

async def resume_course(course: Optional[str] = None, **kw) -> Dict[str, Any]:
    """'Resume my course' — dashboard (health-gated) → current lesson item.
    Returns {ok, state, course, item_title, item_type, current_url, message}."""
    d = _Driver(**kw)
    if not d.sid:
        return _no_session_result()
    try:
        await _show_view(d)
        fail = await _gated_entry(d, course)
        if fail:
            return fail
        return await d.describe_item(await d.url())
    except _Offline as e:
        return _offline_result(e.error)
    except Exception as e:  # never let the study loop take the tool loop down
        logger.exception("[coursera_driver] resume_course raised")
        return _result("error", error=f"resume_course failed: {e}")


async def read_transcript(**kw) -> Dict[str, Any]:
    """Read the current lesson's transcript (video) or article body
    (reading) as plain DOM text — no vision. Resumes first if the view
    isn't on a lesson item."""
    d = _Driver(**kw)
    if not d.sid:
        return _no_session_result()
    try:
        await _show_view(d)
        cur = await d.url()
        if not _is_item(cur):
            fail = await _gated_entry(d, None)
            if fail:
                return fail
            cur = await d.url()
        elif await d.logged_out_markers_present():
            from app.web_automation.coursera_reader import LOGIN_NEEDED_MESSAGE
            return _result("logged_out", message=LOGIN_NEEDED_MESSAGE, current_url=cur)

        itype = item_type_from_url(cur)
        if itype in BLOCKED_TYPES:
            return _result(
                "ok", course=_course_from_url(cur), item_type=itype,
                current_url=cur, message=QUIZ_STOP_MESSAGE,
            )

        if itype == "video":
            # Open the transcript panel if it isn't already rendered.
            if not await d.runner.gate("transcript_body", timeout_ms=3000):
                await d.runner.act(
                    "transcript_toggle", "click", verify=False, timeout_ms=8000,
                    goal="the Transcript tab on a Coursera lecture page",
                )
                await d.runner.gate("transcript_body", timeout_ms=10000)
            text = await d.extract_first("transcript_body")
            what = "transcript"
        elif itype == "reading":
            text = await d.extract_first("reading_body")
            what = "article body"
        else:
            text = await d.extract_first("reading_body") or await d.extract_first("transcript_body")
            what = "content"

        title = await d.title()
        if not text:
            return _result(
                "error",
                course=_course_from_url(cur), item_title=title, item_type=itype,
                current_url=cur,
                error=f"no {what} text found on the page",
                message=f"I couldn't find the {what} on this page — the panel may not have rendered. A recon dump will confirm the selectors.",
            )

        truncated = len(text) > TRANSCRIPT_CHAR_CAP
        if truncated:
            text = text[:TRANSCRIPT_CHAR_CAP] + "\n… [transcript truncated]"
        return _result(
            "ok",
            course=_course_from_url(cur), item_title=title, item_type=itype,
            current_url=cur, transcript_text=text, truncated=truncated,
            message=f"Read the {what} of '{title or 'the current item'}'.",
        )
    except _Offline as e:
        return _offline_result(e.error)
    except Exception as e:
        logger.exception("[coursera_driver] read_transcript raised")
        return _result("error", error=f"read_transcript failed: {e}")


async def next_item(**kw) -> Dict[str, Any]:
    """Advance to the next lesson item and report it. Quizzes are
    announced and never attempted."""
    d = _Driver(**kw)
    if not d.sid:
        return _no_session_result()
    try:
        await _show_view(d)
        cur = await d.url()
        if not _is_item(cur):
            fail = await _gated_entry(d, None)
            if fail:
                return fail
            cur = await d.url()
        elif await d.logged_out_markers_present():
            from app.web_automation.coursera_reader import LOGIN_NEEDED_MESSAGE
            return _result("logged_out", message=LOGIN_NEEDED_MESSAGE, current_url=cur)

        old = cur
        r = await d.runner.act(
            "next_item", "click", verify=False, timeout_ms=10000,
            goal="the Next button that advances to the next lesson item",
        )
        if not r["ok"]:
            return _result(
                "error", current_url=old,
                error=r.get("error") or "Next control not found",
                message="I couldn't find the Next control on this lesson — it may be the end of the course, or the layout changed.",
            )

        landed = await d.poll_url(lambda u: u and u != old)
        if landed == old:
            # Policy: a missed advance is a re-orient signal, not a
            # blind-retry signal.
            return _result(
                "error", current_url=old,
                error="page did not advance after clicking Next",
                message="The page didn't move on after Next — this may be the last item, or the click needs a fresh look. I won't blind-retry.",
            )
        return await d.describe_item(landed)
    except _Offline as e:
        return _offline_result(e.error)
    except Exception as e:
        logger.exception("[coursera_driver] next_item raised")
        return _result("error", error=f"next_item failed: {e}")
