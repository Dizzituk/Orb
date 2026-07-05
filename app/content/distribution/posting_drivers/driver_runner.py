# FILE: app/content/distribution/posting_drivers/driver_runner.py
# Purpose: Shared deterministic step-executor for composite posting drivers.
# Called-by: app.content.distribution.posting_drivers.meta_driver, .coursera_driver, tests
# Depends-on: app.web_automation.bridge (injectable), .selector_maps/*.json, .self_heal (injectable)
# Last-renovated: 2026-07-03
"""
Driver runner — walks a selector map one named step at a time.

Every step is gated by `wait_for` BEFORE it acts (the fix for the old
"agent acted before the page was ready" failure), tries its candidate
locators in order, retries a failed step once after a settle, and only
then hands to self-heal. Nothing here improvises: no screenshots, no
vision, no coordinate guessing on the happy path — css selectors gate
and click; the text/dom_snapshot fallback exists only for resilience
and is what self-heal upgrades to a css selector.

Self-confirming map (2026-07-03): every candidate that actually WORKS is
promoted to the front of its step and stamped verified + observed, so
the map converges to proven selectors through ordinary use — self-heal's
philosophy applied to the happy path. (The first live Coursera resume
burned a full 10s wait window on a dead first candidate before the
second matched; promotion makes that a one-time cost.) `persist_heal`
gates ALL map persistence: healed candidates and observed stamps alike.

Injectable `bridge` and `heal` make the whole thing unit-testable with
a fake bridge and no live browser.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import date
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_MAPS_DIR = Path(__file__).parent / "selector_maps"

# heal(session_id, step_name, goal_text, dom_elements) -> Optional[css_selector]
HealFn = Callable[[str, str, str, List[dict]], Awaitable[Optional[str]]]


def load_selector_map(name: str) -> dict:
    """Load selector_maps/<name>.json."""
    path = _MAPS_DIR / f"{name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_selector_map(name: str, data: dict) -> str:
    """Persist a selector map back to disk (self-heal writes here)."""
    path = _MAPS_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return str(path)


def find_in_snapshot(
    elements: List[dict], text: str, role: str = ""
) -> Optional[Dict[str, int]]:
    """Find the best element whose text/name contains `text` (and role, if given).

    Returns click-centre coords, preferring the largest matching element
    (mirrors web_click's "when text collides, prefer the bigger box" rule).
    """
    needle = (text or "").strip().lower()
    if not needle:
        return None
    best = None
    best_area = -1
    for el in elements or []:
        if role and (el.get("role") or "") != role and (el.get("tag") or "") != role:
            continue
        hay = f"{el.get('text') or ''} {el.get('name') or ''}".lower()
        if needle not in hay:
            continue
        w = int(el.get("w") or 0)
        h = int(el.get("h") or 0)
        area = w * h
        if area >= best_area:
            best_area = area
            best = el
    if best is None:
        return None
    return {
        "x": int(best.get("x", 0)) + int(best.get("w", 0)) // 2,
        "y": int(best.get("y", 0)) + int(best.get("h", 0)) // 2,
    }


class StepRunner:
    """Executes named steps from a selector map against one session."""

    def __init__(
        self,
        session_id: str,
        platform: str,
        map_name: str,
        *,
        bridge: Any,
        heal: Optional[HealFn] = None,
        pace_range: tuple = (0.8, 2.5),
        settle_s: float = 2.0,
        persist_heal: bool = True,
        selector_map: Optional[dict] = None,
    ):
        self.session_id = session_id
        self.platform = platform
        self.map_name = map_name
        self.bridge = bridge
        self.heal = heal
        self.pace_range = pace_range
        self.settle_s = settle_s
        self.persist_heal = persist_heal
        self.map = selector_map if selector_map is not None else load_selector_map(map_name)
        self.trace: List[Dict[str, Any]] = []

    # ── public API ───────────────────────────────────────────────────

    def candidates(self, step: str) -> List[dict]:
        return list((self.map.get("steps") or {}).get(step) or [])

    async def pace(self) -> None:
        """Human-ish gap between steps. Range collapses to 0 in tests."""
        lo, hi = self.pace_range
        if hi <= 0:
            return
        await asyncio.sleep(random.uniform(lo, hi))

    async def gate(
        self, step: str, *, state: str = "visible", timeout_ms: int = 15000
    ) -> bool:
        """Wait until any candidate for `step` is present. No action taken."""
        for cand in self.candidates(step):
            if await self._wait(cand, state=state, timeout_ms=timeout_ms):
                return True
        return False

    async def act(
        self,
        step: str,
        action: str,
        *,
        value: Optional[str] = None,
        gate_state: str = "visible",
        timeout_ms: int = 15000,
        verify: bool = True,
        goal: str = "",
    ) -> Dict[str, Any]:
        """Gate then perform `action` (click|type|upload|none) for `step`.

        Retries the whole candidate list once after a settle, then hands
        to self-heal (which prepends a fresh css candidate and retries
        once more). Returns {ok, step, candidate?, error?}.
        """
        cands = self.candidates(step)
        res = await self._try_candidates(step, cands, action, value, gate_state, timeout_ms, verify)
        if res["ok"]:
            return res

        await asyncio.sleep(self.settle_s)
        res = await self._try_candidates(step, cands, action, value, gate_state, timeout_ms, verify)
        if res["ok"]:
            return res

        # Self-heal: the ONLY place a big model enters the flow.
        if self.heal is not None:
            elements = await self._dom_elements()
            new_css = await self.heal(self.session_id, step, goal or step, elements)
            if new_css:
                cand = {"css": new_css, "cite": "self-heal", "verified": False}
                self._prepend_candidate(step, cand)
                logger.info("[driver_runner] self-heal added css for %s: %s", step, new_css)
                healed = await self._try_candidates(
                    step, [cand], action, value, gate_state, timeout_ms, verify
                )
                if healed["ok"]:
                    healed["healed"] = True
                    return healed

        return {"ok": False, "step": step, "error": res.get("error") or "all candidates failed"}

    # ── internals ────────────────────────────────────────────────────

    async def _try_candidates(
        self, step, cands, action, value, gate_state, timeout_ms, verify
    ) -> Dict[str, Any]:
        last_err = "no candidates"
        for cand in cands:
            matched = await self._wait(cand, state=gate_state, timeout_ms=timeout_ms)
            if not matched:
                last_err = f"{step}: candidate not present ({_desc(cand)})"
                continue
            outcome = await self._do(cand, action, value, verify)
            self.trace.append({"step": step, "candidate": _desc(cand), **outcome})
            if outcome["ok"]:
                self._record_success(step, cand)
                return {"ok": True, "step": step, "candidate": _desc(cand)}
            last_err = f"{step}: {outcome.get('error') or 'action failed'} ({_desc(cand)})"
        return {"ok": False, "step": step, "error": last_err}

    async def _wait(self, cand: dict, *, state: str, timeout_ms: int) -> bool:
        payload: Dict[str, Any] = {"state": state, "timeout_ms": timeout_ms}
        if cand.get("css"):
            payload["selector"] = cand["css"]
        elif cand.get("text"):
            payload["text"] = cand["text"]
        else:
            return False
        r = await self.bridge.execute_action(
            self.session_id, "wait_for", payload,
            timeout_seconds=timeout_ms / 1000.0 + 8.0,
        )
        res = r.get("result") or {}
        return bool(r.get("ok")) and bool(res.get("matched"))

    async def _do(self, cand: dict, action: str, value, verify: bool) -> Dict[str, Any]:
        if action == "none":
            return {"ok": True}
        if action == "click":
            return await self._click(cand, verify)
        if action == "type":
            return await self._type(cand, value)
        if action == "upload":
            return await self._upload(cand, value)
        return {"ok": False, "error": f"unknown action '{action}'"}

    async def _click(self, cand: dict, verify: bool) -> Dict[str, Any]:
        if cand.get("css"):
            r = await self.bridge.execute_action(
                self.session_id, "click", {"selector": cand["css"]}, timeout_seconds=10.0
            )
        else:
            # Text candidates resolve IN-PAGE at click time (click.js v4
            # role/name mode): getByRole-style match, scrollIntoView to
            # centre, then real mouse events at the LIVE element centre.
            # The old path — our own dom_snapshot -> viewport coords ->
            # raw (x, y) click — missed anything below the fold: the
            # snapshot reports rect.y past the viewport, nothing sits at
            # that point, and the mouse events land in dead space while
            # still reporting ok (live incident 2026-07-03 16:00, Coursera
            # resume_entry at y=1063: changed=false, element=null). A
            # role/name miss now returns ok=false, so candidate iteration
            # and self-heal genuinely engage instead of hollow-succeeding.
            r = await self.bridge.execute_action(
                self.session_id, "click",
                {"name": cand.get("text", ""), "role": cand.get("role", "") or ""},
                timeout_seconds=10.0,
            )
        if not r.get("ok"):
            return {"ok": False, "error": r.get("error") or "click failed"}
        if verify:
            changed = bool((r.get("result") or {}).get("changed", True))
            if not changed:
                return {"ok": False, "error": "click landed but page did not change"}
        return {"ok": True}

    async def _type(self, cand: dict, value) -> Dict[str, Any]:
        if not cand.get("css"):
            return {"ok": False, "error": "type requires a css candidate (contenteditable/input)"}
        r = await self.bridge.execute_action(
            self.session_id, "type",
            {"selector": cand["css"], "text": value or "", "clear_first": True},
            timeout_seconds=15.0,
        )
        return {"ok": bool(r.get("ok")), "error": r.get("error") or ""}

    async def _upload(self, cand: dict, value) -> Dict[str, Any]:
        if cand.get("css"):
            payload = {"selector": cand["css"], "file_path": value}
        else:
            coords = await self._coords_for(cand)
            if not coords:
                return {"ok": False, "error": "upload button not found in dom_snapshot"}
            payload = {"file_path": value, **coords}
        r = await self.bridge.execute_action(
            self.session_id, "upload_file", payload, timeout_seconds=30.0
        )
        return {"ok": bool(r.get("ok")), "error": r.get("error") or ""}

    async def _coords_for(self, cand: dict) -> Optional[Dict[str, int]]:
        elements = await self._dom_elements()
        return find_in_snapshot(elements, cand.get("text", ""), cand.get("role", ""))

    async def _dom_elements(self) -> List[dict]:
        r = await self.bridge.execute_action(
            self.session_id, "dom_snapshot", {}, timeout_seconds=20.0
        )
        return ((r.get("result") or {}).get("elements")) or []

    def _prepend_candidate(self, step: str, cand: dict) -> None:
        steps = self.map.setdefault("steps", {})
        steps[step] = [cand] + list(steps.get(step) or [])
        if self.persist_heal:
            try:
                save_selector_map(self.map_name, self.map)
            except Exception as e:  # never let a disk write kill a live post
                logger.warning("[driver_runner] could not persist healed map: %s", e)

    def _record_success(self, step: str, cand: dict) -> None:
        """Winning candidate → front of its step, stamped verified/observed.

        Converges the map to live-proven selectors through normal use and
        makes the proven candidate first (no more burning a full wait
        window on a dead draft ahead of it). Disk write only when
        something actually changed, and only when persist_heal is on.
        """
        steps = self.map.setdefault("steps", {})
        cands = list(steps.get(step) or [])
        idx = next((i for i, c in enumerate(cands) if c is cand), None)
        if idx is None:
            return  # candidate not from this map — nothing to converge
        changed = False
        if not cand.get("verified"):
            cand["verified"] = True
            changed = True
        if not cand.get("observed"):
            cand["observed"] = date.today().isoformat()
            changed = True
        if idx != 0:
            cands.pop(idx)
            cands.insert(0, cand)
            steps[step] = cands
            changed = True
        if changed and self.persist_heal:
            try:
                save_selector_map(self.map_name, self.map)
            except Exception as e:  # never let a disk write kill a live run
                logger.warning("[driver_runner] could not persist observed map: %s", e)


def _desc(cand: dict) -> str:
    if cand.get("css"):
        return f"css:{cand['css']}"
    if cand.get("text"):
        role = cand.get("role") or "*"
        return f"text:{role}:{cand['text']}"
    return "empty-candidate"
