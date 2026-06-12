# FILE: app/pipeline_v2/verifier_agent/perception_tools.py
# Purpose: JOB 10 (2026-06-10) - Perception-action toolset for the agentic verifier.
# Called-by: app.pipeline_v2.verifier_agent.agent
# Depends-on: app.pipeline_v2.bvl, app.pipeline_v2.bvl.emulator_bridge, app.pipeline_v2.test_env
# Last-renovated: 2026-06-11
"""
JOB 10 (2026-06-10) - Perception-action toolset for the agentic verifier.

Exposes the emulator bridge (app/pipeline_v2/bvl/emulator_bridge.py) as
model-callable tools, so the verifier works turn-by-turn like a human
tester: LOOK (screenshot / ui tree) -> DECIDE -> ACT (tap / type / swipe)
-> LOOK again. This replaces blind pre-scripted flows as the primary
verification mechanism; tier-2 scripted flows remain as a cheap regression
layer.

adb_screenshot returns an IMAGE tool_result (the Anthropic adapter, Job 8,
converts it to an image block) - the model genuinely sees the screen.
Screenshots are also saved to an evidence directory when one is set, so the
final checkout report can cite them.

Single responsibility: tool schemas + executor mapping. No LLM calls, no
stage orchestration (that is Job 11's verifier module).
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Evidence directory - set by the verifier stage per job. When set, every
# screenshot is also written here as evidence_NNN.png for the report.
_evidence_dir: Optional[str] = None
_evidence_count = 0


def set_evidence_dir(path: Optional[str]) -> None:
    global _evidence_dir, _evidence_count
    _evidence_dir = path
    _evidence_count = 0
    if path:
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as exc:
            logger.warning("[perception] could not create evidence dir %s: %s", path, exc)


def _save_evidence(b64_png: str, label: str = "") -> str:
    """Save a screenshot to the evidence dir. Returns the file path or ''."""
    global _evidence_count
    if not _evidence_dir or not b64_png:
        return ""
    try:
        import base64
        _evidence_count += 1
        safe = "".join(c for c in (label or "shot") if c.isalnum() or c in "-_")[:40]
        fname = f"evidence_{_evidence_count:03d}_{safe}.png"
        fpath = os.path.join(_evidence_dir, fname)
        with open(fpath, "wb") as f:
            f.write(base64.b64decode(b64_png))
        return fpath
    except Exception as exc:
        logger.warning("[perception] evidence save failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function format; Job 8 converts for Anthropic)
# ---------------------------------------------------------------------------

PERCEPTION_TOOLS: List[Dict] = [
    {"type": "function", "function": {
        "name": "adb_screenshot",
        "description": (
            "Take a screenshot of the emulator screen and SEE it (returned as an "
            "image). Use after every action to observe the result. Optional label "
            "names the evidence file."
        ),
        "parameters": {"type": "object", "properties": {
            "label": {"type": "string", "description": "Short label for the evidence file, e.g. 'after_login'"},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "adb_ui_tree",
        "description": (
            "Dump the current UI hierarchy as compact text: one line per element "
            "with text, content-desc, class, bounds and clickability. Use to find "
            "exact tap targets and to read on-screen text reliably."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "adb_tap",
        "description": "Tap the screen at pixel coordinates (x, y).",
        "parameters": {"type": "object", "properties": {
            "x": {"type": "integer"}, "y": {"type": "integer"},
        }, "required": ["x", "y"]},
    }},
    {"type": "function", "function": {
        "name": "adb_tap_element",
        "description": (
            "Find an element by visible text or content-description (fuzzy match) "
            "and tap its centre. More reliable than raw coordinates."
        ),
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "Visible text to match"},
            "content_desc": {"type": "string", "description": "Content description to match"},
            "resource_id": {"type": "string", "description": "Resource id substring to match"},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "adb_type",
        "description": "Type text into the currently focused input field.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"},
        }, "required": ["text"]},
    }},
    {"type": "function", "function": {
        "name": "adb_swipe",
        "description": "Swipe from (x1,y1) to (x2,y2). Use for scrolling lists.",
        "parameters": {"type": "object", "properties": {
            "x1": {"type": "integer"}, "y1": {"type": "integer"},
            "x2": {"type": "integer"}, "y2": {"type": "integer"},
            "duration_ms": {"type": "integer", "description": "Default 300"},
        }, "required": ["x1", "y1", "x2", "y2"]},
    }},
    {"type": "function", "function": {
        "name": "adb_back",
        "description": "Press the Android back button.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "adb_enter",
        "description": "Press the enter/return key.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "launch_app",
        "description": "Launch (or relaunch) the app under test's main activity.",
        "parameters": {"type": "object", "properties": {
            "force_restart": {"type": "boolean", "description": "Force-stop first for a cold start"},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "logcat_tail",
        "description": "Capture recent error-level logcat for the app under test (crashes, exceptions).",
        "parameters": {"type": "object", "properties": {
            "duration_sec": {"type": "integer", "description": "Capture window, default 6"},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "wait",
        "description": "Wait N seconds (max 15) for the UI to settle or content to load.",
        "parameters": {"type": "object", "properties": {
            "seconds": {"type": "number"},
        }, "required": ["seconds"]},
    }},
]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

def _compact_ui_tree(xml: str, cap: int = 4000) -> str:
    """Render the UI dump as one line per meaningful node."""
    from app.pipeline_v2.bvl.emulator_bridge import parse_ui_nodes
    nodes = parse_ui_nodes(xml or "")
    lines: List[str] = []
    for n in nodes:
        bits: List[str] = []
        if n.text:
            bits.append(f'text="{n.text[:60]}"')
        if n.content_desc:
            bits.append(f'desc="{n.content_desc[:60]}"')
        if not bits:
            continue
        cls = n.class_name.rsplit(".", 1)[-1] if n.class_name else "?"
        bits.append(cls)
        bits.append(n.bounds)
        if n.clickable:
            bits.append("CLICKABLE")
        if n.focused:
            bits.append("FOCUSED")
        lines.append("  ".join(bits))
    out = "\n".join(lines)
    if not out.strip():
        return "(no text/desc-bearing elements on screen - app may be loading, crashed, or showing pure imagery)"
    if len(out) > cap:
        out = out[:cap] + f"\n[TRUNCATED - {len(lines)} elements total]"
    return out


async def execute_perception_tool(
    name: str,
    args: Dict[str, Any],
    profile: Any,
) -> Any:
    """Execute one perception/action tool. Returns str or {'text','image'}.

    Never raises - errors come back as strings the model can react to.
    """
    from app.pipeline_v2.bvl import emulator_bridge as eb

    try:
        if name == "adb_screenshot":
            b64 = await eb.screenshot_as_base64()
            if not b64:
                return "ERROR: screenshot failed (emulator running?)"
            saved = _save_evidence(b64, args.get("label", ""))
            note = f"Screenshot captured ({time.strftime('%H:%M:%S')})"
            if saved:
                note += f" - saved as {os.path.basename(saved)}"
            return {"text": note, "image": {"data": b64, "media_type": "image/png"}}

        if name == "adb_ui_tree":
            xml = await eb.dump_ui_tree()
            return _compact_ui_tree(xml)

        if name == "adb_tap":
            r = await eb.tap(int(args["x"]), int(args["y"]))
            return "OK: tapped" if r.ok else f"ERROR: tap failed: {r.stderr[:120]}"

        if name == "adb_tap_element":
            node = await eb.find_element(
                resource_id=args.get("resource_id", ""),
                text=args.get("text", ""),
                content_desc=args.get("content_desc", ""),
            )
            if node is None:
                return ("ERROR: element not found. Use adb_ui_tree to see what is "
                        "actually on screen, then match its exact text.")
            r = await eb.tap_element(node)
            label = node.text or node.content_desc or node.resource_id
            return f"OK: tapped '{label[:60]}'" if r.ok else f"ERROR: tap failed: {r.stderr[:120]}"

        if name == "adb_type":
            r = await eb.type_text(str(args.get("text", "")))
            return "OK: typed" if r.ok else f"ERROR: type failed: {r.stderr[:120]}"

        if name == "adb_swipe":
            r = await eb.swipe(
                int(args["x1"]), int(args["y1"]), int(args["x2"]), int(args["y2"]),
                int(args.get("duration_ms", 300)),
            )
            return "OK: swiped" if r.ok else f"ERROR: swipe failed: {r.stderr[:120]}"

        if name == "adb_back":
            r = await eb.press_back()
            return "OK: back" if r.ok else "ERROR: back failed"

        if name == "adb_enter":
            r = await eb.press_enter()
            return "OK: enter" if r.ok else "ERROR: enter failed"

        if name == "launch_app":
            package = str(getattr(profile, "package_name", "") or "")
            if not package:
                return "ERROR: profile has no package_name"
            activity = ".MainActivity"
            try:
                from app.pipeline_v2.test_env import get_test_env_for_profile
                env = get_test_env_for_profile(profile)
                if env and env.main_activity:
                    activity = env.main_activity
            except Exception:
                pass
            if args.get("force_restart"):
                await eb.force_stop(package)
                await asyncio.sleep(1)
            r = await eb.launch_app(package, activity)
            await asyncio.sleep(2.5)
            return "OK: launched" if r.ok else f"ERROR: launch failed: {r.stderr[:150]}"

        if name == "logcat_tail":
            package = str(getattr(profile, "package_name", "") or "")
            text = await eb.capture_logcat(package, int(args.get("duration_sec", 6)), level="E")
            return text[:3000] if text.strip() else "(no error-level log lines captured)"

        if name == "wait":
            secs = min(float(args.get("seconds", 1)), 15.0)
            await asyncio.sleep(secs)
            return f"OK: waited {secs:.1f}s"

        return f"ERROR: unknown perception tool: {name}"

    except Exception as exc:
        logger.warning("[perception] %s crashed: %s", name, exc)
        return f"ERROR: {name} crashed: {exc}"
