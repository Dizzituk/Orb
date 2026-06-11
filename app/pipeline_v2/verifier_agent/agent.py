# FILE: app/pipeline_v2/verifier_agent/agent.py
"""
JOB 11 (2026-06-10) - The Claude agentic verifier: final checkout that tests
like a human and fixes surgically.

Flow (list / check / fill - the same discipline as SpecGate's self-review):
  1. Load intent + spec + acceptance criteria + merged build evidence
  2. ENUMERATE every acceptance criterion
  3. CHECK each one by driving the real app (perception-action tools, Job 10)
     with screenshot evidence saved per criterion
  4. Mark VERIFIED / FAILED / BLOCKED
  5. For FAILED with an obvious code cause: surgical edit_file fix (Job 9),
     rebuild if needed, re-test - bounded by a hard edit budget
  6. Emit the evidence-backed checkout report for human sign-off

Budgets (env-configurable; this is the most expensive stage):
  ASTRA_VERIFIER_MAX_TOOL_CALLS  (default 60)
  ASTRA_VERIFIER_MAX_SECONDS     (default 900)
  ASTRA_VERIFIER_MAX_EDITS       (default 3)
  ASTRA_VERIFIER_MAX_TOKENS      (default 16384 per call)

Gate: the pipeline only runs this stage when ASTRA_AGENTIC_VERIFIER=1
(default OFF - it spends real model tokens).

Model: config.VERIFIER_AGENT_MODEL (Job 12), claude-fable-5 by default,
falling back to claude-opus-4-8 on first-call failure.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = int(os.getenv("ASTRA_VERIFIER_MAX_TOOL_CALLS", "60"))
MAX_SECONDS = int(os.getenv("ASTRA_VERIFIER_MAX_SECONDS", "900"))
MAX_EDITS = int(os.getenv("ASTRA_VERIFIER_MAX_EDITS", "3"))
MAX_TOKENS = int(os.getenv("ASTRA_VERIFIER_MAX_TOKENS", "16384"))

VERDICT_LINE = re.compile(r"^\s*[-*]?\s*\[(VERIFIED|FAILED|BLOCKED)\]\s*(.+)$", re.IGNORECASE)
OVERALL_LINE = re.compile(r"OVERALL\s*:\s*(VERIFIED|FAILED|BLOCKED)", re.IGNORECASE)


@dataclass
class CriterionResult:
    criterion: str
    status: str          # VERIFIED / FAILED / BLOCKED
    evidence: str = ""


@dataclass
class VerifierReport:
    ran: bool = False
    overall: str = "BLOCKED"
    criteria: List[CriterionResult] = field(default_factory=list)
    final_text: str = ""
    skipped_reason: str = ""
    model: str = ""
    tool_calls: int = 0
    edits_made: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_s: float = 0.0
    evidence_dir: str = ""

    def render_lines(self) -> List[str]:
        lines: List[str] = []
        lines.append("")
        lines.append("=" * 60)
        lines.append("AGENTIC VERIFIER CHECKOUT (Claude)")
        lines.append("=" * 60)
        if not self.ran:
            lines.append(f"   SKIPPED: {self.skipped_reason}")
            lines.append("=" * 60)
            return lines
        for c in self.criteria:
            lines.append(f"   [{c.status}] {c.criterion[:130]}" + (f" - {c.evidence[:110]}" if c.evidence else ""))
        lines.append(f"   OVERALL: {self.overall}")
        lines.append(
            f"   budget: {self.tool_calls} tool calls, {self.edits_made} edits, "
            f"{self.input_tokens}+{self.output_tokens} tokens, {self.duration_s:.0f}s ({self.model})"
        )
        if self.evidence_dir:
            lines.append(f"   evidence: {self.evidence_dir}")
        lines.append("=" * 60)
        return lines


def _extract_criteria(spec: Optional[Dict]) -> List[str]:
    """Same criterion sources as final_verdict (Job 3) for consistency."""
    try:
        from app.pipeline_v2.final_verdict import _extract_criteria as fv_extract
        return fv_extract(spec)
    except Exception:
        return []


def _system_prompt(profile: Any) -> str:
    """Mode dispatch: Android targets get ADB hands; self-builds get clone hands."""
    try:
        from app.pipeline_v2.clone_freshness import is_self_build
        if is_self_build(profile):
            return _system_prompt_clone(profile)
    except Exception:
        pass
    return _system_prompt_android(profile)


def _system_prompt_clone(profile: Any) -> str:
    root = str(getattr(profile, "project_root", "") or "D:/Orb")
    return f"""You are ASTRA's final-checkout verifier for a SELF-BUILD. The code under
test is a CLONE OF YOURSELF running inside an isolated sandbox VM. The live
system is OUT OF BOUNDS and unreachable from here - everything you touch
(files via read_file/edit_file, shells via run_shell, the backend via
backend_* and http_request) is the clone at {root} inside the VM.

CLONE AWARENESS:
- The clone holds NO live secrets: its master key is a throwaway, so the
  encrypted store decrypts to placeholders and key-dependent features
  (real LLM calls, external APIs) will degrade. That is BY DESIGN.
- If a criterion cannot be exercised because a capability is absent in the
  clone (missing API key, no live data, no TTS hardware), mark it BLOCKED
  and name the missing capability precisely - "works in theory, dependency
  absent in clone". Do NOT mark it FAILED, and never mark it VERIFIED
  without seeing it work.

DISCIPLINE - list / check / fill:
1. LIST every acceptance criterion, numbered.
2. CHECK each one against the RUNNING clone: backend_start once, then
   http_request per criterion, backend_logs when responses look wrong.
   Evidence = actual status codes, response bodies and log excerpts.
3. FILL: mark each [VERIFIED], [FAILED] or [BLOCKED].

FIXING: for a FAILED criterion with an obvious small code fault, read_file
the source, apply a SURGICAL edit_file patch (exact unique match), check
syntax with run_shell (python -m py_compile <file>), backend_stop then
backend_start to reload, and re-test that criterion. Strict edit budget;
touch nothing unrelated.

HARD RULES:
- Evidence over inference: VERIFIED only for behaviour you SAW this session.
- One thing at a time; re-check after every action.
- Stay inside {root}; the sandbox agent on port 8765 is off limits.

FINAL ANSWER - end with exactly this block (no tools after it):
CHECKOUT VERDICT
- [VERIFIED|FAILED|BLOCKED] <criterion 1 text> - <one-line evidence>
- [VERIFIED|FAILED|BLOCKED] <criterion 2 text> - <one-line evidence>
OVERALL: <VERIFIED if all verified, FAILED if any failed, else BLOCKED>"""


def _system_prompt_android(profile: Any) -> str:
    pkg = str(getattr(profile, "package_name", "") or "(unknown package)")
    root = str(getattr(profile, "project_root", "") or "(unknown root)")
    return f"""You are ASTRA's final-checkout verifier. You test a freshly built Android
feature ON A REAL EMULATOR the way a meticulous human QA engineer would, then
fix small code faults surgically, then re-test.

App under test: {pkg}   Project root: {root}

DISCIPLINE - list / check / fill:
1. LIST: enumerate every acceptance criterion you were given, numbered.
2. CHECK each criterion one at a time by DRIVING THE APP: adb_screenshot to
   look, adb_ui_tree for exact tap targets, adb_tap_element / adb_type to
   act, then adb_screenshot again to confirm. Take a labelled screenshot as
   evidence for every criterion (label it after the criterion number).
3. FILL: mark each criterion [VERIFIED], [FAILED] or [BLOCKED] (blocked =
   environment prevented testing; say what was missing).

FIXING: if a criterion FAILED and the cause is an obvious small code fault,
you may read_file the relevant source, apply a SURGICAL edit_file patch
(exact unique match - never rewrite whole files), run_shell the project's
compile command, relaunch, and re-test that criterion. You have a strict
edit budget; do not refactor, do not fix style, do not touch anything
unrelated to a failed criterion.

HARD RULES:
- Evidence over inference: a criterion is VERIFIED only if you SAW it work
  on screen this session. Never mark VERIFIED from reading code alone.
- One thing at a time; re-look after every action.
- If the app is on a login screen or the backend is unreachable, that is
  BLOCKED (environment), not FAILED - report it precisely and move on.
- Stay inside the project root for all file operations.

FINAL ANSWER - end with exactly this block (no tools after it):
CHECKOUT VERDICT
- [VERIFIED|FAILED|BLOCKED] <criterion 1 text> - <one-line evidence>
- [VERIFIED|FAILED|BLOCKED] <criterion 2 text> - <one-line evidence>
OVERALL: <VERIFIED if all verified, FAILED if any failed, else BLOCKED>"""


def _user_prompt(intent_text: str, spec: Dict, criteria: List[str], evidence_summary: str) -> str:
    crit_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(criteria)) or "(no structured criteria - derive 3-6 testable criteria from the spec, list them first, then test those)"
    spec_text = json.dumps(spec, indent=2, ensure_ascii=False)[:18000] if isinstance(spec, dict) else str(spec)[:18000]
    return f"""USER INTENT:
{(intent_text or '(none)')[:4000]}

ACCEPTANCE CRITERIA TO VERIFY:
{crit_block}

SPEC (agreed scope):
{spec_text}

PRIOR BUILD EVIDENCE (merged verdict from BVL / contract verifier / spec review):
{evidence_summary[:4000] or '(none)'}

Begin: take a screenshot to see the current app state, then work the criteria
list in order."""


async def run_verifier_agent(
    spec: Dict,
    profile: Any,
    intent_text: str = "",
    evidence_summary: str = "",
    job_id: Optional[str] = None,
    emit: Optional[Callable[[str], None]] = None,
) -> VerifierReport:
    """Run the agentic verifier stage. Never raises."""
    emit = emit or (lambda m: None)
    report = VerifierReport()
    t0 = time.time()
    try:
        return await _run(spec, profile, intent_text, evidence_summary, job_id, emit, report, t0)
    except Exception as exc:
        logger.exception("[verifier_agent] crashed: %s", exc)
        report.ran = bool(report.tool_calls)
        report.skipped_reason = report.skipped_reason or f"verifier crashed: {exc}"
        report.duration_s = time.time() - t0
        return report


async def _run(spec, profile, intent_text, evidence_summary, job_id, emit, report, t0) -> VerifierReport:
    from app.pipeline_v2.config import VERIFIER_AGENT_ENABLED, VERIFIER_AGENT_MODEL, VERIFIER_FALLBACK_MODEL

    if not VERIFIER_AGENT_ENABLED:
        report.skipped_reason = "ASTRA_AGENTIC_VERIFIER is not enabled (set =1 to run)"
        return report
    if not os.environ.get("ANTHROPIC_API_KEY"):
        report.skipped_reason = "ANTHROPIC_API_KEY not set"
        return report

    criteria = _extract_criteria(spec)

    # Evidence directory under the job folder
    evidence_dir = ""
    if job_id:
        evidence_dir = os.path.join("D:\\Orb", "jobs", str(job_id), "verifier_evidence")
    from app.pipeline_v2.verifier_agent.perception_tools import (
        PERCEPTION_TOOLS, execute_perception_tool, set_evidence_dir,
    )
    set_evidence_dir(evidence_dir or None)
    report.evidence_dir = evidence_dir

    # Toolset: action hands depend on the target. Android -> ADB perception;
    # SELF-BUILD -> clone backend driving (PROVING RUN 2026-06-10). Plus
    # read/edit/shell in both modes (NO write_file - surgical only).
    from app.pipeline_v2.clone_freshness import is_self_build
    if is_self_build(profile):
        from app.pipeline_v2.verifier_agent.backend_tools import BACKEND_TOOLS, execute_backend_tool
        action_tools = BACKEND_TOOLS

        async def _action_exec(name, args):
            return await execute_backend_tool(name, args, profile)
    else:
        action_tools = PERCEPTION_TOOLS

        async def _action_exec(name, args):
            return await execute_perception_tool(name, args, profile)

    from app.pipeline_v2.llm_tool_defs import TOOLS as FILE_TOOLS
    file_tools = [t for t in FILE_TOOLS if t["function"]["name"] in ("read_file", "edit_file", "run_shell")]
    tools = action_tools + file_tools
    action_names = {t["function"]["name"] for t in action_tools}

    # Route file tools to the right repo
    from app.pipeline_v2.llm_tools import _execute_tool as file_executor, set_tool_profile
    set_tool_profile(profile)

    counters = {"tool_calls": 0, "edits": 0}

    async def executor(name: str, args: Dict) -> Any:
        counters["tool_calls"] += 1
        if counters["tool_calls"] > MAX_TOOL_CALLS:
            return ("BUDGET EXCEEDED: tool-call budget spent. Stop testing and "
                    "produce your CHECKOUT VERDICT block now from the evidence you have.")
        if name == "edit_file":
            if counters["edits"] >= MAX_EDITS:
                return ("EDIT BUDGET EXCEEDED: no more fixes allowed. Mark the "
                        "criterion FAILED with what you found and finish the report.")
            counters["edits"] += 1
        if name in action_names:
            return await _action_exec(name, args)
        return await file_executor(name, args)

    final_texts: List[str] = []

    def on_text(text: str) -> None:
        final_texts.append(text)

    def on_tool_call(name: str, args: Dict) -> None:
        brief = ""
        if name == "adb_tap_element":
            brief = args.get("text") or args.get("content_desc") or ""
        elif name == "adb_type":
            brief = (args.get("text", "")[:30] + "...") if len(args.get("text", "")) > 30 else args.get("text", "")
        elif name in ("read_file", "edit_file"):
            brief = args.get("path", "")
        emit(f"   [verifier] {name}" + (f": {brief}" if brief else ""))

    from app.pipeline_v2.llm_tools_anthropic import run_anthropic_tool_loop

    sys_prompt = _system_prompt(profile)
    user_prompt = _user_prompt(intent_text, spec, criteria, evidence_summary)

    emit(f"   [verifier] starting on {VERIFIER_AGENT_MODEL} (budget: {MAX_TOOL_CALLS} calls / {MAX_EDITS} edits / {MAX_SECONDS}s)")

    async def _loop(model: str):
        return await run_anthropic_tool_loop(
            system_prompt=sys_prompt,
            initial_user_message=user_prompt,
            model=model,
            max_iterations=MAX_TOOL_CALLS + 10,
            max_tokens=MAX_TOKENS,
            on_tool_call=on_tool_call,
            on_text=on_text,
            tools=tools,
            tool_executor=executor,
            thinking=True,
        )

    model_used = VERIFIER_AGENT_MODEL
    try:
        messages, in_tok, out_tok = await asyncio.wait_for(_loop(model_used), timeout=MAX_SECONDS)
    except asyncio.TimeoutError:
        report.ran = True
        report.skipped_reason = f"wall-clock budget exceeded ({MAX_SECONDS}s)"
        messages, in_tok, out_tok = [], 0, 0
    if not messages and not final_texts and report.skipped_reason == "":
        # Primary model never answered - try the fallback once
        emit(f"   [verifier] {model_used} failed - falling back to {VERIFIER_FALLBACK_MODEL}")
        model_used = VERIFIER_FALLBACK_MODEL
        try:
            messages, in_tok, out_tok = await asyncio.wait_for(_loop(model_used), timeout=MAX_SECONDS)
        except asyncio.TimeoutError:
            report.skipped_reason = f"wall-clock budget exceeded on fallback ({MAX_SECONDS}s)"
            messages, in_tok, out_tok = [], 0, 0

    report.ran = True
    report.model = model_used
    report.tool_calls = counters["tool_calls"]
    report.edits_made = counters["edits"]
    report.input_tokens = in_tok
    report.output_tokens = out_tok
    report.final_text = "\n\n".join(final_texts)[-12000:]
    report.duration_s = time.time() - t0

    _parse_verdict(report, criteria)
    for line in report.render_lines():
        emit(line)
    _persist(report, job_id)
    return report


def _parse_verdict(report: VerifierReport, criteria: List[str]) -> None:
    """Parse the CHECKOUT VERDICT block from the model's final text."""
    text = report.final_text or ""
    block = text
    if "CHECKOUT VERDICT" in text:
        block = text.rsplit("CHECKOUT VERDICT", 1)[1]
    rows: List[CriterionResult] = []
    for raw_line in block.splitlines():
        m = VERDICT_LINE.match(raw_line)
        if not m:
            continue
        status = m.group(1).upper()
        rest = m.group(2).strip()
        crit, _, ev = rest.partition(" - ")
        rows.append(CriterionResult(criterion=crit.strip()[:300], status=status, evidence=ev.strip()[:300]))
    report.criteria = rows

    m_overall = OVERALL_LINE.search(block)
    if m_overall:
        report.overall = m_overall.group(1).upper()
    elif rows:
        statuses = {r.status for r in rows}
        report.overall = "FAILED" if "FAILED" in statuses else ("BLOCKED" if "BLOCKED" in statuses else "VERIFIED")
    else:
        report.overall = "BLOCKED"
        if not report.skipped_reason:
            report.skipped_reason = "model produced no parseable CHECKOUT VERDICT block"


def _persist(report: VerifierReport, job_id: Optional[str]) -> None:
    """Save the report JSON next to the evidence (best-effort)."""
    if not job_id:
        return
    try:
        out_dir = os.path.join("D:\\Orb", "jobs", str(job_id))
        os.makedirs(out_dir, exist_ok=True)
        payload = {
            "overall": report.overall,
            "criteria": [c.__dict__ for c in report.criteria],
            "model": report.model,
            "tool_calls": report.tool_calls,
            "edits_made": report.edits_made,
            "input_tokens": report.input_tokens,
            "output_tokens": report.output_tokens,
            "duration_s": report.duration_s,
            "evidence_dir": report.evidence_dir,
            "skipped_reason": report.skipped_reason,
            "final_text": report.final_text[-8000:],
        }
        with open(os.path.join(out_dir, "verifier_report.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.warning("[verifier_agent] report persist failed: %s", exc)
