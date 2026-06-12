# FILE: app/pipeline_v2/stages/verifier.py
# Purpose: Stage 5 — The Verifier.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.pipeline_v2.config, app.pipeline_v2.llm_caller, app.pipeline_v2.models, app.pipeline_v2.sandbox_tools
# Last-renovated: 2026-06-11
"""
Stage 5 — The Verifier.

A multimodal model that acts as the client. It NEVER reads code.
It NEVER touches code. It looks at the running product and asks:
  "Does this match what the user asked for?"

Uses Gemini 3.1 Pro for multimodal visual assessment.
Has tool access: screenshot capture, browser automation.

If the product passes, the build is complete.
If not, it produces a visual feedback report in plain language
that goes back to the Builder for another pass.

v1.0 (2026-03-06): Initial implementation for ASTRA v2.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from app.pipeline_v2.config import (
    VERIFIER_PROVIDER, VERIFIER_MODEL, VERIFIER_MAX_OUTPUT,
)
from app.pipeline_v2.models import VerifyResult, VisualIssue

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM = """You are ASTRA's Verifier. You are the CLIENT.

You never read code. You never touch code. You never suggest code fixes.

You receive:
1. The original user intent (what they asked for)
2. The specification (what should have been built)
3. Boot/build test results (does the app start without errors?)
4. Screenshots of the running application (when available)

Your ONE job: Does the built product match what the user asked for?

OUTPUT FORMAT (JSON):

{
  "passed": true/false,
  "confidence": 0.0-1.0,
  "summary": "One sentence overall assessment",
  "issues": [
    {
      "description": "What's wrong, in plain language",
      "severity": "low/medium/high",
      "spec_reference": "Which requirement it violates"
    }
  ],
  "feedback_for_builder": "Plain language description of what needs fixing. No code. No file paths. Just what's wrong from the user's perspective."
}

RULES:
- NEVER mention file names, function names, or code.
- NEVER suggest how to fix things technically.
- Describe problems the way a non-technical user would:
  "The login page is missing" not "LoginPage.tsx is not rendering"
  "The dashboard doesn't show weekly stats" not "the WeeklyStats component is unmounted"
- If boot/build tests fail, that's an automatic FAIL — the app doesn't even start.
- Be honest. Don't pass a build that doesn't meet the spec.
"""


async def run_verifier(
    intent_text: str,
    spec_text: str,
    files_written: List[str],
    on_progress: Optional[Callable[[str], None]] = None,
) -> VerifyResult:
    """Run the Verifier stage.

    Boots the app, captures evidence, compares against intent and spec.
    """
    emit = on_progress or (lambda msg: None)
    t_start = time.time()

    emit("✅ Verifier: Checking the built product...")

    from app.pipeline_v2.sandbox_tools import boot_check, build_check

    # --- Boot check ---
    emit("   🔄 Running boot check...")
    boot_ok, boot_output = await boot_check()
    emit(f"   {'✅' if boot_ok else '❌'} Boot: {'PASS' if boot_ok else 'FAIL'}")

    # --- Build check ---
    emit("   🔄 Running build check (TypeScript)...")
    build_ok, build_output = await build_check()
    emit(f"   {'✅' if build_ok else '❌'} Build: {'PASS' if build_ok else 'FAIL'}")

    # --- If basic checks fail, auto-fail ---
    if not boot_ok or not build_ok:
        issues = []
        if not boot_ok:
            issues.append(VisualIssue(
                description="The application fails to start.",
                severity="high",
                spec_reference="Basic functionality",
            ))
        if not build_ok:
            issues.append(VisualIssue(
                description="The application has compilation errors and cannot be built.",
                severity="high",
                spec_reference="Basic functionality",
            ))

        result = VerifyResult(
            passed=False,
            confidence=1.0,
            issues=issues,
            feedback_for_builder=(
                f"The application {'does not boot' if not boot_ok else 'has build errors'}. "
                f"It needs to start cleanly before visual verification can proceed. "
                f"Boot output: {boot_output[:200]}"
            ),
        )

        duration = time.time() - t_start
        emit(f"✅ Verifier: FAIL (basic checks failed, {duration:.1f}s)")
        return result

    # --- LLM visual assessment ---
    emit("   🤖 Running visual assessment...")

    # Build the prompt with intent, spec, and test results
    user_prompt = (
        f"# USER INTENT (what they asked for)\n\n{intent_text}\n\n"
        f"# SPECIFICATION (what should have been built)\n\n{spec_text}\n\n"
        f"# TEST RESULTS\n\n"
        f"Backend boot: PASS\n"
        f"Frontend build: PASS\n"
        f"Files written: {len(files_written)}\n\n"
        f"# FILES PRODUCED\n\n"
    )
    for fp in files_written:
        user_prompt += f"- {fp}\n"
    user_prompt += (
        f"\n# YOUR TASK\n\n"
        f"Based on the intent and spec, does this build appear complete? "
        f"Are the right files in place? Does the structure match what was requested? "
        f"Output your assessment as JSON."
    )

    # TODO: When screenshot capture is available, add captured images here.
    # For now, the Verifier works from file lists and test results.

    from app.pipeline_v2.llm_caller import call_llm
    try:
        raw = await call_llm(
            provider=VERIFIER_PROVIDER,
            model=VERIFIER_MODEL,
            system_prompt=VERIFIER_SYSTEM,
            user_prompt=user_prompt,
            max_tokens=VERIFIER_MAX_OUTPUT,
        )
    except RuntimeError as e:
        emit(f"   ❌ Verifier LLM failed: {e}")
        return VerifyResult(
            passed=False, confidence=0.0,
            feedback_for_builder=f"Verifier could not run: {e}",
        )

    # Parse the JSON response
    result = _parse_verifier_response(raw)

    duration = time.time() - t_start
    emit(f"✅ Verifier: {'PASS' if result.passed else 'FAIL'} "
         f"(confidence={result.confidence:.0%}, issues={len(result.issues)}, {duration:.1f}s)")

    if result.issues:
        for issue in result.issues:
            emit(f"   {'🔴' if issue.severity == 'high' else '🟡'} {issue.description}")

    return result


def _parse_verifier_response(raw: str) -> VerifyResult:
    """Parse the Verifier's JSON response."""
    import json

    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return VerifyResult(
                    passed=False, confidence=0.0,
                    feedback_for_builder=f"Verifier output was not parseable: {text[:200]}",
                )
        else:
            return VerifyResult(
                passed=False, confidence=0.0,
                feedback_for_builder=f"Verifier output was not parseable: {text[:200]}",
            )

    issues = []
    for issue_data in data.get("issues", []):
        issues.append(VisualIssue(
            description=issue_data.get("description", ""),
            severity=issue_data.get("severity", "medium"),
            spec_reference=issue_data.get("spec_reference", ""),
        ))

    return VerifyResult(
        passed=data.get("passed", False),
        confidence=data.get("confidence", 0.0),
        issues=issues,
        feedback_for_builder=data.get("feedback_for_builder", ""),
    )
