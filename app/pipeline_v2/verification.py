# FILE: app/pipeline_v2/verification.py
"""
ASTRA v2.1 Visual Verification — cheap model with eyes.

Takes a screenshot of the running app + the spec, sends to a
lightweight vision model (Gemini Flash / Haiku), and gets back
PASS or plain-language feedback.

The Verification Model does NOT need frontier reasoning.
It just needs to look at a screenshot and answer:
"Does this match the spec? If not, what's wrong?"

Cost per verification: ~$0.01

v1.0 (2026-03-07): Initial implementation for ASTRA v2.1.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.pipeline_v2.config import VERIFIER_PROVIDER, VERIFIER_MODEL, VERIFIER_MAX_OUTPUT
from app.pipeline_v2.models import VerifyResult
from app.pipeline_v2.screenshot import capture_screenshot

logger = logging.getLogger(__name__)

VERIFY_SYSTEM = """You are a visual QA inspector for a desktop application.

You will receive:
1. A screenshot of the running application
2. The specification describing what the application should look like and do

Your job is simple: compare the screenshot against the spec.

Respond with EXACTLY this format:

RESULT: PASS
(if the screenshot matches the spec)

OR:

RESULT: FAIL
ISSUES:
- [describe what's wrong in plain language]
- [describe another issue]
- [etc.]

Rules:
- Be specific: "the Debug tab shows a blank screen" not "something looks wrong"
- Focus on visible UI issues: missing components, wrong layout, error messages, blank screens
- Ignore minor styling differences (colours, fonts, spacing) unless the spec is explicit
- If you see an error message or stack trace on screen, always report FAIL
- If the app clearly hasn't loaded (blank/white screen), report FAIL
"""


async def verify_visually(
    spec_text: str,
    attempt: int = 1,
    emit=None,
) -> VerifyResult:
    """Take a screenshot and verify against the spec.

    Args:
        spec_text: The spec (or relevant excerpt) to verify against.
        attempt: Which verification attempt this is.
        emit: Progress callback.

    Returns:
        VerifyResult with pass/fail and feedback.
    """
    emit = emit or (lambda msg: None)

    # 1. Capture screenshot
    emit("   📸 Capturing screenshot...")
    b64_png, screenshot_path = await capture_screenshot()

    if b64_png is None:
        emit(f"   ⚠️ Screenshot failed: {screenshot_path}")
        return VerifyResult(
            passed=False,
            feedback=f"Could not capture screenshot: {screenshot_path}",
            attempt=attempt,
        )

    emit(f"   📸 Screenshot captured ({len(b64_png):,} bytes)")

    # 2. Send to verification model
    emit(f"   🔍 Sending to {VERIFIER_PROVIDER}/{VERIFIER_MODEL}...")

    user_prompt = (
        f"SPECIFICATION:\n{spec_text[:8000]}\n\n"
        f"Compare the attached screenshot against this specification. "
        f"Does the app match? Report PASS or FAIL with specific issues."
    )

    try:
        response = await _call_vision_model(
            system_prompt=VERIFY_SYSTEM,
            user_prompt=user_prompt,
            image_base64=b64_png,
        )
    except RuntimeError as e:
        emit(f"   ❌ Verification model failed: {e}")
        return VerifyResult(
            passed=False,
            feedback=f"Verification model error: {e}",
            attempt=attempt,
        )

    # 3. Parse result
    passed = "RESULT: PASS" in response.upper()
    feedback = ""

    if not passed:
        # Extract issues section
        if "ISSUES:" in response:
            feedback = response.split("ISSUES:", 1)[1].strip()
        else:
            feedback = response.strip()

    status = "✅ PASS" if passed else "❌ FAIL"
    emit(f"   {status}")
    if feedback:
        emit(f"   Feedback: {feedback[:200]}")

    return VerifyResult(
        passed=passed,
        feedback=feedback,
        screenshot_path=screenshot_path,
        attempt=attempt,
    )


async def _call_vision_model(
    system_prompt: str,
    user_prompt: str,
    image_base64: str,
) -> str:
    """Call the vision model with an image.

    Currently uses the same call_llm infrastructure but with
    the image embedded in the user prompt for vision-capable models.

    TODO: When the LLM caller supports native image inputs,
    switch to passing the image as a separate content block.
    """
    from app.pipeline_v2.llm_caller import call_llm

    # For now, append image reference to prompt
    # The actual image sending depends on provider support
    # Google Gemini and GPT-5.4 both support base64 images natively
    prompt_with_image = (
        f"{user_prompt}\n\n"
        f"[SCREENSHOT: base64 PNG image attached, {len(image_base64)} bytes]"
    )

    return await call_llm(
        provider=VERIFIER_PROVIDER,
        model=VERIFIER_MODEL,
        system_prompt=system_prompt,
        user_prompt=prompt_with_image,
        max_tokens=VERIFIER_MAX_OUTPUT,
    )
