# FILE: app/llm/weaver_job_class.py
# Purpose: Strict parser for the Weaver "Job class" output field.
# Called-by: app.llm._spec_gate_stream_utils_2, app.pot_spec.grounded._greenfield_gate, tests
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-07-04
"""
Strict parser for the Weaver v3.0 "Job class" output field (JOB 3 of the
greenfield build-target routing fixpack, 2026-07-04).

The Weaver system prompts instruct the LLM to emit one line:

    **Job class**: greenfield_new_app | modify_existing | unknown

greenfield_new_app  -> brand-new standalone app/project (own new folder)
modify_existing     -> changes to existing code (ASTRA repos or a registered target)
unknown             -> Weaver could not tell (also the default when the line
                       is missing or malformed — older Weaver outputs)

Parsing is deliberately STRICT: only those three tokens are accepted;
anything else degrades to "unknown" so downstream falls back to
deterministic domain detection. Never raises.
"""
from __future__ import annotations

import re

GREENFIELD_NEW_APP = "greenfield_new_app"
MODIFY_EXISTING = "modify_existing"
UNKNOWN = "unknown"

VALID_JOB_CLASSES = (GREENFIELD_NEW_APP, MODIFY_EXISTING, UNKNOWN)

# Accepts the markdown variants the Weaver can plausibly emit:
#   **Job class**: greenfield_new_app
#   - **Job class**: modify_existing
#   Job class: unknown
#   JOB_CLASS: greenfield_new_app
# Anchored to line start; the value must be one of the three exact tokens.
_JOB_CLASS_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?job[ _]?class(?:\*\*)?\s*[:=]\s*(?:\*\*)?\s*"
    r"(greenfield_new_app|modify_existing|unknown)\b",
    re.IGNORECASE | re.MULTILINE,
)


def parse_weaver_job_class(text: str | None) -> str:
    """Extract the job class from Weaver output text.

    Returns one of VALID_JOB_CLASSES; "unknown" when the field is absent,
    malformed, or carries any value outside the allowed set.
    """
    if not text or not isinstance(text, str):
        return UNKNOWN
    match = _JOB_CLASS_RE.search(text)
    if not match:
        return UNKNOWN
    return match.group(1).lower()
