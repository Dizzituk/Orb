# FILE: app/pipeline_v2/bvl/tier2_test_generator.py
"""
Tier 2 Test Generator — LLM-powered behavioral test script generation.

Takes a spec's user stories and acceptance criteria, sends them to the
Builder model (GPT-5.4), and gets back structured TestFlow objects that
the Tier 2 executor can run via ADB/UI Automator.

The generator also produces a FlowCoverage matrix mapping every user
story to its test sequence.

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-BVL-001.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from app.pipeline_v2.bvl.bvl_models import (
    FlowCoverage,
    TestAction,
    TestAssertion,
    TestFlow,
)
from app.pipeline_v2.config import BUILDER_PROVIDER, BUILDER_MODEL

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile

logger = logging.getLogger(__name__)

GENERATOR_SYSTEM = """You are a test engineer generating behavioral test scripts for an Android app.

You receive a specification describing the app's features, screens, and user stories.
For each screen/component, you produce a structured test flow.

OUTPUT FORMAT — respond with ONLY a JSON array of test flows:

[
  {
    "flow_id": "unique_snake_case_id",
    "description": "Human-readable description of what this tests",
    "user_story_ref": "Which requirement/user story this covers",
    "setup_actions": [
      {"action_type": "tap|type|swipe|scroll|wait|back", "target": "element description", "value": "optional"}
    ],
    "actions": [
      {"action_type": "tap", "target": "Login button", "value": ""},
      {"action_type": "type", "target": "Email field", "value": "test@example.com"},
      {"action_type": "wait", "target": "", "value": "2000"}
    ],
    "assertions": [
      {"assertion_type": "element_visible|text_matches|screen_changed|data_persisted", "target": "element", "expected": "value"}
    ],
    "teardown_actions": [
      {"action_type": "back", "target": "", "value": ""}
    ]
  }
]

RULES:
- One test flow per user story or acceptance criterion
- Actions use ADB input commands (tap, type, swipe, scroll, wait, back)
- Targets should reference UI elements by their text label, content description, or resource ID
- Each flow should be self-contained: setup → actions → assertions → teardown
- Keep flows focused on ONE behavior each
- Include navigation to the target screen in setup_actions
- Include returning to a known state in teardown_actions
- Wait values are in milliseconds
- For tap targets, describe the element: "the Save button", "Settings icon", "first item in list"
- Respond with ONLY the JSON array, no markdown fences, no explanation
"""


async def generate_test_flows(
    spec: Dict[str, Any],
    profile: "BuildTargetProfile",
    emit: Optional[Callable[[str], None]] = None,
) -> List[TestFlow]:
    """Generate behavioral test flows from a spec.

    Sends the spec to the LLM and parses structured TestFlow objects.

    Args:
        spec: The verified spec from SpecGate.
        profile: Build target profile.
        emit: Progress callback.

    Returns:
        List of TestFlow objects ready for execution.
    """
    emit = emit or (lambda msg: None)
    emit("   🧪 T2: Generating test flows from spec...")

    spec_text = json.dumps(spec, indent=2) if isinstance(spec, dict) else str(spec)

    user_prompt = (
        f"Generate behavioral test flows for this Android app.\n\n"
        f"Package: {profile.package_name}\n"
        f"Framework: {profile.framework}\n"
        f"Architecture: {profile.architecture_pattern}\n\n"
        f"SPECIFICATION:\n{spec_text[:12000]}\n\n"
        f"Generate test flows covering every user story and acceptance criterion."
    )

    from app.pipeline_v2.llm_caller import call_llm

    try:
        raw = await call_llm(
            provider=BUILDER_PROVIDER,
            model=BUILDER_MODEL,
            system_prompt=GENERATOR_SYSTEM,
            user_prompt=user_prompt,
            max_tokens=8000,
            temperature=0.0,
        )
    except RuntimeError as e:
        emit(f"   ❌ Test generation failed: {e}")
        logger.error("[tier2_gen] LLM call failed: %s", e)
        return []

    flows = _parse_flows(raw)
    emit(f"   🧪 Generated {len(flows)} test flows")

    for flow in flows:
        emit(f"      • {flow.flow_id}: {flow.description[:60]}")

    return flows


def build_coverage_matrix(
    flows: List[TestFlow],
) -> List[FlowCoverage]:
    """Build a coverage matrix mapping user stories to test flows.

    Args:
        flows: Generated test flows.

    Returns:
        List of FlowCoverage entries.
    """
    matrix = []
    for flow in flows:
        matrix.append(FlowCoverage(
            user_story=flow.user_story_ref or flow.description,
            flow_id=flow.flow_id,
            tested=False,
            passed=False,
        ))
    return matrix


def update_coverage(
    matrix: List[FlowCoverage],
    flow_id: str,
    passed: bool,
    failure_reason: str = "",
) -> None:
    """Update the coverage matrix after executing a flow."""
    for entry in matrix:
        if entry.flow_id == flow_id:
            entry.tested = True
            entry.passed = passed
            entry.failure_reason = failure_reason
            break


# ═══════════════════════════════════════════════════════════════════
# Parsing
# ═══════════════════════════════════════════════════════════════════

def _parse_flows(raw: str) -> List[TestFlow]:
    """Parse LLM response into TestFlow objects.

    v1.1: More robust parsing. GPT-5.4 often wraps JSON in markdown
    fences, adds explanatory text, or nests arrays in wrapper objects.
    Handles all common patterns gracefully.
    """
    text = raw.strip()

    # Strip ALL markdown fence blocks — may be multiple
    text = re.sub(r'```(?:json|JSON)?\s*\n?', '', text).strip()

    # Try direct JSON parse
    data = _try_json_parse(text)

    # If it parsed as a dict with a list inside, extract the list
    if isinstance(data, dict):
        for key in ("flows", "test_flows", "tests", "testFlows"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if data is None:
        # Try bracket-matching instead of greedy regex
        # Find the first [ and match its closing ]
        start = text.find('[')
        if start >= 0:
            depth = 0
            end = start
            for i in range(start, len(text)):
                if text[i] == '[':
                    depth += 1
                elif text[i] == ']':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            candidate = text[start:end]
            data = _try_json_parse(candidate)

    # If still a dict, try extracting list
    if isinstance(data, dict):
        for val in data.values():
            if isinstance(val, list) and len(val) > 0:
                data = val
                break

    if not isinstance(data, list):
        logger.warning(
            "[tier2_gen] Could not parse flows from LLM output (%d chars, first 200: %s)",
            len(raw), raw[:200],
        )
        return []

    flows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            flow = TestFlow(
                flow_id=item.get("flow_id", f"flow_{len(flows)}"),
                description=item.get("description", ""),
                user_story_ref=item.get("user_story_ref", ""),
                setup_actions=_parse_actions(item.get("setup_actions", [])),
                actions=_parse_actions(item.get("actions", [])),
                assertions=_parse_assertions(item.get("assertions", [])),
                teardown_actions=_parse_actions(item.get("teardown_actions", [])),
            )
            flows.append(flow)
        except Exception as e:
            logger.warning("[tier2_gen] Skipping malformed flow: %s", e)

    return flows


def _parse_actions(raw_actions: list) -> List[TestAction]:
    """Parse action dicts into TestAction objects."""
    actions = []
    for item in raw_actions:
        if not isinstance(item, dict):
            continue
        actions.append(TestAction(
            action_type=item.get("action_type", "wait"),
            target=item.get("target", ""),
            value=item.get("value", ""),
            timeout_ms=int(item.get("timeout_ms", 5000)),
        ))
    return actions


def _parse_assertions(raw_assertions: list) -> List[TestAssertion]:
    """Parse assertion dicts into TestAssertion objects."""
    assertions = []
    for item in raw_assertions:
        if not isinstance(item, dict):
            continue
        assertions.append(TestAssertion(
            assertion_type=item.get("assertion_type", "element_visible"),
            target=item.get("target", ""),
            expected=item.get("expected", ""),
        ))
    return assertions


def _try_json_parse(text: str) -> Any:
    """Attempt JSON parse, return None on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
