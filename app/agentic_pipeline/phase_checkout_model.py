# FILE: app/agentic_pipeline/phase_checkout_model.py
"""
Phase Checkout — Big Model Verification (Stage 3).

One big model reads everything and confirms the implementation is correct.
If PASS: job complete. If FAIL: targeted fix list routed to either
deterministic fix or agentic loop regeneration.

v1.0 (2026-03-05): Initial implementation.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class FixType(str, Enum):
    DETERMINISTIC = "deterministic"
    AGENTIC_LOOP = "agentic_loop"
    MANUAL = "manual"


@dataclass
class FixItem:
    file_path: str
    description: str
    fix_type: FixType
    suggested_code: Optional[str] = None
    affected_segments: List[str] = field(default_factory=list)
    severity: str = "error"


@dataclass
class CheckoutVerdict:
    passed: bool = False
    confidence: float = 0.0
    summary: str = ""
    fix_items: List[FixItem] = field(default_factory=list)
    raw_model_output: str = ""
    boot_passed: bool = False
    build_passed: bool = False

    @property
    def has_deterministic_fixes(self) -> bool:
        return any(f.fix_type == FixType.DETERMINISTIC for f in self.fix_items)

    @property
    def has_structural_fixes(self) -> bool:
        return any(f.fix_type == FixType.AGENTIC_LOOP for f in self.fix_items)

    @property
    def deterministic_fixes(self) -> List[FixItem]:
        return [f for f in self.fix_items if f.fix_type == FixType.DETERMINISTIC]

    @property
    def structural_fixes(self) -> List[FixItem]:
        return [f for f in self.fix_items if f.fix_type == FixType.AGENTIC_LOOP]


_MAX_FILE_CHARS = 8_000


def _build_checkout_context(
    spec_summary: str, written_files: Dict[str, str],
    boot_result: str, build_result: str,
    import_graph: Optional[Dict[str, List[str]]] = None,
) -> str:
    parts = []
    parts.append("# ORIGINAL INTENT (POT Spec)\n" + spec_summary + "\n---\n")

    parts.append(f"# ALL WRITTEN FILES ({len(written_files)} files)\n")
    for path, content in sorted(written_files.items()):
        display = content[:_MAX_FILE_CHARS] + f"\n... [truncated]" if len(content) > _MAX_FILE_CHARS else content
        parts.append(f"### `{path}`\n```\n{display}\n```\n")
    parts.append("\n---\n")

    parts.append("# BOOT CHECK RESULT\n" + (boot_result or "(not run)") + "\n---\n")
    parts.append("# FRONTEND BUILD RESULT\n" + (build_result or "(not run)") + "\n---\n")

    if import_graph:
        parts.append("# IMPORT GRAPH (written files only)\n")
        for path, imports in sorted(import_graph.items()):
            parts.append(f"- `{path}` imports: {imports}")
        parts.append("")

    return "\n".join(parts)


_CHECKOUT_SYSTEM_PROMPT = """You are ASTRA's Phase Checkout Verifier. Read everything built, confirm correct or prescribe fixes.

You receive: spec intent, all written files, boot results, build results, import graph.

OUTPUT FORMAT (JSON):
{
  "passed": true/false,
  "confidence": 0.0-1.0,
  "summary": "one paragraph assessment",
  "fixes": [
    {
      "file_path": "path/to/file.ext",
      "description": "specific problem",
      "fix_type": "deterministic" or "agentic_loop" or "manual",
      "suggested_code": "optional replacement",
      "affected_segments": ["seg-01-..."],
      "severity": "error" or "warning"
    }
  ]
}

fix_type rules:
- "deterministic": Simple fix (import rename, typo). Include suggested_code.
- "agentic_loop": Structural problem. List affected_segments.
- "manual": Cannot be auto-fixed.

Be SPECIFIC with file + line references.
"""


async def run_phase_checkout(
    spec_summary: str, written_files: Dict[str, str],
    boot_result: str, build_result: str,
    llm_call_fn: Callable,
    provider_id: str = "anthropic",
    model_id: str = "claude-sonnet-4-20250514",
    import_graph: Optional[Dict[str, List[str]]] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> CheckoutVerdict:
    """Run the big-model phase checkout verification."""
    def _progress(msg):
        if on_progress:
            on_progress(msg)
        logger.info("[phase_checkout_model] %s", msg)

    _progress(f"Running phase checkout: {len(written_files)} files")

    context = _build_checkout_context(
        spec_summary, written_files, boot_result, build_result, import_graph,
    )

    try:
        response = await llm_call_fn(
            provider_id=provider_id, model_id=model_id,
            messages=[
                {"role": "system", "content": _CHECKOUT_SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            max_tokens=8_000, timeout_seconds=120,
        )
    except Exception as e:
        _progress(f"Checkout model call failed: {e}")
        return CheckoutVerdict(
            summary=f"Checkout model failed: {e}",
            boot_passed="PASS" in (boot_result or ""),
            build_passed="PASS" in (build_result or ""),
        )

    from app.agentic_pipeline.loop_controller import _extract_content
    raw = _extract_content(response)
    verdict = _parse_verdict(raw)
    verdict.raw_model_output = raw
    verdict.boot_passed = "PASS" in (boot_result or "")
    verdict.build_passed = "PASS" in (build_result or "")

    _progress(f"Verdict: {'PASS' if verdict.passed else 'FAIL'} (confidence={verdict.confidence:.2f}, fixes={len(verdict.fix_items)})")
    return verdict


def _parse_verdict(raw_output: str) -> CheckoutVerdict:
    verdict = CheckoutVerdict()
    json_str = _extract_json_block(raw_output)
    if not json_str:
        verdict.summary = "Could not parse checkout output as JSON"
        return verdict
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        verdict.summary = f"JSON parse error: {e}"
        return verdict

    verdict.passed = data.get("passed", False)
    verdict.confidence = float(data.get("confidence", 0.0))
    verdict.summary = data.get("summary", "")

    for fd in data.get("fixes", []):
        try:
            ft = FixType(fd.get("fix_type", "manual"))
        except ValueError:
            ft = FixType.MANUAL
        verdict.fix_items.append(FixItem(
            file_path=fd.get("file_path", ""), description=fd.get("description", ""),
            fix_type=ft, suggested_code=fd.get("suggested_code"),
            affected_segments=fd.get("affected_segments", []),
            severity=fd.get("severity", "error"),
        ))
    return verdict


def _extract_json_block(text: str) -> Optional[str]:
    import re
    # Try markdown-fenced JSON first (most reliable)
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: find the outermost balanced JSON object
    # Use the first '{' and find its matching '}' by brace counting
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1].strip()
    return None
