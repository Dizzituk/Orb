# FILE: app/llm/pipeline/_high_stakes_prompt_build.py
"""
High-stakes pipeline: Draft message building.

Injects POT spec, foundation templates, spec metadata, evidence contract,
and other context into the draft messages for the architecture LLM.
Extracted from high_stakes.py.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def build_draft_messages(
    envelope_messages: List[Dict],
    spec_markdown: Optional[str],
    spec_json: Optional[str],
    spec_id: Optional[str],
    spec_hash: Optional[str],
    transcripts_text: str,
    file_map: Optional[str],
) -> List[Dict]:
    """Build the full draft message list with all injections."""
    draft_messages = list(envelope_messages)

    # v5.0: Inject POT spec markdown
    if spec_markdown:
        draft_messages.append({"role": "system", "content": _build_pot_spec_instruction(spec_markdown)})
        logger.info("[high_stakes] v5.0 Injected POT spec (%d chars)", len(spec_markdown))

    # v5.5: Foundation templates
    _inject_foundation_templates(draft_messages, spec_json, spec_markdown)

    # v4.2: Spec metadata anchoring
    if spec_json:
        _inject_spec_anchoring(draft_messages, spec_json)

    # Stage 3 spec echo
    if spec_id and spec_hash:
        _inject_spec_echo(draft_messages, spec_id, spec_hash)

    # Video transcripts
    if transcripts_text:
        draft_messages.append({"role": "system", "content": f"Video context:\n{transcripts_text.strip()}"})

    # File map
    if file_map:
        draft_messages.append({"role": "system", "content": f"{file_map}\n\nRefer to files using [FILE_X] identifiers."})

    # v2.0: Evidence contract
    _inject_evidence_contract(draft_messages)

    return draft_messages


def _build_pot_spec_instruction(spec_markdown: str) -> str:
    sep = '=' * 70
    return f"""{sep}
POT SPEC - AUTHORITATIVE SOURCE OF TRUTH (GROUNDED EVIDENCE)
{sep}

The following POT spec contains VERIFIED information:
- Real file paths that have been confirmed to exist
- Real line numbers pointing to actual code
- Real content excerpts from the codebase

Your architecture MUST:
1. Address EVERY item in the "Change" section below
2. NOT modify items in the "Skip" section
3. Follow the exact file paths and line numbers provided
4. NOT invent features, files, or changes beyond this spec
5. Treat ALL sections in this markdown as binding — including Acceptance Criteria,
   Constraints, Evidence Requests, and Implementation Steps.
6. FILE SIZE CONSTRAINT: Design all output files to be under 20 KB (~500 lines)
   each. Prefer single-responsibility modules. If a file would exceed 20 KB,
   decompose it into smaller focused modules.

{spec_markdown}

{sep}
END OF POT SPEC - Architecture must implement EXACTLY the above
{sep}
"""


def _inject_foundation_templates(
    draft_messages: List[Dict],
    spec_json: Optional[str],
    spec_markdown: Optional[str],
) -> None:
    """v5.5: Inject foundation templates for greenfield CREATE jobs."""
    try:
        _spec_data = {}
        if spec_json:
            try:
                _spec_data = json.loads(spec_json) if isinstance(spec_json, str) else (spec_json or {})
            except Exception:
                pass

        _job_kind = _spec_data.get("job_kind", "")
        _impl_stack = _spec_data.get("implementation_stack", {})

        if _job_kind in ("architecture", "create", "") and spec_markdown:
            from app.llm.critical_pipeline.foundation_templates import match_templates
            _tech_dict = {k: str(v) for k, v in _impl_stack.items() if v} if isinstance(_impl_stack, dict) else {}
            _matched = match_templates(tech_stack=_tech_dict, spec_text=spec_markdown, max_templates=4)
            if _matched.count > 0:
                draft_messages.append({"role": "system", "content": _matched.format_for_prompt()})
                logger.info("[high_stakes] v5.5 Foundation templates: %d injected", _matched.count)
    except ImportError:
        pass
    except Exception as err:
        logger.warning("[high_stakes] v5.5 Foundation template error: %s", err)


def _inject_spec_anchoring(draft_messages: List[Dict], spec_json) -> None:
    """v4.2: Extract and inject spec metadata as anchoring context."""
    try:
        spec_data = json.loads(spec_json) if isinstance(spec_json, str) else spec_json
        parts = []
        parts.append("=" * 60)
        parts.append("AUTHORITATIVE SPEC (PoT) - YOU MUST HONOR THESE CONSTRAINTS")
        parts.append("=" * 60)

        if spec_data.get("goal"):
            parts.append(f"\nGOAL: {spec_data.get('goal')}")

        impl_stack = spec_data.get("implementation_stack")
        if impl_stack and isinstance(impl_stack, dict):
            parts.append("\nIMPLEMENTATION STACK:")
            for key in ("language", "framework", "runtime"):
                if impl_stack.get(key):
                    parts.append(f"  {key.title()}: {impl_stack[key]}")
            parts.append(f"  Source: {impl_stack.get('source', 'user discussion')}")
            if impl_stack.get("stack_locked"):
                parts.append("  ⚠️  STACK LOCKED: Use this exact technology stack.")
            else:
                parts.append("  Stack discussed but not locked.")

        requirements = spec_data.get("requirements", {})
        for level, label in [("must", "MUST"), ("should", "SHOULD")]:
            reqs = requirements.get(level, [])
            if reqs:
                parts.append(f"\n{label} REQUIREMENTS:")
                for i, req in enumerate(reqs[:10], 1):
                    parts.append(f"  {i}. {req}")

        constraints = spec_data.get("constraints", {})
        if constraints:
            parts.append("\nCONSTRAINTS:")
            for key, value in list(constraints.items())[:10]:
                parts.append(f"  {key}: {value}")

        parts.append("\n" + "=" * 60)
        parts.append("YOUR ARCHITECTURE MUST ALIGN WITH THE ABOVE SPEC.")
        parts.append("=" * 60)

        draft_messages.append({"role": "system", "content": "\n".join(parts)})
        logger.info("[high_stakes] v4.2 Spec anchoring injected")
    except Exception as e:
        logger.warning("[high_stakes] v4.2 Spec injection failed: %s", e)


def _inject_spec_echo(draft_messages: List[Dict], spec_id: str, spec_hash: str) -> None:
    """Inject Stage 3 spec echo instruction."""
    try:
        from app.jobs.stage3_locks import build_spec_echo_instruction
        draft_messages.append({"role": "system", "content": build_spec_echo_instruction(spec_id, spec_hash)})
    except ImportError:
        pass


def _inject_evidence_contract(draft_messages: List[Dict]) -> None:
    """v2.0: Inject evidence-or-request contract prompt."""
    try:
        from app.llm.pipeline.evidence_contract_prompt import EVIDENCE_CONTRACT_PROMPT
        if EVIDENCE_CONTRACT_PROMPT:
            draft_messages.append({"role": "system", "content": EVIDENCE_CONTRACT_PROMPT})
            logger.info("[high_stakes] v2.0 Evidence contract injected (%d chars)", len(EVIDENCE_CONTRACT_PROMPT))
    except ImportError:
        pass
