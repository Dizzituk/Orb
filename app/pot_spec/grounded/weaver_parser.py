# FILE: app/pot_spec/grounded/weaver_parser.py
"""
Weaver Intent Parser for SpecGate

This module handles parsing of Weaver output to extract intent components
for downstream spec generation.

Responsibilities:
- Parse Weaver v3.0 simple text format (weaver_job_description_text)
- Parse Weaver v2.x full spec JSON format (weaver_spec_json)
- Extract goals, constraints, scope markers from Weaver output
- Handle various goal extraction patterns including inline formats

Key Features:
- v1.39: Handle "What is being built:" extraction with regex
- Support for both structured and unstructured Weaver output

Used by:
- spec_runner.py for intent parsing

Version: v2.0 (2026-02-01) - Extracted from spec_generation.py
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


__all__ = [
    "parse_weaver_intent",
]


def parse_weaver_intent(constraints_hint: Optional[Dict]) -> Dict[str, Any]:
    """
    Parse Weaver output to extract intent components.
    
    Handles both:
    - v3.0 simple text (weaver_job_description_text)
    - v2.x full spec JSON (weaver_spec_json)
    
    Args:
        constraints_hint: Dict containing Weaver output and other hints
        
    Returns:
        Dict with extracted intent components:
        - raw_text: Original Weaver text
        - source: Source identifier
        - goal: Extracted goal/objective
        - constraints: List of constraints
        - scope_in: List of in-scope items
        - scope_out: List of out-of-scope items
        - weaver_steps: Steps from Weaver (if available)
        - weaver_outputs: Outputs from Weaver (if available)
        - weaver_acceptance: Acceptance criteria from Weaver (if available)
    """
    if not constraints_hint:
        logger.warning("[weaver_parser] parse_weaver_intent: constraints_hint is empty/None")
        return {}
    
    result = {}
    
    logger.info(
        "[weaver_parser] parse_weaver_intent: constraints_hint keys=%s",
        list(constraints_hint.keys())
    )
    
    # v3.0: Simple Weaver text
    job_desc_text = constraints_hint.get("weaver_job_description_text")
    if job_desc_text:
        result["raw_text"] = job_desc_text
        result["source"] = "weaver_simple"
        logger.info(
            "[weaver_parser] parse_weaver_intent: set raw_text from weaver_job_description_text (%d chars)",
            len(job_desc_text)
        )
        
        # v1.39: Extract goal from "What is being built" section
        # CRITICAL FIX: Handle text like "Astra, command: send to spec gate What is being built: Multi-file reader..."
        # We need to find "What is being built:" specifically, not just split on first colon
        lines = job_desc_text.strip().split("\n")
        goal_found = False
        
        # v1.39: First try direct regex extraction (handles inline format)
        what_is_being_built_match = re.search(
            r'what\s+is\s+being\s+built\s*[:\-]\s*(.+?)(?:\n|$)',
            job_desc_text,
            re.IGNORECASE
        )
        if what_is_being_built_match:
            goal_text = what_is_being_built_match.group(1).strip()
            # Clean up: remove trailing section markers
            goal_text = re.split(r'\*\*|\n', goal_text)[0].strip()
            if goal_text:
                result["goal"] = goal_text
                goal_found = True
                logger.info(
                    "[weaver_parser] v1.39 Extracted goal via regex: %s",
                    result["goal"][:100]
                )
        
        # Fallback: line-by-line parsing (v1.12 behavior)
        if not goal_found:
            for i, line in enumerate(lines):
                line_lower = line.lower().strip()
                if "what is being built" in line_lower:
                    # Check if goal is on same line (after the phrase)
                    match = re.search(r'what\s+is\s+being\s+built\s*[:\-]\s*(.+)', line, re.IGNORECASE)
                    if match and match.group(1).strip():
                        result["goal"] = match.group(1).strip()
                        goal_found = True
                        logger.info(
                            "[weaver_parser] v1.39 Extracted goal from 'What is being built' line: %s",
                            result["goal"][:100]
                        )
                        break
                    # Otherwise, goal might be on next line
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line and not next_line.lower().startswith(("intended", "unresolved", "questions", "-", "*", "**")):
                            result["goal"] = next_line.lstrip("- ").strip()
                            goal_found = True
                            logger.info(
                                "[weaver_parser] v1.39 Extracted goal from line after 'What is being built': %s",
                                result["goal"][:100]
                            )
                            break
        
        # Fallback: first non-header line (old behavior)
        if not goal_found and lines:
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#") and not line.lower().startswith("what is being built"):
                    # Skip generic headers
                    if line.lower() not in ("job description", "job description from weaver"):
                        result["goal"] = line
                        break
        
        # Look for constraints/scope markers
        result["constraints"] = []
        result["scope_in"] = []
        result["scope_out"] = []
        
        for line in lines:
            line_lower = line.lower().strip()
            if "constraint" in line_lower or "must" in line_lower or "require" in line_lower:
                result["constraints"].append(line.strip())
            if "in scope" in line_lower or "should" in line_lower:
                result["scope_in"].append(line.strip())
            if "out of scope" in line_lower or "should not" in line_lower or "don't" in line_lower:
                result["scope_out"].append(line.strip())
    else:
        logger.warning("[weaver_parser] parse_weaver_intent: weaver_job_description_text not found in constraints_hint")
    
    # v2.x: Full spec JSON
    weaver_spec = constraints_hint.get("weaver_spec_json")
    if isinstance(weaver_spec, dict):
        result["source"] = weaver_spec.get("source", "weaver_spec")
        result["goal"] = (
            weaver_spec.get("objective") or
            weaver_spec.get("title") or
            weaver_spec.get("job_description", "")[:200]
        )
        
        # Extract metadata
        metadata = weaver_spec.get("metadata", {}) or {}
        result["content_verbatim"] = (
            metadata.get("content_verbatim") or
            weaver_spec.get("content_verbatim")
        )
        result["location"] = (
            metadata.get("location") or
            weaver_spec.get("location")
        )
        result["scope_constraints"] = (
            metadata.get("scope_constraints") or
            weaver_spec.get("scope_constraints", [])
        )
        
        # Steps and outputs
        result["weaver_steps"] = weaver_spec.get("steps", [])
        result["weaver_outputs"] = weaver_spec.get("outputs", [])
        result["weaver_acceptance"] = weaver_spec.get("acceptance_criteria", [])
    
    return result
