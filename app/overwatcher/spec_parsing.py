# FILE: app/overwatcher/spec_parsing.py
"""Spec Content Parsing: Extracts deliverables from spec JSON/markdown.

Parses spec content to extract:
- Target filename
- Content to write
- Action (add/modify/delete)
- Target location (DESKTOP, etc.)
- Must-exist constraint
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_TARGET = "DESKTOP"


@dataclass
class ParsedDeliverable:
    """Parsed deliverable from spec content."""
    filename: str
    content: str
    action: str  # "add" | "modify" | "delete"
    target: str  # "DESKTOP" | path
    must_exist: bool = False  # If True, file must exist before operation
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "content": self.content,
            "action": self.action,
            "target": self.target,
            "must_exist": self.must_exist,
        }


def parse_spec_content(content: str) -> Optional[ParsedDeliverable]:
    """Parse spec content (markdown or JSON) to extract deliverable.
    
    Args:
        content: Raw spec content (markdown, JSON, or text)
    
    Returns:
        ParsedDeliverable if parsing succeeds, None otherwise
    """
    if not content:
        logger.error("[parse_spec] No content provided")
        return None
    
    logger.info(f"[parse_spec] Parsing content ({len(content)} chars)")
    logger.info(f"[parse_spec] Content preview: {content[:300]}...")
    
    # Try JSON first
    try:
        spec_json = json.loads(content)
        result = _parse_json_spec(spec_json)
        if result:
            logger.info(f"[parse_spec] JSON parse succeeded: {result.to_dict()}")
            return result
    except json.JSONDecodeError:
        logger.debug("[parse_spec] Not JSON, trying markdown")
    
    # Parse as markdown
    result = _parse_markdown_spec(content)
    if result:
        logger.info(f"[parse_spec] Markdown parse succeeded: {result.to_dict()}")
    else:
        logger.error("[parse_spec] Failed to parse deliverable from content")
        logger.error(f"[parse_spec] Full content:\n{content}")
    
    return result


def _parse_json_spec(spec: Dict[str, Any]) -> Optional[ParsedDeliverable]:
    """Parse JSON spec to extract deliverable."""
    
    # =========================================================================
    # v1.1: SANDBOX MICRO-EXECUTION FALLBACK (SpecGate schema compatibility)
    # =========================================================================
    # SpecGate micro-execution specs use sandbox_* prefixed fields.
    # Map these to ParsedDeliverable for Overwatcher compatibility.
    #
    # SpecGate fields used:
    #   - sandbox_output_path → target file
    #   - sandbox_input_path → fallback target file
    #   - sandbox_generated_reply → content to write
    # =========================================================================
    
    sandbox_output_path = spec.get("sandbox_output_path")
    sandbox_input_path = spec.get("sandbox_input_path")
    sandbox_generated_reply = spec.get("sandbox_generated_reply")
    
    # Determine target file: output_path preferred, input_path as fallback
    target_file = sandbox_output_path or sandbox_input_path
    
    if target_file:
        logger.info(
            "[parse_spec] Sandbox micro-execution: file=%s, content=%d chars",
            target_file,
            len(sandbox_generated_reply) if sandbox_generated_reply else 0,
        )
        
        return ParsedDeliverable(
            filename=target_file,
            content=sandbox_generated_reply or "",
            action="modify",
            target=DEFAULT_TARGET,
            must_exist=True,
        )
    
    # =========================================================================
    # EXISTING PARSING LOGIC (unchanged below this line)
    # =========================================================================
    
    # Check for explicit deliverables array
    deliverables = spec.get("deliverables", [])
    if deliverables and len(deliverables) > 0:
        d = deliverables[0]
        filename = d.get("filename")
        if not filename:
            return None
        
        action = d.get("action", "add")
        must_exist = d.get("must_exist", action == "modify")
        
        return ParsedDeliverable(
            filename=filename,
            content=d.get("content", ""),
            action=action,
            target=d.get("target", DEFAULT_TARGET),
            must_exist=must_exist,
        )
    
    # Parse from requirements
    requirements = spec.get("requirements", {})
    if isinstance(requirements, dict):
        functional = requirements.get("functional", [])
        for req in functional:
            if isinstance(req, str):
                parsed = _parse_requirement_text(req)
                if parsed:
                    return parsed
    
    # Check acceptance criteria
    acceptance = spec.get("acceptance_criteria", [])
    for criterion in acceptance:
        if isinstance(criterion, str):
            parsed = _parse_requirement_text(criterion)
            if parsed:
                return parsed
    
    # Try steps array (Weaver spec format)
    steps = spec.get("steps", [])
    for step in steps:
        if isinstance(step, dict):
            desc = step.get("description", "")
            if desc:
                parsed = _parse_requirement_text(desc)
                if parsed:
                    return parsed
        elif isinstance(step, str):
            parsed = _parse_requirement_text(step)
            if parsed:
                return parsed
    
    # Try outputs array
    outputs = spec.get("outputs", [])
    for output in outputs:
        if isinstance(output, dict):
            if output.get("type") == "file":
                filename = output.get("name") or output.get("filename") or output.get("path")
                if filename:
                    return ParsedDeliverable(
                        filename=filename,
                        content=output.get("content", ""),
                        action=output.get("action", "add"),
                        target=output.get("target", DEFAULT_TARGET),
                        must_exist=output.get("must_exist", False),
                    )
            desc = output.get("description", "")
            if desc:
                parsed = _parse_requirement_text(desc)
                if parsed:
                    return parsed
    
    return None


def _parse_markdown_spec(content: str) -> Optional[ParsedDeliverable]:
    """Parse markdown spec to extract deliverable."""
    deliverable = None
    
    # Extract all list items
    list_items = re.findall(r'^[-*]\s+(.+)$', content, re.MULTILINE)
    
    logger.info(f"[parse_markdown] Found {len(list_items)} list items")
    
    for item in list_items:
        logger.debug(f"[parse_markdown] Parsing item: {item}")
        parsed = _parse_requirement_text(item)
        if parsed:
            logger.info(f"[parse_markdown] Parsed from item: {parsed.to_dict()}")
            if deliverable is None:
                deliverable = parsed
            else:
                if parsed.content and not deliverable.content:
                    deliverable.content = parsed.content
                if parsed.filename and not deliverable.filename:
                    deliverable.filename = parsed.filename
    
    # Also look for content in acceptance criteria section
    acceptance_match = re.search(
        r'(?:Acceptance Criteria|acceptance_criteria)(.*?)(?:##|\Z)',
        content,
        re.IGNORECASE | re.DOTALL
    )
    if acceptance_match:
        acceptance_text = acceptance_match.group(1)
        logger.info(f"[parse_markdown] Found acceptance section: {acceptance_text[:200]}...")
        for item in re.findall(r'^[-*]\s+(.+)$', acceptance_text, re.MULTILINE):
            parsed = _parse_requirement_text(item)
            if parsed:
                logger.info(f"[parse_markdown] From acceptance: {parsed.to_dict()}")
                if deliverable and parsed.content:
                    deliverable.content = parsed.content
                elif parsed:
                    deliverable = parsed
    
    return deliverable


def _parse_requirement_text(text: str) -> Optional[ParsedDeliverable]:
    """Parse a single requirement text to extract file operation."""
    filename = None
    file_content = None
    action = "modify"
    target = DEFAULT_TARGET
    must_exist = False
    
    text_lower = text.lower()
    
    # === Extract filename ===
    patterns = [
        (r"file\s+named\s+['\"]([^'\"]+)['\"]", "file named 'X'"),
        (r"file\s+called\s+['\"]([^'\"]+)['\"]", "file called 'X'"),
        (r"['\"]([^'\"]+)['\"](?:\s+file)", "'X' file"),
        (r"into\s+(?:the\s+)?['\"]([^'\"]+)['\"](?:\s+file)?", "into 'X'"),
        (r"(?:the|a)\s+['\"]([^'\"]+)['\"](?:\s+file)", "the 'X' file"),
        (r"file\s+['\"]([^'\"]+)['\"]", "file 'X'"),
        (r"called\s+['\"]([^'\"]+)['\"]", "called 'X'"),
    ]
    
    for pattern, desc in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            filename = match.group(1)
            logger.debug(f"[parse_req] Found filename '{filename}' via pattern: {desc}")
            break
    
    # === Extract content ===
    content_patterns = [
        (r"string\s+['\"]([^'\"]+)['\"]", "string 'X'"),
        (r"content\s+['\"]([^'\"]+)['\"]", "content 'X'"),
        (r"contains?\s+(?:the\s+)?(?:string\s+)?['\"]([^'\"]+)['\"]", "contains 'X'"),
        (r"with\s+(?:content\s+)?['\"]([^'\"]+)['\"]", "with 'X'"),
        (r"write\s+['\"]([^'\"]+)['\"]", "write 'X'"),
        (r"saying[,:]?\s+['\"]([^'\"]+)['\"]", "saying 'X'"),
        (r"text\s+['\"]([^'\"]+)['\"]", "text 'X'"),
    ]
    
    for pattern, desc in content_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            file_content = match.group(1)
            file_content = file_content.rstrip('.,;')
            logger.debug(f"[parse_req] Found content '{file_content[:50]}...' via pattern: {desc}")
            break
    
    # === Determine action and must_exist ===
    if "create" in text_lower and "new" in text_lower:
        action = "add"
        must_exist = False
    elif "locate" in text_lower or "find" in text_lower or "existing" in text_lower:
        action = "modify"
        must_exist = True
    elif "cannot create" in text_lower or "must not create" in text_lower or "no new" in text_lower:
        action = "modify"
        must_exist = True
    elif "write" in text_lower and ("into" in text_lower or "to the" in text_lower):
        action = "modify"
        must_exist = True
    
    # === Determine target ===
    if "sandbox" in text_lower and "desktop" in text_lower:
        target = "DESKTOP"
    elif "desktop" in text_lower:
        target = "DESKTOP"
    
    if filename:
        return ParsedDeliverable(
            filename=filename,
            content=file_content or "",
            action=action,
            target=target,
            must_exist=must_exist,
        )
    
    return None


__all__ = [
    "ParsedDeliverable",
    "parse_spec_content",
    "DEFAULT_TARGET",
]
