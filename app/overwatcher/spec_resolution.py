# FILE: app/overwatcher/spec_resolution.py
"""Spec Resolution: Resolves specs from database and creates ResolvedSpec objects.

Handles:
- Loading validated specs from DB via specs.service
- Parsing content_json to extract deliverables
- Creating smoke test specs for explicit testing

v1.2 (2026-01-24): Added get_output_mode() and get_insertion_format() accessors
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from .spec_parsing import ParsedDeliverable, parse_spec_content, DEFAULT_TARGET
from .pot_spec_parser import is_pot_spec_format, parse_pot_spec_markdown, POTParseResult

logger = logging.getLogger(__name__)

# Smoke test constants - ONLY used by create_smoke_test_spec()
_SMOKE_TEST_FILENAME = "hello.txt"
_SMOKE_TEST_CONTENT = "ASTRA OK"


class SpecMissingDeliverableError(Exception):
    """Raised when spec has no concrete file deliverable."""
    pass


@dataclass
class ResolvedSpec:
    """Resolved spec context from database.
    
    v1.2: Added get_output_mode() and get_insertion_format() accessor methods.
    v2.0 (2026-02-02): Added POT spec support with is_pot_spec and pot_tasks fields.
    v3.0 (2026-02-06): Added architecture spec support for Critical Pipeline outputs.
    """
    spec_id: str
    spec_hash: str
    project_id: int
    title: Optional[str] = None
    created_at: Optional[str] = None
    spec_content: Optional[str] = None
    deliverable: Optional[ParsedDeliverable] = None
    is_pot_spec: bool = False
    pot_tasks: Optional[POTParseResult] = None
    # v3.0: Architecture spec support
    is_architecture_spec: bool = False
    architecture_markdown: Optional[str] = None
    
    @property
    def is_smoke_test(self) -> bool:
        """Check if this is a smoke test spec."""
        return self.spec_id.startswith("smoke-test-")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "spec_hash": self.spec_hash,
            "project_id": self.project_id,
            "title": self.title,
            "created_at": self.created_at,
            "spec_content": self.spec_content[:500] if self.spec_content else None,
            "deliverable": self.deliverable.to_dict() if self.deliverable else None,
            "is_smoke_test": self.is_smoke_test,
        }
    
    def get_target_file(self) -> Tuple[str, str, str]:
        """Get target filename, content, and action from parsed spec.
        
        Returns:
            Tuple of (filename, content, action)
        
        Raises:
            SpecMissingDeliverableError: If real spec has no deliverable
        """
        if self.deliverable:
            return (
                self.deliverable.filename,
                self.deliverable.content,
                self.deliverable.action,
            )
        
        # ONLY for explicit smoke test specs
        if self.is_smoke_test:
            return _SMOKE_TEST_FILENAME, _SMOKE_TEST_CONTENT, "add"
        
        # HARD FAIL for real specs without deliverable
        raise SpecMissingDeliverableError(
            f"Spec {self.spec_id} has no parsed deliverable. "
            f"Cannot determine target file. "
            f"Spec content preview: {self.spec_content[:200] if self.spec_content else 'None'}..."
        )
    
    def get_target(self) -> str:
        """Get sandbox target (e.g., 'DESKTOP')."""
        if self.deliverable:
            return self.deliverable.target
        return DEFAULT_TARGET
    
    def get_must_exist(self) -> bool:
        """Check if target file must exist before operation."""
        if self.deliverable:
            return self.deliverable.must_exist
        return False
    
    def get_output_mode(self) -> Optional[str]:
        """Get output mode for file write operations.
        
        v1.2: Added for APPEND_IN_PLACE support.
        
        Returns:
            Output mode string: "append_in_place", "separate_reply_file", "chat_only", or None
        """
        result = None
        if self.deliverable:
            result = self.deliverable.output_mode
        # AGGRESSIVE DEBUG
        print(f">>> [SPEC_RESOLUTION] get_output_mode() -> {result!r} (deliverable.output_mode={self.deliverable.output_mode if self.deliverable else 'NO_DELIVERABLE'})")
        return result
    
    def get_insertion_format(self) -> Optional[str]:
        """Get insertion format string for append operations.
        
        v1.2: Added for APPEND_IN_PLACE support.
        
        Returns:
            Insertion format string (e.g., "\\n\\nAnswer:\\n{reply}\\n") or None
        """
        if self.deliverable:
            return self.deliverable.insertion_format
        return None
    
    def get_task_description(self) -> str:
        """Generate task description from spec content."""
        if self.deliverable:
            action_verb = {
                "add": "Create",
                "modify": "Modify existing",
                "delete": "Delete",
            }.get(self.deliverable.action, "Process")
            
            desc = f"{action_verb} file '{self.deliverable.filename}'"
            if self.deliverable.content:
                desc += f" with content '{self.deliverable.content}'"
            return desc
        
        return self.title or f"Execute spec {self.spec_id}"


# =============================================================================
# v3.0: Architecture Spec Detection
# =============================================================================

_ARCHITECTURE_MARKERS = [
    "## Goal",
    "## Implementation Steps",
    "## New Files to Create",
    "## Files to Modify",
    "## Acceptance Criteria",
    "## LLM Architecture Analysis",
    "EVIDENCE_REQUEST",
]


def is_architecture_spec_format(content_markdown: Optional[str], content_json: Optional[str]) -> bool:
    """Detect if a spec is an architecture/grounded-create spec.
    
    v3.0: Architecture specs from the SpecGate grounded-create path have:
    - Substantial content_markdown (>500 chars) with architecture markers
    - content_json that is either empty or has all-empty structured fields
    - NOT a POT spec (no ## Change section)
    
    Returns True if this looks like an architecture spec that needs
    routing through the Critical Pipeline architecture document.
    """
    if not content_markdown or len(content_markdown) < 500:
        return False
    
    # Check for architecture markers in markdown
    marker_count = sum(
        1 for marker in _ARCHITECTURE_MARKERS
        if marker in content_markdown
    )
    
    # Need at least 2 markers to be confident
    if marker_count < 2:
        return False
    
    # Verify JSON is empty/minimal (the root cause of why this spec can't parse)
    if content_json:
        try:
            import json as _json
            parsed = _json.loads(content_json) if isinstance(content_json, str) else content_json
            if isinstance(parsed, dict):
                # Check if key structured fields are all empty
                empty_indicators = [
                    not parsed.get("proposed_steps"),
                    not parsed.get("acceptance_criteria"),
                    not parsed.get("deliverables"),
                    not parsed.get("steps"),
                    not parsed.get("sandbox_output_path"),
                ]
                if not all(empty_indicators):
                    # JSON actually has content — not an architecture spec gap
                    return False
        except Exception:
            pass  # If JSON is unparseable, markdown is even more likely the source of truth
    
    return True


def resolve_latest_spec(
    project_id: int,
    db_session=None,
) -> Optional[ResolvedSpec]:
    """Resolve the latest validated spec for a project.
    
    Returns None (causing command to fail) if:
    - No spec found
    - Spec has no content
    - Content cannot be parsed to extract deliverable (for non-smoke specs)
    """
    try:
        from app.specs.service import get_latest_validated_spec
        
        spec = get_latest_validated_spec(db_session, project_id)
        if not spec:
            logger.warning(f"[resolve_spec] No validated spec for project {project_id}")
            return None
        
        logger.info(f"[resolve_spec] Found spec {spec.spec_id}")
        
        # Log all available attributes for debugging
        attrs = [a for a in dir(spec) if not a.startswith('_')]
        logger.info(f"[resolve_spec] Spec attributes: {attrs}")
        
        # Get spec content - Use correct field names from models.py
        # The Spec model has: content_json (canonical JSON) and content_markdown
        spec_content = None
        tried_fields = []
        
        # v2.0: POT SPEC DETECTION - Check both content_json and content_markdown
        # POT specs may have empty content_json but real data in content_markdown
        content_json = getattr(spec, 'content_json', None) if hasattr(spec, 'content_json') else None
        content_markdown = getattr(spec, 'content_markdown', None) if hasattr(spec, 'content_markdown') else None
        
        # DEBUG PRINTS (will definitely show in console)
        print(f">>> [POT_DETECT] content_json exists: {content_json is not None}, len: {len(content_json) if content_json else 0}")
        print(f">>> [POT_DETECT] content_markdown exists: {content_markdown is not None}, len: {len(content_markdown) if content_markdown else 0}")
        if content_markdown:
            print(f">>> [POT_DETECT] content_markdown preview: {content_markdown[:300]}...")
        else:
            print(f">>> [POT_DETECT] content_markdown is NULL or empty!")
        
        # Detect if this is a POT spec by checking markdown format
        is_pot_spec = False
        if content_markdown and is_pot_spec_format(content_markdown):
            is_pot_spec = True
            print(f">>> [POT_DETECT] ✓ POT SPEC DETECTED - will use content_markdown")
        else:
            print(f">>> [POT_DETECT] Not a POT spec (content_markdown empty or no ## Change section)")
        
        # Priority order depends on spec type
        if is_pot_spec:
            # POT specs: prioritize content_markdown
            field_names = [
                'content_markdown',  # PRIMARY for POT specs
                'content_json',      # Fallback
                'content',
                'spec_content',
                'markdown',
                'raw_content',
            ]
        else:
            # Regular specs: prioritize content_json
            field_names = [
                'content_json',      # PRIMARY for regular specs
                'content_markdown',
                'content',
                'spec_content',
                'markdown',
                'raw_content',
            ]
        
        for field_name in field_names:
            tried_fields.append(field_name)
            if hasattr(spec, field_name):
                value = getattr(spec, field_name)
                if value and isinstance(value, str) and len(value) > 10:
                    spec_content = value
                    logger.info(f"[resolve_spec] Found content in '{field_name}': {len(spec_content)} chars")
                    logger.info(f"[resolve_spec] Content preview: {spec_content[:200]}...")
                    break
        
        if not spec_content:
            logger.error(f"[resolve_spec] Spec {spec.spec_id} has NO content!")
            logger.error(f"[resolve_spec] Tried fields: {tried_fields}")
            logger.error(f"[resolve_spec] Actual attributes with values:")
            for attr in attrs:
                try:
                    val = getattr(spec, attr)
                    if val and not callable(val):
                        val_str = str(val)[:100]
                        logger.error(f"  {attr}: {type(val).__name__} = {val_str}")
                except Exception as e:
                    logger.error(f"  {attr}: <error reading: {e}>")
            return None
        
        # v2.0: Parse spec content - different logic for POT vs regular specs
        logger.info(f"[resolve_spec] Parsing spec content (is_pot_spec={is_pot_spec})...")
        
        deliverable = None
        pot_tasks = None
        
        if is_pot_spec:
            # Parse POT markdown into atomic tasks
            logger.info(f"[resolve_spec] Parsing as POT spec...")
            pot_tasks = parse_pot_spec_markdown(
                markdown=spec_content,
                spec_content=content_json or spec_content,
            )
            
            if not pot_tasks.is_valid:
                logger.error(f"[resolve_spec] POT parsing failed: {pot_tasks.errors}")
                return ResolvedSpec(
                    spec_id=spec.spec_id,
                    spec_hash=spec.spec_hash,
                    project_id=project_id,
                    title=getattr(spec, 'title', None),
                    created_at=spec.created_at.isoformat() if hasattr(spec, 'created_at') and spec.created_at else None,
                    spec_content=spec_content,
                    deliverable=None,
                    is_pot_spec=True,
                    pot_tasks=pot_tasks,
                )
            
            logger.info(
                f"[resolve_spec] POT parsing SUCCESS: {len(pot_tasks.tasks)} tasks, "
                f"search='{pot_tasks.search_term}', replace='{pot_tasks.replace_term}'"
            )
        else:
            # Parse regular spec (JSON/markdown)
            deliverable = parse_spec_content(spec_content)
            
            if not deliverable:
                # v3.0: Before failing, check if this is an architecture spec
                # Architecture specs from grounded-create have rich content_markdown
                # but empty content_json — they need routing via architecture doc
                if is_architecture_spec_format(content_markdown, content_json):
                    logger.info(
                        f"[resolve_spec] v3.0 ARCHITECTURE SPEC DETECTED for {spec.spec_id}: "
                        f"content_markdown={len(content_markdown)} chars, "
                        f"content_json has empty structured fields"
                    )
                    print(
                        f">>> [ARCH_DETECT] \u2713 Architecture spec detected — "
                        f"will route via Critical Pipeline architecture document"
                    )
                    return ResolvedSpec(
                        spec_id=spec.spec_id,
                        spec_hash=spec.spec_hash,
                        project_id=project_id,
                        title=getattr(spec, 'title', None),
                        created_at=spec.created_at.isoformat() if hasattr(spec, 'created_at') and spec.created_at else None,
                        spec_content=spec_content,
                        deliverable=None,
                        is_pot_spec=False,
                        pot_tasks=None,
                        is_architecture_spec=True,
                        architecture_markdown=content_markdown,
                    )
                
                logger.error(f"[resolve_spec] Failed to parse deliverable from spec {spec.spec_id}")
                logger.error(f"[resolve_spec] Full content:\n{spec_content}")
                return ResolvedSpec(
                    spec_id=spec.spec_id,
                    spec_hash=spec.spec_hash,
                    project_id=project_id,
                    title=getattr(spec, 'title', None),
                    created_at=spec.created_at.isoformat() if hasattr(spec, 'created_at') and spec.created_at else None,
                    spec_content=spec_content,
                    deliverable=None,
                    is_pot_spec=False,
                    pot_tasks=None,
                )
            
            logger.info(f"[resolve_spec] Regular spec SUCCESS: deliverable={deliverable.to_dict()}")
        
        return ResolvedSpec(
            spec_id=spec.spec_id,
            spec_hash=spec.spec_hash,
            project_id=project_id,
            title=getattr(spec, 'title', None),
            created_at=spec.created_at.isoformat() if hasattr(spec, 'created_at') and spec.created_at else None,
            spec_content=spec_content,
            deliverable=deliverable,
            is_pot_spec=is_pot_spec,
            pot_tasks=pot_tasks,
        )
        
    except ImportError as e:
        logger.error(f"[resolve_spec] specs.service not available: {e}")
        return None
    except Exception as e:
        logger.exception(f"[resolve_spec] Failed: {e}")
        return None


def create_smoke_test_spec() -> ResolvedSpec:
    """Create spec for explicit smoke testing ONLY.
    
    This is the ONLY place that uses _SMOKE_TEST_FILENAME/_SMOKE_TEST_CONTENT.
    """
    spec_content = json.dumps({
        "title": "Smoke Test: Sandbox Hello",
        "summary": f"Create {_SMOKE_TEST_FILENAME} with '{_SMOKE_TEST_CONTENT}' on sandbox desktop",
        "deliverables": [{
            "type": "file",
            "target": "DESKTOP",
            "filename": _SMOKE_TEST_FILENAME,
            "content": _SMOKE_TEST_CONTENT,
            "action": "add",
            "must_exist": False,
        }],
    }, sort_keys=True)
    
    spec_hash = hashlib.sha256(spec_content.encode()).hexdigest()
    spec_id = f"smoke-test-{uuid4().hex[:8]}"
    
    deliverable = ParsedDeliverable(
        filename=_SMOKE_TEST_FILENAME,
        content=_SMOKE_TEST_CONTENT,
        action="add",
        target="DESKTOP",
        must_exist=False,
    )
    
    logger.info(f"[create_smoke_test_spec] Created smoke test spec {spec_id}")
    
    return ResolvedSpec(
        spec_id=spec_id,
        spec_hash=spec_hash,
        project_id=0,
        title="Smoke Test: Sandbox Hello",
        created_at=datetime.now(timezone.utc).isoformat(),
        spec_content=spec_content,
        deliverable=deliverable,
    )


__all__ = [
    "ResolvedSpec",
    "SpecMissingDeliverableError",
    "resolve_latest_spec",
    "create_smoke_test_spec",
    "is_architecture_spec_format",
]
