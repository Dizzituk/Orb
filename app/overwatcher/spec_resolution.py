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
    """
    spec_id: str
    spec_hash: str
    project_id: int
    title: Optional[str] = None
    created_at: Optional[str] = None
    spec_content: Optional[str] = None
    deliverable: Optional[ParsedDeliverable] = None
    
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
        
        # Priority order: content_json first (canonical), then content_markdown
        field_names = [
            'content_json',      # Canonical JSON - PRIMARY SOURCE
            'content_markdown',  # Markdown version
            'content',           # Generic fallback
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
        
        # Parse spec content to get deliverable
        logger.info(f"[resolve_spec] Parsing spec content...")
        deliverable = parse_spec_content(spec_content)
        
        if not deliverable:
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
            )
        
        logger.info(f"[resolve_spec] SUCCESS: deliverable={deliverable.to_dict()}")
        
        return ResolvedSpec(
            spec_id=spec.spec_id,
            spec_hash=spec.spec_hash,
            project_id=project_id,
            title=getattr(spec, 'title', None),
            created_at=spec.created_at.isoformat() if hasattr(spec, 'created_at') and spec.created_at else None,
            spec_content=spec_content,
            deliverable=deliverable,
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
]
