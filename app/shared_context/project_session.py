# FILE: app/shared_context/project_session.py
# Purpose: Lightweight session state for project-aware pipeline routing.
# Called-by: app.builds.pipeline_bridge, app.llm._stream_router_utils, app.llm.project_scoping_stream, app.llm.spec_gate_stream (+3 more)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Lightweight session state for project-aware pipeline routing.

Holds the active project context so the Weaver and pipeline know
where to target builds. Lives in memory for the duration of a chat
session — not persisted across restarts.

v1.0 (2026-03-11): Initial implementation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProjectSession:
    """Active project context for a chat session."""

    project_id: Optional[str] = None
    project_name: Optional[str] = None
    project_type: Optional[str] = None  # "android", "web", "backend", "frontend"
    project_root: Optional[str] = None
    scoping_active: bool = False
    scoping_messages: List[Dict[str, str]] = field(default_factory=list)

    @property
    def is_set(self) -> bool:
        """True if a project is currently active."""
        return self.project_id is not None

    def clear(self) -> None:
        """Reset the session to no active project."""
        self.project_id = None
        self.project_name = None
        self.project_type = None
        self.project_root = None
        self.scoping_active = False
        self.scoping_messages.clear()
        logger.info("[project_session] Session cleared")

    def start_scoping(self) -> None:
        """Enter scoping mode for a new project."""
        self.clear()
        self.scoping_active = True
        logger.info("[project_session] Scoping mode started")

    def add_scoping_message(self, role: str, content: str) -> None:
        """Add a message to the scoping conversation."""
        self.scoping_messages.append({"role": role, "content": content})

    def finish_scoping(
        self,
        project_id: str,
        project_name: str,
        project_type: str,
        project_root: str,
    ) -> None:
        """Finalise scoping and set the active project."""
        self.project_id = project_id
        self.project_name = project_name
        self.project_type = project_type
        self.project_root = project_root
        self.scoping_active = False
        logger.info(
            "[project_session] Scoping complete: %s (%s) at %s",
            project_name, project_type, project_root,
        )

    def to_context_dict(self) -> Dict[str, Optional[str]]:
        """Export as a dict for injection into Weaver/pipeline context."""
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "project_type": self.project_type,
            "project_root": self.project_root,
            "scoping_active": self.scoping_active,
        }


# ═══════════════════════════════════════════════════════════════════
# Global session store (keyed by conversation_id)
# ═══════════════════════════════════════════════════════════════════

_sessions: Dict[str, ProjectSession] = {}


def get_project_session(conversation_id: str) -> ProjectSession:
    """Get or create a ProjectSession for a conversation."""
    if conversation_id not in _sessions:
        _sessions[conversation_id] = ProjectSession()
    return _sessions[conversation_id]


def clear_project_session(conversation_id: str) -> None:
    """Clear the project session for a conversation."""
    if conversation_id in _sessions:
        _sessions[conversation_id].clear()
        del _sessions[conversation_id]


def get_active_session_for_project(project_id: str) -> Optional[ProjectSession]:
    """Find a session that has this project active (any conversation)."""
    for session in _sessions.values():
        if session.project_id == project_id:
            return session
    return None
