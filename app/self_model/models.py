# FILE: app/self_model/models.py
"""
Data models for the Self-Model & Evolution Layer.

Defines the core data structures used across capability mapping,
user modelling, pattern observation, suggestions, and the evolution journal.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class SuggestionCategory(str, Enum):
    CAPABILITY = "capability"
    QUALITY = "quality"
    PERSONALISATION = "personalisation"
    MAINTENANCE = "maintenance"
    OPPORTUNITY = "opportunity"


class SuggestionStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class TrustStage(str, Enum):
    EARLY = "early"
    MIDDLE = "middle"
    MATURE = "mature"


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class CapabilityEntry:
    """A single capability that ASTRA knows it has."""
    domain: str
    name: str
    description: str
    status: str = "active"  # active, stubbed, in_development, broken
    module_path: str = ""
    external_services: List[str] = field(default_factory=list)
    last_verified: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UserFact:
    """A single fact ASTRA knows about the user."""
    category: str  # biographical, preference, philosophy, project, pattern
    key: str
    value: str
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    source: str = ""  # which session/conversation this came from
    reinforcement_count: int = 0
    first_seen: str = ""
    last_seen: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["confidence"] = self.confidence.value
        return data


@dataclass
class ObservedPattern:
    """A pattern the Self-Model has noticed."""
    pattern_id: str = field(default_factory=lambda: str(uuid4())[:8])
    domain: str = ""
    description: str = ""
    frequency: int = 1
    first_observed: str = ""
    last_observed: str = ""
    evidence: List[str] = field(default_factory=list)
    actionable: bool = False
    suggested: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Suggestion:
    """A suggestion from the Self-Model to the user."""
    suggestion_id: str = field(default_factory=lambda: str(uuid4())[:8])
    category: SuggestionCategory = SuggestionCategory.QUALITY
    title: str = ""
    description: str = ""
    reasoning: str = ""
    what_we_gain: str = ""
    status: SuggestionStatus = SuggestionStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: str = ""
    source_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["category"] = self.category.value
        data["status"] = self.status.value
        return data


@dataclass
class JournalEntry:
    """A single entry in the evolution journal."""
    entry_id: str = field(default_factory=lambda: str(uuid4())[:8])
    date: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    event_type: str = ""  # observation, suggestion, approval, rejection, deferral, learning, capability_change
    summary: str = ""
    details: str = ""
    suggestion_id: Optional[str] = None
    outcome: str = ""  # what changed as a result

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
