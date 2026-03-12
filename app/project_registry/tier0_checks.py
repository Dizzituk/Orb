# FILE: app/project_registry/tier0_checks.py
"""
Tier 0 rule checks for project registry commands.

Detects project scan, register, and list commands in natural
language. Integrates with the wake word classifier to require
ASTRA to be addressed for command execution.

v1.0 (2026-03): Initial implementation.
"""
from __future__ import annotations

import re
from typing import Optional

from app.translation.tier0_rules import Tier0RuleResult
from app.translation.schemas import CanonicalIntent


# ── Patterns ────────────────────────────────────────────────────────────

# Project scan patterns (after wake word / command prefix stripped)
_SCAN_PROJECT_PATTERNS = [
    re.compile(r"scan\s+(?:the\s+)?(?:project\s+)?architecture", re.IGNORECASE),
    re.compile(r"(?:re)?scan\s+(?:the\s+)?project", re.IGNORECASE),
    re.compile(r"refresh\s+(?:the\s+)?architecture", re.IGNORECASE),
    re.compile(r"update\s+(?:the\s+)?(?:project\s+)?architecture", re.IGNORECASE),
    re.compile(r"scan\s+(?:the\s+)?(?:driver|copilot|android)", re.IGNORECASE),
    re.compile(r"rescan\s+(?:the\s+)?(?:driver|copilot|android)", re.IGNORECASE),
]

# Register project patterns
_REGISTER_PATTERNS = [
    re.compile(r"register\s+(?:a\s+)?(?:new\s+)?project", re.IGNORECASE),
    re.compile(r"add\s+(?:a\s+)?(?:new\s+)?project", re.IGNORECASE),
]

# List projects patterns
_LIST_PATTERNS = [
    re.compile(r"list\s+(?:registered\s+)?projects", re.IGNORECASE),
    re.compile(r"show\s+(?:registered\s+)?projects", re.IGNORECASE),
    re.compile(r"what\s+projects\s+(?:are|do)", re.IGNORECASE),
]


# ── Check functions ─────────────────────────────────────────────────────

def check_project_scan_trigger(text: str) -> Tier0RuleResult:
    """Check if the message is a project architecture scan command."""
    for pattern in _SCAN_PROJECT_PATTERNS:
        if pattern.search(text):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.SCAN_PROJECT_ARCHITECTURE,
                confidence=0.9,
                rule_name="project_scan_trigger",
                reason=f"Matched project scan pattern: {pattern.pattern}",
            )
    return Tier0RuleResult(matched=False)


def check_register_project_trigger(text: str) -> Tier0RuleResult:
    """Check if the message is a register project command."""
    for pattern in _REGISTER_PATTERNS:
        if pattern.search(text):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.REGISTER_PROJECT,
                confidence=0.9,
                rule_name="register_project_trigger",
                reason=f"Matched register project pattern: {pattern.pattern}",
            )
    return Tier0RuleResult(matched=False)


def check_list_projects_trigger(text: str) -> Tier0RuleResult:
    """Check if the message is a list projects query."""
    for pattern in _LIST_PATTERNS:
        if pattern.search(text):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.LIST_REGISTERED_PROJECTS,
                confidence=0.9,
                rule_name="list_projects_trigger",
                reason=f"Matched list projects pattern: {pattern.pattern}",
            )
    return Tier0RuleResult(matched=False)