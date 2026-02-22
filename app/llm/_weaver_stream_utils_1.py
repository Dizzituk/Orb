from __future__ import annotations
from typing import Any, Dict, List, Set


def _hash_messages(messages: List[Dict[str, Any]]) -> Set[str]:
    """Hash a list of messages, returning a set of hashes."""
    return {_hash_message(m) for m in messages}

LEAKAGE_PATTERNS = [
    r"^EXISTING JOB DESCRIPTION:\s*",
    r"^---\s*$",
    r"^NEW USER REQUIREMENTS.*?:\s*",
    r"^NEW USER MESSAGE.*?:\s*",
    r"^PREVIOUS SPEC:\s*",
    r"^UPDATED JOB DESCRIPTION:\s*",
]

INTENT_GOAL_PATTERNS = [
    r"\bi\s+want\b",
    r"\bi\s+need\b",
    r"\bi'm\s+trying\b",
    r"\bi\s+am\s+trying\b",
    r"\bi'd\s+like\b",
    r"\bi\s+would\s+like\b",
]

NEGATION_PATTERNS = [
    r"\bdon'?t\s+",
    r"\bdo\s+not\s+",
    r"\bnever\s+",
    r"\bno\s+need\s+to\s+",
    r"\bwithout\s+",
    r"\bavoid\s+",
    r"\bskip\s+",
]

MICRO_FILE_INDICATORS = [
    "desktop", "folder", "file", "txt", "read", "open", "find",
    "answer", "reply", "write", "document", "documents", "message",
    "look at", "check", "locate", "search",
]

NON_MICRO_INDICATORS = [
    "app", "application", "website", "page", "component", "feature",
    "game", "dashboard", "ui", "interface", "api", "endpoint", "service",
    "database", "design", "develop", "implement",
    "prototype", "demo", "refactor", "restructure", "migrate",
]

REFACTOR_INDICATORS = [
    "rename", "rebrand", "refactor", "replace all", "change all",
    "all occurrences", "codebase",
    # NOTE: "astra", "orb to astra", "branding", "front-end ui",
    # "across", "everywhere" were REMOVED in v3.10 — too generic,
    # caused false positives on any message mentioning the app name.
]

BUILD_VERBS = [
    "build", "create", "make", "develop", "implement", "prototype",
]
