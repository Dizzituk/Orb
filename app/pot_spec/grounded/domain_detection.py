# FILE: app/pot_spec/grounded/domain_detection.py
"""
Domain Detection and Decision Fork System (v1.2, v1.10, v1.12, v1.18)

Detects job domains (mobile_app, greenfield_build, game, scan_only, sandbox_file)
and extracts bounded decision fork questions.

Version Notes:
-------------
v1.2 (2026-01): Decision forks replace lazy questions
v1.10 (2026-01): Added greenfield_build domain
v1.12 (2026-01): Domain drift fix - excludes question text from detection
v1.18 (2026-01): Added scan_only domain for filesystem scans
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

from .spec_models import (
    GroundedQuestion,
    GroundedAssumption,
    QuestionCategory,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DOMAIN KEYWORDS (v1.10, v1.12, v1.18)
# =============================================================================

DOMAIN_KEYWORDS = {
    "mobile_app": [
        "mobile app", "phone app", "android", "ios", "iphone",
        "offline-first", "offline first", "sync", "ocr", "screenshot",
        "voice", "push-to-talk", "push to talk", "wake word", "wakeword",
        "encryption", "encrypted", "trusted wi-fi", "trusted wifi",
        "in-van", "in van", "delivery", "parcels", "shift",
    ],
    "greenfield_build": [
        "build me", "make me", "create a", "build a", "make a",
        "i want a", "i need a", "i'd like a", "i would like a",
        "new app", "new project", "new game", "new tool",
        "tetris", "snake", "pong", "game", "clone", "prototype",
        "from scratch", "brand new", "greenfield",
    ],
    "game": [
        "tetris", "snake", "pong", "breakout", "asteroids",
        "game", "gameplay", "playfield", "playable",
        "tetrominoes", "tetromino", "grid", "board",
        "score", "level", "high score", "game over",
        "player", "controls", "line clear", "line clearing",
        "gravity", "drop", "hard drop", "soft drop",
        "rotation", "rotate", "spawn", "next piece",
        "arcade", "puzzle game", "casual game",
        "classic game", "retro game", "web game",
    ],
    "scan_only": [
        "scan the", "scan all", "scan for", "scan folder", "scan folders",
        "scan drive", "scan d:", "scan c:", "scan directory", "scan directories",
        "scan entire", "scan whole", "scan project", "scan codebase",
        "find all occurrences", "find all references", "find all files",
        "find all folders", "find all instances", "find all mentions",
        "search for", "search the", "search entire", "search across",
        "search all", "search whole", "search project",
        "list all", "list references", "list files", "list folders",
        "enumerate", "report full paths", "report all",
        "where is", "locate all", "show me all",
        "references to", "mentions of", "occurrences of",
        "and d:", "and c:", "both d:", "both c:",
    ],
    "sandbox_file": [
        "sandbox desktop", "sandbox task", "sandbox folder",
        "in the sandbox", "from sandbox",
        "find the file", "find file", "find by discovery", "discovery",
        "read the file", "read file", "read the question", "read the message",
        "open the file", "open file",
        "reply to file", "reply to the", "include the reply", "include reply",
        "respond to file", "answer the file",
        "desktop folder", "folder on desktop", "folder called", "folder named",
        "documents folder", "folder in documents",
        "text file in", "file on the desktop", "file in desktop",
        "file on desktop", "file in documents",
        ".txt file", "test.txt", "message.txt",
    ],
}


# =============================================================================
# FORK QUESTION BANK (v1.2, v1.4)
# =============================================================================

MOBILE_APP_FORK_BANK = [
    {
        "id": "platform_v1",
        "question": "Platform for v1 release?",
        "why_it_matters": "Determines SDK choice, build tooling, and timeline. iOS adds ~40% development time.",
        "options": ["Android-only first", "Android + iOS from day 1"],
        "triggers": ["android", "ios", "iphone", "mobile", "phone"],
        "blocking": True,
        "default_value": None,
        "default_reason": None,
    },
    {
        "id": "offline_storage",
        "question": "Offline data storage approach?",
        "why_it_matters": "Affects data model complexity, encryption implementation, and sync conflict resolution.",
        "options": [
            "Room/SQLite + SQLCipher (structured, queryable)",
            "Encrypted file store (JSON + crypto, simpler but less queryable)",
        ],
        "triggers": ["offline", "encryption", "encrypted", "storage", "local"],
        "blocking": False,
        "default_value": "Local encrypted storage (SQLite + encryption for v1)",
        "default_reason": "Standard v1 approach; can upgrade to SQLCipher later if needed",
    },
    {
        "id": "input_mode_v1",
        "question": "Primary input mode for v1?",
        "why_it_matters": "Voice requires speech-to-text integration and error handling. Manual is simpler but slower in-van.",
        "options": [
            "Push-to-talk voice + manual fallback",
            "Voice-only (no manual entry)",
            "Manual-only (voice deferred to v2)",
        ],
        "triggers": ["voice", "push-to-talk", "push to talk", "manual", "input", "talk"],
        "blocking": True,
        "default_value": None,
        "default_reason": None,
    },
    {
        "id": "ocr_scope_v1",
        "question": "Screenshot OCR scope for v1?",
        "why_it_matters": "Multiple formats require more OCR training/templates. Single format is faster to ship.",
        "options": [
            "Finish Tour screenshot only",
            "Multiple screen formats (Finish Tour + route summary + others)",
        ],
        "triggers": ["ocr", "screenshot", "parse", "finish tour", "scan"],
        "blocking": True,
        "default_value": None,
        "default_reason": None,
    },
    {
        "id": "sync_behaviour",
        "question": "Data sync behaviour?",
        "why_it_matters": "Auto-sync needs background service and battery optimization. Manual is simpler but requires user action.",
        "options": [
            "Manual sync only (user taps 'Sync now')",
            "Auto-sync on trusted Wi-Fi",
            "Both (manual + optional auto on trusted Wi-Fi)",
        ],
        "triggers": ["sync", "wi-fi", "wifi", "upload", "background"],
        "blocking": False,
        "default_value": "Local-only for v1 (no cloud sync)",
        "default_reason": "v1 focus is local capture; sync deferred or via export",
    },
    {
        "id": "sync_target",
        "question": "Sync target for v1?",
        "why_it_matters": "Live endpoint requires server setup and auth. File export is portable but no real-time.",
        "options": [
            "Private ASTRA endpoint over LAN/VPN",
            "Export/import file only (no live endpoint yet)",
        ],
        "triggers": ["astra", "endpoint", "server", "export", "private", "lan", "vpn"],
        "blocking": False,
        "default_value": "Export file for ASTRA integration (no live endpoint for v1)",
        "default_reason": "ASTRA integration via file export/import; live endpoint deferred",
    },
    {
        "id": "knockon_tracking",
        "question": "Knock-on day tracking method?",
        "why_it_matters": "Inferred rules need pattern detection logic. Manual toggle is explicit but requires user discipline.",
        "options": [
            "Manual toggle per day ('Today is a knock-on day: yes/no')",
            "Inferred from Tue/Thu pattern (auto-detected)",
        ],
        "triggers": ["knock-on", "knockon", "knock on", "tuesday", "thursday", "reschedule"],
        "blocking": False,
        "default_value": "Manual toggle per day",
        "default_reason": "Simpler for v1; pattern detection can be added later",
    },
    {
        "id": "pay_variation",
        "question": "Pay-per-parcel variation handling?",
        "why_it_matters": "Daily override needs UI for quick rate changes. Fixed default is simpler but less accurate.",
        "options": [
            "Default rate with quick voice/tap override ('Pay is 2.00 today')",
            "Forced daily confirmation before shift start",
        ],
        "triggers": ["pay", "rate", "parcel", "1.85", "2.00", "variation", "override"],
        "blocking": False,
        "default_value": "Default rate (£1.85) with optional override in settings",
        "default_reason": "Use provided rate as default; override in settings if needed",
    },
]


# =============================================================================
# FUNCTIONS
# =============================================================================

def detect_domains(text: str, exclude_questions: bool = True) -> List[str]:
    """
    Detect which domains are mentioned in the text.
    
    v1.12: By default, excludes question/ambiguity sections from detection.
    
    Args:
        text: The text to analyze
        exclude_questions: If True, strips out Questions and Unresolved ambiguities sections
        
    Returns:
        List of domain keys (e.g., ["mobile_app"])
    """
    if not text:
        return []
    
    analysis_text = text
    if exclude_questions:
        analysis_text = _extract_job_description_only(text)
    
    text_lower = analysis_text.lower()
    detected = []
    
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                detected.append(domain)
                break
    
    return detected


def _extract_job_description_only(text: str) -> str:
    """
    v1.12: Extract only the job description parts of Weaver output,
    excluding Questions and Unresolved ambiguities sections.
    """
    if not text:
        return ""
    
    lines = text.split("\n")
    result_lines = []
    in_excluded_section = False
    
    excluded_headers = [
        "questions",
        "unresolved ambiguities",
        "unresolved ambigu",
    ]
    
    included_headers = [
        "what is being built",
        "intended outcome",
        "target platform:",
        "color scheme:",
        "scope:",
        "layout:",
        "platform:",
    ]
    
    for line in lines:
        line_lower = line.lower().strip()
        
        if any(header in line_lower for header in excluded_headers):
            in_excluded_section = True
            continue
        
        if any(header in line_lower for header in included_headers):
            in_excluded_section = False
            result_lines.append(line)
            continue
        
        if line.strip().startswith("#") or (line.strip().startswith("**") and line.strip().endswith("**")):
            if any(header in line_lower for header in excluded_headers):
                in_excluded_section = True
                continue
            else:
                in_excluded_section = False
        
        if not in_excluded_section:
            result_lines.append(line)
    
    result = "\n".join(result_lines)
    
    if len(result) != len(text):
        logger.info(
            "[domain_detection] v1.12 _extract_job_description_only: "
            "Excluded %d chars of question/ambiguity text (kept %d of %d chars)",
            len(text) - len(result), len(result), len(text)
        )
    
    return result


def extract_unresolved_ambiguities(weaver_text: str) -> List[str]:
    """Extract the 'Unresolved ambiguities' section from Weaver output."""
    if not weaver_text:
        return []
    
    ambiguities = []
    in_section = False
    
    for line in weaver_text.split("\n"):
        line_stripped = line.strip()
        line_lower = line_stripped.lower()
        
        if "unresolved ambigu" in line_lower:
            in_section = True
            continue
        
        if in_section:
            if line_stripped.startswith("#") or line_stripped.startswith("**") and not line_stripped.startswith("**-"):
                break
            if line_stripped.startswith("-") or line_stripped.startswith("*"):
                content = line_stripped.lstrip("-*").strip()
                if content:
                    ambiguities.append(content)
            elif line_stripped and not line_stripped.startswith("#"):
                ambiguities.append(line_stripped)
    
    return ambiguities


def extract_decision_forks(
    weaver_text: str,
    detected_domains: List[str],
    max_questions: int = 7,
) -> Tuple[List[GroundedQuestion], List[GroundedAssumption]]:
    """
    Extract bounded decision fork questions from Weaver text.
    
    v1.4: Only returns blocking questions; non-blocking get safe defaults.
    
    Args:
        weaver_text: Full Weaver job description text
        detected_domains: List of detected domain keys
        max_questions: Maximum questions to return
        
    Returns:
        Tuple of (blocking_questions, assumptions)
    """
    if not weaver_text:
        return [], []
    
    questions: List[GroundedQuestion] = []
    assumptions: List[GroundedAssumption] = []
    text_lower = weaver_text.lower()
    
    ambiguities = extract_unresolved_ambiguities(weaver_text)
    
    if "mobile_app" in detected_domains:
        for fork in MOBILE_APP_FORK_BANK:
            triggered = any(trigger in text_lower for trigger in fork["triggers"])
            
            if not triggered:
                continue
            
            is_blocking = fork.get("blocking", True)
            
            if is_blocking:
                evidence = "Detected from Weaver intent"
                for amb in ambiguities:
                    amb_lower = amb.lower()
                    if any(trigger in amb_lower for trigger in fork["triggers"]):
                        evidence = f"Weaver ambiguity: '{amb[:100]}...'" if len(amb) > 100 else f"Weaver ambiguity: '{amb}'"
                        break
                
                questions.append(GroundedQuestion(
                    question=fork["question"],
                    category=QuestionCategory.DECISION_FORK,
                    why_it_matters=fork["why_it_matters"],
                    evidence_found=evidence,
                    options=fork["options"],
                ))
                
                logger.info("[domain_detection] v1.4 BLOCKING question: %s", fork["question"])
            else:
                default_value = fork.get("default_value")
                default_reason = fork.get("default_reason")
                
                if default_value and default_reason:
                    assumptions.append(GroundedAssumption(
                        topic=fork["id"],
                        assumed_value=default_value,
                        reason=default_reason,
                        can_override=True,
                    ))
                    
                    logger.info(
                        "[domain_detection] v1.4 ASSUMED (not blocking): %s -> %s",
                        fork["id"], default_value
                    )
            
            if len(questions) >= max_questions:
                break
    
    return questions[:max_questions], assumptions
