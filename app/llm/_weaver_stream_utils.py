from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List


def _serialize_sse(data: Dict[str, Any]) -> bytes:
    """Serialize dict to SSE format."""
    return f"data: {json.dumps(data)}\n\n".encode("utf-8")

def _get_weaver_config() -> tuple[str, str]:
    """Get provider and model for weaver from environment."""
    provider = os.getenv("WEAVER_PROVIDER", "openai")
    model = os.getenv("WEAVER_MODEL", "gpt-4.1-mini")
    return provider, model

def _is_control_message(role: str, content: str) -> bool:
    """Check if message is a control/system message to skip."""
    c = (content or "").strip()
    rl = (role or "").strip().lower()
    
    if not c:
        return True
    
    if rl == "system":
        return True
    
    # Skip command triggers
    if rl == "user":
        lc = c.lower()
        if any(lc.startswith(prefix) for prefix in [
            "astra, command:", "astra command:", "astra, cmd:", "orb, command:",
            "how does that look all together",
        ]):
            return True
    
    # Skip Weaver/Orb output messages
    if rl in ("assistant", "orb"):
        markers = (
            "🧵 weaving", "📋 spec", "📋 job description",
            "shall i send", "say yes to proceed", "⚠️ weak spots",
            "ready for spec gate", "provenance",
            "🎨 design preferences", "design preferences needed",
            "🎨 got it", "🎨 perfect",
        )
        lc = c.lower()
        if any(m in lc for m in markers):
            return True
    
    return False

def _format_ramble(messages: List[Dict[str, Any]]) -> str:
    """Format messages into a ramble text block."""
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if not content:
            continue
        speaker = "Human" if role == "user" else "Assistant"
        lines.append(f"[{speaker}]: {content}")
    return "\n\n".join(lines)

def _format_execution_mode(extracted_modes: List[str]) -> str:
    """
    Format extracted meta-mode phrases into a clean execution mode string.
    """
    if not extracted_modes:
        return ""
    
    # Normalize and deduplicate
    normalized = []
    for mode in extracted_modes:
        mode_lower = mode.lower().strip()
        if "no code" in mode_lower or "don't build" in mode_lower:
            if "No coding yet" not in normalized:
                normalized.append("No coding yet")
        elif "talk about" in mode_lower or "discuss" in mode_lower:
            if "Discussion only" not in normalized:
                normalized.append("Discussion only")
        elif "planning" in mode_lower:
            if "Planning phase" not in normalized:
                normalized.append("Planning phase")
        elif "questions first" in mode_lower or "don't assume" in mode_lower:
            if "Clarification needed first" not in normalized:
                normalized.append("Clarification needed first")
        else:
            # Keep as-is if no normalization rule
            cap_mode = mode.strip().capitalize()
            if cap_mode not in normalized:
                normalized.append(cap_mode)
    
    return ", ".join(normalized) if normalized else ""

def _enforce_deduplication(output: str) -> str:
    """
    Enforce deduplication: Same sentence should not appear in multiple sections (Bug 3 fix).
    
    Specifically checks if "What is being built" and "Intended outcome" are identical
    or near-identical, and rewrites Outcome if so.
    """
    lines = output.split("\n")
    
    # Extract What and Outcome values
    what_value = ""
    outcome_value = ""
    what_line_idx = -1
    outcome_line_idx = -1
    
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        
        # Look for "What is being built" section
        if line_lower.startswith("what is being built") or line_lower.startswith("**what is being built"):
            # Value is on same line after colon, or next line
            if ":" in line:
                what_value = line.split(":", 1)[1].strip()
                what_line_idx = i
            elif i + 1 < len(lines):
                what_value = lines[i + 1].strip()
                what_line_idx = i + 1
        
        # Look for "Intended outcome" section
        elif line_lower.startswith("intended outcome") or line_lower.startswith("**intended outcome"):
            if ":" in line:
                outcome_value = line.split(":", 1)[1].strip()
                outcome_line_idx = i
            elif i + 1 < len(lines):
                outcome_value = lines[i + 1].strip()
                outcome_line_idx = i + 1
    
    # Check for duplication
    if what_value and outcome_value:
        # Normalize for comparison (lowercase, strip punctuation)
        what_normalized = re.sub(r'[^\w\s]', '', what_value.lower()).strip()
        outcome_normalized = re.sub(r'[^\w\s]', '', outcome_value.lower()).strip()
        
        # Check if identical or very similar
        is_duplicate = (
            what_normalized == outcome_normalized or
            what_normalized in outcome_normalized or
            outcome_normalized in what_normalized
        )
        
        if is_duplicate and outcome_line_idx >= 0:
            print(f"[WEAVER] Deduplication: What and Outcome were identical/similar")
            # Rewrite outcome to be different
            # Simple heuristic: prepend "Functional" or "Working" + add "implementation"
            if ":" in lines[outcome_line_idx]:
                prefix = lines[outcome_line_idx].split(":")[0] + ":"
                lines[outcome_line_idx] = f"{prefix} Functional {what_value.lower()} implementation"
            else:
                lines[outcome_line_idx] = f"Functional {what_value.lower()} implementation"
            print(f"[WEAVER] Rewrote Outcome to: {lines[outcome_line_idx]}")
    
    return "\n".join(lines)

_SLOT_RECONCILIATION_REMOVED = True  # Marker for grep/search

def _get_blocking_questions(text: str, is_micro_task: bool) -> List[str]:
    """
    Only return questions that would BLOCK execution (v3.6.0).
    
    For micro tasks:
    - read + write is NOT a conflict (normal output flow)
    - delete IS a blocker (dangerous, must confirm)
    - move/copy without destination IS a blocker
    
    For non-micro tasks, returns empty (uses existing shallow question logic).
    """
    if not is_micro_task:
        return []  # Non-micro tasks use existing shallow question logic
    
    text_lower = text.lower()
    questions = []
    
    # Check for ACTUALLY conflicting/dangerous actions
    has_delete = any(w in text_lower for w in ["delete", "remove", "erase"])
    has_move = any(w in text_lower for w in ["move", "copy", "transfer"])
    
    # Blocker: delete is mentioned (dangerous, must confirm)
    if has_delete:
        questions.append("You mentioned deleting - should I delete the file, or just read it?")
    
    # Blocker: move/copy with unclear destination
    if has_move:
        # Check if destination is specified
        has_destination = any(w in text_lower for w in ["to ", "into ", "destination"])
        if not has_destination:
            questions.append("Where should I move/copy the file to?")
    
    # NON-blockers (do NOT ask):
    # - read + write (normal output flow)
    # - read + answer/reply (normal response flow)
    # - OS/platform (sandbox handles it)
    # - Which desktop (only one accessible)
    # - Exact filename (search and pick)
    # - Multiple files (use default selection rules)
    
    return questions
