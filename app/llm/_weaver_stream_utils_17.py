from __future__ import annotations
import logging
import re
from app.llm._weaver_stream_utils_13 import INTENT_GOAL_PATTERNS, NEGATION_PATTERNS
from app.llm._weaver_stream_utils_14 import CORE_GOAL_VERBS
from app.llm._weaver_stream_utils_16 import CONCRETE_TARGETS
logger = logging.getLogger(__name__)


CORE_GOAL_TARGETS = [
    # Technical targets
    "app", "application", "website", "page", "component", "feature", "function",
    "api", "endpoint", "service", "database", "table", "file", "folder", "code",
    "script", "module", "class", "method", "button", "form", "ui", "interface",
    "dashboard", "panel", "modal", "menu", "navbar", "sidebar", "widget",
    # Content targets
    "message", "email", "reply", "response", "document", "report", "spec",
    "text", "content", "data", "config", "settings", "theme", "style", "layout",
    # Project targets (v3.4.1)
    "tracker", "tool", "integration", "screen", "overlay", "plan", "flow",
    "logger", "monitor", "viewer", "editor", "builder", "generator",
    # Creative/project targets (v3.5.0 - Bug 1 fix)
    "game", "prototype", "demo", "simulator", "visualizer", "calculator",
    "timer", "clock", "todo", "calendar", "planner", "clone", "replica",
    # Abstract targets (only valid with action verbs, not intent patterns)
    "it", "this", "that", "one", "something", "thing", "system", "process",
]

def _has_core_goal(ramble_text: str) -> bool:
    """
    Check if ramble has a clear action/goal using 2-factor heuristic.
    
    v3.5.0: Now recognizes creative targets like "game", "prototype", "demo".
    
    Logic:
    - PASS if: (action_verb + any_target) OR (intent_pattern + concrete_target)
    - FAIL if: negated OR no valid pattern found
    
    NOTE: In v3.5.0, even if this returns False, Weaver should STILL produce
    a structured outline with ambiguities listed. This function is now used
    only for logging/debugging, not for blocking weave.
    """
    text_lower = ramble_text.lower()
    
    # --- Check for ACTION VERB + TARGET ---
    has_action_verb = False
    for verb in CORE_GOAL_VERBS:
        verb_pattern = rf"\b{re.escape(verb)}\b"
        verb_match = re.search(verb_pattern, text_lower)
        
        if not verb_match:
            continue
        
        verb_pos = verb_match.start()
        prefix_start = max(0, verb_pos - 20)
        prefix = text_lower[prefix_start:verb_pos]
        
        is_negated = any(re.search(neg, prefix) for neg in NEGATION_PATTERNS)
        
        if not is_negated:
            has_action_verb = True
            break
    
    if has_action_verb:
        # Action verb found - check for ANY target (including abstract ones)
        for target in CORE_GOAL_TARGETS:
            target_pattern = rf"\b{re.escape(target)}\b"
            if re.search(target_pattern, text_lower):
                print("[WEAVER] Core goal detected (action verb + target)")
                return True
    
    # --- Check for INTENT PATTERN + CONCRETE TARGET ---
    has_intent_pattern = False
    intent_negated = False
    
    for pattern in INTENT_GOAL_PATTERNS:
        intent_match = re.search(pattern, text_lower)
        if intent_match:
            # Check for negation before the intent pattern
            intent_pos = intent_match.start()
            prefix_start = max(0, intent_pos - 20)
            prefix = text_lower[prefix_start:intent_pos]
            
            if any(re.search(neg, prefix) for neg in NEGATION_PATTERNS):
                intent_negated = True
                continue
            
            has_intent_pattern = True
            break
    
    if has_intent_pattern and not intent_negated:
        # Intent pattern found - check for CONCRETE targets only
        # (prevents "I want something" from passing)
        for target in CONCRETE_TARGETS:
            target_pattern = rf"\b{re.escape(target)}\b"
            if re.search(target_pattern, text_lower):
                print(f"[WEAVER] Core goal detected (intent pattern + concrete target: '{target}')")
                return True
    
    # No valid goal pattern found
    if has_action_verb:
        print("[WEAVER] Action verb found but no target")
    elif has_intent_pattern:
        print("[WEAVER] Intent pattern found but no concrete target")
    else:
        print("[WEAVER] No action verb or intent pattern found")
    
    return False
