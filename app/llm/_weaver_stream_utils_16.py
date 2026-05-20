from __future__ import annotations
import logging
import re
from app.llm._weaver_stream_utils_15 import DESIGN_PREF_WHITELIST_PATTERNS, REFACTOR_ACTION_PATTERNS
logger = logging.getLogger(__name__)


VISION_CONTEXT_PATTERNS = [
    # Screenshot/image descriptions
    r"screenshot",
    r"image shows",
    r"i can see",
    r"i see a",
    r"the image",
    r"in the picture",
    r"looking at the",
    # UI element descriptions (from vision analysis)
    r"title bar",
    r"window title",
    r"menu bar",
    r"status bar",
    r"status indicator",
    r"toolbar",
    r"heading.*says",
    r"button.*labeled",
    r"text.*reads",
    r"displays.*text",
    r"shows.*logo",
    r"cyan.*text",
    r"blue.*text",
    r"icon.*shows",
    # Visual descriptions
    r"ui shows",
    r"ui elements",
    r"visible.*elements",
    r"display shows",
    r"interface shows",
    r"window shows",
    r"window contains",
    # Color/appearance descriptions
    r"dark\s*(?:theme|mode|background)",
    r"light\s*(?:theme|mode|background)",
    r"colored.*(?:text|background|border)",
    # Position descriptions
    r"top.*(?:left|right|corner)",
    r"bottom.*(?:left|right|corner)",
    r"center of",
    r"sidebar",
    # Action analysis phrases
    r"appears to be",
    r"looks like",
    r"seems to show",
]

def _is_vision_context(content: str) -> bool:
    """
    Detect if an assistant message contains vision/image analysis.
    
    v3.9.0: Vision analysis from Gemini should NOT be filtered out.
    This context is valuable for downstream stages (SpecGate classifier)
    to understand which matches are USER-VISIBLE UI elements.
    
    Returns True if the message likely contains vision analysis.
    """
    if not content:
        return False
    
    content_lower = content.lower()
    
    # Check for vision context patterns
    for pattern in VISION_CONTEXT_PATTERNS:
        if re.search(pattern, content_lower, re.IGNORECASE):
            print(f"[WEAVER] v3.9 Vision context detected (pattern: {pattern[:30]}...)")
            return True
    
    return False

DESIGN_PREF_BLACKLIST_PATTERNS = [
    # Calculations & logic
    r"\bcalculat", r"\bcompute\b", r"\bformula\b", r"\baverag",
    r"\bper\s*(day|week|parcel)\b", r"\b/\s*day\b", r"\bdaily\s+cost\b",
    # Data handling
    r"\bsync\b", r"\bexport\b", r"\bimport\b", r"\bapi\b",
    r"\btrack\b", r"\brecord\b", r"\blog\b", r"\binput\b",
    r"\bextract", r"\bformat\b", r"\bpars",  # parse, parsing
    # Screenshot / OCR
    r"\bocr\b", r"\bscreenshot", r"\bphoto\b", r"\bimage\b",
    r"\bdetect", r"\brecogni",  # detect, detection, recognize, recognition
    # Business / financial
    r"\bprofit\b", r"\bpay\b", r"\bfuel\b", r"\bwear\b", r"\bcost\b",
    r"\bparcel\b", r"\bdelivery\b", r"\bdeliveries\b", r"\bearning",
    # UI elements that are functional, not visual
    r"\bhistory\s*(list|row|screen)\b", r"\bshow\s*(gross|net|costs?)\b",
    r"\bimport\s*button\b", r"\bstart\s*day\b", r"\bfinish\s*day\b",
    # Workflow / method preferences
    r"\bworkflow\b", r"\bmethod\b", r"\bhandling\b", r"\bmodel\b",
    r"\bpriority\b", r"\bpriorities\b", r"\bprefer\b",
    r"\bauto\b", r"\bautomat",  # auto, automatic, automatically
    # Integration / system
    r"\bastra\b", r"\bweekly\s*breakdown\b",
    r"\bno\s*(manual|export|chart)\b",  # functional constraints
    # Misc functional terms
    r"\bvs\.?\b", r"\bversus\b",  # "X vs Y" is a functional choice
    r"\brequire", r"\bshould\b", r"\bmust\b",  # requirement language
]

def _enforce_design_pref_hygiene(output: str) -> str:
    """
    Enforce section hygiene: Design preferences should only contain visual/UI prefs.
    
    v3.4.2: Removes functional requirements that were incorrectly bucketed into 
    Design preferences. This prevents duplication across sections during UPDATE merges.
    
    Keeps: color, layout, style, "big buttons", "no clutter", "dead simple"
    Removes: calculations, sync rules, tracking, OCR, profit/pay/fuel, history contents
    """
    lines = output.split("\n")
    result_lines = []
    in_design_section = False
    removed_count = 0
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Detect section headers (various formats)
        is_design_header = any([
            line_lower.startswith("design preferences"),
            line_lower.startswith("**design preferences"),
            line_lower.startswith("## design preferences"),
            line_lower.startswith("### design preferences"),
        ])
        
        # Detect other section headers (to know when we've left design prefs)
        is_other_header = any([
            line_lower.startswith("constraints"),
            line_lower.startswith("**constraints"),
            line_lower.startswith("platform"),
            line_lower.startswith("**platform"),
            line_lower.startswith("priority"),
            line_lower.startswith("**priority"),
            line_lower.startswith("unresolved"),
            line_lower.startswith("**unresolved"),
            line_lower.startswith("intended outcome"),
            line_lower.startswith("**intended outcome"),
            line_lower.startswith("what is being"),
            line_lower.startswith("**what is being"),
            line_lower.startswith("execution mode"),
            line_lower.startswith("**execution mode"),
            line_lower.startswith("specgate must resolve"),
            line_lower.startswith("**specgate must resolve"),
            line_lower.startswith("questions for user"),
            line_lower.startswith("**questions for user"),
        ])
        
        if is_design_header:
            in_design_section = True
            result_lines.append(line)
            continue
        
        if is_other_header and in_design_section:
            in_design_section = False
        
        if in_design_section and line_lower:
            # Skip structural/formatting lines - keep them
            if line_lower in ["---", "***", "___"] or line_lower.startswith("(if "):
                result_lines.append(line)
                continue
            
            # Check if this line belongs in design preferences
            has_whitelist = any(re.search(p, line_lower) for p in DESIGN_PREF_WHITELIST_PATTERNS)
            has_blacklist = any(re.search(p, line_lower) for p in DESIGN_PREF_BLACKLIST_PATTERNS)
            
            # Keep line if it's a valid preference line (starts with Color:/Layout:/Style:)
            is_valid_pref_line = any([
                line_lower.startswith("color"),
                line_lower.startswith("- color"),
                line_lower.startswith("* color"),
                line_lower.startswith("layout"),
                line_lower.startswith("- layout"),
                line_lower.startswith("* layout"),
                line_lower.startswith("style"),
                line_lower.startswith("- style"),
                line_lower.startswith("* style"),
                line_lower.startswith("ui element"),
                line_lower.startswith("- ui element"),
                line_lower.startswith("* ui element"),
            ])
            
            # v3.4.2 logic: Be strict about what stays in Design preferences
            # - Valid pref line (Color:/Layout:/Style:) -> KEEP
            # - Whitelist match without blacklist -> KEEP  
            # - Blacklist match -> REMOVE
            # - Neither (ambiguous) -> REMOVE (stricter than before)
            if is_valid_pref_line:
                result_lines.append(line)
            elif has_whitelist and not has_blacklist:
                result_lines.append(line)
            elif has_blacklist:
                # Skip this line - it's a functional requirement
                removed_count += 1
                preview = line.strip()[:60]
                print(f"[WEAVER] Removed from Design prefs (functional): {preview}...")
            else:
                # Ambiguous line in Design prefs - remove it (be strict)
                # This catches things that don't match whitelist visual patterns
                removed_count += 1
                preview = line.strip()[:60]
                print(f"[WEAVER] Removed from Design prefs (ambiguous): {preview}...")
        else:
            result_lines.append(line)
    
    if removed_count > 0:
        print(f"[WEAVER] Design pref hygiene: removed {removed_count} functional requirement(s)")
    
    return "\n".join(result_lines)

CONCRETE_TARGETS = [
    "app", "application", "website", "page", "component", "feature", "function",
    "api", "endpoint", "service", "database", "table", "file", "folder", "code",
    "script", "module", "class", "method", "button", "form", "ui", "interface",
    "dashboard", "panel", "modal", "menu", "navbar", "sidebar", "widget",
    "message", "email", "reply", "response", "document", "report", "spec",
    "tracker", "tool", "integration", "screen", "overlay", "plan", "flow",
    "logger", "monitor", "viewer", "editor", "builder", "generator",
    # Creative/project targets (v3.5.0 - Bug 1 fix)
    "game", "prototype", "demo", "simulator", "visualizer", "calculator",
    "timer", "clock", "todo", "calendar", "planner", "clone", "replica",
]

REFACTOR_TASK_SYSTEM_PROMPT = """You are Weaver for REFACTOR/RENAME TASKS.

Your job: Produce a FOCUSED job outline for text replacement / rename operations.

## CRITICAL RULES FOR REFACTOR TASKS:
1. NO design questions (dark mode, light mode, controls, layout, etc.) - IRRELEVANT
2. NO UI/UX questions - this is a TEXT REPLACEMENT task
3. NO platform questions - the pipeline knows the platform
4. Focus ONLY on: what to search, what to replace, where to search
5. Questions section should say "none" - the pipeline handles discovery

## WHAT TO EXTRACT:
- Search term: What text/string to find (e.g., "Orb")
- Replace term: What to replace it with (e.g., "Astra")
- Scope: Where to search (folder path, file types)
- Constraints: What NOT to change (e.g., no logos, text-only)

## OUTPUT FORMAT:

What is being built: [Short description] (refactor task)
Intent: Rename/replace "[SEARCH]" with "[REPLACE]" in [SCOPE]
Execution type: REFACTOR_TASK
Search term: [exact text to find]
Replace term: [exact text to replace with]
Scope: [folder/path to search]
Constraints:
- [constraint 1]
- [constraint 2]
Questions: none

## EXAMPLE:

Input: "Change the front-end UI so it's called Astra instead of Orb. Look in Orb Desktop on D drive. Text only, no logos."

Output:
What is being built: Text rebrand from Orb to Astra (refactor task)
Intent: Rename all occurrences of "Orb" to "Astra" in D:\\Orb Desktop front-end files
Execution type: REFACTOR_TASK
Search term: Orb (case-preserving: Orb→Astra, ORB→ASTRA, orb→astra)
Replace term: Astra
Scope: D:\\Orb Desktop (front-end UI files)
Constraints:
- Text-only changes (no logos or icons)
- Case-preserving replacement
- Front-end UI files only
Questions: none

CRITICAL:
- Keep output under 20 lines
- Questions section MUST say "none" - design questions are NEVER relevant for refactor tasks
- The Implementer will handle file discovery and show matches for confirmation
- DO NOT ask about colors, themes, controls, layout, scope preferences, etc."""

def _is_refactor_task(text: str) -> bool:
    """
    Detect refactor/rename operations that need special handling.
    
    v3.10: Now uses PATTERN-BASED detection instead of keyword matching.
    Requires actual rename/replace ACTION + SCOPE/TARGET context.
    
    This prevents false positives like:
    - "Add voice-to-text to the ASTRA desktop app" (mentions app name)
    - "Update the branding page" (mentions branding as a feature)
    - "Improve the front-end UI" (mentions UI as a feature target)
    
    Only triggers on actual refactor language like:
    - "Rename Orb to Astra across the codebase"
    - "Replace all occurrences of X with Y"
    - "Rebrand from Orb to Astra"
    - "Find and replace in all files"

    v3.11 (2026-04-17): Added size gate. A genuine pure-rename request is
    always short. If the input is > MAX_REFACTOR_INPUT_LEN chars, the input
    is a capability spec / design doc / project plan that happens to contain
    a rename pattern somewhere in it — NOT a refactor task. This prevents
    the whole document being thrown away because one section mentions
    renaming a nav label.

    Returns True if a refactor ACTION PATTERN matches AND the input is
    short enough to plausibly be a refactor request.
    """
    # v3.11 size gate — real rename requests are short.
    MAX_REFACTOR_INPUT_LEN = 4000
    if len(text) > MAX_REFACTOR_INPUT_LEN:
        print(
            f"[WEAVER] v3.11 NOT refactor task — input too long "
            f"({len(text)} > {MAX_REFACTOR_INPUT_LEN} chars). Refactor patterns "
            f"in large documents are almost always incidental references, "
            f"not the primary intent."
        )
        return False

    text_lower = text.lower()
    
    for pattern in REFACTOR_ACTION_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            matched_text = match.group(0).strip()
            print(f"[WEAVER] v3.10 REFACTOR_TASK detected (pattern: '{matched_text}')")
            return True
    
    print(f"[WEAVER] v3.10 NOT refactor task (no action patterns matched)")
    return False

MICRO_TASK_SYSTEM_PROMPT = """You are Weaver for MICRO FILE TASKS.

Your job: Produce a SHORT, minimal job outline (10-20 lines max) for simple file operations.

## ABSOLUTE RULES FOR MICRO FILE TASKS:
1. NO questions about OS/platform - it's always Windows
2. NO questions about desktop location - there's only one accessible
3. NO questions about file extensions - the system will search
4. NO questions about paths - the system will find them
5. NO questions about file format - default is plain text (.txt)
6. NO questions about overwriting - default is overwrite if exists
7. NO questions about exact filenames - the system searches
8. Questions section should say "none" unless execution would truly FAIL

## THE ONLY BLOCKING QUESTIONS (rare):
- DELETE operations need confirmation ("Should I really delete X?")
- MOVE without destination needs clarification ("Where to?")
- NOTHING ELSE is a blocker

## OUTPUT FORMAT (keep it short!):

What is being built: [Short description] (micro file task)
Intent: [One line - what to find and what to do with it]
Execution type: MICRO_FILE_TASK
Planned steps:
- [Step 1: Locate]
- [Step 2: Action]
- [Step 3: Output/Return]
Questions: none

That's the entire output. No ambiguities section. No extra sections.

## EXAMPLES:

Input: "Find test1, test2, test3, test4 on my system, read them, create reply file on desktop, write a reply"
Output:
What is being built: Multi-file reader with reply synthesis (micro file task)
Intent: Find test1-test4 anywhere on system, read content, create Desktop/reply.txt with response
Execution type: MICRO_FILE_TASK
Planned steps:
- System-wide search for test1, test2, test3, test4
- Read all found files
- Synthesize content into a reply
- Create Desktop/reply.txt with synthesized response
Questions: none

CRITICAL: 
- Keep output under 15 lines
- Questions section must say "none" unless DELETE or unclear MOVE destination
- SpecGate handles all file discovery - don't ask about paths/filenames/locations"""
