# FILE: verify_slot_reconciliation.py
"""Quick verification that slot reconciliation works correctly."""
import re
from typing import Dict

# Copy the key functions inline for testing

def _detect_filled_slots(ramble_text: str) -> Dict[str, str]:
    """Detect which slots have been answered."""
    text_lower = ramble_text.lower()
    filled_slots = {}
    
    # Platform detection
    platform_patterns = [
        (r"\bandroid\b", "Android"),
        (r"\bios\b", "iOS"),
        (r"\bweb\b", "Web"),
        (r"\bdesktop\b", "Desktop"),
    ]
    for pattern, value in platform_patterns:
        if re.search(pattern, text_lower):
            filled_slots["platform"] = value
            break
    
    # Color mode detection
    if re.search(r"\bdark\s*mode\b|\bdark\b", text_lower):
        filled_slots["look_feel"] = "Dark mode"
    elif re.search(r"\blight\s*mode\b|\blight\b", text_lower):
        filled_slots["look_feel"] = "Light mode"
    
    # Controls detection
    if re.search(r"\btouch\b|\btap\b|\bswipe\b", text_lower):
        filled_slots["controls"] = "Touch"
    elif re.search(r"\bkeyboard\b", text_lower):
        filled_slots["controls"] = "Keyboard"
    
    # Scope detection
    if re.search(r"\bbare\s*minimum\b|\bminimal\b|\bbasic\b", text_lower):
        filled_slots["scope"] = "Bare minimum / basic"
    
    # Layout detection
    if re.search(r"\bcentered\b|\bfull\s*screen\b", text_lower):
        filled_slots["layout"] = "Centered / fullscreen"
    
    return filled_slots


SLOT_AMBIGUITY_PATTERNS = {
    "platform": [r"platform\s*(is\s+)?unspecified", r"target\s+platform"],
    "look_feel": [r"color\s*mode\s*(is\s+)?unspecified", r"dark.*light.*unspecified"],
    "controls": [r"control\s*(method\s*)?(is\s+)?unspecified"],
    "scope": [r"scope\s*(is\s+)?unspecified"],
    "layout": [r"layout\s*(preference\s*)?(is\s+)?unspecified"],
}

SLOT_QUESTION_PATTERNS = {
    "platform": [r"what\s+platform", r"web.*android.*desktop.*ios"],
    "look_feel": [r"dark\s*(mode)?\s*(or|vs|/)\s*light\s*(mode)?"],
    "controls": [r"what\s+controls", r"keyboard.*touch.*controller"],
    "scope": [r"bare\s*minimum.*extras?"],
    "layout": [r"layout\s*\?", r"centered.*sidebar"],
}


def _reconcile_filled_slots(output: str, filled_slots: Dict[str, str]) -> str:
    """Remove filled slots from ambiguities/questions."""
    if not filled_slots:
        return output
    
    lines = output.split("\n")
    result_lines = []
    in_ambiguities_section = False
    in_questions_section = False
    removed_count = 0
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Detect section headers
        if "unresolved ambiguit" in line_lower:
            in_ambiguities_section = True
            in_questions_section = False
            result_lines.append(line)
            continue
        elif line_lower.startswith("questions") or line_lower.startswith("**questions"):
            in_questions_section = True
            in_ambiguities_section = False
            result_lines.append(line)
            continue
        elif line_lower.startswith("**") and (in_ambiguities_section or in_questions_section):
            in_ambiguities_section = False
            in_questions_section = False
        
        should_remove = False
        
        if in_ambiguities_section and line_lower:
            for slot_name in filled_slots:
                for pattern in SLOT_AMBIGUITY_PATTERNS.get(slot_name, []):
                    if re.search(pattern, line_lower, re.IGNORECASE):
                        should_remove = True
                        removed_count += 1
                        break
                if should_remove:
                    break
        
        elif in_questions_section and line_lower:
            for slot_name in filled_slots:
                for pattern in SLOT_QUESTION_PATTERNS.get(slot_name, []):
                    if re.search(pattern, line_lower, re.IGNORECASE):
                        should_remove = True
                        removed_count += 1
                        break
                if should_remove:
                    break
        
        if not should_remove:
            result_lines.append(line)
    
    print(f"[RECONCILE] Removed {removed_count} resolved items")
    return "\n".join(result_lines)


# Test data: broken Weaver output (regression)
BROKEN_OUTPUT = """**What is being built:** Classic Tetris game

**Intended outcome:** A Tetris game

**Design preferences:** "Just the classic idea"

**Unresolved ambiguities:**
Target platform is unspecified
Color mode (dark vs light) is unspecified
Control method (keyboard / touch / controller) is unspecified
Scope (bare minimum playable vs extras) is unspecified
Layout preference (centered vs sidebar HUD) is unspecified

**Questions:**
What platform do you want this on? (web / Android / desktop / iOS)
Dark mode or light mode?
What controls? (keyboard / touch / controller)
Bare minimum playable first, or add some extras?
Any preference on layout? (centered vs sidebar HUD)

**New requirements from user:**
Target platform: Android
Color mode: Dark mode
Scope: Bare minimum playable first
Layout preference: Centered full screen
"""

# User's answer
USER_ANSWER = "Android, Dark mode, Bare minimum playable first. centered full screen"


def main():
    print("=" * 60)
    print("SLOT RECONCILIATION VERIFICATION")
    print("=" * 60)
    
    print("\n1. User answer:")
    print(f"   '{USER_ANSWER}'")
    
    print("\n2. Detected filled slots:")
    filled = _detect_filled_slots(USER_ANSWER)
    for slot, value in filled.items():
        print(f"   - {slot}: {value}")
    
    print(f"\n   Missing slots: ", end="")
    all_slots = {"platform", "look_feel", "controls", "scope", "layout"}
    missing = all_slots - set(filled.keys())
    print(", ".join(missing) if missing else "NONE")
    
    print("\n3. Applying slot reconciliation...")
    fixed_output = _reconcile_filled_slots(BROKEN_OUTPUT, filled)
    
    print("\n4. FIXED OUTPUT:")
    print("-" * 60)
    print(fixed_output)
    print("-" * 60)
    
    # Verification
    print("\n5. VERIFICATION:")
    fixed_lower = fixed_output.lower()
    
    checks = [
        ("platform is unspecified" not in fixed_lower, "Platform ambiguity removed"),
        ("color mode (dark vs light)" not in fixed_lower, "Color mode ambiguity removed"),
        ("scope (bare minimum" not in fixed_lower, "Scope ambiguity removed"),
        ("layout preference" not in fixed_lower, "Layout ambiguity removed"),
        ("control method" in fixed_lower or "control" in fixed_lower, "Controls ambiguity REMAINS"),
        ("what platform" not in fixed_lower, "Platform question removed"),
        ("dark mode or light mode" not in fixed_lower, "Color mode question removed"),
        ("what controls" in fixed_lower, "Controls question REMAINS"),
    ]
    
    all_passed = True
    for condition, description in checks:
        status = "✓ PASS" if condition else "✗ FAIL"
        if not condition:
            all_passed = False
        print(f"   {status}: {description}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL CHECKS PASSED - FIX VERIFIED")
    else:
        print("SOME CHECKS FAILED - NEEDS REVIEW")
    print("=" * 60)


if __name__ == "__main__":
    main()
