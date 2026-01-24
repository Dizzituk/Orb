# FILE: verify_v352_fix.py
"""
Quick verification script for Weaver v3.5.2 slot reconciliation fix.

Run this to confirm the bug is fixed:
    python verify_v352_fix.py
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.llm.weaver_stream import (
    _detect_filled_slots,
    _reconcile_filled_slots,
    _add_known_requirements_section,
)


def main():
    print("=" * 70)
    print("Weaver v3.5.2 Slot Reconciliation Fix Verification")
    print("=" * 70)
    
    # User's message that answers the questions
    user_message = """Desktop, dark mode, keyboard controls (arrow keys + space to hard drop, P to pause), and keep the layout centered with a simple HUD — score, level, and next piece preview."""
    
    # The buggy output that should be fixed
    buggy_output = """What is being built
Classic Tetris game

Intended outcome
Clean, playable classic Tetris implementation

Unresolved ambiguities
Target platform not specified
Visual theme (dark vs light) not specified
Exact control method and key mappings are unspecified ("basic controls you listed" is unclear)
Layout/HUD placement not specified

Questions
What platform do you want this on? (web / Android / desktop / iOS)
Dark mode or light mode?
What controls? (keyboard / touch / controller)
Any preference on layout? (centered vs sidebar HUD)"""

    print("\n1. DETECTING FILLED SLOTS FROM USER MESSAGE:")
    print("-" * 50)
    filled = _detect_filled_slots(user_message)
    for slot, value in filled.items():
        print(f"   ✓ {slot}: {value}")
    
    print("\n2. RECONCILING (REMOVING ANSWERED SLOTS):")
    print("-" * 50)
    result = _reconcile_filled_slots(buggy_output, filled)
    
    print("\n3. ADDING KNOWN REQUIREMENTS SECTION:")
    print("-" * 50)
    final = _add_known_requirements_section(result, filled)
    
    print("\n4. FINAL OUTPUT:")
    print("=" * 70)
    print(final)
    print("=" * 70)
    
    # Verify the fix
    print("\n5. VERIFICATION CHECKS:")
    print("-" * 50)
    
    checks = [
        ("Platform ambiguity removed", "Target platform not specified" not in final),
        ("Theme ambiguity removed", "Visual theme (dark vs light) not specified" not in final),
        ("Layout ambiguity removed", "Layout/HUD placement not specified" not in final),
        ("Platform question removed", "What platform do you want this on" not in final),
        ("Theme question removed", "Dark mode or light mode" not in final),
        ("Layout question removed", "preference on layout" not in final.lower()),
        ("Known requirements added", "Known requirements" in final),
    ]
    
    all_passed = True
    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL CHECKS PASSED - v3.5.2 fix is working correctly!")
    else:
        print("SOME CHECKS FAILED - review the output above")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
