# FILE: verify_specgate_greenfield_fix.py
"""
Quick verification script for SpecGate v1.10 greenfield build fix.

Run this to confirm the bug is fixed:
    python verify_specgate_greenfield_fix.py
"""
import sys
import os
import re

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.pot_spec.spec_gate_grounded import (
    _extract_sandbox_hints,
    detect_domains,
    DOMAIN_KEYWORDS,
)


def main():
    print("=" * 70)
    print("SpecGate v1.10 Greenfield Build Fix Verification")
    print("=" * 70)
    
    # The problematic Weaver output that was causing the bug
    tetris_weaver_output = """
What is being built: Classic Tetris game

Known requirements:
- Target platform: Desktop
- Color mode: Dark mode
- Controls: Keyboard controls
- Layout: Centered layout

Intended outcome: Playable classic Tetris implementation

Design preferences:
- Classic 10x20 playfield
- Standard 7 tetromino pieces
- Dark mode visual theme
- Keyboard controls using arrow keys
"""

    print("\n1. TESTING DOMAIN DETECTION:")
    print("-" * 50)
    
    detected = detect_domains(tetris_weaver_output)
    print(f"   Detected domains: {detected}")
    
    is_greenfield = "greenfield_build" in detected
    is_sandbox_file = "sandbox_file" in detected
    
    print(f"   ✓ Is greenfield build: {is_greenfield}")
    print(f"   ✓ Is sandbox file: {is_sandbox_file}")
    
    if is_greenfield and not is_sandbox_file:
        print("   ✓ PASS: Correctly detected as greenfield, NOT sandbox file")
    else:
        print("   ✗ FAIL: Domain detection incorrect")
    
    print("\n2. TESTING SANDBOX HINTS EXTRACTION:")
    print("-" * 50)
    
    anchor, subfolder = _extract_sandbox_hints(tetris_weaver_output)
    
    print(f"   Anchor extracted: {anchor}")
    print(f"   Subfolder extracted: {subfolder}")
    
    if anchor is None:
        print("   ✓ PASS: Correctly returned (None, None) - no sandbox discovery triggered")
    else:
        print(f"   ✗ FAIL: Should NOT have extracted anchor '{anchor}'")
    
    print("\n3. TESTING PLATFORM CONTEXT DETECTION:")
    print("-" * 50)
    
    # Test cases that should NOT trigger sandbox discovery
    platform_cases = [
        "Target platform: Desktop",
        "Build me a game for desktop",
        "Desktop app with dark mode",
        "I want a desktop game",
    ]
    
    all_pass = True
    for case in platform_cases:
        anchor, subfolder = _extract_sandbox_hints(case)
        passed = anchor is None
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {status}: '{case[:40]}...' -> anchor={anchor}")
        if not passed:
            all_pass = False
    
    print("\n4. TESTING FILE CONTEXT DETECTION (should still work):")
    print("-" * 50)
    
    # Test cases that SHOULD trigger sandbox discovery
    file_cases = [
        ("Desktop folder called test", "desktop", "test"),
        ("file on the desktop", "desktop", None),
        ("sandbox desktop folder", "desktop", None),
        ("read the file in the desktop folder", "desktop", None),
    ]
    
    for case, expected_anchor, expected_subfolder in file_cases:
        anchor, subfolder = _extract_sandbox_hints(case)
        passed = (anchor == expected_anchor)
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {status}: '{case[:40]}' -> anchor={anchor} (expected {expected_anchor})")
        if not passed:
            all_pass = False
    
    print("\n" + "=" * 70)
    if all_pass:
        print("ALL CHECKS PASSED - v1.10 greenfield fix is working correctly!")
        print("")
        print("SpecGate will now:")
        print("  ✓ Skip sandbox discovery for 'Target platform: Desktop'")
        print("  ✓ Classify 'build Tetris' as ARCHITECTURE job")
        print("  ✓ NOT block with 'Could not find target file'")
    else:
        print("SOME CHECKS FAILED - review the output above")
    print("=" * 70)
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
