# FILE: tests/test_weaver_v352_regression.py
"""
Regression tests for Weaver v3.5.2 - Slot Reconciliation Pattern Fix

This test uses the EXACT output format from the bug report where:
- User answered: Desktop, Dark mode, Keyboard, Centered HUD
- But output still showed those as "unspecified" in ambiguities

The v3.5.1 patterns used "unspecified" but LLM was outputting "not specified".
v3.5.2 fixes this by matching both variations.
"""
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.llm.weaver_stream import (
    _detect_filled_slots,
    _reconcile_filled_slots,
    _add_known_requirements_section,
)


class TestV352RegressionExactBugOutput:
    """Test with the EXACT output format from the Tetris bug report."""
    
    # This is the exact problematic output from the bug report
    BUG_OUTPUT = """What is being built
Classic Tetris game

Known requirements:
- Scope: Bare minimum / basic

Intended outcome
Clean, playable classic Tetris implementation

Execution mode
For now: discussion only — no code or file creation yet

Design preferences
Classic 10x20 playfield
Standard 7 tetromino pieces
Scoring and level speed-up
Next-piece preview
Basic controls as previously referenced
Keep it minimal: no hold piece, no ghost piece, no sound, no extras
Emphasis on core gameplay feeling right and playing clean
Dark mode visual theme
Keyboard controls using arrow keys
Space bar performs hard drop
P key pauses the game
Layout centered
Simple HUD showing score
Simple HUD showing level
Simple HUD showing next-piece preview

Constraints
Do not generate code or create files until basics are agreed
Keep feature set minimal (no fancy additions)

Unresolved ambiguities
Target platform not specified
Visual theme (dark vs light) not specified
Exact control method and key mappings are unspecified ("basic controls you listed" is unclear)
Layout/HUD placement not specified
Precise scoring rules and level progression details are unspecified
Persistence (high scores, settings) and pause/restart behavior not defined

Questions
What platform do you want this on? (web / Android / desktop / iOS)
Dark mode or light mode?
What controls? (keyboard / touch / controller)
Any preference on layout? (centered vs sidebar HUD)"""

    # This is the user's message that answered the questions
    USER_MESSAGE = """Desktop, dark mode, keyboard controls (arrow keys + space to hard drop, P to pause), and keep the layout centered with a simple HUD — score, level, and next piece preview."""

    def test_detect_filled_slots_from_user_message(self):
        """Test that we detect all answered slots from user message."""
        filled = _detect_filled_slots(self.USER_MESSAGE)
        
        # Must detect all four answered slots
        assert "platform" in filled, "Should detect 'Desktop' as platform"
        assert "look_feel" in filled, "Should detect 'dark mode' as look_feel"
        assert "controls" in filled, "Should detect 'keyboard' as controls"
        assert "layout" in filled, "Should detect 'centered' as layout"
        
        # Values should be reasonable
        assert filled["platform"] == "Desktop"
        assert filled["look_feel"] == "Dark mode"
        assert filled["controls"] == "Keyboard"
        assert "Centered" in filled["layout"]

    def test_reconciliation_removes_platform_ambiguity(self):
        """Test that 'Target platform not specified' is removed."""
        filled = {"platform": "Desktop"}
        result = _reconcile_filled_slots(self.BUG_OUTPUT, filled)
        
        assert "Target platform not specified" not in result, \
            "Should remove 'Target platform not specified' when platform is filled"
        assert "What platform do you want this on" not in result, \
            "Should remove platform question when platform is filled"

    def test_reconciliation_removes_theme_ambiguity(self):
        """Test that 'Visual theme (dark vs light) not specified' is removed."""
        filled = {"look_feel": "Dark mode"}
        result = _reconcile_filled_slots(self.BUG_OUTPUT, filled)
        
        assert "Visual theme (dark vs light) not specified" not in result, \
            "Should remove visual theme ambiguity when look_feel is filled"
        assert "Dark mode or light mode" not in result, \
            "Should remove dark/light question when look_feel is filled"

    def test_reconciliation_removes_controls_ambiguity(self):
        """Test that control method ambiguity is removed."""
        filled = {"controls": "Keyboard"}
        result = _reconcile_filled_slots(self.BUG_OUTPUT, filled)
        
        # The exact line is: 'Exact control method and key mappings are unspecified ("basic controls you listed" is unclear)'
        assert "control method" not in result.lower() or "unspecified" not in result.lower(), \
            "Should remove control method ambiguity when controls is filled"
        assert "What controls" not in result, \
            "Should remove controls question when controls is filled"

    def test_reconciliation_removes_layout_ambiguity(self):
        """Test that layout ambiguity is removed."""
        filled = {"layout": "Centered / fullscreen"}
        result = _reconcile_filled_slots(self.BUG_OUTPUT, filled)
        
        assert "Layout/HUD placement not specified" not in result, \
            "Should remove layout ambiguity when layout is filled"
        assert "preference on layout" not in result.lower(), \
            "Should remove layout question when layout is filled"

    def test_full_reconciliation_all_slots(self):
        """Test complete reconciliation with all four answered slots."""
        filled = _detect_filled_slots(self.USER_MESSAGE)
        result = _reconcile_filled_slots(self.BUG_OUTPUT, filled)
        
        # All these should be GONE
        problematic_lines = [
            "Target platform not specified",
            "Visual theme (dark vs light) not specified",
            "Layout/HUD placement not specified",
            "What platform do you want this on",
            "Dark mode or light mode",
            "What controls",
            "preference on layout",
        ]
        
        for line in problematic_lines:
            assert line not in result, f"'{line}' should be removed after reconciliation"

    def test_known_requirements_added(self):
        """Test that Known requirements section is added with filled values."""
        filled = _detect_filled_slots(self.USER_MESSAGE)
        result = _reconcile_filled_slots(self.BUG_OUTPUT, filled)
        result_with_known = _add_known_requirements_section(result, filled)
        
        # Known requirements should be present
        assert "Known requirements" in result_with_known, \
            "Should have Known requirements section"
        
        # And it should show what was filled
        result_lower = result_with_known.lower()
        assert "platform" in result_lower and "desktop" in result_lower
        assert "dark" in result_lower

    def test_unresolved_items_remain(self):
        """Test that truly unresolved items are NOT removed."""
        filled = _detect_filled_slots(self.USER_MESSAGE)
        result = _reconcile_filled_slots(self.BUG_OUTPUT, filled)
        
        # These should REMAIN (not answered by user)
        items_to_keep = [
            "Precise scoring rules",
            "Persistence (high scores, settings)",
        ]
        
        for item in items_to_keep:
            assert item in result, f"'{item}' should remain (not answered)"


class TestPatternMatching:
    """Test that the regex patterns match expected variations."""
    
    def test_platform_not_specified_pattern(self):
        """Test platform pattern matches 'not specified' variant."""
        from app.llm.weaver_stream import SLOT_AMBIGUITY_PATTERNS
        import re
        
        test_lines = [
            "Target platform not specified",
            "platform is not specified",
            "Platform unspecified",
            "platform is unspecified",
            "target platform is unclear",
        ]
        
        for line in test_lines:
            matched = False
            for pattern in SLOT_AMBIGUITY_PATTERNS["platform"]:
                if re.search(pattern, line.lower(), re.IGNORECASE):
                    matched = True
                    break
            assert matched, f"Pattern should match: '{line}'"

    def test_visual_theme_pattern(self):
        """Test look_feel pattern matches 'Visual theme' variant."""
        from app.llm.weaver_stream import SLOT_AMBIGUITY_PATTERNS
        import re
        
        test_lines = [
            "Visual theme (dark vs light) not specified",
            "dark vs light not specified",
            "color mode unspecified",
            "theme is unclear",
        ]
        
        for line in test_lines:
            matched = False
            for pattern in SLOT_AMBIGUITY_PATTERNS["look_feel"]:
                if re.search(pattern, line.lower(), re.IGNORECASE):
                    matched = True
                    break
            assert matched, f"Pattern should match: '{line}'"

    def test_controls_pattern(self):
        """Test controls pattern matches various formats."""
        from app.llm.weaver_stream import SLOT_AMBIGUITY_PATTERNS
        import re
        
        test_lines = [
            "Exact control method and key mappings are unspecified",
            "control method is not specified",
            "controls unspecified",
            "input method unclear",
        ]
        
        for line in test_lines:
            matched = False
            for pattern in SLOT_AMBIGUITY_PATTERNS["controls"]:
                if re.search(pattern, line.lower(), re.IGNORECASE):
                    matched = True
                    break
            assert matched, f"Pattern should match: '{line}'"


class TestIdempotency:
    """Test that reconciliation is idempotent (safe to run multiple times)."""
    
    OUTPUT_AFTER_FIRST_RUN = """What is being built
Classic Tetris game

Unresolved ambiguities
Precise scoring rules not specified

Questions
(none remaining)"""

    def test_double_reconciliation_same_result(self):
        """Running reconciliation twice should give same result."""
        filled = {"platform": "Desktop", "look_feel": "Dark mode"}
        
        result1 = _reconcile_filled_slots(self.OUTPUT_AFTER_FIRST_RUN, filled)
        result2 = _reconcile_filled_slots(result1, filled)
        
        assert result1 == result2, "Double reconciliation should be idempotent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
