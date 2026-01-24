# FILE: tests/test_weaver_slot_reconciliation.py
"""
Unit tests for Weaver Slot Reconciliation (v3.5.1)

Tests the fix for the question regression bug where answered questions
were being repeated in subsequent Weaver runs.

Run with: pytest tests/test_weaver_slot_reconciliation.py -v
"""
import pytest
import re


# Import the functions under test
from app.llm.weaver_stream import (
    _detect_filled_slots,
    _reconcile_filled_slots,
    _add_known_requirements_section,
    _get_shallow_questions,
    SHALLOW_QUESTIONS,
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_weaver_output_all_unresolved():
    """Weaver output with ALL slots unresolved (initial state)."""
    return """**What is being built:** Classic Tetris game

**Intended outcome:** Playable Tetris implementation

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
"""


@pytest.fixture
def sample_weaver_output_with_new_requirements():
    """Weaver output after UPDATE mode (has New requirements but still shows old ambiguities - BUG)."""
    return """**What is being built:** Classic Tetris game

**Intended outcome:** Playable Tetris implementation

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


# ---------------------------------------------------------------------------
# Test 1: Partial Answers
# ---------------------------------------------------------------------------

class TestPartialAnswers:
    """When user answers some but not all slots, only unanswered ones remain."""
    
    def test_detect_partial_slots(self):
        """User says 'Android, Dark mode, centered full screen' - controls remains unanswered."""
        ramble = "[Human]: Android, Dark mode, Bare minimum playable first. centered full screen"
        
        filled = _detect_filled_slots(ramble)
        
        assert "platform" in filled
        assert filled["platform"] == "Android"
        
        assert "look_feel" in filled
        assert filled["look_feel"] == "Dark mode"
        
        assert "scope" in filled
        assert "minimum" in filled["scope"].lower() or "basic" in filled["scope"].lower()
        
        assert "layout" in filled
        assert "centered" in filled["layout"].lower() or "fullscreen" in filled["layout"].lower()
        
        # Controls NOT answered
        assert "controls" not in filled
    
    def test_reconcile_partial_answers(self, sample_weaver_output_all_unresolved):
        """After partial answers, only controls ambiguity/question remains."""
        filled_slots = {
            "platform": "Android",
            "look_feel": "Dark mode",
            "scope": "Bare minimum / basic",
            "layout": "Centered / fullscreen",
        }
        
        result = _reconcile_filled_slots(sample_weaver_output_all_unresolved, filled_slots)
        
        # Should still have controls-related items
        assert "control" in result.lower()
        
        # Should NOT have platform/color/scope/layout ambiguities
        assert "platform is unspecified" not in result.lower()
        assert "color mode" not in result.lower() or "dark mode" in result.lower()  # dark mode mention is OK in Known Requirements
        assert "dark vs light" not in result.lower()
        assert "scope (bare minimum" not in result.lower()
        assert "layout preference" not in result.lower() or "centered" in result.lower()


# ---------------------------------------------------------------------------
# Test 2: Full Answers
# ---------------------------------------------------------------------------

class TestFullAnswers:
    """When user answers ALL slots, no ambiguities/questions remain."""
    
    def test_detect_all_slots(self):
        """User answers all 5 slots."""
        ramble = "[Human]: I want a Tetris game for Android, dark mode, touch controls, bare minimum, centered fullscreen"
        
        filled = _detect_filled_slots(ramble)
        
        assert len(filled) >= 4  # At least platform, look_feel, controls, layout
        assert "platform" in filled
        assert "look_feel" in filled
        assert "controls" in filled
        assert "layout" in filled
    
    def test_reconcile_all_answers(self, sample_weaver_output_all_unresolved):
        """After all answers, no unresolved ambiguities should remain."""
        filled_slots = {
            "platform": "Android",
            "look_feel": "Dark mode",
            "controls": "Touch",
            "scope": "Bare minimum / basic",
            "layout": "Centered / fullscreen",
        }
        
        result = _reconcile_filled_slots(sample_weaver_output_all_unresolved, filled_slots)
        
        # Count remaining items in ambiguities section
        ambig_section = re.search(
            r'\*\*Unresolved ambiguities:\*\*(.*?)(\*\*|$)',
            result,
            re.DOTALL | re.IGNORECASE
        )
        
        if ambig_section:
            # The section exists but should be empty or have no slot-related items
            ambig_text = ambig_section.group(1).strip()
            # Should not have any of our slot-related ambiguities
            assert "platform" not in ambig_text.lower()
            assert "color mode" not in ambig_text.lower()
            assert "control" not in ambig_text.lower()
            assert "scope" not in ambig_text.lower()
            assert "layout" not in ambig_text.lower()


# ---------------------------------------------------------------------------
# Test 3: Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    """Running reconciliation twice should not re-add resolved items."""
    
    def test_double_reconcile_same_result(self, sample_weaver_output_all_unresolved):
        """Reconciling twice with same filled slots gives identical result."""
        filled_slots = {
            "platform": "Android",
            "look_feel": "Dark mode",
            "layout": "Centered / fullscreen",
        }
        
        result1 = _reconcile_filled_slots(sample_weaver_output_all_unresolved, filled_slots)
        result2 = _reconcile_filled_slots(result1, filled_slots)
        
        # Results should be identical
        assert result1 == result2
    
    def test_known_requirements_not_duplicated(self, sample_weaver_output_all_unresolved):
        """Adding known requirements section twice should not duplicate it."""
        filled_slots = {
            "platform": "Android",
            "look_feel": "Dark mode",
        }
        
        result1 = _add_known_requirements_section(sample_weaver_output_all_unresolved, filled_slots)
        result2 = _add_known_requirements_section(result1, filled_slots)
        
        # Count occurrences of "Known requirements"
        count1 = result1.lower().count("known requirements")
        count2 = result2.lower().count("known requirements")
        
        assert count1 == 1
        assert count2 == 1  # Should not duplicate


# ---------------------------------------------------------------------------
# Test 4: No False Removal
# ---------------------------------------------------------------------------

class TestNoFalseRemoval:
    """Unanswered slots must remain in output."""
    
    def test_empty_filled_slots_preserves_all(self, sample_weaver_output_all_unresolved):
        """With no filled slots, ALL ambiguities/questions remain."""
        filled_slots = {}
        
        result = _reconcile_filled_slots(sample_weaver_output_all_unresolved, filled_slots)
        
        # All original ambiguities should remain
        assert "platform is unspecified" in result.lower()
        assert "color mode" in result.lower()
        assert "control" in result.lower()
        assert "scope" in result.lower()
        assert "layout" in result.lower()
    
    def test_partial_filled_preserves_unfilled(self, sample_weaver_output_all_unresolved):
        """Only platform filled - other 4 slots remain."""
        filled_slots = {"platform": "Android"}
        
        result = _reconcile_filled_slots(sample_weaver_output_all_unresolved, filled_slots)
        
        # Platform should be removed
        assert "platform is unspecified" not in result.lower()
        
        # Others should remain
        assert "color mode" in result.lower() or "dark" in result.lower()
        assert "control" in result.lower()
        assert "scope" in result.lower()
        assert "layout" in result.lower()


# ---------------------------------------------------------------------------
# Test 5: Slot Detection Patterns
# ---------------------------------------------------------------------------

class TestSlotDetectionPatterns:
    """Test that various user phrasings are correctly detected."""
    
    @pytest.mark.parametrize("text,expected_slot,expected_value", [
        # Platform variants
        ("I want it on Android", "platform", "Android"),
        ("for iOS please", "platform", "iOS"),
        ("make it a web app", "platform", "Web"),
        ("desktop application", "platform", "Desktop"),
        ("mobile first", "platform", "Mobile"),
        
        # Look/feel variants
        ("dark mode", "look_feel", "Dark mode"),
        ("light theme", "look_feel", "Light mode"),
        ("I prefer dark", "look_feel", "Dark mode"),
        
        # Controls variants
        ("touch controls", "controls", "Touch"),
        ("keyboard input", "controls", "Keyboard"),
        ("swipe gestures", "controls", "Touch"),
        ("gamepad support", "controls", "Controller"),
        
        # Scope variants
        ("bare minimum first", "scope", "Bare minimum / basic"),
        ("just the basics", "scope", "Bare minimum / basic"),
        ("minimal version", "scope", "Bare minimum / basic"),
        ("with all the features", "scope", "With extras / features"),
        
        # Layout variants
        ("centered on screen", "layout", "Centered / fullscreen"),
        ("fullscreen mode", "layout", "Centered / fullscreen"),
        ("with a sidebar", "layout", "Sidebar / HUD"),
    ])
    def test_slot_detection_patterns(self, text, expected_slot, expected_value):
        """Various phrasings should be detected correctly."""
        filled = _detect_filled_slots(text)
        
        assert expected_slot in filled, f"Failed to detect {expected_slot} from: {text}"
        # Check that value contains expected substring (case-insensitive)
        assert expected_value.lower() in filled[expected_slot].lower() or \
               filled[expected_slot].lower() in expected_value.lower()


# ---------------------------------------------------------------------------
# Test 6: Integration Test
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end test simulating the Tetris reproduction case."""
    
    def test_tetris_reproduction_case(self, sample_weaver_output_with_new_requirements):
        """
        Reproduction case from the bug report:
        - Initial Weaver asks all 5 questions
        - User replies: "Android, Dark mode, Bare minimum playable first. centered full screen"
        - UPDATE mode adds "New requirements" but keeps old ambiguities (BUG)
        - FIXED: Slot reconciliation removes resolved items
        """
        # User's answer
        user_answer = "Android, Dark mode, Bare minimum playable first. centered full screen"
        
        # Detect filled slots
        filled = _detect_filled_slots(user_answer)
        
        # Should detect 4 slots
        assert "platform" in filled
        assert "look_feel" in filled
        assert "scope" in filled
        assert "layout" in filled
        assert "controls" not in filled  # User didn't specify controls
        
        # Apply reconciliation
        result = _reconcile_filled_slots(sample_weaver_output_with_new_requirements, filled)
        
        # Add known requirements section
        final_result = _add_known_requirements_section(result, filled)
        
        # Verify: resolved items removed
        lines_lower = final_result.lower()
        
        # Platform ambiguity/question should be GONE
        assert "platform is unspecified" not in lines_lower
        assert "what platform do you want" not in lines_lower
        
        # Color mode ambiguity/question should be GONE
        assert "dark vs light" not in lines_lower
        assert "dark mode or light mode?" not in lines_lower
        
        # Scope ambiguity/question should be GONE
        assert "scope (bare minimum" not in lines_lower
        assert "bare minimum playable first, or add" not in lines_lower
        
        # Layout ambiguity/question should be GONE
        assert "layout preference (centered vs sidebar" not in lines_lower
        
        # Controls should REMAIN (was not answered)
        assert "control" in lines_lower
        
        # Known requirements should be present
        assert "known requirements" in lines_lower
        assert "android" in lines_lower
        assert "dark mode" in lines_lower


# ---------------------------------------------------------------------------
# Test 7: Shallow Question Generation Integration
# ---------------------------------------------------------------------------

class TestShallowQuestionGeneration:
    """Test that _get_shallow_questions correctly identifies answered slots."""
    
    def test_no_questions_when_all_answered(self):
        """When all slots are answered, no questions should be generated."""
        ramble = "Build a Tetris for Android, dark mode, touch, minimal, centered fullscreen"
        
        questions = _get_shallow_questions(ramble)
        
        # Should have very few or no questions
        assert len(questions) <= 1  # At most style might remain
    
    def test_all_questions_when_none_answered(self):
        """When no slots are answered, all questions should be generated."""
        ramble = "I want a Tetris game"
        
        questions = _get_shallow_questions(ramble)
        
        # Should have most/all questions
        assert len(questions) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
