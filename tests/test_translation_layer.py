# FILE: tests/test_translation_layer.py
"""
Comprehensive tests for ASTRA Translation Layer.

Tests cover:
1. Mode classification (Chat/Command/Feedback)
2. Directive vs Story gate (questions, past tense, etc.)
3. Tier 0 rule matching
4. Phrase cache functionality
5. Context gate
6. Confirmation gate
7. End-to-end translation
8. Specific misfiring scenarios (the "tell me about Overwatch" bug)
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import pytest
import sys

# Make sure local app package is importable when running tests directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.translation import (
    # Schemas
    TranslationMode,
    CanonicalIntent,
    LatencyTier,
    TranslationResult,
    UIContext,
    # Mode classification
    classify_mode,
    classify_mode_with_ui,
    # Gates
    check_directive_gate,
    check_context_gate,
    is_obvious_chat,
    # Tier 0 rules
    tier0_classify,
    is_user_chat_pattern,
    # Phrase cache
    PhraseCache,
    normalize_phrase,
    extract_pattern,
    # Main translator
    Translator,
    translate_message_sync,
)


# =============================================================================
# MODE CLASSIFICATION TESTS
# =============================================================================


class TestModeClassification:
    """Tests for first-stage mode classification."""

    def test_default_to_chat_mode(self):
        """Plain messages should default to chat mode."""
        mode, wake, text = classify_mode("Hello, how are you?")
        assert mode == TranslationMode.CHAT
        assert wake is None
        assert text == "Hello, how are you?"

    def test_command_wake_phrase(self):
        """Messages with command wake phrase enter command mode."""
        mode, wake, text = classify_mode("Astra, command: run tests")
        assert mode == TranslationMode.COMMAND_CAPABLE
        assert "command" in wake.lower()
        assert text == "run tests"

    def test_command_wake_phrase_case_insensitive(self):
        """Wake phrase detection is case-insensitive."""
        for phrase in ["ASTRA, command:", "astra, command:", "Astra, Command:"]:
            mode, wake, text = classify_mode(f"{phrase} do something")
            assert mode == TranslationMode.COMMAND_CAPABLE

    def test_feedback_wake_phrase(self):
        """Messages with feedback wake phrase enter feedback mode."""
        mode, wake, text = classify_mode("Astra, feedback: that was wrong")
        assert mode == TranslationMode.FEEDBACK
        assert "feedback" in wake.lower()
        assert text == "that was wrong"

    def test_feedback_takes_priority(self):
        """Feedback mode takes priority if wake phrase matches."""
        mode, wake, text = classify_mode("Astra, feedback: command was wrong")
        assert mode == TranslationMode.FEEDBACK

    def test_ui_context_enables_command_mode(self):
        """UI context can enable command mode without wake phrase."""
        ui_context = UIContext(in_job_config=True)
        mode, wake, text = classify_mode_with_ui("Create architecture map", ui_context)
        assert mode == TranslationMode.COMMAND_CAPABLE
        assert wake is None  # No wake phrase needed


# =============================================================================
# DIRECTIVE VS STORY GATE TESTS
# =============================================================================


class TestDirectiveGate:
    """Tests for directive vs story gate - CRITICAL for preventing misfires."""

    # Questions should NOT be commands
    def test_blocks_questions(self):
        """Questions should be blocked as chat."""
        questions = [
            "How do you map the architecture?",
            "What does the Overwatch system do?",
            "Can you tell me about the pipeline?",
            "Where is the sandbox config?",
            "Why did it fail?",
        ]
        for q in questions:
            result = check_directive_gate(q)
            assert not result.passed, f"Should block question: {q}"
            assert result.detected_pattern == "question"

    def test_blocks_tell_me_about(self):
        """'Tell me about X' should be blocked - this was the original bug."""
        result = check_directive_gate("Tell me about the Overwatch subsystem")
        assert not result.passed
        assert result.detected_pattern in ("question", "meta_discussion")

    # Past tense should NOT be commands
    def test_blocks_past_tense(self):
        """Past tense references should be blocked."""
        past_refs = [
            "That time you mapped the repo",
            "When you updated the architecture yesterday",
            "You created a map before",
            "We discussed this earlier",
            "Remember when you ran the pipeline?",
        ]
        for ref in past_refs:
            result = check_directive_gate(ref)
            assert not result.passed, f"Should block past tense: {ref}"

    # Future planning should NOT be commands
    def test_blocks_future_planning(self):
        """Future planning should be blocked."""
        future_refs = [
            "Next week we'll map the architecture",
            "We'll run the pipeline tomorrow",
            "I'm going to start the zombie later",
            "Maybe we should update architecture soon",
            "Eventually we'll need to run this",
        ]
        for ref in future_refs:
            result = check_directive_gate(ref)
            assert not result.passed, f"Should block future planning: {ref}"

    # Meta-discussion should NOT be commands
    def test_blocks_meta_discussion(self):
        """Talking ABOUT commands should be blocked."""
        meta = [
            "When we run start your zombie...",
            "If you start your zombie, what happens?",
            "About the create architecture command...",
            "The pipeline system is interesting",
            "Your Overwatch subsystem looks good",
        ]
        for m in meta:
            result = check_directive_gate(m)
            assert not result.passed, f"Should block meta-discussion: {m}"

    # True imperatives SHOULD pass
    def test_allows_true_imperatives(self):
        """True imperative commands should pass."""
        imperatives = [
            "CREATE ARCHITECTURE MAP",
            "Start your zombie",
            "update architecture",
            "Run critical pipeline for job abc123",
            "SCAN SANDBOX STRUCTURE",
        ]
        for imp in imperatives:
            result = check_directive_gate(imp)
            assert result.passed, f"Should allow imperative: {imp}"


# =============================================================================
# TIER 0 RULE TESTS
# =============================================================================


class TestTier0Rules:
    """Tests for Tier 0 rule-based classification."""

    def test_exact_trigger_phrase_all_caps(self):
        """ALL CAPS trigger phrases should match exactly."""
        result = tier0_classify("CREATE ARCHITECTURE MAP")
        assert result.matched
        assert result.intent == CanonicalIntent.ARCHITECTURE_MAP_WITH_FILES
        assert result.confidence == 1.0

    def test_all_caps_vs_normal_case(self):
        """ALL CAPS should give different result than normal case."""
        all_caps = tier0_classify("CREATE ARCHITECTURE MAP")
        normal = tier0_classify("Create architecture map")

        assert all_caps.intent == CanonicalIntent.ARCHITECTURE_MAP_WITH_FILES
        assert normal.intent == CanonicalIntent.ARCHITECTURE_MAP_STRUCTURE_ONLY

    def test_zombie_start_variants(self):
        """Various zombie start phrasings should match."""
        variants = [
            "Start your zombie",
            "start your zombie",
        ]
        for v in variants:
            result = tier0_classify(v)
            assert result.matched, f"Should match: {v}"
            assert result.intent == CanonicalIntent.START_SANDBOX_ZOMBIE_SELF

    def test_sandbox_structure_variants(self):
        """Sandbox structure scan phrases should match."""
        variants = [
            "SCAN SANDBOX STRUCTURE",
            "Scan sandbox structure",
            "Scan the sandbox structure",
        ]
        for v in variants:
            result = tier0_classify(v)
            assert result.matched, f"Should match: {v}"
            assert result.intent == CanonicalIntent.SCAN_SANDBOX_STRUCTURE

    def test_update_architecture_variants(self):
        """Update architecture variants should match."""
        variants = [
            "update architecture",
            "Update architecture",
        ]
        for v in variants:
            result = tier0_classify(v)
            assert result.matched, f"Should match: {v}"
            assert result.intent == CanonicalIntent.ARCHITECTURE_UPDATE_ATLAS_ONLY

    def test_obvious_chat_short_circuit(self):
        """Obvious chat should be caught at Tier 0."""
        chat_messages = [
            "Hello!",
            "Thanks for that",
            "I think we should discuss...",
            "How does the Overwatch system work?",
        ]
        for msg in chat_messages:
            result = tier0_classify(msg)
            assert result.matched, f"Should match as chat: {msg}"
            assert result.intent == CanonicalIntent.CHAT_ONLY

    def test_no_match_for_ambiguous(self):
        """Ambiguous messages should not match as commands at Tier 0."""
        ambiguous = [
            "architecture stuff",
            "maybe create a map",
            "something about zombies",
        ]
        for msg in ambiguous:
            result = tier0_classify(msg)
            if result.matched:
                assert result.intent == CanonicalIntent.CHAT_ONLY


# =============================================================================
# MULTI-FILE TRIGGER TESTS (v5.10 - Level 3)
# =============================================================================


class TestMultiFileTrigger:
    """Tests for multi-file operation detection (Level 3)."""
    
    # Import the specific function for testing
    @pytest.fixture(autouse=True)
    def setup(self):
        from app.translation.tier0_rules import check_multi_file_trigger
        self.check_multi_file_trigger = check_multi_file_trigger
    
    # =========================================================================
    # SEARCH PATTERNS (MULTI_FILE_SEARCH)
    # =========================================================================
    
    def test_find_all_todos(self):
        """'find all TODO' should trigger MULTI_FILE_SEARCH."""
        result = self.check_multi_file_trigger("find all TODO comments")
        assert result.matched
        assert result.intent == CanonicalIntent.MULTI_FILE_SEARCH
        assert "TODO" in result.reason.lower() or "todo" in result.reason
    
    def test_find_all_in_codebase(self):
        """'find all X in the codebase' should trigger MULTI_FILE_SEARCH."""
        result = self.check_multi_file_trigger("find all deprecated functions in the codebase")
        assert result.matched
        assert result.intent == CanonicalIntent.MULTI_FILE_SEARCH
    
    def test_list_files_containing(self):
        """'list files containing X' should trigger MULTI_FILE_SEARCH."""
        result = self.check_multi_file_trigger("list all files containing import os")
        assert result.matched
        assert result.intent == CanonicalIntent.MULTI_FILE_SEARCH
    
    def test_search_codebase_for(self):
        """'search codebase for X' should trigger MULTI_FILE_SEARCH."""
        result = self.check_multi_file_trigger("search the codebase for async def")
        assert result.matched
        assert result.intent == CanonicalIntent.MULTI_FILE_SEARCH
    
    def test_count_occurrences(self):
        """'count occurrences of X' should trigger MULTI_FILE_SEARCH."""
        result = self.check_multi_file_trigger("count all occurrences of DEBUG")
        assert result.matched
        assert result.intent == CanonicalIntent.MULTI_FILE_SEARCH
    
    # =========================================================================
    # REFACTOR PATTERNS (MULTI_FILE_REFACTOR)
    # =========================================================================
    
    def test_replace_everywhere(self):
        """'replace X with Y everywhere' should trigger MULTI_FILE_REFACTOR."""
        result = self.check_multi_file_trigger("replace Orb with Astra everywhere")
        assert result.matched
        assert result.intent == CanonicalIntent.MULTI_FILE_REFACTOR
        assert "Orb" in result.reason or "orb" in result.reason.lower()
    
    def test_replace_in_all_files(self):
        """'replace X with Y in all files' should trigger MULTI_FILE_REFACTOR."""
        result = self.check_multi_file_trigger("replace DEBUG = True with DEBUG = False in all files")
        assert result.matched
        assert result.intent == CanonicalIntent.MULTI_FILE_REFACTOR
    
    def test_change_all_to(self):
        """'change all X to Y' should trigger MULTI_FILE_REFACTOR."""
        result = self.check_multi_file_trigger("change all print statements to logging")
        assert result.matched
        assert result.intent == CanonicalIntent.MULTI_FILE_REFACTOR
    
    def test_rename_across_codebase(self):
        """'rename X to Y across the codebase' should trigger MULTI_FILE_REFACTOR."""
        result = self.check_multi_file_trigger("rename old_function to new_function across the codebase")
        assert result.matched
        assert result.intent == CanonicalIntent.MULTI_FILE_REFACTOR
    
    def test_remove_from_codebase(self):
        """'remove X from codebase' should trigger MULTI_FILE_REFACTOR."""
        result = self.check_multi_file_trigger("remove all console.log from the codebase")
        assert result.matched
        assert result.intent == CanonicalIntent.MULTI_FILE_REFACTOR
    
    # =========================================================================
    # NEGATIVE CASES (should NOT trigger)
    # =========================================================================
    
    def test_no_match_regular_chat(self):
        """Regular chat should not trigger multi-file."""
        result = self.check_multi_file_trigger("what is a TODO comment?")
        assert not result.matched
    
    def test_no_match_single_file_find(self):
        """'find X' without scope keyword should not trigger."""
        result = self.check_multi_file_trigger("find the config file")
        assert not result.matched
    
    def test_no_match_simple_replace(self):
        """'replace X with Y' without scope should not trigger."""
        result = self.check_multi_file_trigger("replace this text")
        assert not result.matched
    
    def test_no_match_short_text(self):
        """Very short text should not trigger."""
        result = self.check_multi_file_trigger("hi")
        assert not result.matched
    
    def test_no_match_question_about_finding(self):
        """Questions about finding should not trigger."""
        result = self.check_multi_file_trigger("how do I find all TODO comments?")
        assert not result.matched
    
    # =========================================================================
    # CASE INSENSITIVITY
    # =========================================================================
    
    def test_case_insensitive_search(self):
        """Search patterns should be case-insensitive."""
        result = self.check_multi_file_trigger("FIND ALL ERRORS IN THE CODEBASE")
        assert result.matched
        assert result.intent == CanonicalIntent.MULTI_FILE_SEARCH
    
    def test_case_insensitive_refactor(self):
        """Refactor patterns should be case-insensitive."""
        result = self.check_multi_file_trigger("REPLACE foo WITH bar EVERYWHERE")
        assert result.matched
        assert result.intent == CanonicalIntent.MULTI_FILE_REFACTOR
    
    # =========================================================================
    # EDGE CASES
    # =========================================================================
    
    def test_pattern_with_special_chars(self):
        """Patterns with special characters should work."""
        result = self.check_multi_file_trigger("find all # TODO: in the codebase")
        assert result.matched
        assert result.intent == CanonicalIntent.MULTI_FILE_SEARCH
    
    def test_multi_word_pattern(self):
        """Multi-word patterns should be captured."""
        result = self.check_multi_file_trigger("find all raise ValueError in the codebase")
        assert result.matched
        assert result.intent == CanonicalIntent.MULTI_FILE_SEARCH


# =============================================================================
# PHRASE CACHE TESTS
# =============================================================================


class TestPhraseCache:
    """Tests for adaptive phrase cache."""

    def test_normalize_phrase(self):
        """Phrase normalization should work correctly."""
        assert normalize_phrase("  Hello  World  ") == "hello world"
        assert normalize_phrase("CREATE ARCHITECTURE MAP") == "create architecture map"

    def test_extract_pattern_with_uuid(self):
        """UUIDs should be replaced with placeholder."""
        text = "Run pipeline for job 12345678-1234-1234-1234-123456789abc"
        pattern = extract_pattern(text)
        assert "{uuid}" in pattern or "{id}" in pattern

    def test_cache_add_and_lookup(self):
        """Adding and looking up entries should work."""
        cache = PhraseCache("test_user")

        cache.add("my custom command", CanonicalIntent.ARCHITECTURE_MAP_WITH_FILES)

        result = cache.lookup("my custom command")
        assert result is not None
        intent, entry = result
        assert intent == CanonicalIntent.ARCHITECTURE_MAP_WITH_FILES

    def test_cache_lookup_normalized(self):
        """Lookup should work with normalized text."""
        cache = PhraseCache("test_user")

        cache.add("My Custom Command", CanonicalIntent.START_SANDBOX_ZOMBIE_SELF)

        # Should find with different casing
        result = cache.lookup("my custom command")
        assert result is not None

    def test_cache_hit_count(self):
        """Hit count should increment on lookup."""
        cache = PhraseCache("test_user")
        cache.add("test phrase", CanonicalIntent.CHAT_ONLY)

        # Multiple lookups
        for _ in range(5):
            cache.lookup("test phrase")

        result = cache.lookup("test phrase")
        assert result is not None
        _, entry = result
        assert entry.hit_count >= 5


# =============================================================================
# CONTEXT GATE TESTS
# =============================================================================


class TestContextGate:
    """Tests for context gate."""

    def test_no_context_required(self):
        """Intents without context requirements should pass."""
        result = check_context_gate(
            CanonicalIntent.ARCHITECTURE_MAP_WITH_FILES,
            provided_context={},
        )
        assert result.passed

    def test_missing_required_context(self):
        """Missing required context should fail."""
        result = check_context_gate(
            CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB,
            provided_context={},
        )
        assert not result.passed
        assert "job_id" in result.missing_context

    def test_provided_context_passes(self):
        """Providing required context should pass."""
        result = check_context_gate(
            CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB,
            provided_context={"job_id": "abc123", "spec_id": "spec_456"},
        )
        assert result.passed


# =============================================================================
# OBVIOUS CHAT DETECTION TESTS
# =============================================================================


class TestObviousChatDetection:
    """Tests for the is_obvious_chat function."""

    def test_questions_are_obvious_chat(self):
        """Questions should be obvious chat."""
        is_chat, reason = is_obvious_chat("What is the architecture?")
        assert is_chat
        assert reason == "question"

    def test_past_tense_is_obvious_chat(self):
        """Past tense is obvious chat."""
        is_chat, reason = is_obvious_chat("That time you mapped the repo")
        assert is_chat
        assert reason == "past_tense"

    def test_meta_discussion_is_obvious_chat(self):
        """Meta discussion is obvious chat."""
        is_chat, reason = is_obvious_chat("Tell me about your Overwatch subsystem")
        assert is_chat
        # Could be "question" or "meta_discussion"
        assert reason in ("question", "meta_discussion")

    def test_commands_not_obvious_chat(self):
        """True commands should not be obvious chat."""
        is_chat, _ = is_obvious_chat("CREATE ARCHITECTURE MAP")
        assert not is_chat


# =============================================================================
# END-TO-END TRANSLATION TESTS
# =============================================================================


class TestEndToEndTranslation:
    """End-to-end tests for the translation layer."""

    def test_chat_mode_stays_chat(self):
        """Messages without wake phrase should stay in chat mode."""
        result = translate_message_sync("Tell me about the Overwatch subsystem")

        assert result.mode == TranslationMode.CHAT
        assert result.resolved_intent == CanonicalIntent.CHAT_ONLY
        assert not result.should_execute

    def test_command_wake_phrase_with_valid_zombie_command(self):
        """Wake phrase + valid zombie command should work."""
        result = translate_message_sync("Astra, command: Start your zombie")

        assert result.mode == TranslationMode.COMMAND_CAPABLE
        assert result.resolved_intent == CanonicalIntent.START_SANDBOX_ZOMBIE_SELF
        assert result.should_execute

    def test_command_wake_phrase_with_sandbox_scan_command(self):
        """Wake phrase + sandbox scan command should work."""
        result = translate_message_sync("Astra, command: SCAN SANDBOX STRUCTURE")

        assert result.mode == TranslationMode.COMMAND_CAPABLE
        assert result.resolved_intent == CanonicalIntent.SCAN_SANDBOX_STRUCTURE
        assert result.should_execute

    def test_command_wake_phrase_with_question_blocked(self):
        """Wake phrase + question should not execute (treated as chat)."""
        result = translate_message_sync("Astra, command: How do you map architecture?")

        assert result.mode == TranslationMode.COMMAND_CAPABLE
        # Should not execute as a real command
        assert result.resolved_intent == CanonicalIntent.CHAT_ONLY
        assert not result.should_execute

    def test_feedback_mode(self):
        """Feedback mode should not trigger commands."""
        result = translate_message_sync(
            "Astra, feedback: that should have been a command",
        )

        assert result.mode == TranslationMode.FEEDBACK
        assert result.resolved_intent == CanonicalIntent.USER_BEHAVIOR_FEEDBACK
        assert not result.should_execute


# =============================================================================
# SPECIFIC BUG REPRODUCTION TESTS
# =============================================================================


class TestMisfireBugFixes:
    """
    Tests for specific misfiring bugs.
    These are regression tests for known issues.
    """

    def test_overwatch_subsystem_bug(self):
        """
        BUG: "Tell me about your Overwatch subsystem" triggered critical pipeline.
        EXPECTED: Should be treated as chat (memory recall).
        """
        result = translate_message_sync("Tell me about your Overwatch subsystem")

        assert result.mode == TranslationMode.CHAT
        assert result.resolved_intent == CanonicalIntent.CHAT_ONLY
        assert not result.should_execute

    def test_explain_architecture_bug(self):
        """
        Questions about architecture should not trigger architecture commands.
        """
        queries = [
            "Can you explain the architecture?",
            "What's in the architecture map?",
            "How does the architecture work?",
            "Tell me about the architecture",
            "Describe the architecture to me",
        ]
        for q in queries:
            result = translate_message_sync(q)
            assert not result.should_execute, f"Should not execute for: {q}"
            assert result.resolved_intent == CanonicalIntent.CHAT_ONLY

    def test_hypothetical_command_mentions(self):
        """
        Hypothetical discussions about commands should not trigger them.
        """
        hypotheticals = [
            "What if I run the critical pipeline?",
            "If we start the zombie, what happens?",
            "Suppose I create architecture map...",
            "When would I use update architecture?",
        ]
        for h in hypotheticals:
            result = translate_message_sync(h)
            assert not result.should_execute, f"Should not execute for: {h}"

    def test_past_command_references(self):
        """
        References to past command executions should not re-trigger them.
        """
        past_refs = [
            "That time you created the architecture map...",
            "When you started the zombie yesterday...",
            "Remember when we ran the pipeline?",
            "You updated the architecture earlier",
        ]
        for ref in past_refs:
            result = translate_message_sync(ref)
            assert not result.should_execute, f"Should not execute for: {ref}"


# =============================================================================
# USER PATTERN TESTS
# =============================================================================


class TestUserPatterns:
    """Tests for common 'User' patterns."""

    def test_user_chat_patterns(self):
        """Common Taz chat patterns should be recognized."""
        chat_patterns = [
            "tell me about the system",
            "explain how this works",
            "show me the architecture",
            "what does this do",
            "hi there",
            "thanks for that",
        ]
        for pattern in chat_patterns:
            assert is_user_chat_pattern(pattern), f"Should be User chat: {pattern}"

    def test_user_command_patterns(self):
        """Common Taz command patterns should not be flagged as chat."""
        command_patterns = [
            "CREATE ARCHITECTURE MAP",
            "Start your zombie",
            "update architecture",
            "Scan sandbox structure",
        ]
        for pattern in command_patterns:
            assert not is_user_chat_pattern(pattern), f"Should NOT be User chat: {pattern}"


# =============================================================================
# LATENCY TIER TESTS
# =============================================================================


class TestLatencyTiers:
    """Tests for latency tier classification."""

    def test_obvious_commands_use_tier0(self):
        """Obvious commands should be Tier 0."""
        result = translate_message_sync("Astra, command: CREATE ARCHITECTURE MAP")
        assert result.latency_tier == LatencyTier.TIER_0_RULES

    def test_obvious_chat_uses_tier0(self):
        """Obvious chat should be Tier 0."""
        result = translate_message_sync("Hello!")
        assert result.latency_tier == LatencyTier.TIER_0_RULES


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
