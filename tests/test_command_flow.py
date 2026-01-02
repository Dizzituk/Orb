# FILE: tests/test_command_flow.py
"""
Tests for ASTRA Command Flow wiring.

Tests cover:
1. Tier 0 detection of all flow commands
2. Translation layer routing for Spec Gate flow
3. Intent-to-handler mapping
4. End-to-end command sequences
5. Edge cases and regression tests

Test Commands:
    pytest tests/test_command_flow.py -v
    pytest tests/test_command_flow.py -v -k "spec_gate"
    pytest tests/test_command_flow.py -v -k "critical_pipeline"
    pytest tests/test_command_flow.py -v -k "overwatcher"
"""
from __future__ import annotations
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.translation import (
    TranslationMode,
    CanonicalIntent,
    LatencyTier,
    translate_message_sync,
)

from app.translation.tier0_rules import (
    tier0_classify,
    check_weaver_trigger,
    check_spec_gate_trigger,
    check_critical_pipeline_trigger,
    check_overwatcher_trigger,
)


# =============================================================================
# WEAVER TRIGGER TESTS
# =============================================================================

class TestWeaverTriggers:
    """Tests for WEAVER_BUILD_SPEC intent detection."""
    
    def test_how_does_that_look_all_together(self):
        """Primary Weaver trigger phrase."""
        result = check_weaver_trigger("how does that look all together")
        assert result.matched
        assert result.intent == CanonicalIntent.WEAVER_BUILD_SPEC
    
    def test_how_does_that_look_all_together_question_mark(self):
        """With question mark."""
        result = check_weaver_trigger("how does that look all together?")
        assert result.matched
        assert result.intent == CanonicalIntent.WEAVER_BUILD_SPEC
    
    def test_weave_this_into_spec(self):
        """Explicit weave command."""
        result = check_weaver_trigger("weave this into a spec")
        assert result.matched
        assert result.intent == CanonicalIntent.WEAVER_BUILD_SPEC
    
    def test_compile_the_spec(self):
        """Compile spec variant."""
        result = check_weaver_trigger("compile the spec")
        assert result.matched
        assert result.intent == CanonicalIntent.WEAVER_BUILD_SPEC
    
    def test_put_that_all_together(self):
        """Put together variant."""
        result = check_weaver_trigger("put that all together")
        assert result.matched
        assert result.intent == CanonicalIntent.WEAVER_BUILD_SPEC
    
    def test_weaver_not_triggered_by_questions(self):
        """Questions about Weaver should not trigger it."""
        result = check_weaver_trigger("what does weaver do?")
        assert not result.matched
    
    def test_weaver_case_insensitive(self):
        """Triggers should be case-insensitive."""
        for variant in ["HOW DOES THAT LOOK ALL TOGETHER", "How Does That Look All Together"]:
            result = check_weaver_trigger(variant)
            assert result.matched, f"Should match: {variant}"


# =============================================================================
# SPEC GATE TRIGGER TESTS
# =============================================================================

class TestSpecGateTriggers:
    """Tests for SEND_TO_SPEC_GATE intent detection."""
    
    def test_critical_architecture(self):
        """Primary Spec Gate trigger - 'critical architecture'."""
        result = check_spec_gate_trigger("critical architecture")
        assert result.matched
        assert result.intent == CanonicalIntent.SEND_TO_SPEC_GATE
    
    def test_send_to_spec_gate(self):
        """Explicit send to spec gate."""
        result = check_spec_gate_trigger("send to spec gate")
        assert result.matched
        assert result.intent == CanonicalIntent.SEND_TO_SPEC_GATE
    
    def test_send_that_to_spec_gate(self):
        """Send that variant."""
        result = check_spec_gate_trigger("send that to spec gate")
        assert result.matched
        assert result.intent == CanonicalIntent.SEND_TO_SPEC_GATE
    
    def test_okay_send_to_spec_gate(self):
        """With okay prefix."""
        result = check_spec_gate_trigger("okay, send that to spec gate")
        assert result.matched
        assert result.intent == CanonicalIntent.SEND_TO_SPEC_GATE
    
    def test_validate_the_spec(self):
        """Validate variant."""
        result = check_spec_gate_trigger("validate the spec")
        assert result.matched
        assert result.intent == CanonicalIntent.SEND_TO_SPEC_GATE
    
    def test_run_spec_gate(self):
        """Run spec gate variant."""
        result = check_spec_gate_trigger("run spec gate")
        assert result.matched
        assert result.intent == CanonicalIntent.SEND_TO_SPEC_GATE
    
    def test_simple_yes_triggers_spec_gate(self):
        """Simple affirmative after Weaver prompt."""
        for affirmative in ["yes", "yep", "sure", "go ahead", "do it", "ok", "y"]:
            result = check_spec_gate_trigger(affirmative)
            assert result.matched, f"Should match affirmative: {affirmative}"
            assert result.intent == CanonicalIntent.SEND_TO_SPEC_GATE
    
    def test_spec_gate_not_triggered_by_questions(self):
        """Questions about Spec Gate should not trigger it."""
        result = check_spec_gate_trigger("what is spec gate?")
        assert not result.matched


# =============================================================================
# CRITICAL PIPELINE TRIGGER TESTS
# =============================================================================

class TestCriticalPipelineTriggers:
    """Tests for RUN_CRITICAL_PIPELINE_FOR_JOB intent detection."""
    
    def test_run_critical_pipeline(self):
        """Primary trigger."""
        result = check_critical_pipeline_trigger("run critical pipeline")
        assert result.matched
        assert result.intent == CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB
    
    def test_run_the_critical_pipeline(self):
        """With 'the'."""
        result = check_critical_pipeline_trigger("run the critical pipeline")
        assert result.matched
        assert result.intent == CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB
    
    def test_execute_critical_pipeline(self):
        """Execute variant."""
        result = check_critical_pipeline_trigger("execute critical pipeline")
        assert result.matched
        assert result.intent == CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB
    
    def test_start_the_pipeline(self):
        """Start pipeline variant."""
        result = check_critical_pipeline_trigger("start the pipeline")
        assert result.matched
        assert result.intent == CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB
    
    def test_critical_pipeline_standalone(self):
        """Just 'critical pipeline'."""
        result = check_critical_pipeline_trigger("critical pipeline")
        assert result.matched
        assert result.intent == CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB
    
    def test_critical_pipeline_case_insensitive(self):
        """Case insensitive matching."""
        for variant in ["RUN CRITICAL PIPELINE", "Run Critical Pipeline"]:
            result = check_critical_pipeline_trigger(variant)
            assert result.matched, f"Should match: {variant}"
    
    def test_pipeline_not_triggered_by_questions(self):
        """Questions about pipeline should not trigger it."""
        result = check_critical_pipeline_trigger("what does the critical pipeline do?")
        assert not result.matched


# =============================================================================
# OVERWATCHER TRIGGER TESTS
# =============================================================================

class TestOverwatcherTriggers:
    """Tests for OVERWATCHER_EXECUTE_CHANGES intent detection."""
    
    def test_send_to_overwatcher(self):
        """Primary trigger."""
        result = check_overwatcher_trigger("send to overwatcher")
        assert result.matched
        assert result.intent == CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES
    
    def test_send_it_to_overwatcher(self):
        """Send it variant."""
        result = check_overwatcher_trigger("send it to overwatcher")
        assert result.matched
        assert result.intent == CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES
    
    def test_send_that_to_overwatcher(self):
        """Send that variant."""
        result = check_overwatcher_trigger("send that to overwatcher")
        assert result.matched
        assert result.intent == CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES
    
    def test_overwatcher_execute(self):
        """Execute variant."""
        result = check_overwatcher_trigger("overwatcher execute")
        assert result.matched
        assert result.intent == CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES
    
    def test_run_overwatcher(self):
        """Run variant."""
        result = check_overwatcher_trigger("run overwatcher")
        assert result.matched
        assert result.intent == CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES
    
    def test_execute_overwatcher(self):
        """Execute overwatcher variant."""
        result = check_overwatcher_trigger("execute overwatcher")
        assert result.matched
        assert result.intent == CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES
    
    def test_overwatcher_standalone(self):
        """Just 'overwatcher'."""
        result = check_overwatcher_trigger("overwatcher")
        assert result.matched
        assert result.intent == CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES
    
    def test_okay_send_to_overwatcher(self):
        """With okay prefix."""
        result = check_overwatcher_trigger("okay, send it to overwatcher")
        assert result.matched
        assert result.intent == CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES
    
    def test_overwatcher_not_triggered_by_questions(self):
        """Questions about Overwatcher should not trigger it."""
        result = check_overwatcher_trigger("what is overwatcher?")
        assert not result.matched
    
    def test_overwatcher_not_triggered_by_tell_me(self):
        """'Tell me about' should not trigger."""
        result = check_overwatcher_trigger("tell me about overwatcher")
        assert not result.matched


# =============================================================================
# TIER 0 CLASSIFY INTEGRATION TESTS
# =============================================================================

class TestTier0ClassifyIntegration:
    """Tests for tier0_classify() correctly routing all flow commands."""
    
    def test_weaver_via_tier0(self):
        """Weaver triggers should be detected by tier0_classify."""
        result = tier0_classify("how does that look all together")
        assert result.matched
        assert result.intent == CanonicalIntent.WEAVER_BUILD_SPEC
    
    def test_spec_gate_via_tier0(self):
        """Spec Gate triggers should be detected by tier0_classify."""
        result = tier0_classify("critical architecture")
        assert result.matched
        assert result.intent == CanonicalIntent.SEND_TO_SPEC_GATE
    
    def test_critical_pipeline_via_tier0(self):
        """Critical Pipeline triggers should be detected by tier0_classify."""
        result = tier0_classify("run critical pipeline")
        assert result.matched
        assert result.intent == CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB
    
    def test_overwatcher_via_tier0(self):
        """Overwatcher triggers should be detected by tier0_classify."""
        result = tier0_classify("send to overwatcher")
        assert result.matched
        assert result.intent == CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES
    
    def test_wake_phrase_stripped(self):
        """Wake phrase prefix should be stripped before matching."""
        result = tier0_classify("Astra, command: run critical pipeline")
        assert result.matched
        assert result.intent == CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB


# =============================================================================
# END-TO-END TRANSLATION TESTS
# =============================================================================

class TestEndToEndCommandFlow:
    """End-to-end tests for the full command flow translation."""
    
    def test_weaver_with_wake_phrase(self):
        """Weaver command with wake phrase."""
        result = translate_message_sync("Astra, command: how does that look all together")
        assert result.mode == TranslationMode.COMMAND_CAPABLE
        assert result.resolved_intent == CanonicalIntent.WEAVER_BUILD_SPEC
        assert result.should_execute
    
    def test_spec_gate_with_wake_phrase(self):
        """Spec Gate command with wake phrase."""
        result = translate_message_sync("Astra, command: critical architecture")
        assert result.mode == TranslationMode.COMMAND_CAPABLE
        assert result.resolved_intent == CanonicalIntent.SEND_TO_SPEC_GATE
        assert result.should_execute
    
    def test_critical_pipeline_with_wake_phrase(self):
        """Critical Pipeline command with wake phrase."""
        result = translate_message_sync("Astra, command: run critical pipeline")
        assert result.mode == TranslationMode.COMMAND_CAPABLE
        assert result.resolved_intent == CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB
        # Note: may require confirmation gate
    
    def test_overwatcher_with_wake_phrase(self):
        """Overwatcher command with wake phrase."""
        result = translate_message_sync("Astra, command: send to overwatcher")
        assert result.mode == TranslationMode.COMMAND_CAPABLE
        assert result.resolved_intent == CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES
        # Note: may require confirmation gate
    
    def test_without_wake_phrase_is_chat(self):
        """Commands without wake phrase should be chat (safety)."""
        # Without wake phrase, these should be chat mode
        result = translate_message_sync("run critical pipeline")
        assert result.mode == TranslationMode.CHAT
        # Unless in command-capable UI context


# =============================================================================
# MISFIRE PREVENTION TESTS
# =============================================================================

class TestMisfirePrevention:
    """Tests to prevent command misfires."""
    
    def test_tell_me_about_spec_gate_is_chat(self):
        """'Tell me about Spec Gate' should NOT trigger Spec Gate."""
        result = translate_message_sync("Tell me about Spec Gate")
        assert result.mode == TranslationMode.CHAT
        assert result.resolved_intent == CanonicalIntent.CHAT_ONLY
        assert not result.should_execute
    
    def test_tell_me_about_overwatcher_is_chat(self):
        """'Tell me about Overwatcher' should NOT trigger Overwatcher."""
        result = translate_message_sync("Tell me about Overwatcher")
        assert result.mode == TranslationMode.CHAT
        assert result.resolved_intent == CanonicalIntent.CHAT_ONLY
        assert not result.should_execute
    
    def test_what_is_critical_pipeline_is_chat(self):
        """'What is the critical pipeline?' should NOT trigger it."""
        result = translate_message_sync("What is the critical pipeline?")
        assert result.mode == TranslationMode.CHAT
        assert not result.should_execute
    
    def test_how_does_overwatcher_work_is_chat(self):
        """'How does Overwatcher work?' should NOT trigger it."""
        result = translate_message_sync("How does Overwatcher work?")
        assert result.mode == TranslationMode.CHAT
        assert not result.should_execute
    
    def test_past_tense_pipeline_reference(self):
        """Past tense references should not trigger."""
        result = translate_message_sync("When you ran the critical pipeline yesterday...")
        assert result.mode == TranslationMode.CHAT
        assert not result.should_execute
    
    def test_hypothetical_overwatcher(self):
        """Hypothetical discussions should not trigger."""
        result = translate_message_sync("What if I send to overwatcher?")
        assert result.mode == TranslationMode.CHAT
        assert not result.should_execute


# =============================================================================
# SEQUENCE TESTS
# =============================================================================

class TestCommandSequence:
    """Tests for the intended command sequence flow."""
    
    def test_full_flow_sequence_intents(self):
        """Test that all four stages have distinct intents."""
        commands = [
            ("how does that look all together", CanonicalIntent.WEAVER_BUILD_SPEC),
            ("critical architecture", CanonicalIntent.SEND_TO_SPEC_GATE),
            ("run critical pipeline", CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB),
            ("send to overwatcher", CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES),
        ]
        
        for cmd, expected_intent in commands:
            result = tier0_classify(cmd)
            assert result.matched, f"Should match: {cmd}"
            assert result.intent == expected_intent, f"Wrong intent for: {cmd}"
    
    def test_flow_requires_wake_phrase_for_execution(self):
        """Without wake phrase, commands should not execute (safety)."""
        commands = [
            "how does that look all together",
            "critical architecture",
            "run critical pipeline",
            "send to overwatcher",
        ]
        
        for cmd in commands:
            result = translate_message_sync(cmd)
            # Without wake phrase, should be chat mode (safe default)
            assert result.mode == TranslationMode.CHAT, f"Should be CHAT without wake: {cmd}"


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""
    
    def test_empty_string(self):
        """Empty string should not crash."""
        result = tier0_classify("")
        assert not result.matched or result.intent == CanonicalIntent.CHAT_ONLY
    
    def test_whitespace_only(self):
        """Whitespace only should not crash."""
        result = tier0_classify("   ")
        assert not result.matched or result.intent == CanonicalIntent.CHAT_ONLY
    
    def test_partial_commands(self):
        """Partial commands should not match."""
        partials = [
            "run critical",  # missing "pipeline"
            "send to over",  # incomplete
            "critical arch",  # incomplete
        ]
        for partial in partials:
            result = tier0_classify(partial)
            # Should either not match or match as chat
            if result.matched:
                assert result.intent == CanonicalIntent.CHAT_ONLY, f"Partial matched wrong: {partial}"
    
    def test_commands_with_extra_whitespace(self):
        """Extra whitespace should be handled."""
        result = tier0_classify("  run critical pipeline  ")
        assert result.matched
        assert result.intent == CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
