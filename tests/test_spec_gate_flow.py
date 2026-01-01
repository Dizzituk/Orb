# FILE: tests/test_spec_gate_flow.py
"""
Tests for ASTRA Spec Gate flow intents.

Tests the Ramble → Weaver → Spec Gate → Pipeline flow:
1. WEAVER_BUILD_SPEC - "How does that look all together?"
2. SEND_TO_SPEC_GATE - "Send to Spec Gate"
3. RUN_CRITICAL_PIPELINE_FOR_JOB - "Run critical pipeline" (requires confirmation)
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
    tier0_classify,
    check_weaver_trigger,
    check_spec_gate_trigger,
    check_critical_pipeline_trigger,
)


# =============================================================================
# WEAVER TRIGGER TESTS
# =============================================================================

class TestWeaverTriggers:
    """Tests for Weaver (spec building) triggers."""
    
    def test_how_does_that_look_all_together(self):
        """Primary natural trigger for Weaver."""
        result = check_weaver_trigger("How does that look all together?")
        assert result.matched
        assert result.intent == CanonicalIntent.WEAVER_BUILD_SPEC
    
    def test_how_does_that_look_all_together_no_question_mark(self):
        """Should work without question mark."""
        result = check_weaver_trigger("How does that look all together")
        assert result.matched
        assert result.intent == CanonicalIntent.WEAVER_BUILD_SPEC
    
    def test_how_does_that_look_lowercase(self):
        """Case insensitive."""
        result = check_weaver_trigger("how does that look all together?")
        assert result.matched
        assert result.intent == CanonicalIntent.WEAVER_BUILD_SPEC
    
    def test_weave_this_into_a_spec(self):
        """Explicit weave command."""
        result = check_weaver_trigger("Weave this into a spec")
        assert result.matched
        assert result.intent == CanonicalIntent.WEAVER_BUILD_SPEC
    
    def test_build_spec_from_ramble(self):
        """Explicit build spec command."""
        result = check_weaver_trigger("Build spec from ramble")
        assert result.matched
        assert result.intent == CanonicalIntent.WEAVER_BUILD_SPEC
    
    def test_compile_the_spec(self):
        """Compile spec command."""
        result = check_weaver_trigger("Compile the spec")
        assert result.matched
        assert result.intent == CanonicalIntent.WEAVER_BUILD_SPEC
    
    def test_put_that_all_together(self):
        """Natural consolidation trigger."""
        result = check_weaver_trigger("Put that all together")
        assert result.matched
        assert result.intent == CanonicalIntent.WEAVER_BUILD_SPEC
    
    def test_consolidate_into_spec(self):
        """Consolidate command."""
        result = check_weaver_trigger("Consolidate that into a spec")
        assert result.matched
        assert result.intent == CanonicalIntent.WEAVER_BUILD_SPEC
    
    def test_turn_this_into_a_spec(self):
        """Turn into spec command."""
        result = check_weaver_trigger("Turn this into a spec")
        assert result.matched
        assert result.intent == CanonicalIntent.WEAVER_BUILD_SPEC
    
    def test_does_not_trigger_on_questions_about_weaving(self):
        """Questions about weaving should NOT trigger."""
        questions = [
            "What does weaving mean?",
            "How do I weave a spec?",
            "Can you explain spec building?",
            "Tell me about the weaver",
        ]
        for q in questions:
            result = check_weaver_trigger(q)
            assert not result.matched, f"Should not match: {q}"
    
    def test_does_not_trigger_on_random_text(self):
        """Random text should not trigger."""
        random_texts = [
            "Hello there",
            "I want to build an app",
            "The architecture looks good",
            "Let's discuss the design",
        ]
        for text in random_texts:
            result = check_weaver_trigger(text)
            assert not result.matched, f"Should not match: {text}"


# =============================================================================
# SPEC GATE TRIGGER TESTS
# =============================================================================

class TestSpecGateTriggers:
    """Tests for Spec Gate (validation) triggers."""
    
    def test_send_to_spec_gate(self):
        """Primary trigger."""
        result = check_spec_gate_trigger("Send to Spec Gate")
        assert result.matched
        assert result.intent == CanonicalIntent.SEND_TO_SPEC_GATE
    
    def test_send_that_to_spec_gate(self):
        """With 'that'."""
        result = check_spec_gate_trigger("Send that to Spec Gate")
        assert result.matched
        assert result.intent == CanonicalIntent.SEND_TO_SPEC_GATE
    
    def test_okay_send_to_spec_gate(self):
        """With confirmation prefix."""
        result = check_spec_gate_trigger("Okay, send that to Spec Gate")
        assert result.matched
        assert result.intent == CanonicalIntent.SEND_TO_SPEC_GATE
    
    def test_ok_send_to_spec_gate(self):
        """With 'ok' prefix."""
        result = check_spec_gate_trigger("Ok, send to Spec Gate")
        assert result.matched
        assert result.intent == CanonicalIntent.SEND_TO_SPEC_GATE
    
    def test_validate_the_spec(self):
        """Validate command."""
        result = check_spec_gate_trigger("Validate the spec")
        assert result.matched
        assert result.intent == CanonicalIntent.SEND_TO_SPEC_GATE
    
    def test_run_spec_gate(self):
        """Run command."""
        result = check_spec_gate_trigger("Run Spec Gate")
        assert result.matched
        assert result.intent == CanonicalIntent.SEND_TO_SPEC_GATE
    
    def test_submit_spec_for_validation(self):
        """Submit command."""
        result = check_spec_gate_trigger("Submit spec for validation")
        assert result.matched
        assert result.intent == CanonicalIntent.SEND_TO_SPEC_GATE
    
    def test_specgate_no_space(self):
        """SpecGate as one word."""
        result = check_spec_gate_trigger("Send to SpecGate")
        assert result.matched
        assert result.intent == CanonicalIntent.SEND_TO_SPEC_GATE
    
    def test_lowercase(self):
        """Case insensitive."""
        result = check_spec_gate_trigger("send to spec gate")
        assert result.matched
        assert result.intent == CanonicalIntent.SEND_TO_SPEC_GATE
    
    def test_does_not_trigger_on_questions(self):
        """Questions about Spec Gate should NOT trigger."""
        questions = [
            "What does Spec Gate do?",
            "How does Spec Gate validate?",
            "Tell me about the Spec Gate",
            "Can you explain Spec Gate?",
        ]
        for q in questions:
            result = check_spec_gate_trigger(q)
            assert not result.matched, f"Should not match: {q}"


# =============================================================================
# CRITICAL PIPELINE TRIGGER TESTS
# =============================================================================

class TestCriticalPipelineTriggers:
    """Tests for critical pipeline triggers."""
    
    def test_run_critical_pipeline(self):
        """Primary trigger."""
        result = check_critical_pipeline_trigger("Run critical pipeline")
        assert result.matched
        assert result.intent == CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB
    
    def test_execute_critical_pipeline(self):
        """Execute variant."""
        result = check_critical_pipeline_trigger("Execute critical pipeline")
        assert result.matched
        assert result.intent == CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB
    
    def test_start_the_pipeline(self):
        """Start variant."""
        result = check_critical_pipeline_trigger("Start the pipeline")
        assert result.matched
        assert result.intent == CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB
    
    def test_does_not_trigger_on_questions(self):
        """Questions about pipeline should NOT trigger."""
        questions = [
            "What does the critical pipeline do?",
            "How does the pipeline work?",
            "Tell me about the critical pipeline",
            "Can you explain the pipeline?",
        ]
        for q in questions:
            result = check_critical_pipeline_trigger(q)
            assert not result.matched, f"Should not match: {q}"


# =============================================================================
# END-TO-END SPEC GATE FLOW TESTS
# =============================================================================

class TestSpecGateFlowEndToEnd:
    """End-to-end tests for the Spec Gate flow."""
    
    def test_weaver_trigger_full_translation(self):
        """Weaver trigger through full translation pipeline."""
        result = translate_message_sync("How does that look all together?")
        # Without wake phrase, should be CHAT mode but intent detected
        # The actual behavior depends on whether we're in command mode
        assert result.resolved_intent in (
            CanonicalIntent.WEAVER_BUILD_SPEC,
            CanonicalIntent.CHAT_ONLY,  # If not in command mode
        )
    
    def test_spec_gate_trigger_with_wake_phrase(self):
        """Spec Gate trigger with wake phrase."""
        result = translate_message_sync("Astra, command: Send to Spec Gate")
        assert result.mode == TranslationMode.COMMAND_CAPABLE
        assert result.resolved_intent == CanonicalIntent.SEND_TO_SPEC_GATE
    
    def test_weaver_trigger_with_wake_phrase(self):
        """Weaver trigger with wake phrase."""
        result = translate_message_sync("Astra, command: How does that look all together?")
        assert result.mode == TranslationMode.COMMAND_CAPABLE
        # Should be blocked by directive gate (it's a question)
        # OR should match as WEAVER_BUILD_SPEC since it's an exact trigger
        # Depends on implementation - either is acceptable
        assert result.resolved_intent in (
            CanonicalIntent.WEAVER_BUILD_SPEC,
            CanonicalIntent.CHAT_ONLY,
        )
    
    def test_questions_about_spec_gate_dont_trigger(self):
        """Talking about Spec Gate should NOT trigger it."""
        questions = [
            "What is the Spec Gate?",
            "How does Spec Gate work?",
            "Tell me about Spec Gate",
            "Explain the Spec Gate validation process",
        ]
        for q in questions:
            result = translate_message_sync(q)
            assert not result.should_execute, f"Should not execute for: {q}"
            assert result.resolved_intent == CanonicalIntent.CHAT_ONLY
    
    def test_questions_about_weaver_dont_trigger(self):
        """Talking about Weaver should NOT trigger it."""
        questions = [
            "What is the Weaver?",
            "How does the spec builder work?",
            "Tell me about weaving specs",
            "What does 'compile the spec' do?",
        ]
        for q in questions:
            result = translate_message_sync(q)
            assert not result.should_execute, f"Should not execute for: {q}"
            assert result.resolved_intent == CanonicalIntent.CHAT_ONLY


# =============================================================================
# TIER 0 INTEGRATION TESTS
# =============================================================================

class TestTier0SpecGateIntegration:
    """Tests that Tier 0 properly handles Spec Gate flow."""
    
    def test_tier0_catches_weaver_trigger(self):
        """Tier 0 should catch Weaver triggers."""
        result = tier0_classify("How does that look all together?")
        assert result.matched
        assert result.intent == CanonicalIntent.WEAVER_BUILD_SPEC
    
    def test_tier0_catches_spec_gate_trigger(self):
        """Tier 0 should catch Spec Gate triggers."""
        result = tier0_classify("Send to Spec Gate")
        assert result.matched
        assert result.intent == CanonicalIntent.SEND_TO_SPEC_GATE
    
    def test_tier0_catches_critical_pipeline_trigger(self):
        """Tier 0 should catch critical pipeline triggers."""
        result = tier0_classify("Run critical pipeline")
        assert result.matched
        assert result.intent == CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB
    
    def test_tier0_blocks_spec_gate_questions(self):
        """Tier 0 should block questions about Spec Gate."""
        result = tier0_classify("What is Spec Gate?")
        assert result.matched
        assert result.intent == CanonicalIntent.CHAT_ONLY


# =============================================================================
# FLOW ORDERING TESTS
# =============================================================================

class TestSpecGateFlowOrdering:
    """Tests for proper flow ordering - can't skip steps."""
    
    def test_pipeline_requires_spec_context(self):
        """Critical pipeline should require spec_id context."""
        from app.translation import check_context_gate
        
        # Without spec_id, should fail context gate
        result = check_context_gate(
            CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB,
            provided_context={}
        )
        assert not result.passed
        assert "spec_id" in result.missing_context or "job_id" in result.missing_context
    
    def test_pipeline_with_spec_context_passes(self):
        """Critical pipeline with proper context should pass."""
        from app.translation import check_context_gate
        
        result = check_context_gate(
            CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB,
            provided_context={"job_id": "abc123", "spec_id": "spec_456"}
        )
        assert result.passed


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
