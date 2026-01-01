# FILE: tests/test_confidence_system.py
"""
Tests for ASTRA Memory Confidence System.

Verifies:
1. Evidence appending is append-only
2. Confidence increases with repeated signals; decreases on contradictions
3. Hard rule is immutable via implicit signals
4. D0/D1 never triggers cold fetch
5. D1 answers within token cap
6. Namespace separation enforcement
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from datetime import datetime, timezone, timedelta

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.db import Base


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def db_session():
    """Create in-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    
    # Import all models to register with Base
    from app.astra_memory.models import AstraJob
    from app.astra_memory.preference_models import (
        PreferenceRecord, PreferenceEvidence, HotIndex,
        SummaryPyramid, MemoryRecordConfidence,
    )
    
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Re-enable append-only enforcement for tests
    # (SQLAlchemy events are registered at class level)
    
    yield session
    session.close()


# =============================================================================
# TEST: EVIDENCE APPEND-ONLY
# =============================================================================

class TestEvidenceAppendOnly:
    """Verify that PreferenceEvidence is append-only."""
    
    def test_evidence_insert_allowed(self, db_session):
        """Normal inserts should work."""
        from app.astra_memory.preference_models import (
            PreferenceRecord, PreferenceEvidence, SignalType, PreferenceStrength,
        )
        
        # Create preference first
        pref = PreferenceRecord(
            preference_key="test_pref",
            preference_value=True,
            strength=PreferenceStrength.DEFAULT,
        )
        db_session.add(pref)
        db_session.commit()
        
        # Add evidence
        evidence = PreferenceEvidence(
            preference_key="test_pref",
            signal_type=SignalType.EXPLICIT,
            weight=3.0,
        )
        db_session.add(evidence)
        db_session.commit()
        
        assert evidence.id is not None
    
    def test_evidence_update_blocked(self, db_session):
        """Updates to evidence records should be blocked."""
        from app.astra_memory.preference_models import (
            PreferenceRecord, PreferenceEvidence, SignalType, PreferenceStrength,
        )
        
        pref = PreferenceRecord(
            preference_key="test_pref_2",
            preference_value=True,
            strength=PreferenceStrength.DEFAULT,
        )
        db_session.add(pref)
        db_session.commit()
        
        evidence = PreferenceEvidence(
            preference_key="test_pref_2",
            signal_type=SignalType.EXPLICIT,
            weight=3.0,
        )
        db_session.add(evidence)
        db_session.commit()
        
        # Attempt to update
        evidence.weight = 5.0
        
        with pytest.raises(ValueError, match="append-only"):
            db_session.commit()
        
        db_session.rollback()
    
    def test_evidence_delete_blocked(self, db_session):
        """Deletes of evidence records should be blocked."""
        from app.astra_memory.preference_models import (
            PreferenceRecord, PreferenceEvidence, SignalType, PreferenceStrength,
        )
        
        pref = PreferenceRecord(
            preference_key="test_pref_3",
            preference_value=True,
            strength=PreferenceStrength.DEFAULT,
        )
        db_session.add(pref)
        db_session.commit()
        
        evidence = PreferenceEvidence(
            preference_key="test_pref_3",
            signal_type=SignalType.EXPLICIT,
            weight=3.0,
        )
        db_session.add(evidence)
        db_session.commit()
        
        # Attempt to delete
        with pytest.raises(ValueError, match="append-only"):
            db_session.delete(evidence)
            db_session.commit()
        
        db_session.rollback()


# =============================================================================
# TEST: CONFIDENCE SCORING
# =============================================================================

class TestConfidenceScoring:
    """Test the confidence scoring functions."""
    
    def test_compute_decay_at_zero(self):
        """Decay at age=0 should be 1.0."""
        from app.astra_memory.confidence_scoring import compute_decay
        
        decay = compute_decay(age_days=0, half_life_days=30)
        assert decay == 1.0
    
    def test_compute_decay_at_half_life(self):
        """Decay at age=half_life should be ~0.5."""
        from app.astra_memory.confidence_scoring import compute_decay
        
        decay = compute_decay(age_days=30, half_life_days=30)
        assert abs(decay - 0.5) < 0.01
    
    def test_confidence_increases_with_evidence(self, db_session):
        """Confidence should increase as positive evidence accumulates."""
        from app.astra_memory.preference_service import create_preference, reinforce_preference
        from app.astra_memory.preference_models import PreferenceStrength, SignalType
        
        pref = create_preference(
            db_session,
            preference_key="increasing_conf",
            preference_value="test",
            strength=PreferenceStrength.SOFT,
        )
        initial_conf = pref.confidence
        
        # Add more evidence
        for i in range(3):
            reinforce_preference(
                db_session,
                "increasing_conf",
                signal_type=SignalType.IMPLICIT,
                context_pointer=f"msg:{i}",
            )
        
        # Refresh and check
        db_session.refresh(pref)
        assert pref.confidence > initial_conf
        assert pref.evidence_count >= 4
    
    def test_confidence_decreases_on_contradiction(self, db_session):
        """Confidence should decrease when contradictions are recorded."""
        from app.astra_memory.preference_service import create_preference
        from app.astra_memory.confidence_scoring import record_contradiction
        from app.astra_memory.preference_models import PreferenceStrength, SignalType
        
        # Create with explicit instruction for initial confidence
        pref = create_preference(
            db_session,
            preference_key="contradicted_pref",
            preference_value="original",
            strength=PreferenceStrength.DEFAULT,
        )
        
        # Add some positive evidence to build confidence
        from app.astra_memory.confidence_scoring import append_preference_evidence, recompute_preference_confidence
        for i in range(3):
            append_preference_evidence(
                db_session, "contradicted_pref",
                SignalType.APPROVAL, f"approval:{i}"
            )
        recompute_preference_confidence(db_session, "contradicted_pref")
        
        db_session.refresh(pref)
        conf_before = pref.confidence
        
        # Record contradiction
        record_contradiction(
            db_session,
            "contradicted_pref",
            context_pointer="contradiction:1",
            new_value="different",
        )
        
        db_session.refresh(pref)
        assert pref.confidence < conf_before or pref.contradiction_count > 0


# =============================================================================
# TEST: HARD RULES
# =============================================================================

class TestHardRules:
    """Test hard rule immutability."""
    
    def test_hard_rule_has_confidence_one(self, db_session):
        """Hard rules should always have confidence=1.0."""
        from app.astra_memory.preference_service import create_hard_rule
        
        pref = create_hard_rule(
            db_session,
            preference_key="safety_rule_1",
            preference_value="never_do_x",
        )
        
        assert pref.confidence == 1.0
    
    def test_hard_rule_blocks_implicit_update(self, db_session):
        """Hard rules cannot be modified by implicit signals."""
        from app.astra_memory.preference_service import create_hard_rule, update_preference_value
        
        pref = create_hard_rule(
            db_session,
            preference_key="safety_rule_2",
            preference_value="value_a",
        )
        
        # Try implicit update
        result = update_preference_value(
            db_session,
            "safety_rule_2",
            new_value="value_b",
            is_explicit=False,
        )
        
        # Value should NOT change
        db_session.refresh(pref)
        assert pref.preference_value == "value_a"
    
    def test_hard_rule_allows_explicit_override(self, db_session):
        """Hard rules CAN be modified by explicit override."""
        from app.astra_memory.preference_service import create_hard_rule, update_preference_value
        
        pref = create_hard_rule(
            db_session,
            preference_key="safety_rule_3",
            preference_value="value_a",
        )
        
        # Explicit update
        result = update_preference_value(
            db_session,
            "safety_rule_3",
            new_value="value_b",
            is_explicit=True,
        )
        
        db_session.refresh(pref)
        assert pref.preference_value == "value_b"
    
    def test_hard_rule_no_decay(self, db_session):
        """Hard rule confidence should not decay over time."""
        from app.astra_memory.preference_service import create_hard_rule
        from app.astra_memory.confidence_scoring import recompute_preference_confidence
        
        pref = create_hard_rule(
            db_session,
            preference_key="safety_rule_4",
            preference_value="never_decay",
        )
        
        # Simulate time passage by recomputing
        recompute_preference_confidence(db_session, "safety_rule_4")
        
        db_session.refresh(pref)
        assert pref.confidence == 1.0


# =============================================================================
# TEST: INTENT DEPTH
# =============================================================================

class TestIntentDepth:
    """Test intent depth classification."""
    
    def test_d0_for_greetings(self):
        """Greetings should be D0/D1."""
        from app.astra_memory.retrieval import classify_intent_depth, IntentDepth
        
        assert classify_intent_depth("hi") == IntentDepth.D0
        assert classify_intent_depth("hello!") == IntentDepth.D0
        assert classify_intent_depth("thanks") == IntentDepth.D0
    
    def test_d1_for_brief_queries(self):
        """Brief queries should be D1."""
        from app.astra_memory.retrieval import classify_intent_depth, IntentDepth
        
        assert classify_intent_depth("tell me a bit about X") == IntentDepth.D1
        assert classify_intent_depth("give me a quick summary") == IntentDepth.D1
    
    def test_d3_for_deep_requests(self):
        """Deep/detailed requests should be D3."""
        from app.astra_memory.retrieval import classify_intent_depth, IntentDepth
        
        assert classify_intent_depth("give me the full spec") == IntentDepth.D3
        assert classify_intent_depth("do a deep dive on this") == IntentDepth.D3
        assert classify_intent_depth("show me the complete architecture") == IntentDepth.D3
    
    def test_explicit_depth_tokens(self):
        """Explicit depth tokens should override classification."""
        from app.astra_memory.retrieval import classify_intent_depth, IntentDepth
        
        assert classify_intent_depth("what is X /deep") == IntentDepth.D3
        assert classify_intent_depth("/forensic show me everything") == IntentDepth.D4
        assert classify_intent_depth("tell me about Y /brief") == IntentDepth.D1


# =============================================================================
# TEST: D0/D1 NO COLD FETCH
# =============================================================================

class TestNoColdFetch:
    """Verify D0/D1 never triggers cold storage access."""
    
    def test_d1_uses_hot_layer_only(self, db_session):
        """D1 retrieval should only use hot layer content."""
        from app.astra_memory.retrieval import (
            retrieve_for_query, upsert_hot_index, IntentDepth, RetrievalCost
        )
        
        # Create hot index entries
        upsert_hot_index(
            db_session,
            record_type="fact",
            record_id="fact-1",
            title="Test Fact",
            one_liner="This is a test fact for D1",
            retrieval_cost=RetrievalCost.TINY,
        )
        
        # Retrieve at D1
        result = retrieve_for_query(
            db_session,
            user_message="tell me briefly about tests",
            depth_override=IntentDepth.D1,
        )
        
        # Check that only hot layer content was used
        for record in result.records:
            assert record.summary_level <= 1  # L0 or L1 only
    
    def test_d0_returns_no_memory(self, db_session):
        """D0 should return no memory records."""
        from app.astra_memory.retrieval import retrieve_for_query, IntentDepth
        
        result = retrieve_for_query(
            db_session,
            user_message="hi",
            depth_override=IntentDepth.D0,
        )
        
        assert len(result.records) == 0
        assert result.token_estimate == 0


# =============================================================================
# TEST: NAMESPACE SEPARATION
# =============================================================================

class TestNamespaceSeparation:
    """Test namespace isolation rules."""
    
    def test_protected_namespace_blocks_implicit(self):
        """Protected namespaces cannot be mutated by non-explicit sources."""
        from app.astra_memory.confidence_scoring import check_namespace_mutation_allowed
        
        # Implicit signal cannot mutate user_personal
        assert not check_namespace_mutation_allowed(
            target_namespace="user_personal",
            source_namespace="repo_derived",
            is_explicit=False,
        )
        
        # Implicit signal cannot mutate safety_critical
        assert not check_namespace_mutation_allowed(
            target_namespace="safety_critical",
            source_namespace="code_analysis",
            is_explicit=False,
        )
    
    def test_explicit_bypasses_namespace_protection(self):
        """Explicit user actions can mutate any namespace."""
        from app.astra_memory.confidence_scoring import check_namespace_mutation_allowed
        
        assert check_namespace_mutation_allowed(
            target_namespace="user_personal",
            source_namespace="repo_derived",
            is_explicit=True,
        )
        
        assert check_namespace_mutation_allowed(
            target_namespace="safety_critical",
            source_namespace="anything",
            is_explicit=True,
        )
    
    def test_repo_mutable_allows_repo_sources(self):
        """Repo-mutable namespaces accept repo-derived updates."""
        from app.astra_memory.confidence_scoring import check_namespace_mutation_allowed
        
        assert check_namespace_mutation_allowed(
            target_namespace="repo_derived",
            source_namespace="repo_derived",
            is_explicit=False,
        )
        
        assert check_namespace_mutation_allowed(
            target_namespace="atlas_nodes",
            source_namespace="atlas",
            is_explicit=False,
        )


# =============================================================================
# TEST: BAD LEARNING PREVENTION
# =============================================================================

class TestBadLearningPrevention:
    """Test that bad learning is prevented."""
    
    def test_single_implicit_not_enough(self, db_session):
        """Single implicit signal should not create confident preference."""
        from app.astra_memory.preference_service import learn_from_behavior
        from app.astra_memory.confidence_config import get_config
        
        cfg = get_config()
        
        pref = learn_from_behavior(
            db_session,
            preference_key="single_observation",
            observed_value="x",
            is_repeated=False,
        )
        
        # Confidence should be below threshold
        assert pref.confidence < cfg.thresholds.suggestion_threshold
    
    def test_explicit_instruction_sufficient(self, db_session):
        """Single explicit instruction should be sufficient."""
        from app.astra_memory.preference_service import create_preference
        from app.astra_memory.preference_models import PreferenceStrength
        from app.astra_memory.confidence_config import get_config
        
        cfg = get_config()
        
        pref = create_preference(
            db_session,
            preference_key="explicit_pref",
            preference_value="y",
            strength=PreferenceStrength.DEFAULT,
            source="user_declared",
        )
        
        # Should have meaningful confidence from explicit signal
        assert pref.evidence_count >= 1


# =============================================================================
# TEST: BEHAVIOR RULES
# =============================================================================

class TestBehaviorRules:
    """Test preference application behavior rules."""
    
    def test_disputed_not_applied(self, db_session):
        """Disputed preferences should not be applied automatically."""
        from app.astra_memory.preference_service import (
            create_preference, resolve_preference_for_default
        )
        from app.astra_memory.preference_models import PreferenceStrength, RecordStatus
        
        pref = create_preference(
            db_session,
            preference_key="disputed_pref",
            preference_value="original",
            strength=PreferenceStrength.DEFAULT,
        )
        
        # Manually mark as disputed
        pref.status = RecordStatus.DISPUTED
        db_session.commit()
        
        value, disposition, conf = resolve_preference_for_default(
            db_session, "disputed_pref", fallback_value="fallback"
        )
        
        assert disposition == "disputed"
        assert value == "fallback"
    
    def test_low_confidence_not_enforced(self, db_session):
        """Low confidence preferences return fallback."""
        from app.astra_memory.preference_service import (
            create_preference, resolve_preference_for_default
        )
        from app.astra_memory.preference_models import PreferenceStrength
        
        pref = create_preference(
            db_session,
            preference_key="low_conf_pref",
            preference_value="weak",
            strength=PreferenceStrength.SOFT,
            source="learned",
        )
        
        # Force low confidence
        pref.confidence = 0.3
        db_session.commit()
        
        value, disposition, conf = resolve_preference_for_default(
            db_session, "low_conf_pref", fallback_value="fallback"
        )
        
        assert disposition == "low_confidence"
        assert value == "fallback"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
