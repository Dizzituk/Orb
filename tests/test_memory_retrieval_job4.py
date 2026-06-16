# FILE: tests/test_memory_retrieval_job4.py
# Purpose: Tests for Job 4 retrieval — confidence gate + bounded graph walk.
# Called-by: pytest
# Depends-on: app.astra_memory.rerank, app.astra_memory.retrieval,
#             app.llm.routing.graph_context, app.llm.routing.memory_injection
# Last-renovated: 2026-06-15
"""
Tests for MEMORY_JOBS_SPEC Job 4 — retrieve by topic and inject.

Two task surfaces, plus the front-door integration that proves both stay inside
the single gated door (no new injection path):

  Task 1 — the confidence gate (app/astra_memory/rerank.py): three sharp items
    beat twenty mushy ones. A clear top cluster sheds its tail; a uniformly-weak
    or tiny pool is left untouched; the top hit never gets gated to nothing.

  Task 2 — the bounded graph walk (app/llm/routing/graph_context.py): a strong
    consolidated cross-domain edge ("investments funds the Portugal move")
    surfaces when its topic is in play; respects the strength threshold; walks
    at most two hops; stays silent with no topic; obeys the kill-switch.

  Task 3 — the front door (build_memory_context include_graph): the graph block
    is folded into the one de-duped memory block, behind the D0-D4 gate.

Run with: pytest tests/test_memory_retrieval_job4.py -v
"""

import pytest


# =========================================================================
# Task 1 — confidence gate
# =========================================================================

def _cands(scores):
    """Build ranked RetrievalCandidates from a list of scores (desc)."""
    from app.astra_memory.retrieval import RetrievalCandidate
    return [
        RetrievalCandidate(
            record_type="memory", record_id=str(i), title=f"r{i}",
            relevance_score=s,
        )
        for i, s in enumerate(scores)
    ]


class TestConfidenceGate:
    def test_keeps_top_cluster_drops_mushy_tail(self):
        from app.astra_memory.rerank import apply_confidence_gate
        # Three sharp (1.5/1.4/1.3), seven mushy base-priority (0.5).
        cands = _cands([1.5, 1.4, 1.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        kept = apply_confidence_gate(cands)
        assert len(kept) == 3
        assert {c.relevance_score for c in kept} == {1.5, 1.4, 1.3}

    def test_noop_for_small_pool(self):
        from app.astra_memory.rerank import apply_confidence_gate
        # <= _MIN_TO_GATE (6) candidates: nothing to prune even with a tail.
        cands = _cands([1.5, 0.5, 0.5, 0.5, 0.5])
        kept = apply_confidence_gate(cands)
        assert len(kept) == 5

    def test_noop_for_uniform_pool(self):
        from app.astra_memory.rerank import apply_confidence_gate
        # A uniformly-weak result set has no clear winner — leave it alone so we
        # never regress the broad-query case down to nothing.
        cands = _cands([0.5] * 10)
        kept = apply_confidence_gate(cands)
        assert len(kept) == 10

    def test_never_empties(self):
        from app.astra_memory.rerank import apply_confidence_gate
        # One strong hit, the rest zero — the top hit must always survive.
        cands = _cands([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        kept = apply_confidence_gate(cands)
        assert len(kept) == 1
        assert kept[0].relevance_score == 1.0

    def test_kill_switch(self, monkeypatch):
        from app.astra_memory.rerank import apply_confidence_gate
        monkeypatch.setenv("ASTRA_RETRIEVAL_CONFIDENCE_GATE", "off")
        cands = _cands([1.5, 1.4, 1.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        kept = apply_confidence_gate(cands)
        assert len(kept) == 10  # disabled → unchanged

    def test_ratio_env_override(self, monkeypatch):
        from app.astra_memory.rerank import apply_confidence_gate
        # A stricter ratio keeps a tighter cluster.
        monkeypatch.setenv("ASTRA_RETRIEVAL_KEEP_RATIO", "0.95")
        cands = _cands([1.5, 1.45, 1.0, 0.9, 0.8, 0.7, 0.6])
        kept = apply_confidence_gate(cands)
        # bar = 1.5 * 0.95 = 1.425 → only 1.5 and 1.45 survive.
        assert len(kept) == 2


# =========================================================================
# Task 2 — bounded graph walk
# =========================================================================

@pytest.fixture
def graph(tmp_path):
    """A throwaway KnowledgeGraph on a temp path (never the live graph json)."""
    from app.intelligent_memory.graph.knowledge_graph import KnowledgeGraph
    g = KnowledgeGraph(graph_path=str(tmp_path / "kg.json"))
    return g


def _add_topic(g, domain, name):
    from app.intelligent_memory.graph import Entity, EntityType
    g.add_entity(Entity(entity_id=f"topic:{domain}", entity_type=EntityType.TOPIC, name=name))


def _add_link(g, src_domain, tgt_domain, relation, weight=2.0):
    from app.intelligent_memory.graph import Relationship, RelationType
    g.add_relationship(Relationship(
        source_id=f"topic:{src_domain}", target_id=f"topic:{tgt_domain}",
        relation_type=RelationType.RELATED_TO, weight=weight,
        metadata={"relation": relation, "occurrences": int(weight)},
    ))


@pytest.fixture
def use_graph(monkeypatch, graph):
    """Point build_graph_context's singleton accessor at the temp graph."""
    monkeypatch.setattr(
        "app.intelligent_memory.graph.get_knowledge_graph", lambda: graph,
    )
    return graph


class TestGraphWalk:
    def test_surfaces_strong_cross_domain_link(self, use_graph):
        from app.llm.routing.graph_context import build_graph_context
        g = use_graph
        _add_topic(g, "investments", "investments")
        _add_topic(g, "portugal", "Portugal move")
        _add_link(g, "investments", "portugal", "funds", weight=2.0)

        block = build_graph_context(["investments"], records=[])
        assert "CONNECTED MEMORY" in block
        assert "investments funds Portugal move" in block

    def test_respects_strength_threshold(self, use_graph, monkeypatch):
        from app.llm.routing.graph_context import build_graph_context
        g = use_graph
        _add_topic(g, "investments", "investments")
        _add_topic(g, "portugal", "Portugal move")
        _add_link(g, "investments", "portugal", "funds", weight=1.0)  # one-off
        # Require recurring links only — the weak one-off must not surface.
        monkeypatch.setenv("ASTRA_GRAPH_WALK_MIN_STRENGTH", "5.0")
        assert build_graph_context(["investments"], records=[]) == ""

    def test_silent_without_topic(self, use_graph):
        from app.llm.routing.graph_context import build_graph_context
        g = use_graph
        _add_topic(g, "investments", "investments")
        _add_topic(g, "portugal", "Portugal move")
        _add_link(g, "investments", "portugal", "funds")
        # No query tags and no records → no anchor → silent.
        assert build_graph_context([], records=[]) == ""

    def test_two_hop_reaches_far_node(self, use_graph):
        from app.llm.routing.graph_context import build_graph_context
        g = use_graph
        _add_topic(g, "investments", "investments")
        _add_topic(g, "portugal", "Portugal move")
        _add_topic(g, "legal", "the legal claim")
        _add_link(g, "investments", "portugal", "funds")        # hop 1
        _add_link(g, "portugal", "legal", "depends on")          # hop 2
        block = build_graph_context(["investments"], records=[])
        # Both the 1-hop and the 2-hop connection should be reachable.
        assert "investments funds Portugal move" in block
        assert "Portugal move depends on the legal claim" in block

    def test_two_hop_cap_blocks_third_hop(self, use_graph, monkeypatch):
        from app.llm.routing.graph_context import build_graph_context
        g = use_graph
        for d, n in [("investments", "investments"), ("portugal", "Portugal move"),
                     ("legal", "the legal claim"), ("work", "the consultancy")]:
            _add_topic(g, d, n)
        _add_link(g, "investments", "portugal", "funds")     # hop 1
        _add_link(g, "portugal", "legal", "depends on")       # hop 2
        _add_link(g, "legal", "work", "paid by")              # hop 3 (beyond cap)
        monkeypatch.setenv("ASTRA_GRAPH_WALK_MAX_LINKS", "10")  # don't cap by count
        block = build_graph_context(["investments"], records=[])
        assert "the legal claim paid by the consultancy" not in block

    def test_kill_switch(self, use_graph, monkeypatch):
        from app.llm.routing.graph_context import build_graph_context
        g = use_graph
        _add_topic(g, "investments", "investments")
        _add_topic(g, "portugal", "Portugal move")
        _add_link(g, "investments", "portugal", "funds")
        monkeypatch.setenv("ASTRA_GRAPH_WALK", "off")
        assert build_graph_context(["investments"], records=[]) == ""

    def test_anchors_on_retrieved_record_tags(self, use_graph):
        """The anchor set includes topics carried by retrieved records, not just
        the literal query tags — so a semantically-rescued record still triggers
        the walk."""
        from app.llm.routing.graph_context import build_graph_context

        class _Rec:
            title = "TAO position sizing"
            content = "notes on the investments thesis and the TAO allocation"

        g = use_graph
        _add_topic(g, "investments", "investments")
        _add_topic(g, "portugal", "Portugal move")
        _add_link(g, "investments", "portugal", "funds")
        # Empty query tags; the record's text tags to 'investments'.
        block = build_graph_context([], records=[_Rec()])
        assert "investments funds Portugal move" in block


# =========================================================================
# Task 3 — front-door integration (graph block folds into the one block)
# =========================================================================

class _FakeResult:
    """Stand-in for retrieve_for_query's result."""
    def __init__(self, records):
        self.records = records
        self.records_expanded = len(records)


@pytest.fixture
def front_door_env(monkeypatch, graph):
    """Neutralise the heavy semantic sources but keep the REAL depth classifier
    and topic tagger, and point the graph walk at the temp graph — so the test
    exercises the actual front-door wiring."""
    import app.llm.routing.memory_injection as mi
    from app.astra_memory import classify_intent_depth, IntentDepth

    monkeypatch.setattr(mi, "_MEMORY_API", {
        "classify_intent_depth": classify_intent_depth,
        "retrieve_for_query": lambda **kw: _FakeResult([]),
        "get_applicable_preferences": lambda *a, **k: [],
        "IntentDepth": IntentDepth,
    })
    monkeypatch.setattr(
        "app.intelligent_memory.graph.get_knowledge_graph", lambda: graph,
    )
    try:
        monkeypatch.setattr(
            "app.rag.retrieval.chat_injection.build_repo_context",
            lambda *a, **k: "",
        )
    except Exception:
        pass
    return graph


class TestFrontDoorGraph:
    def test_graph_block_folds_into_memory_block(self, front_door_env):
        from app.llm.routing.memory_injection import build_memory_context
        g = front_door_env
        _add_topic(g, "investments", "investments")
        _add_topic(g, "portugal", "Portugal move")
        _add_link(g, "investments", "portugal", "funds", weight=2.0)

        ctx = build_memory_context(
            None,
            messages=[{"role": "user",
                       "content": "how are my investments looking this month?"}],
            include_graph=True,
        )
        block = ctx.format_for_system_prompt()
        assert ctx.graph_text  # the walk fired
        assert "CONNECTED MEMORY" in block
        assert "investments funds Portugal move" in block

    def test_graph_off_by_default(self, front_door_env):
        """Envelope path (include_graph defaulted off) gets no graph block —
        zero blast radius on the call_llm path."""
        from app.llm.routing.memory_injection import build_memory_context
        g = front_door_env
        _add_topic(g, "investments", "investments")
        _add_topic(g, "portugal", "Portugal move")
        _add_link(g, "investments", "portugal", "funds")

        ctx = build_memory_context(
            None,
            messages=[{"role": "user", "content": "how are my investments?"}],
        )
        assert ctx.graph_text == ""

    def test_d0_suppresses_graph(self, front_door_env):
        """The D0 gate governs the whole block, graph included."""
        from app.llm.routing.memory_injection import build_memory_context
        g = front_door_env
        _add_topic(g, "investments", "investments")
        _add_topic(g, "portugal", "Portugal move")
        _add_link(g, "investments", "portugal", "funds")

        ctx = build_memory_context(
            None,
            messages=[{"role": "user", "content": "/nomem how are my investments?"}],
            include_graph=True,
        )
        assert ctx.depth == "D0"
        assert ctx.graph_text == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
