# FILE: tests/test_memory_consolidation.py
# Purpose: Tests for the Job 3 consolidation membrane (encode + graph + dedupe).
# Called-by: pytest
# Depends-on: app.memory.consolidation_encode, app.memory.consolidation_graph, app.memory.knowledge_extractor
# Last-renovated: 2026-06-15
"""
Tests for MEMORY_JOBS_SPEC Job 3 — the consolidation membrane.

Covers the four acceptance criteria:
  - a finalised fact is TAGGED (HotIndex) and EMBEDDED (embeddings table);
  - encoding is idempotent (re-consolidation never duplicates the embedding);
  - a cross-domain connection becomes a GRAPH EDGE (investments -> portugal);
  - the edge STRENGTHENS on recurrence (and a single-domain session never
    spends the LLM call);
  - repeating a known fact creates NO DUPLICATE, and an exact-key conflict
    routes through the overwrite path.

Run with: pytest tests/test_memory_consolidation.py -v
"""

import pytest
from unittest.mock import AsyncMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base

# Import model modules so their tables register on Base before create_all().
from app.memory import models as _memory_models  # noqa: F401
from app.memory import conversation_models as _conv_models  # noqa: F401
from app.astra_memory import preference_models as _pref_models  # noqa: F401
from app.embeddings import models as _embedding_models  # noqa: F401


@pytest.fixture
def db():
    """In-memory SQLite with encryption stubbed (mirrors test_conversation_memory)."""
    engine = create_engine("sqlite:///:memory:")
    with patch("app.crypto.types.is_encryption_ready", return_value=True), \
         patch("app.crypto.types.encrypt_string", side_effect=lambda x: f"ENC:{x}"), \
         patch("app.crypto.types.decrypt_string",
               side_effect=lambda x: x[4:] if x.startswith("ENC:") else x):
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        from app.memory.models import Project
        project = Project(
            id=1, name="test-project", description="Test",
            project_key="test-project", type="development", status="active",
        )
        session.add(project)
        session.commit()

        yield session
        session.close()


# Cross-domain summary the user actually connected (used by the graph tests).
_LINKED_SUMMARY = (
    "current_topic: using investment gains to fund the move to Portugal. "
    "key_decisions: sell down part of the TAO position to fund the Portugal "
    "land purchase and the D7 visa application. "
    "biographical_facts: pursuing a passive income visa."
)
_LINK_JSON = '[{"source": "investments", "relation": "funds", "target": "Portugal move"}]'


# =========================================================================
# Task 4 — encode for retrieval (tag + embed)
# =========================================================================

class TestEncodeForRetrieval:
    def test_writes_hotindex_and_embedding(self, db):
        from app.memory import consolidation_encode as ce
        from app.astra_memory.preference_models import HotIndex
        from app.embeddings.models import Embedding

        with patch("app.embeddings.service.generate_embedding", return_value=[0.1] * 8):
            ok = ce.encode_memory_item(
                db, project_id=1, pref_id=4242,
                key="legal_challenge",
                value="Pursuing worker status claim against InPost via Leigh Day",
                reembed=True,
            )
        assert ok is True

        hot = db.query(HotIndex).filter(
            HotIndex.record_type == "memory", HotIndex.record_id == "4242",
        ).first()
        assert hot is not None
        assert hot.one_liner.startswith("Pursuing worker status")
        # Tagged with the shared topic_tagger vocabulary (legal claim -> 'legal').
        assert "legal" in (hot.tags or [])

        emb = db.query(Embedding).filter(
            Embedding.source_type == "memory", Embedding.source_id == 4242,
        ).first()
        assert emb is not None

    def test_idempotent_no_duplicate_embedding(self, db):
        from app.memory import consolidation_encode as ce
        from app.astra_memory.preference_models import HotIndex
        from app.embeddings.models import Embedding

        with patch("app.embeddings.service.generate_embedding", return_value=[0.2] * 8):
            ce.encode_memory_item(db, 1, 7, "loc", "value one", reembed=True)
            ce.encode_memory_item(db, 1, 7, "loc", "value two", reembed=True)

        embs = db.query(Embedding).filter(
            Embedding.source_type == "memory", Embedding.source_id == 7,
        ).all()
        assert len(embs) == 1  # delete-then-store keeps exactly one
        assert embs[0].content == "value two"

        hots = db.query(HotIndex).filter(
            HotIndex.record_type == "memory", HotIndex.record_id == "7",
        ).all()
        assert len(hots) == 1  # upsert in place
        assert hots[0].one_liner == "value two"

    def test_reembed_false_still_embeds_when_missing(self, db):
        from app.memory import consolidation_encode as ce
        from app.embeddings.models import Embedding

        with patch("app.embeddings.service.generate_embedding", return_value=[0.3] * 8):
            ce.encode_memory_item(db, 1, 9, "k", "first value", reembed=False)
        # reembed=False must still create the missing embedding (at-least-once).
        assert db.query(Embedding).filter(
            Embedding.source_type == "memory", Embedding.source_id == 9,
        ).count() == 1


# =========================================================================
# Task 3 — cross-domain links into the knowledge graph
# =========================================================================

class TestCrossDomainGraph:
    @pytest.mark.asyncio
    async def test_edge_created_between_domains(self, tmp_path):
        pytest.importorskip("networkx")
        from app.intelligent_memory.graph.knowledge_graph import KnowledgeGraph
        from app.memory import consolidation_graph as cg

        graph = KnowledgeGraph(graph_path=str(tmp_path / "g.json"))
        with patch("app.llm._streaming_utils_3.call_llm_text",
                   new=AsyncMock(return_value=_LINK_JSON)), \
             patch("app.intelligent_memory.graph.get_knowledge_graph",
                   return_value=graph):
            written = await cg.detect_and_write_links(_LINKED_SUMMARY, session_id=99)

        assert written == 1
        rels = graph.get_relationships("topic:investments", direction="outgoing")
        targets = [r.target_id for r in rels]
        assert "topic:portugal" in targets

    @pytest.mark.asyncio
    async def test_edge_strengthens_on_recurrence(self, tmp_path):
        pytest.importorskip("networkx")
        from app.intelligent_memory.graph.knowledge_graph import KnowledgeGraph
        from app.memory import consolidation_graph as cg

        graph = KnowledgeGraph(graph_path=str(tmp_path / "g.json"))
        with patch("app.llm._streaming_utils_3.call_llm_text",
                   new=AsyncMock(return_value=_LINK_JSON)), \
             patch("app.intelligent_memory.graph.get_knowledge_graph",
                   return_value=graph):
            await cg.detect_and_write_links(_LINKED_SUMMARY, session_id=1)
            await cg.detect_and_write_links(_LINKED_SUMMARY, session_id=2)

        edges = [r for r in graph.get_relationships("topic:investments", direction="outgoing")
                 if r.target_id == "topic:portugal"]
        assert len(edges) == 1  # strengthened, not duplicated
        assert edges[0].weight == 2.0
        assert edges[0].metadata.get("occurrences") == 2

    @pytest.mark.asyncio
    async def test_single_domain_skips_llm(self, tmp_path):
        pytest.importorskip("networkx")
        from app.intelligent_memory.graph.knowledge_graph import KnowledgeGraph
        from app.memory import consolidation_graph as cg

        graph = KnowledgeGraph(graph_path=str(tmp_path / "g.json"))
        llm = AsyncMock(return_value="[]")
        single_domain = (
            "current_topic: investments. Discussed the TAO position, Bittensor "
            "subnet rebalance, and portfolio allocation only."
        )
        with patch("app.llm._streaming_utils_3.call_llm_text", new=llm), \
             patch("app.intelligent_memory.graph.get_knowledge_graph",
                   return_value=graph):
            written = await cg.detect_and_write_links(single_domain, session_id=3)

        assert written == 0
        llm.assert_not_called()  # deterministic gate short-circuits before the LLM


# =========================================================================
# Dedupe + conflict resolution in the fact-writer
# =========================================================================

class TestDedupeAndConflict:
    def test_repeat_same_fact_no_duplicate(self, db):
        from app.memory.knowledge_extractor import _write_to_astra_memory
        from app.astra_memory.preference_models import PreferenceRecord

        fact = {"category": "project", "key": "legal_challenge",
                "value": "Worker status claim via Leigh Day", "weight": 4}
        t1 = _write_to_astra_memory(db, [dict(fact)], session_id=1)
        assert len(t1) == 1 and t1[0]["changed"] is True

        t2 = _write_to_astra_memory(db, [dict(fact)], session_id=1)
        recs = db.query(PreferenceRecord).filter(
            PreferenceRecord.preference_key == "conv_extract:project:legal_challenge",
        ).all()
        assert len(recs) == 1  # re-archiving a known fact creates no duplicate
        assert t2[0]["changed"] is False  # same value => no needless re-embed

    def test_exact_key_conflict_routes_through_overwrite(self, db):
        from app.memory.knowledge_extractor import _write_to_astra_memory
        from app.astra_memory.preference_models import PreferenceRecord
        import app.astra_memory.preference_service as ps

        _write_to_astra_memory(
            db, [{"category": "project", "key": "legal_challenge",
                  "value": "claim A", "weight": 3}], session_id=1,
        )
        with patch.object(ps, "update_preference_value",
                          wraps=ps.update_preference_value) as upd:
            _write_to_astra_memory(
                db, [{"category": "project", "key": "legal_challenge",
                      "value": "claim B", "weight": 3}], session_id=1,
            )
        upd.assert_called_once()  # exact-key + changed value -> overwrite path

        recs = db.query(PreferenceRecord).filter(
            PreferenceRecord.preference_key == "conv_extract:project:legal_challenge",
        ).all()
        assert len(recs) == 1  # still no duplicate
