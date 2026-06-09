"""End-to-end test: does query-time tag/entity extraction now drive retrieval?"""
import sys
sys.path.insert(0, r'D:\Orb')

from app.db import get_db_session
from app.astra_memory.topic_tagger import extract_tags, extract_entities
from app.astra_memory.retrieval import stage1_candidate_selection

db = get_db_session()
try:
    queries = [
        "What did we talk about regarding TAO?",
        "I need to think about my Yodel work hours.",
        "How is the Leigh Day claim progressing?",
        "Tell me about the AGI displacement thesis.",
        "Should I rebalance my investments?",
    ]
    for q in queries:
        tags = extract_tags(q)
        ents = extract_entities(q)
        # Drop 'general'
        filter_tags = [t for t in tags if t != 'general'] or None
        candidates = stage1_candidate_selection(
            db=db,
            query_tags=filter_tags,
            query_entities=ents or None,
            max_candidates=5,
        )
        print(f"\nQ: {q}")
        print(f"  tags filter:     {filter_tags}")
        print(f"  entities filter: {ents or None}")
        print(f"  top {len(candidates)} candidates (score, type, title):")
        for c in candidates:
            title = (c.title or '').encode('ascii', 'replace').decode('ascii')
            print(f"    {c.relevance_score:.3f}  [{c.record_type}]  {title[:70]}")
finally:
    db.close()
