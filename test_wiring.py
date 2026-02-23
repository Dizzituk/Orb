import sys
sys.path.insert(0, r"D:\Orb")

from app.memory.integration import on_intent_confirmed, after_user_message, enrich_routing, inject_memory_context
from app.translation.feedback import FeedbackLogger, _CONFIDENCE_AVAILABLE
from app.memory.migrations import run_memory_migrations, migrate_rag_entries_package_role

print(f"Confidence available in feedback: {_CONFIDENCE_AVAILABLE}")

# Test enrich_routing
result = enrich_routing("redesign the RAG architecture", job_type="orchestrator")
tier = result.get("model_tier", "?")
upgraded = result.get("was_upgraded", "?")
rag = result.get("rag_needed", "?")
print(f"Complexity routing: tier={tier}, upgraded={upgraded}, rag={rag}")

# Test inject_memory_context (will return empty since no data)
ctx = inject_memory_context("test query")
if ctx:
    print(f"Memory context: {ctx[:50]}")
else:
    print("Memory context: (empty - expected, no data)")

print("ALL DEEP IMPORTS + FUNCTIONAL OK")
