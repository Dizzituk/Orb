"""Test memory system integration."""
import sys, os
sys.path.insert(0, r"D:\Orb")
os.chdir(r"D:\Orb")
from dotenv import load_dotenv
load_dotenv(r"D:\Orb\.env")

from app.memory.router import memory_router, MemoryRouter, DomainStore
from app.memory.domains.architecture import ArchitectureStore
from app.memory.domains.knowledge import KnowledgeStore

# Test protocol compliance
arch = ArchitectureStore()
know = KnowledgeStore()

print(f"ArchitectureStore.domain_name = {arch.domain_name}")
print(f"KnowledgeStore.domain_name = {know.domain_name}")
print(f"ArchitectureStore is DomainStore: {isinstance(arch, DomainStore)}")
print(f"KnowledgeStore is DomainStore: {isinstance(know, DomainStore)}")

# Test registration
memory_router.register(arch)
memory_router.register(know)
print(f"Registered domains: {memory_router.registered_domains}")

# Test query
results = memory_router.query("weaver streaming", limit=3)
print(f"Query results: {len(results)}")
for r in results:
    print(f"  [{r.domain}] {r.content[:60]}... (score={r.relevance:.2f})")

# Test stats
stats = memory_router.get_stats()
for domain, s in stats.items():
    print(f"  {domain}: {s.active_entries} active, {s.quarantined_entries} quarantined")

print("\nDONE - all checks passed")
