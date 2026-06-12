# Purpose: Test full scan_imports via sandbox.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.memory.domains.dependency_scanner
# Last-renovated: 2026-06-11
"""Test full scan_imports via sandbox."""
from app.memory.domains.dependency_scanner import scan_imports, summarise_graph
import time

start = time.time()
print("Running scan_imports via sandbox...")
graph = scan_imports(r"D:\Orb")
elapsed = time.time() - start

stats = summarise_graph(graph)
print(f"Done in {elapsed:.1f}s")
print(f"Modules: {stats['total_modules']}")
print(f"Edges: {stats['total_edges']}")
print("Top 5 most depended on:")
for item in stats["most_depended_on"][:5]:
    print(f"  {item['module']} ({item['dependents']} dependents)")
