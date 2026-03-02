"""Test cross_segment_interfaces parser."""
import sys
sys.path.insert(0, r"D:\Orb")

from app.orchestrator.cross_segment_interfaces import (
    extract_exports_from_arch,
    build_sibling_interface_section,
    validate_cross_segment_interfaces,
    _extract_imports_from_arch,
)

JOB_DIR = r"D:\Orb\jobs\jobs\sg-ddbbc5b4"

# Test 1: Extract exports from seg-01
print("=" * 60)
print("TEST 1: Extract exports from seg-01")
print("=" * 60)
with open(JOB_DIR + r"\segments\seg-01-education-feature-base-compone\arch\arch_v1.md") as f:
    arch1 = f.read()

exports1 = extract_exports_from_arch(arch1)
for path, info in exports1.items():
    print(f"\n  {path}:")
    print(f"    exports: {info['exports']}")
    for n, b in info.get("interfaces", {}).items():
        print(f"    interface {n}: {b[:80]}...")
    for n, b in info.get("props", {}).items():
        print(f"    props {n}: {b[:80]}...")

if not exports1:
    print("  (EMPTY - parser found nothing)")

# Test 2: Extract imports from seg-02
print("\n" + "=" * 60)
print("TEST 2: Extract imports from seg-02")
print("=" * 60)
with open(JOB_DIR + r"\segments\seg-02-education-feature-orchestrator\arch\arch_v1.md") as f:
    arch2 = f.read()

imports2 = _extract_imports_from_arch(arch2)
for path, names in imports2.items():
    print(f"  {path}: {names}")

if not imports2:
    print("  (EMPTY - parser found nothing)")

# Test 3: Build sibling interface section
print("\n" + "=" * 60)
print("TEST 3: Sibling interface section for seg-02")
print("=" * 60)
section = build_sibling_interface_section(
    "seg-02-education-feature-orchestrator",
    ["seg-01-education-feature-base-compone"],
    JOB_DIR,
)
if section:
    print(section[:500])
else:
    print("  (EMPTY - no section generated)")

# Test 4: Cross-segment interface validation
print("\n" + "=" * 60)
print("TEST 4: Interface validation")
print("=" * 60)
import json
with open(JOB_DIR + r"\segments\manifest.json") as f:
    manifest = json.load(f)

issues = validate_cross_segment_interfaces(JOB_DIR, manifest["segments"])
if issues:
    for iss in issues:
        print(f"  [{iss['severity']}] {iss['description'][:120]}")
else:
    print("  (NO ISSUES - all interfaces match)")
