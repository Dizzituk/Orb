"""Test 3: Full end-to-end test of all 3 fixes."""
import json
import sys
sys.path.insert(0, r"D:\Orb")

from app.orchestrator.cross_segment_interfaces import (
    extract_exports_from_arch,
    build_sibling_interface_section,
    validate_cross_segment_interfaces,
    _extract_imports_from_arch,
)

JOB_DIR = r"D:\Orb\jobs\jobs\sg-ddbbc5b4"

# ================================================================
# FIX 1: Sibling interface injection
# ================================================================
print("=" * 60)
print("FIX 1: Sibling Interface Injection")
print("=" * 60)

section = build_sibling_interface_section(
    "seg-02-education-feature-orchestrator",
    ["seg-01-education-feature-base-compone"],
    JOB_DIR,
)
if section:
    print("Section generated:")
    for line in section.split("\n")[:20]:
        print(f"  {line}")
    print(f"  ... ({len(section)} chars total)")
else:
    print("FAIL: No section generated")

# ================================================================
# FIX 2: Cross-segment interface validation
# ================================================================
print("\n" + "=" * 60)
print("FIX 2: Interface Validation (should catch mismatches)")
print("=" * 60)

with open(JOB_DIR + r"\segments\manifest.json") as f:
    manifest = json.load(f)

issues = validate_cross_segment_interfaces(JOB_DIR, manifest["segments"])
if issues:
    print(f"Found {len(issues)} issue(s):")
    for iss in issues:
        print(f"  [{iss['severity'].upper()}] {iss['category']}")
        print(f"    {iss['description'][:150]}")
        print(f"    fix: {iss['suggested_fix'][:100]}")
else:
    print("No issues found (unexpected — should catch mockCourses/courses mismatch)")

# ================================================================
# FIX 3: Skeleton back-propagation
# ================================================================
print("\n" + "=" * 60)
print("FIX 3: Skeleton Back-Propagation (dry run)")
print("=" * 60)

skel_path = JOB_DIR + r"\segments\skeleton_contract.json"
with open(skel_path) as f:
    skel = json.load(f)

# Show current state of exports
for s in skel.get("skeletons", []):
    sid = s.get("segment_id", "")
    for exp in s.get("exports", []):
        names = exp.get("names", [])
        fpath = exp.get("file_path", "")
        print(f"  {sid}: {fpath} -> names={names}")

# Test back-propagation (actually write it)
from app.orchestrator.cross_segment_interfaces import backpropagate_exports_to_skeleton
with open(JOB_DIR + r"\segments\seg-01-education-feature-base-compone\arch\arch_v1.md") as f:
    arch1 = f.read()
count = backpropagate_exports_to_skeleton(
    "seg-01-education-feature-base-compone",
    arch1,
    skel_path,
)
print(f"\nBack-propagated {count} export binding(s)")

# Verify
with open(skel_path) as f:
    skel_after = json.load(f)
for s in skel_after.get("skeletons", []):
    sid = s.get("segment_id", "")
    for exp in s.get("exports", []):
        names = exp.get("names", [])
        fpath = exp.get("file_path", "")
        if names:
            print(f"  UPDATED {sid}: {fpath} -> names={names}")
