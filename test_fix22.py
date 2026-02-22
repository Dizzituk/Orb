"""Verify Fix 22: Cohesion check no longer produces false cross-package issues."""
import sys, json
sys.path.insert(0, r'D:\Orb')

# Load the manifest from the failed run
with open(r'D:\Orb\jobs\jobs\sg-8c31153e\segments\manifest.json') as f:
    manifest = json.load(f)

# Load architectures
architectures = {}
import os
for seg in manifest['segments']:
    seg_id = seg['segment_id']
    arch_path = os.path.join(r'D:\Orb\jobs\jobs\sg-8c31153e\segments', seg_id, 'arch', 'arch_v1.md')
    if os.path.isfile(arch_path):
        with open(arch_path, encoding='utf-8') as f:
            architectures[seg_id] = f.read()

print(f"Segments: {len(manifest['segments'])}")
print(f"Architectures: {len(architectures)}")
for seg_id in architectures:
    print(f"  {seg_id}")

# Run cohesion check
from app.orchestrator.cohesion_check import run_skeleton_compliance

issues = run_skeleton_compliance(
    architectures=architectures,
    skeleton_json=None,
    manifest_dict=manifest,
)

blocking = [i for i in issues if i.severity == 'blocking']
warnings = [i for i in issues if i.severity == 'warning']

print(f"\nTotal issues: {len(issues)} ({len(blocking)} blocking, {len(warnings)} warning)")
for issue in issues:
    print(f"  [{issue.severity}] {issue.category}: {issue.description[:140]}")

if not blocking:
    print("\nNO BLOCKING ISSUES - Fix 22 working!")
else:
    print(f"\nSTILL {len(blocking)} BLOCKING - needs investigation")
