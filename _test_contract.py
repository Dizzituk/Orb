import sys, json
sys.path.insert(0, "D:\\Orb")
sys.stdout.reconfigure(encoding='utf-8')
from app.orchestrator.skeleton_contracts import load_skeleton_contract
cs = load_skeleton_contract("D:\\Orb\\jobs\\jobs\\sg-755fbc70")
if cs:
    md = cs.format_contract_for_segment("seg-03-segment-state-updates")
    print("=== CONTRACT MARKDOWN ===")
    print(md)
    print("=== END ===")
    print(f"\nLength: {len(md)} chars")
else:
    print("Failed to load contract set")
