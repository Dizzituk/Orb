import sys, json
sys.path.insert(0, "D:\\Orb")
sys.stdout.reconfigure(encoding='utf-8')

from app.orchestrator.skeleton_contracts import load_skeleton_contract
from app.overwatcher.signature_checker import extract_contract_signatures_for_file

cs = load_skeleton_contract("D:\\Orb\\jobs\\jobs\\sg-755fbc70")
if cs:
    md = cs.format_contract_for_segment("seg-03-segment-state-updates")
    sigs = extract_contract_signatures_for_file(md, "app/orchestrator/segment_loop/_state_updates.py")
    print(f"Contract length: {len(md)} chars")
    print(f"Signatures found: {len(sigs)}")
    for s in sigs:
        print(f"  -> {s}")
    if not sigs:
        print("\nDEBUG: Searching manually...")
        for i, line in enumerate(md.split("\\n")):
            if "_state_updates" in line or "MUST" in line or "def update" in line:
                print(f"  Line {i}: {line.rstrip()}")
else:
    print("Failed to load contract set")
