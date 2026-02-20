import sys, json, re
sys.path.insert(0, "D:\\Orb")
sys.stdout.reconfigure(encoding='utf-8')

from app.orchestrator.skeleton_contracts import load_skeleton_contract

cs = load_skeleton_contract("D:\\Orb\\jobs\\jobs\\sg-755fbc70")
md = cs.format_contract_for_segment("seg-03-segment-state-updates")

file_path = "app/orchestrator/segment_loop/_state_updates.py"
file_path_norm = file_path.replace("\\\\", "/").strip()

lines = md.split("\n")
print(f"Total lines in contract: {len(lines)}")
print(f"Looking for: {file_path_norm}")

in_file_section = False
in_exports = False

for i, line in enumerate(lines):
    stripped = line.strip()
    stripped_norm = stripped.replace("\\\\", "/")
    
    if f"{file_path_norm}" in stripped_norm:
        print(f"\n>>> MATCH at line {i}: {stripped}")
        in_file_section = True
        in_exports = False
        continue

    if in_file_section:
        if "MUST EXPORT" in stripped:
            print(f">>> EXPORT HEADER at line {i}: {stripped}")
            in_exports = True
            continue

        if stripped.startswith("###") or stripped.startswith("## "):
            print(f">>> SECTION END at line {i}: {stripped}")
            in_file_section = False
            in_exports = False
            continue

        if stripped.startswith("- \") and "\" in stripped[3:]:
            match = re.match(r'^-\s*\([^\]+)\', stripped)
            if match:
                candidate = match.group(1).strip().replace("\\\\", "/")
                is_file_path = ("/" in candidate or candidate.endswith(".py"))
                is_signature = candidate.startswith("def ") or candidate.startswith("async def ")
                if is_file_path and not is_signature:
                    if candidate != file_path_norm:
                        print(f">>> NEW FILE at line {i}: {candidate} (leaving section)")
                        in_file_section = False
                        in_exports = False
                        continue

        if in_exports and stripped.startswith("- \"):
            match = re.match(r'^-\s*\([^\]+)\', stripped)
            if match:
                sig = match.group(1).strip()
                print(f">>> CANDIDATE SIG at line {i}: {sig}")
                if sig.startswith("def ") or sig.startswith("async def "):
                    print(f">>> !!! FOUND SIGNATURE: {sig}")

print("\nDone.")
