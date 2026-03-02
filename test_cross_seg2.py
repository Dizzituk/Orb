"""Test 2: Debug the parser against actual arch files."""
import re
import sys
sys.path.insert(0, r"D:\Orb")

ARCH_PATH = r"D:\Orb\jobs\jobs\sg-ddbbc5b4\segments\seg-01-education-feature-base-compone\arch\arch_v1.md"

with open(ARCH_PATH, "r", encoding="utf-8") as f:
    content = f.read()

print(f"File length: {len(content)}")

# The arch file contains stream chrome before the actual arch.
# Find the architecture section
arch_start = content.find("### Architecture Document")
print(f"Architecture starts at: {arch_start}")

if arch_start >= 0:
    arch_part = content[arch_start:]
else:
    arch_part = content

# Count backticks
bt_count = arch_part.count("`")
print(f"Backticks in arch: {bt_count}")

# Find code fences
fences = re.findall(r"```(\w+)", arch_part)
print(f"Code fence languages: {fences}")

# Find file sections
file_splits = re.split(r"###\s+File:\s*`([^`]+)`", arch_part)
n_files = (len(file_splits) - 1) // 2
print(f"File sections: {n_files}")
for i in range(1, len(file_splits) - 1, 2):
    print(f"  {file_splits[i].strip()}")

# Now test the actual parser
from app.orchestrator.cross_segment_interfaces import extract_exports_from_arch

# The parser needs to handle the stream chrome prefix
# Let's test with just the arch section
exports = extract_exports_from_arch(arch_part)
print(f"\nExports found: {len(exports)} files")
for path, info in exports.items():
    print(f"  {path}:")
    print(f"    exports: {info['exports']}")
    for n, b in info.get("interfaces", {}).items():
        print(f"    interface {n}: {b[:60]}...")
    for n, b in info.get("props", {}).items():
        print(f"    props {n}: {b[:60]}...")

# Also test against full content (with chrome)
print(f"\n--- Full content (with chrome) ---")
exports_full = extract_exports_from_arch(content)
print(f"Exports found: {len(exports_full)} files")
