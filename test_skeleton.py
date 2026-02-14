import json, sys
sys.path.insert(0, r'D:\Orb')
from app.orchestrator.skeleton_contracts import generate_skeleton_contract

with open(r'D:\Orb\jobs\jobs\sg-2ec370b6\segments\manifest.json') as f:
    manifest = json.load(f)

contract = generate_skeleton_contract(manifest, 'sg-2ec370b6')
print('Segments:', contract.total_segments)
print('Bindings:', len(contract.cross_segment_bindings))

for s in contract.skeletons:
    print()
    print('===', s.segment_id, '===')
    print('  Scope:', s.file_scope)
    print('  Exports:', len(s.exports))
    for e in s.exports:
        print('    ->', e.file_path, 'consumed by', e.consumed_by)
    if s.imports_from:
        print('  Imports:')
        for k, v in s.imports_from.items():
            print('    <-', k, ':', v)

print()
print('=== BINDINGS ===')
for b in contract.cross_segment_bindings:
    print(' ', b['from_segment'], '->', b['to_segment'], ':', b['file_path'])

print()
print('=== SAMPLE CONTRACT MARKDOWN (seg-03) ===')
md = contract.format_contract_for_segment('seg-03-architecture-document-parsing-')
print(md[:2000])
