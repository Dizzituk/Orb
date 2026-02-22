import sys, os
sys.path.insert(0, r'D:\Orb')

from app.orchestrator.refactor_pipeline import run_deterministic_refactor

sources = [
    ('app/overwatcher/conduct_policy.py', 'app/overwatcher/conduct_policy/'),
    ('app/translation/intents.py', 'app/translation/intents/'),
    ('app/overwatcher/sandbox_build_validator.py', 'app/overwatcher/sandbox_build_validator/'),
]

all_ids = []
for src, tgt in sources:
    plan, archs, manifest_data = run_deterministic_refactor(
        source_file_path=src,
        architecture_file_inventory='',
        target_package=tgt,
        job_dir=r'D:\Orb\jobs\temp_test',
        spec_id='test-fix20',
    )
    print(f'\n=== {src} ===')
    for seg in manifest_data['segments']:
        seg_id = seg['segment_id']
        files = [f.split('/')[-1] for f in seg.get('file_scope', [])]
        all_ids.append(seg_id)
        print(f'  {seg_id} -> {files}')
    
    # Check for empty file warnings in the plan
    if plan.warnings:
        for w in plan.warnings:
            print(f'  WARNING: {w}')

    # Show re-exports in __init__.py
    for seg_id, arch_text in archs.items():
        for line in arch_text.split('\n'):
            s = line.strip()
            if s.startswith('from .') or s.startswith('from ..'):
                print(f'  [IMPORT] {s}')

print(f'\nTotal segments: {len(all_ids)}')
dupes = [x for x in all_ids if all_ids.count(x) > 1]
if dupes:
    print(f'COLLISION! Duplicate IDs: {set(dupes)}')
else:
    print('ALL UNIQUE')
