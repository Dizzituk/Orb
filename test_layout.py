import sys, os, traceback; sys.path.insert(0, r'D:\Orb')
from app.orchestrator.codebase_scanner import scan_file
from app.orchestrator.refactor_segmenter import _auto_layout_files

test_files = [
    ('app/overwatcher/conduct_policy.py', 'app/overwatcher/conduct_policy'),
    ('app/translation/intents.py', 'app/translation/intents'),
    ('app/overwatcher/sandbox_build_validator.py', 'app/overwatcher/sandbox_build_validator'),
    ('app/overwatcher/signature_checker.py', 'app/overwatcher/signature_checker'),
    ('app/pot_spec/grounded/multi_file_detection.py', 'app/pot_spec/grounded/multi_file_detection'),
]

for rel, pkg in test_files:
    try:
        abs_path = os.path.join('D:\\Orb', rel.replace('/', os.sep))
        with open(abs_path, encoding='utf-8') as f:
            source = f.read()
        scan = scan_file(rel, source)
        nodes = _auto_layout_files(scan.symbols, pkg, rel)
        file_kb = len(source) / 1024
        print(f"\n{rel} ({file_kb:.1f} KB, {len(scan.symbols)} symbols)")
        for path, node in nodes.items():
            fname = path.split('/')[-1]
            print(f"  {fname:25s}  {node.description[:60]}")
        print(f"  Total: {len(nodes)} files")
    except Exception as e:
        traceback.print_exc()
