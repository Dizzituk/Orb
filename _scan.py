"""Full scan including underscore files."""
import os

MAX_KB = 30
results = []

for root, dirs, files in os.walk('app'):
    dirs[:] = [d for d in dirs if d != '__pycache__']
    for fname in files:
        if fname.endswith('.py'):
            path = os.path.join(root, fname).replace('\\', '/')
            sz = os.path.getsize(path) / 1024
            if sz >= MAX_KB:
                results.append((path, round(sz, 1)))

# Add specific known files
for p in ['app/translation/intents.py', 'app/llm/pipeline/high_stakes.py', 
          'app/pot_spec/grounded/spec_runner.py', 'app/llm/weaver_stream.py',
          'app/translation/tier0_rules.py']:
    sz = os.path.getsize(p) / 1024
    entry = (p, round(sz, 1))
    if entry not in results and sz >= MAX_KB:
        results.append(entry)

results.sort(key=lambda x: x[1], reverse=True)
print(f"Files >= {MAX_KB}KB:")
print(f"{'File':<70} {'KB':>7}")
print("-" * 78)
for path, sz in results:
    print(f"{path:<70} {sz:>6.1f}K")
print(f"\nTotal: {len(results)} files")
