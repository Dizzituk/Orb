import sys; sys.path.insert(0, r'D:\Orb')
from app.overwatcher.sandbox_client import SandboxClient
client = SandboxClient()

# Read the 3 largest files and break down what's taking space
files = [
    ('intents/models.py', r'D:\Orb\app\translation\intents\models.py'),
    ('conduct_policy/models.py', r'D:\Orb\app\overwatcher\conduct_policy\models.py'),
    ('sbv/models.py', r'D:\Orb\app\overwatcher\sandbox_build_validator\models.py'),
    ('sbv/core.py', r'D:\Orb\app\overwatcher\sandbox_build_validator\core.py'),
]

for label, fpath in files:
    r = client.shell_run(f'Get-Content "{fpath}" -Raw', timeout_seconds=10)
    content = r.stdout or ''
    lines = content.split('\n')
    
    print(f"\n{'='*60}")
    print(f"{label} — {len(content)/1024:.1f} KB, {len(lines)} lines")
    print(f"{'='*60}")
    
    # Break down by symbol — find each def/class and measure its size
    current_name = None
    current_start = 0
    current_indent = 0
    symbols = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        
        # Top-level or class-level definition
        is_def = stripped.startswith(('def ', 'async def ', 'class '))
        # Top-level constant (ALL_CAPS = ...)
        is_const = (len(line) > 0 and line[0] != ' ' and '=' in stripped 
                    and stripped.split('=')[0].strip().replace('_','').isupper()
                    and not stripped.startswith(('from ', 'import ')))
        
        if is_def or is_const:
            indent = len(line) - len(line.lstrip())
            if indent == 0:  # top-level only
                if current_name:
                    size = sum(len(l)+1 for l in lines[current_start:i])
                    symbols.append((current_name, current_start, i, size))
                if is_const:
                    name = stripped.split('=')[0].strip().split(':')[0].strip()
                else:
                    name = stripped.split('(')[0].split(':')[0].replace('def ','').replace('async ','').replace('class ','').strip()
                current_name = name
                current_start = i
    
    # Last symbol
    if current_name:
        size = sum(len(l)+1 for l in lines[current_start:])
        symbols.append((current_name, current_start, len(lines), size))
    
    # Sort by size descending
    symbols.sort(key=lambda x: -x[3])
    
    # Show breakdown
    import_size = 0
    for line in lines:
        if line.strip().startswith(('from ', 'import ')):
            import_size += len(line) + 1
    
    print(f"  Imports: {import_size/1024:.1f} KB")
    print(f"  Top symbols by size:")
    for name, start, end, size in symbols[:10]:
        pct = size / len(content) * 100
        print(f"    {name:40s}  {size/1024:6.1f} KB  ({end-start:4d} lines, {pct:4.1f}%)")
    
    if len(symbols) > 10:
        rest = sum(s[3] for s in symbols[10:])
        print(f"    {'... remaining ' + str(len(symbols)-10):40s}  {rest/1024:6.1f} KB")
