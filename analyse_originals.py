import sys; sys.path.insert(0, r'D:\Orb')
from app.overwatcher.sandbox_client import SandboxClient
client = SandboxClient()

# Read the quarantined originals
files = [
    ('intents.py ORIGINAL', r'D:\Orb\app\translation\.quarantined\intents.py'),
    ('conduct_policy.py ORIGINAL', r'D:\Orb\app\overwatcher\.quarantined\conduct_policy.py'),
    ('sandbox_build_validator.py ORIGINAL', r'D:\Orb\app\overwatcher\.quarantined\sandbox_build_validator.py'),
]

for label, fpath in files:
    r = client.shell_run(f'Get-Content "{fpath}" -Raw', timeout_seconds=10)
    content = r.stdout or ''
    lines = content.split('\n')
    
    print(f"\n{label} — {len(content)/1024:.1f} KB, {len(lines)} lines")
    
    # Count top-level symbols and their sizes
    current_name = None
    current_start = 0
    symbols = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        
        is_def = stripped.startswith(('def ', 'async def ', 'class '))
        is_const = (len(line) > 0 and line[0] != ' ' and '=' in stripped 
                    and stripped.split('=')[0].strip().replace('_','').isupper()
                    and not stripped.startswith(('from ', 'import ')))
        
        if is_def or is_const:
            indent = len(line) - len(line.lstrip())
            if indent == 0:
                if current_name:
                    size = sum(len(l)+1 for l in lines[current_start:i])
                    symbols.append((current_name, size))
                if is_const:
                    name = stripped.split('=')[0].strip().split(':')[0].strip()
                else:
                    name = stripped.split('(')[0].split(':')[0].replace('def ','').replace('async ','').replace('class ','').strip()
                current_name = name
                current_start = i
    if current_name:
        size = sum(len(l)+1 for l in lines[current_start:])
        symbols.append((current_name, size))
    
    symbols.sort(key=lambda x: -x[1])
    
    print(f"  {len(symbols)} top-level symbols")
    for name, size in symbols[:8]:
        print(f"    {name:40s}  {size/1024:6.1f} KB")
    if len(symbols) > 8:
        rest = sum(s[1] for s in symbols[8:])
        print(f"    {'... remaining ' + str(len(symbols)-8):40s}  {rest/1024:6.1f} KB")
