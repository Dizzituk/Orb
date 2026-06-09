"""Find the tagger that's producing these limited tag values."""
import sys, os, re
sys.path.insert(0, r'D:\Orb')

# Walk app/, grep for the suspicious tag literals
patterns = ["architecture", "debugging", "testing", "documentation"]
hits = {}
for root, dirs, files in os.walk(r'D:\Orb\app'):
    if '__pycache__' in root:
        continue
    for fn in files:
        if not fn.endswith('.py'):
            continue
        p = os.path.join(root, fn)
        try:
            text = open(p, encoding='utf-8').read()
        except Exception:
            continue
        score = sum(1 for pat in patterns if pat in text)
        if score >= 3:  # Likely a tagger if it mentions 3+ of these
            # Count occurrences of each as string literal
            n_arch = len(re.findall(r"['\"]architecture['\"]", text))
            n_dbg = len(re.findall(r"['\"]debugging['\"]", text))
            n_test = len(re.findall(r"['\"]testing['\"]", text))
            if n_arch + n_dbg + n_test >= 3:
                hits[p] = (n_arch, n_dbg, n_test)

for p, (a, d, t) in sorted(hits.items(), key=lambda kv: -sum(kv[1])):
    print(f'{p}  arch={a} debug={d} test={t}')
