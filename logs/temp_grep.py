import sys
log_file = r"D:\Orb\logs\astra.log"
with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()[-5000:]
    
matches = [i for i, line in enumerate(lines) if "ASGI" in line or "Traceback" in line or "500" in line]
if matches:
    idx = matches[-1]
    for j in range(max(0, idx-10), min(len(lines), idx+30)):
        print(lines[j].strip())
else:
    print("No matches found")