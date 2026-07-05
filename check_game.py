import glob

paths = glob.glob("C:/Users/dizzi/OneDrive/Documents/Games/Tazza*/index.html")
if not paths:
    print("No file found")
else:
    path = paths[0]
    print(f"Reading {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    
    # Let's find the controls section and the mute button
    for line in content.splitlines():
        if "mute-btn" in line or "key-badge" in line or "muteBtn.textContent" in line:
            print(line.strip())
