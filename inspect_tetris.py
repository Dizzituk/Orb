import os

path = r"C:\Users\dizzi\OneDrive\Documents\Games\Tazza's Tetris\index.html"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
print("Last 20 lines:")
for line in lines[-20:]:
    print(line, end="")
