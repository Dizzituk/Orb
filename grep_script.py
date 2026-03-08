import sys
with open('D:/Orb/app/debug/debug_chat.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if 'genai' in line:
            print(f"{i+1}: {line.strip()}")
