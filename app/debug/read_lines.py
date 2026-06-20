with open('D:/Orb/app/lifestyle/service.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i in range(510, 560):
    if i < len(lines):
        print(f"{i+1}: {lines[i]}", end='')
