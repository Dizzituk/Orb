with open('D:/Orb/Aldi+Create+Thai+Green+Curry+Kit+224g.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Let's find occurrences of "1.2g", "3.6g", "8.2g", "1.6g", "4.8g", "11g" and print surrounding text.
import re

for match in re.finditer(r'(?:Protein|Carbs|Fat|Energy|Calories|Sugar|Saturates|Fibre|Salt)\s*\d+(?:\.\d+)?\s*[a-zA-Z]*', html, re.IGNORECASE):
    start = max(0, match.start() - 100)
    end = min(len(html), match.end() + 100)
    print("MATCH SURROUNDING:")
    print(html[start:end].replace('\n', ' ').strip())
    print("-" * 50)
