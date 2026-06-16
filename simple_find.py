with open('D:/Orb/Aldi+Create+Thai+Green+Curry+Kit+224g.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Simple search for "8.2g"
for term in ["8.2g", "3.6g", "1.2g", "11g", "4.8g", "1.6g"]:
    idx = html.find(term)
    if idx != -1:
        print(f"Found {term} at {idx}:")
        print(html[idx-100:idx+150].replace('\n', ' ').strip())
        print("="*60)
