import re

with open('D:/Orb/Aldi+Create+Thai+Green+Curry+Kit+224g.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Let's clean up and output sections. Let's just find all divs with class="showps" or "showps selected" or id starting with prodDetails
# Since bs4 was not found, let's use a simple regex to get div contents.
divs = re.findall(r'<div id="prodDetails\d+"[^>]*>.*?</div>', html, re.DOTALL)
print(f"Found {len(divs)} prodDetails divs (non-greedy):")
for i, d in enumerate(divs):
    print(f"Div {i+1}:")
    # clean HTML tags
    text = re.sub(r'<[^>]+>', ' ', d).strip()
    text = re.sub(r'\s+', ' ', text)
    print(text)

# Let's write a more precise regex to catch the div with its contents (greedy but restricted to reasonable size)
divs_blocks = re.findall(r'(<div id="prodDetails\d+".*?</table>)', html, re.DOTALL)
for i, block in enumerate(divs_blocks):
    print(f"\n--- BLOCK {i+1} ---")
    clean = re.sub(r'<[^>]+>', ' ', block).strip()
    clean = re.sub(r'\s+', ' ', clean)
    print(clean)
