import urllib.parse
import re

print("Parsing ddg_full.html...")
with open('D:/Orb/ddg_full.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Let's find all organic links. In DDG HTML (non-JS), organic results look like:
# <a class="result__url" href="//duckduckgo.com/l/?uddg=HTTP_URL...">
# or similar.
# Let's search for result__snippet or result__body or search-result.
# Let's look for result__snippet first.
snippets = re.findall(r'<td class="result-snippet">.*?</td>', html, re.DOTALL)
print("Result snippets (td class=result-snippet):", len(snippets))

# Let's search for result-link or result__url or result-snippet in a robust way:
# Let's list some snippets or raw lines of interest:
lines = html.split('\n')
print(f"Total lines: {len(lines)}")
for line in lines:
    if 'class="result-link"' in line or 'result-snippet' in line:
        print(line[:300])

# Let's print out the first few <a> tags with class containing "result"
a_tags = re.findall(r'<a\s+[^>]*class="[^"]*result[^"]*"[^>]*>.*?</a>', html, re.DOTALL)
print(f"Found a_tags: {len(a_tags)}")
for a in a_tags[:10]:
    print("---")
    print(a[:300])
