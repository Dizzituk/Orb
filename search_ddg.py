import urllib.request
import urllib.parse
import re

query = 'Ready Set Cook Thai Green Curry Meal Kit'
url = 'https://html.duckduckgo.com/html/?q=' + urllib.parse.quote(query)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
}

print("Searching DDG...")
try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        html = response.read().decode('utf-8')
        print("HTML length:", len(html))
        # Look for results in html
        if "ddg" in html:
            print("DDG found in HTML")
        # Let's save a snippet to analyze why it didn't find results
        with open('D:/Orb/ddg_result.html', 'w', encoding='utf-8') as f:
            f.write(html[:5000])
        
        # Let's see if there are standard links
        matches = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL)
        print("All links found:", len(matches))
        for m in matches[:20]:
            print(m)
except Exception as e:
    import traceback
    traceback.print_exc()
