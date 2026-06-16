import urllib.request
import urllib.parse
import re

query = 'Aldi Ready Set Cook Thai Green Curry Meal Kit calories'
url = 'https://html.duckduckgo.com/html/?q=' + urllib.parse.quote(query)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
}

print("Searching DDG...")
try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        html = response.read().decode('utf-8')
        
        # Write full HTML to file to inspect classes
        with open('D:/Orb/ddg_full.html', 'w', encoding='utf-8') as f:
            f.write(html)
            
        print("Wrote full HTML to ddg_full.html. Parsing...")
        
        # Let's find result links
        # Under html.duckduckgo.com, structure is:
        # <a class="result__snippet" href="...">...</a>
        # Or <a class="result__snippet" href="...">...</a>
        # Actually let's search for result elements
        results = re.findall(r'<div class="web-result.*?">.*?</div>', html, re.DOTALL)
        print("web-result divs:", len(results))
        
        # Let's find all hrefs containing 'uddg='
        hrefs = re.findall(r'href="([^"]+uddg=[^"]+)"', html)
        print("Found matching hrefs:", len(hrefs))
        for href in hrefs[:10]:
            clean_url = href.split('uddg=')[1].split('&')[0]
            clean_url = urllib.parse.unquote(clean_url)
            print("Link:", clean_url)
            
except Exception as e:
    import traceback
    traceback.print_exc()
