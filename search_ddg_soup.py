import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
import json

query = 'Aldi "Ready, Set... Cook!" Thai Green Curry Meal Kit calories'
url = 'https://html.duckduckgo.com/html/?q=' + urllib.parse.quote(query)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
}

print("Searching DDG...")
try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        html = response.read().decode('utf-8')
        soup = BeautifulSoup(html, 'html.parser')
        results = soup.find_all('a', class_='result__snippet')
        print(f"Snippets: {len(results)}")
        
        # Look for result divs
        links = soup.find_all('a', class_='result__url')
        print(f"Links: {len(links)}")
        
        for i, (l, s) in enumerate(zip(links, results)):
            url_text = l.text.strip()
            href = l.get('href')
            if 'uddg=' in href:
                href = href.split('uddg=')[1].split('&')[0]
                href = urllib.parse.unquote(href)
            snippet = s.text.strip()
            print(f"\n[{i+1}] {url_text}")
            print(f"URL: {href}")
            print(f"Snippet: {snippet}")
except Exception as e:
    import traceback
    traceback.print_exc()
