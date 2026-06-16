import urllib.request
import re

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
}

urls = [
    'https://www.aldi.co.uk/product/ready-set-cook-thai-green-curry-kit-000000000336498004',
    'https://www.nutracheck.co.uk/CaloriesIn/Product/84/Aldi+Create+Thai+Green+Curry+Kit+224g'
]

for url in urls:
    print(f"\nFetching {url}...")
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8', errors='ignore')
            # Save raw html for analysis
            filename = url.split('/')[-1] + '.html'
            with open('D:/Orb/' + filename, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"Saved {filename}. HTML length: {len(html)}")
            
            # Simple check for nutritional terms
            for term in ['Fat', 'Carbohydrate', 'Protein', 'Saturates', 'Sugar', 'kcal']:
                matches = re.findall(rf'(\d+(?:\.\d+)?\s*(?:g|kcal))\s*.*?{term}', html, re.IGNORECASE)
                matches_reverse = re.findall(rf'{term}\s*.*?(\d+(?:\.\d+)?\s*(?:g|kcal))', html, re.IGNORECASE)
                print(f"Term '{term}': matches={matches[:3]}, reverse={matches_reverse[:3]}")
    except Exception as e:
        print(f"Error fetching {url}: {e}")
