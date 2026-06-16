import urllib.request
import json
import urllib.parse

query = 'Ready Set Cook Thai Green Curry'
url = 'https://world.openfoodfacts.org/cgi/search.pl?search_terms=' + urllib.parse.quote(query) + '&search_simple=1&action=process&json=1'
print("Searching OFF for:", query)
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Astra/1.0'})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        products = data.get('products', [])
        print(f"Found {len(products)} products.")
        for p in products[:5]:
            print("---")
            print("Name:", p.get('product_name'))
            print("Brand:", p.get('brands'))
            print("Quantity:", p.get('quantity'))
            nutr = p.get('nutriments', {})
            print("Energy/100g:", nutr.get('energy-kcal_100g'), "kcal")
            print("Protein/100g:", nutr.get('proteins_100g'), "g")
            print("Carbs/100g:", nutr.get('carbohydrates_100g'), "g")
            print("Sugars/100g:", nutr.get('sugars_100g'), "g")
            print("Fat/100g:", nutr.get('fat_100g'), "g")
            print("Saturates/100g:", nutr.get('saturated-fat_100g'), "g")
            print("Salt/100g:", nutr.get('salt_100g'), "g")
except Exception as e:
    print('Error:', e)
