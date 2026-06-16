import re

with open('D:/Orb/Aldi+Create+Thai+Green+Curry+Kit+224g.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Let's search for standard Nutracheck patterns like calories, protein, carbs, fat.
# Usually, they have elements like:
# "Energy: 123 kcal", "Carbohydrates: 12g", etc.
# Let's print out lines matching numbers or key nutrition words.

print("--- RAW CONTENT ANALYSIS ---")
# Let's print all sentences or HTML pieces that contain numbers and g or kcal
matches = re.findall(r'([^<>]{1,100}(?:g|kcal|Calories)[^<>]{1,100})', html)
print(f"Found {len(matches)} potential text fragments.")
for m in matches[:50]:
    if any(k in m.lower() for k in ['fat', 'carb', 'protein', 'sugar', 'kcal', 'cal', 'weight', 'size']):
        print(m.strip())

print("\n--- TABLE OR LIST ANALYSIS ---")
# Let's find any tables or structures with nutritional details
for row in re.findall(r'<tr.*?>.*?</tr>', html, re.DOTALL):
    if any(k in row.lower() for k in ['fat', 'carb', 'protein', 'kcal']):
        # strip html tags to read text
        clean_row = re.sub(r'<[^>]+>', ' ', row).strip()
        # collapse spaces
        clean_row = re.sub(r'\s+', ' ', clean_row)
        print("Row:", clean_row)

for div in re.findall(r'<div[^>]*class="[^"]*nutrition[^"]*"[^>]*>.*?</div>', html, re.DOTALL):
    clean_div = re.sub(r'<[^>]+>', ' ', div).strip()
    clean_div = re.sub(r'\s+', ' ', clean_div)
    print("Div:", clean_div)

# Let's search for the word 'pouch' or 'pack' or grams/calories numbers
# In the front of pack picture, 1/2 pack as sold is 147 kcal, Fat 12.4g, Saturates 10.2g, Sugars 3.5g, Salt 1.36g
# Let's search the HTML for '147' or '117' (since per 100g is 117 kcal)
print("\n--- SEARCHING SPECIFIC NUMBERS ---")
for line in html.split('\n'):
    if '147' in line or '117' in line or '12.4' in line or '10.2' in line or '1.36' in line:
        print(line[:200].strip())
