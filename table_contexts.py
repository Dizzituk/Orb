with open('D:/Orb/Aldi+Create+Thai+Green+Curry+Kit+224g.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Let's extract the context around the first set of values (protein 1.2g, carbs 3.6g, fat 8.2g)
print("--- TABLE 1 CONTEXT ---")
print(html[24000:27200].replace('\n', ' '))

print("\n--- TABLE 2 CONTEXT ---")
print(html[27200:30000].replace('\n', ' '))
