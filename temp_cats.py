import sqlite3
conn = sqlite3.connect(r'D:\Orb\data\orb_memory.db')

print("=== Existing categories ===")
for r in conn.execute("SELECT id, name, parent_id FROM finance_categories ORDER BY parent_id, id").fetchall():
    print(f"  cat#{r[0]}: {r[1]} (parent={r[2]})")

print()
print("=== Merchant patterns ===")
for r in conn.execute("SELECT id, pattern, category_id, default_scope FROM finance_merchant_patterns ORDER BY id").fetchall():
    print(f"  pat#{r[0]}: '{r[1]}' -> cat={r[2]}, scope={r[3]}")

print()
print("=== Moneybarn-like transactions ===")
for r in conn.execute("""
    SELECT id, date, description, amount, category_name, expense_scope
    FROM finance_transactions
    WHERE LOWER(description) LIKE '%moneybarn%' OR LOWER(description) LIKE '%money barn%'
    ORDER BY date DESC LIMIT 10
""").fetchall():
    print(f"  tx#{r[0]}: {r[1]} | {r[2][:60]} | £{r[3]} | cat={r[4]} | scope={r[5]}")

conn.close()
