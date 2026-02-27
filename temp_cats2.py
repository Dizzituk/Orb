import sqlite3
conn = sqlite3.connect(r'D:\Orb\data\orb_memory.db')

# Check if categories table exists
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'finance%'").fetchall()]
print(f"Finance tables: {tables}")

# Check transaction columns
cols = [r[1] for r in conn.execute("PRAGMA table_info(finance_transactions)").fetchall()]
print(f"\nTransaction columns: {cols}")

# Check for category_name directly on transactions
for r in conn.execute("SELECT DISTINCT category_name FROM finance_transactions WHERE category_name IS NOT NULL AND category_name != '' LIMIT 20").fetchall():
    print(f"  Used category: '{r[0]}'")

# Check merchant patterns table
if 'finance_merchant_patterns' in tables:
    cols2 = [r[1] for r in conn.execute("PRAGMA table_info(finance_merchant_patterns)").fetchall()]
    print(f"\nMerchant pattern columns: {cols2}")
    for r in conn.execute("SELECT * FROM finance_merchant_patterns LIMIT 10").fetchall():
        print(f"  pattern: {r}")

# Moneybarn transactions
print("\n=== Moneybarn ===")
for r in conn.execute("""
    SELECT id, date, description, amount, category_name, expense_scope
    FROM finance_transactions
    WHERE LOWER(description) LIKE '%moneybarn%'
    LIMIT 10
""").fetchall():
    print(f"  tx#{r[0]}: {r[1]} | {r[2][:70]} | £{r[3]} | cat={r[4]} | scope={r[5]}")

conn.close()
