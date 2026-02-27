import sqlite3
conn = sqlite3.connect(r'D:\Orb\data\orb_memory.db')

# category_name isn't a column - it must come from a JOIN
# Check categories table
cols = [r[1] for r in conn.execute("PRAGMA table_info(finance_categories)").fetchall()]
print(f"Category columns: {cols}")
for r in conn.execute("SELECT * FROM finance_categories").fetchall():
    print(f"  {r}")

# merchant patterns
cols2 = [r[1] for r in conn.execute("PRAGMA table_info(finance_merchant_patterns)").fetchall()]
print(f"\nMerchant pattern columns: {cols2}")
for r in conn.execute("SELECT * FROM finance_merchant_patterns").fetchall():
    print(f"  {r}")

# Moneybarn by category_id
print("\n=== Moneybarn txs ===")
for r in conn.execute("""
    SELECT t.id, t.transaction_date, t.description, t.amount, t.category_id, t.expense_scope,
           c.name as cat_name
    FROM finance_transactions t
    LEFT JOIN finance_categories c ON t.category_id = c.id
    WHERE LOWER(t.description) LIKE '%moneybarn%'
    ORDER BY t.transaction_date DESC LIMIT 10
""").fetchall():
    print(f"  tx#{r[0]}: {r[1]} | {r[2][:60]} | £{r[3]} | cat_id={r[4]} cat={r[6]} | scope={r[5]}")

# Van-related
print("\n=== Van-related txs (fuel, repair, insurance, MOT) ===")
for r in conn.execute("""
    SELECT t.id, t.transaction_date, t.description, t.amount, t.expense_scope, c.name
    FROM finance_transactions t
    LEFT JOIN finance_categories c ON t.category_id = c.id
    WHERE LOWER(t.description) LIKE '%fuel%'
       OR LOWER(t.description) LIKE '%halfords%'
       OR LOWER(t.description) LIKE '%mot%'
       OR LOWER(t.description) LIKE '%insurance%'
       OR LOWER(t.description) LIKE '%garage%'
       OR LOWER(t.description) LIKE '%moneybarn%'
       OR LOWER(t.description) LIKE '%tyre%'
       OR LOWER(t.description) LIKE '%tire%'
    ORDER BY t.transaction_date DESC LIMIT 20
""").fetchall():
    print(f"  tx#{r[0]}: {r[1]} | {r[2][:55]} | £{r[3]} | scope={r[4]} | cat={r[5]}")

conn.close()
