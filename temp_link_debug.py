import sqlite3
conn = sqlite3.connect(r'D:\Orb\data\orb_memory.db')

# Get all NatWest transactions that might be card payments
print("=== NatWest transactions with card-like descriptions ===")
for r in conn.execute("""
    SELECT id, description, amount, expense_scope, linked_card_id
    FROM finance_transactions
    WHERE LOWER(description) LIKE '%jaja%'
       OR LOWER(description) LIKE '%fluid%'
       OR LOWER(description) LIKE '%zable%'
       OR LOWER(description) LIKE '%capital one%'
       OR LOWER(description) LIKE '%capital%one%'
    ORDER BY id
    LIMIT 20
""").fetchall():
    print(f"  tx#{r[0]}: {r[1][:80]} | amt={r[2]} | scope={r[3]} | linked={r[4]}")

print()
print("=== Credit card natwest_descriptions ===")
for r in conn.execute("SELECT id, name, natwest_description FROM finance_credit_cards").fetchall():
    print(f"  card#{r[0]}: {r[1]} -> natwest_desc='{r[2]}'")

conn.close()
