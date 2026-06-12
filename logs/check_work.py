import sqlite3

c = sqlite3.connect(r"file:D:\Orb\data\orb_memory.db?mode=ro", uri=True)
rows = c.execute(
    "SELECT id, transaction_date, transaction_type, amount, description, input_source "
    "FROM finance_transactions WHERE is_deleted = 0 ORDER BY id DESC LIMIT 6"
).fetchall()
print("Newest 6 transactions:")
for r in rows:
    print("  ", r)
