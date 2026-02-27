import sqlite3
conn = sqlite3.connect(r'D:\Orb\data\orb_memory.db')

# Fix the van finance record
conn.execute("""
    UPDATE finance_van_finance SET
        purchase_price = 13594.0,
        finance_amount = 12594.0
    WHERE is_active = 1
""")
conn.commit()

# Verify
row = conn.execute("SELECT purchase_price, deposit_paid, finance_amount FROM finance_van_finance WHERE is_active = 1").fetchone()
print(f"Purchase price: £{row[0]} (AIA claimable)")
print(f"Deposit: £{row[1]}")
print(f"Financed (Moneybarn): £{row[2]}")
print(f"Check: {row[1]} + {row[2]} = £{row[1] + row[2]}")

conn.close()
