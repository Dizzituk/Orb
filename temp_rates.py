import sqlite3
conn = sqlite3.connect(r'D:\Orb\data\orb_memory.db')

# Check if there's a rate config table
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'finance%rate%'").fetchall()]
print(f"Rate tables: {tables}")

# For now, store rates as a simple config in a new table
conn.execute("""
    CREATE TABLE IF NOT EXISTS finance_delivery_rates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        effective_from DATE NOT NULL,
        rate_per_parcel REAL NOT NULL,
        rate_per_collection REAL DEFAULT 0.0,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

# Insert rate history
conn.execute("DELETE FROM finance_delivery_rates")  # clean slate
conn.execute("""
    INSERT INTO finance_delivery_rates (effective_from, rate_per_parcel, rate_per_collection, notes)
    VALUES ('2025-04-06', 1.85, 0.0, 'Standard rate from start of tax year 2025/26')
""")
conn.execute("""
    INSERT INTO finance_delivery_rates (effective_from, rate_per_parcel, rate_per_collection, notes)
    VALUES ('2026-01-20', 2.35, 0.0, 'Pay increase from third week of January 2026')
""")
conn.commit()

# Verify
for r in conn.execute("SELECT * FROM finance_delivery_rates ORDER BY effective_from").fetchall():
    print(f"  From {r[1]}: £{r[2]}/parcel, £{r[3]}/collection - {r[4]}")

conn.close()
print("Rate history stored")
