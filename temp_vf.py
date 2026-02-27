import sqlite3
conn = sqlite3.connect(r'D:\Orb\data\orb_memory.db')

cols = [r[1] for r in conn.execute("PRAGMA table_info(finance_van_finance)").fetchall()]
print(f"Van finance columns: {cols}")
for r in conn.execute("SELECT * FROM finance_van_finance").fetchall():
    print(f"  {r}")

conn.close()
