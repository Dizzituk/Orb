import sqlite3
conn = sqlite3.connect(r'D:\Orb\orb_memory.db')
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
for r in c.fetchall():
    print(r[0])
conn.close()
