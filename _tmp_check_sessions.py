import sqlite3

conn = sqlite3.connect('D:/Orb/data/orb_memory.db')
cur = conn.cursor()

# Enable FK cascades explicitly (SQLite default is off)
cur.execute('PRAGMA foreign_keys = ON')

before = cur.execute("SELECT platform, label FROM web_sessions ORDER BY platform").fetchall()
print('Before:')
for row in before:
    print(f'  {row}')

# Delete the two consolidated-away sessions
for platform in ('facebook_page', 'instagram_astraukai'):
    cur.execute("DELETE FROM web_sessions WHERE platform = ?", (platform,))
    print(f'Deleted {platform}: rowcount={cur.rowcount}')

conn.commit()

after = cur.execute("SELECT platform, label FROM web_sessions ORDER BY platform").fetchall()
print('After:')
for row in after:
    print(f'  {row}')

conn.close()
