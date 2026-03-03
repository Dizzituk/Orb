import sqlite3, json, os, sys

db_path = r'D:\Orb\orb_memory.db'
out_path = r'D:\Orb\logs\_arch_query_results.json'

if not os.path.exists(db_path):
    print(f"DB not found: {db_path}", file=sys.stderr)
    sys.exit(1)

sz = os.path.getsize(db_path)
if sz == 0:
    print(f"DB is empty (0 bytes)", file=sys.stderr)
    sys.exit(1)

conn = sqlite3.connect(db_path)
c = conn.cursor()

# Get tables
c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [r[0] for r in c.fetchall()]

# Get recent messages containing architecture
results = {"tables": tables}

if "messages" in tables:
    c.execute("""SELECT id, role, content, provider, model, created_at 
                 FROM messages 
                 WHERE content LIKE '%architecture%' OR content LIKE '%Architecture%'
                 ORDER BY created_at DESC LIMIT 10""")
    rows = c.fetchall()
    results["arch_messages"] = [{"id": r[0], "role": r[1], "content_len": len(r[2] or ""), 
                                  "content_preview": (r[2] or "")[:300], "provider": r[3],
                                  "model": r[4], "created_at": r[5]} for r in rows]

# Check for builds/specs/segments tables
for t in tables:
    if any(k in t.lower() for k in ['build', 'spec', 'segment', 'arch', 'artefact']):
        c.execute(f"SELECT COUNT(*) FROM [{t}]")
        cnt = c.fetchone()[0]
        results.setdefault("relevant_tables", {})[t] = cnt

conn.close()

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, default=str)

print(f"Written to {out_path}")