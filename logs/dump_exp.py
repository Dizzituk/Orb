import sqlite3
db = sqlite3.connect(r'D:\Orb\data\orb_memory.db')
rows = db.execute('SELECT id, category, stage, description, root_cause, resolution, confidence, source_job_id FROM experience_patterns ORDER BY id').fetchall()
for r in rows:
    desc = (r[3] or '')[:140]
    root = (r[4] or '')[:100]
    fix = (r[5] or '')[:100]
    print(f'#{r[0]} [{r[1]}|{r[2]}] conf={r[6]} job={r[7]}')
    print(f'  DESC: {desc}')
    if root: print(f'  ROOT: {root}')
    if fix: print(f'  FIX:  {fix}')
    print()
db.close()
