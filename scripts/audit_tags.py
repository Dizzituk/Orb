"""Tag/entity inspection — are HotIndex records actually tagged?"""
import sys, json
sys.path.insert(0, r'D:\Orb')
from app.db import get_db_session
from sqlalchemy import text

db = get_db_session()
try:
    print('=== TAG COVERAGE: how many records have non-empty tags? ===')
    n_total = db.execute(text("SELECT COUNT(*) FROM astra_hot_index")).scalar()
    n_with_tags = db.execute(text(
        "SELECT COUNT(*) FROM astra_hot_index "
        "WHERE tags IS NOT NULL AND tags != '[]' AND tags != ''"
    )).scalar()
    n_with_entities = db.execute(text(
        "SELECT COUNT(*) FROM astra_hot_index "
        "WHERE entities IS NOT NULL AND entities != '[]' AND entities != ''"
    )).scalar()
    print(f'  total records:     {n_total}')
    print(f'  with tags:         {n_with_tags} ({n_with_tags*100//max(n_total,1)}%)')
    print(f'  with entities:     {n_with_entities} ({n_with_entities*100//max(n_total,1)}%)')

    print()
    print('=== Sample 5 TAO records: title + tags + entities ===')
    rows = db.execute(text(
        "SELECT id, title, tags, entities FROM astra_hot_index "
        "WHERE LOWER(title) LIKE '%tao%' OR LOWER(one_liner) LIKE '%tao%' "
        "ORDER BY created_at DESC LIMIT 5"
    )).fetchall()
    for r in rows:
        print(f'  id={r[0]}')
        print(f'    title:    {(r[1] or "")[:90]}')
        print(f'    tags:     {r[2]!r}')
        print(f'    entities: {r[3]!r}')

    print()
    print('=== Most common tag values (top 15) ===')
    rows = db.execute(text(
        "SELECT tags, COUNT(*) FROM astra_hot_index "
        "WHERE tags IS NOT NULL AND tags != '[]' AND tags != '' "
        "GROUP BY tags ORDER BY COUNT(*) DESC LIMIT 15"
    )).fetchall()
    for r in rows:
        print(f'  ({r[1]}x)  {r[0]}')

finally:
    db.close()
