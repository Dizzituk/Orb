"""Sanity check: count records mentioning 'claude' vs 'anthropic' vs counted entities."""
import sys
sys.path.insert(0, r'D:\Orb')
from app.db import get_db_session
from sqlalchemy import text

db = get_db_session()
try:
    for kw in ['claude', 'anthropic', 'asi', 'agi', 'astra', 'bittensor', 'tao', 'openai']:
        n = db.execute(text(
            "SELECT COUNT(*) FROM astra_hot_index "
            "WHERE LOWER(title) LIKE :p OR LOWER(one_liner) LIKE :p"
        ), {'p': f'%{kw}%'}).scalar()
        print(f"  {kw}: {n}")
finally:
    db.close()
