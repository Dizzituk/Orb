"""Audit script v2: retrieval quality check for Astra memory."""
import sys, json
sys.path.insert(0, r'D:\Orb')

from app.db import get_db_session
from sqlalchemy import text

db = get_db_session()
try:
    print('=== HOT INDEX record_type distribution ===')
    rows = db.execute(text(
        "SELECT record_type, COUNT(*) FROM astra_hot_index "
        "GROUP BY record_type ORDER BY COUNT(*) DESC"
    )).fetchall()
    for r in rows:
        print(f'  {r[0]}: {r[1]}')

    print()
    print('=== HOT INDEX age distribution ===')
    rows = db.execute(text(
        "SELECT CASE "
        "  WHEN created_at > datetime('now','-7 days') THEN '1_last_week' "
        "  WHEN created_at > datetime('now','-30 days') THEN '2_last_month' "
        "  WHEN created_at > datetime('now','-90 days') THEN '3_last_quarter' "
        "  ELSE '4_older' END as age, COUNT(*) "
        "FROM astra_hot_index GROUP BY age ORDER BY age"
    )).fetchall()
    for r in rows:
        print(f'  {r[0]}: {r[1]}')

    print()
    print('=== PREFERENCE namespace distribution (top 15) ===')
    rows = db.execute(text(
        "SELECT namespace, COUNT(*) FROM astra_preferences "
        "GROUP BY namespace ORDER BY COUNT(*) DESC LIMIT 15"
    )).fetchall()
    for r in rows:
        print(f'  {r[0]}: {r[1]}')

    print()
    print('=== TOPIC SPOT CHECKS (keyword search) ===')
    print('  topic                     kw              hot  prefs  summaries')
    topics = [
        ('TAO', 'tao'),
        ('Bittensor', 'bittensor'),
        ('immigration', 'immigration'),
        ('AI displacement', 'displacement'),
        ('Reform party', 'reform'),
        ('Polanski/Green', 'polanski'),
        ('quantum', 'quantum'),
        ('robotics', 'robotics'),
        ('Yodel', 'yodel'),
        ('Leigh Day', 'leigh'),
        ('Portugal', 'portugal'),
        ('Cornwall', 'cornwall'),
        ('Anthropic', 'anthropic'),
    ]
    for label, kw in topics:
        n_hot = db.execute(text(
            "SELECT COUNT(*) FROM astra_hot_index "
            "WHERE LOWER(title) LIKE :pat OR LOWER(one_liner) LIKE :pat "
            "  OR LOWER(bullets_5) LIKE :pat OR LOWER(tags) LIKE :pat"
        ), {'pat': f'%{kw}%'}).scalar() or 0
        n_prefs = db.execute(text(
            "SELECT COUNT(*) FROM astra_preferences "
            "WHERE LOWER(preference_key) LIKE :pat OR LOWER(preference_value) LIKE :pat"
        ), {'pat': f'%{kw}%'}).scalar() or 0
        n_summaries = db.execute(text(
            "SELECT COUNT(*) FROM conversation_summaries "
            "WHERE LOWER(summary_json) LIKE :pat"
        ), {'pat': f'%{kw}%'}).scalar() or 0
        print(f'  {label:25s} {kw:15s} {n_hot:4d}  {n_prefs:5d}  {n_summaries:5d}')

    # Sample a TAO hot_index hit to see what shape the actual records take
    print()
    print('=== SAMPLE: 3 hot_index records mentioning "tao" ===')
    rows = db.execute(text(
        "SELECT id, record_type, title, one_liner, retrieval_priority, created_at "
        "FROM astra_hot_index "
        "WHERE LOWER(title) LIKE '%tao%' OR LOWER(one_liner) LIKE '%tao%' "
        "ORDER BY created_at DESC LIMIT 3"
    )).fetchall()
    for r in rows:
        print(f'  id={r[0]} type={r[1]} prio={r[4]} at={r[5]}')
        print(f'    title: {(r[2] or "")[:90]}')
        print(f'    line:  {(r[3] or "")[:90]}')

finally:
    db.close()
