import sqlite3
conn = sqlite3.connect(r'D:\Orb\data\orb_memory.db')
cursor = conn.cursor()

# 1. Add van HP categories
categories_to_add = [
    ('van_hp_interest', 'Van HP Interest', 'Interest on bank and business loans', 'business', 1, 100.0, 'percent', '#E74C3C', 6),
    ('van_hp_capital', 'Van HP Capital Repayment', None, 'business', 0, 0.0, 'banknote', '#95A5A6', 7),
]

for name, display, hmrc_cat, scope, deductible, pct, icon, colour, sort in categories_to_add:
    cursor.execute("""
        INSERT OR IGNORE INTO finance_categories (name, display_name, hmrc_category, default_scope, is_deductible, deductible_percentage, icon, colour, sort_order, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
    """, (name, display, hmrc_cat, scope, deductible, pct, icon, colour, sort))
    print(f"Added category: {display}")

# 2. Add merchant patterns for van-related expenses
patterns = [
    ('moneybarn', 'Moneybarn (Van HP)', None, 'business', 0.95),  # category set dynamically per split
    ('rac motoring', 'RAC Breakdown Cover', 2, 'business', 0.95),  # vehicle_maintenance
    ('acorn insurance', 'Acorn Insurance (Van)', 3, 'business', 0.95),  # vehicle_insurance
    ('halfords', 'Halfords', 2, 'business', 0.85),  # vehicle_maintenance
    ('taylors tyres', 'Taylors Tyres', 2, 'business', 0.90),
    ('turnpike garage', 'Turnpike Garage', 1, 'business', 0.85),  # fuel
    ('rock insurance', 'Rock Insurance', 3, 'business', 0.85),  # vehicle_insurance
]

for pattern, display, cat_id, scope, confidence in patterns:
    cursor.execute("""
        INSERT OR IGNORE INTO finance_merchant_patterns 
        (merchant_pattern, merchant_display_name, category_id, default_scope, confidence_score, match_count, is_active)
        VALUES (?, ?, ?, ?, ?, 0, 1)
    """, (pattern, display, cat_id, scope, confidence))
    print(f"Added pattern: {pattern} -> {display}")

conn.commit()

# Verify
print("\n=== All categories ===")
for r in conn.execute("SELECT id, name, display_name, hmrc_category, is_deductible FROM finance_categories ORDER BY sort_order").fetchall():
    print(f"  #{r[0]}: {r[1]} ({r[2]}) | HMRC: {r[3]} | deductible: {r[4]}")

print("\n=== All patterns ===")
for r in conn.execute("SELECT id, merchant_pattern, merchant_display_name, category_id FROM finance_merchant_patterns ORDER BY id").fetchall():
    print(f"  #{r[0]}: '{r[1]}' -> {r[2]} (cat={r[3]})")

conn.close()
