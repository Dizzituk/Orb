import sqlite3

conn = sqlite3.connect(r'D:\Orb\data\orb_memory.db')

# Van finance details
van = conn.execute("SELECT * FROM finance_van_finance WHERE is_active = 1").fetchone()
cols = [r[1] for r in conn.execute("PRAGMA table_info(finance_van_finance)").fetchall()]
van_dict = dict(zip(cols, van))

finance_amount = van_dict['finance_amount']  # 11594 (what moneybarn lent)
purchase_price = van_dict['purchase_price']  # 12594 (total van cost = AIA claimable)
apr = van_dict['apr']  # 41.9
monthly_rate = apr / 100.0 / 12.0
monthly_payment = van_dict['monthly_payment']  # 417.87
total_payments = van_dict['total_payments']  # 60
biz_pct = van_dict['business_use_percentage'] / 100.0  # 1.0

# Build proper amortisation on FINANCED amount (not purchase price)
balance = finance_amount
schedule = []
for month in range(1, total_payments + 1):
    interest = round(balance * monthly_rate, 2)
    capital = round(monthly_payment - interest, 2)
    closing = round(balance - capital, 2)
    schedule.append((month, interest, capital, closing))
    balance -= capital

# Get Moneybarn transactions 
mb_txs = conn.execute("""
    SELECT id, transaction_date, description, amount
    FROM finance_transactions
    WHERE LOWER(description) LIKE '%moneybarn%'
    AND is_deleted = 0
    ORDER BY transaction_date ASC
""").fetchall()

print(f"Found {len(mb_txs)} Moneybarn transactions")
print(f"Purchase price (AIA): £{purchase_price}")
print(f"Financed: £{finance_amount}")
print(f"Deposit: £{van_dict['deposit_paid']}")
print()

total_interest = 0
total_capital = 0

for i, tx in enumerate(mb_txs):
    if i >= len(schedule):
        print(f"WARNING: tx#{tx[0]} beyond schedule!")
        continue
    
    month_num, interest, capital, closing_bal = schedule[i]
    deductible = round(interest * biz_pct, 2)
    total_interest += interest
    total_capital += capital
    
    notes = (
        f"HP Payment #{month_num}: "
        f"Interest £{interest:.2f} (claimable) + "
        f"Capital £{capital:.2f} (not claimable). "
        f"Balance: £{closing_bal:.2f}"
    )
    
    conn.execute("""
        UPDATE finance_transactions SET
            category_id = 24,
            expense_scope = 'business',
            is_tax_deductible = 1,
            deductible_amount = ?,
            merchant_name = 'Moneybarn (Van HP)',
            auto_categorised = 1,
            categorisation_confidence = 0.95,
            notes = ?
        WHERE id = ?
    """, (deductible, notes, tx[0]))
    
    print(f"  #{month_num} | {tx[1]} | £{tx[3]:.2f} -> int=£{interest:.2f} cap=£{capital:.2f} | ded=£{deductible:.2f} | bal=£{closing_bal:.2f}")

conn.commit()

print(f"\nTotal interest paid so far: £{total_interest:.2f}")
print(f"Total capital repaid so far: £{total_capital:.2f}")
print(f"Remaining balance: £{schedule[len(mb_txs)-1][3]:.2f}")

# Now categorise other van expenses
print("\n=== Auto-categorising van expenses ===")
patterns = conn.execute("""
    SELECT id, merchant_pattern, merchant_display_name, category_id, default_scope, confidence_score
    FROM finance_merchant_patterns WHERE is_active = 1
""").fetchall()

uncategorised = conn.execute("""
    SELECT id, description FROM finance_transactions 
    WHERE category_id IS NULL AND is_deleted = 0
""").fetchall()

updated = 0
for tx_id, desc in uncategorised:
    desc_lower = desc.lower()
    for pat in patterns:
        pat_str = pat[1].lower()
        if pat_str == 'moneybarn':
            continue  # handled above
        if pat_str in desc_lower:
            conn.execute("""
                UPDATE finance_transactions SET
                    category_id = ?,
                    merchant_name = ?,
                    expense_scope = ?,
                    is_tax_deductible = CASE WHEN ? = 'business' THEN 1 ELSE 0 END,
                    auto_categorised = 1,
                    categorisation_confidence = ?
                WHERE id = ?
            """, (pat[3], pat[2], pat[4], pat[4], pat[5], tx_id))
            print(f"  tx#{tx_id}: {desc[:50]} -> {pat[2]}")
            updated += 1
            break

conn.commit()
print(f"Updated {updated} van-related transactions")
conn.close()
