import sqlite3
conn = sqlite3.connect(r'D:\Orb\data\orb_memory.db')

finance_amount = 12594.0  # corrected
apr = 41.9
monthly_rate = apr / 100.0 / 12.0
monthly_payment = 417.87
total_payments = 60

# Rebuild schedule with correct finance amount
balance = finance_amount
schedule = []
for month in range(1, total_payments + 1):
    interest = round(balance * monthly_rate, 2)
    capital = round(monthly_payment - interest, 2)
    closing = round(balance - capital, 2)
    schedule.append((month, interest, capital, closing))
    balance -= capital

# Re-categorise Moneybarn transactions
mb_txs = conn.execute("""
    SELECT id, transaction_date, amount
    FROM finance_transactions
    WHERE LOWER(description) LIKE '%moneybarn%'
    AND is_deleted = 0
    ORDER BY transaction_date ASC
""").fetchall()

total_interest = 0
total_capital = 0

for i, tx in enumerate(mb_txs):
    if i >= len(schedule):
        break
    month_num, interest, capital, closing_bal = schedule[i]
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
            deductible_amount = ?,
            notes = ?
        WHERE id = ?
    """, (interest, notes, tx[0]))
    
    print(f"  #{month_num} | {tx[1]} | int=£{interest:.2f} cap=£{capital:.2f} | bal=£{closing_bal:.2f}")

conn.commit()

print(f"\nCorrected totals (9 payments):")
print(f"  Interest paid: £{total_interest:.2f}")
print(f"  Capital repaid: £{total_capital:.2f}")
print(f"  Remaining balance: £{schedule[len(mb_txs)-1][3]:.2f}")

# Show full term summary
total_int_all = sum(s[1] for s in schedule)
total_cap_all = sum(s[2] for s in schedule)
final_bal = schedule[-1][3]
print(f"\nFull 60-month term:")
print(f"  Total interest: £{total_int_all:.2f}")
print(f"  Total capital repaid: £{total_cap_all:.2f}")
print(f"  Balloon at end: £{final_bal:.2f}")
print(f"  Total cost: £{1000 + monthly_payment * 60:.2f} (deposit + payments)")

conn.close()
