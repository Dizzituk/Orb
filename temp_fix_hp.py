import sqlite3
conn = sqlite3.connect(r'D:\Orb\data\orb_memory.db')

# With interest > payment, the ENTIRE payment is interest
# (the unpaid interest gets added to the balance)
# So deductible_amount = full payment amount

mb_txs = conn.execute("""
    SELECT id, transaction_date, amount
    FROM finance_transactions
    WHERE LOWER(description) LIKE '%moneybarn%'
    AND is_deleted = 0
    ORDER BY transaction_date ASC
""").fetchall()

finance_amount = 12594.0
apr = 41.9
monthly_rate = apr / 100.0 / 12.0

balance = finance_amount
for i, tx in enumerate(mb_txs):
    actual_interest = round(balance * monthly_rate, 2)
    payment = tx[2]
    # Payment doesn't cover interest, so entire payment is interest
    # Unpaid interest (actual_interest - payment) rolls into balance
    unpaid_interest = round(actual_interest - payment, 2)
    balance = round(balance + unpaid_interest, 2)
    
    notes = (
        f"HP Payment #{i+1}: "
        f"Full £{payment:.2f} is interest (claimable). "
        f"Actual interest due: £{actual_interest:.2f}, "
        f"unpaid interest £{unpaid_interest:.2f} added to balance. "
        f"Outstanding: £{balance:.2f}"
    )
    
    conn.execute("""
        UPDATE finance_transactions SET
            deductible_amount = ?,
            notes = ?
        WHERE id = ?
    """, (payment, notes, tx[0]))
    
    print(f"  #{i+1} | {tx[1]} | paid=£{payment:.2f} | actual int=£{actual_interest:.2f} | unpaid=£{unpaid_interest:.2f} | balance=£{balance:.2f}")

conn.commit()

total_paid = sum(tx[2] for tx in mb_txs)
print(f"\nSummary:")
print(f"  9 payments × £417.87 = £{total_paid:.2f} (ALL claimable as interest)")
print(f"  Balance now: £{balance:.2f} (started at £12,594)")
print(f"  Balance has GROWN by £{balance - 12594:.2f} due to unpaid interest")
print(f"\n  When you refinance in Oct, the new loan amount covers this inflated balance.")

conn.close()
