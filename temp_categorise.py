import sys
sys.path.insert(0, r'D:\Orb')

from app.database import SessionLocal
from app.finance.services.hp_amortisation_service import (
    categorise_moneybarn_transactions,
    auto_categorise_van_expenses,
)

db = SessionLocal()

# 1. Categorise Moneybarn HP payments
print("=== Categorising Moneybarn HP payments ===")
result = categorise_moneybarn_transactions(db)
print(f"Updated: {result.get('updated', 0)} transactions")
print(f"Total interest paid: £{result.get('total_interest_paid', 0):.2f}")
print(f"Total capital repaid: £{result.get('total_capital_repaid', 0):.2f}")
print(f"Total deductible: £{result.get('total_deductible', 0):.2f}")
if 'payments' in result:
    for p in result['payments']:
        print(f"  #{p['payment_num']}: {p['date']} | int=£{p['interest']:.2f} cap=£{p['capital']:.2f} | deductible=£{p['deductible']:.2f} | bal=£{p['balance']:.2f}")

# 2. Categorise other van expenses
print("\n=== Categorising van-related expenses ===")
result2 = auto_categorise_van_expenses(db)
print(f"Updated: {result2.get('updated', 0)} transactions")
for m in result2.get('matches', []):
    print(f"  tx#{m['tx_id']}: {m['description'][:50]} -> {m['category']} ({m['pattern']})")

db.close()
