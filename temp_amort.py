import sqlite3

# Calculate HP amortisation schedule
finance_amount = 11594.0
apr = 41.9 / 100  # 41.9%
monthly_rate = apr / 12
monthly_payment = 417.87
total_payments = 60

balance = finance_amount
total_interest = 0
total_capital = 0

print("Month | Payment | Interest | Capital | Balance")
print("-" * 55)
for month in range(1, total_payments + 1):
    interest = balance * monthly_rate
    capital = monthly_payment - interest
    balance -= capital
    total_interest += interest
    total_capital += capital
    if month <= 12 or month == total_payments:
        print(f"  {month:3d}  | £{monthly_payment:.2f} | £{interest:.2f}  | £{capital:.2f}  | £{max(0,balance):.2f}")
    elif month == 13:
        print("  ...  |         |          |          |")

print()
print(f"Total interest over 60 months: £{total_interest:.2f}")
print(f"Total capital repaid: £{total_capital:.2f}")
print(f"Total paid: £{monthly_payment * total_payments:.2f}")

# For the 9 payments made so far
balance = finance_amount
ytd_interest = 0
ytd_capital = 0
for month in range(1, 10):
    interest = balance * monthly_rate
    capital = monthly_payment - interest
    balance -= capital
    ytd_interest += interest
    ytd_capital += capital

print(f"\n=== 9 payments made so far ===")
print(f"Interest paid: £{ytd_interest:.2f} (claimable as revenue expense)")
print(f"Capital repaid: £{ytd_capital:.2f} (NOT a separate expense - covered by AIA)")
print(f"Remaining balance: £{balance:.2f}")

# Tax year split (Apr 2025 - Apr 2026)
# First payment June 2025, so in tax year 2025/26:
# June 2025 to March 2026 = 10 months, but only 9 paid so far
balance = finance_amount
ty_interest = 0
ty_capital = 0
for month in range(1, 10):  # 9 payments Jun-Feb
    interest = balance * monthly_rate
    capital = monthly_payment - interest
    balance -= capital
    ty_interest += interest
    ty_capital += capital

print(f"\n=== Tax year 2025/26 (9 payments Jun 25 - Feb 26) ===")
print(f"Interest claimable: £{ty_interest:.2f}")
print(f"Van AIA (one-off): £12,594.00 (full purchase price)")

