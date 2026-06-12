"""Fuel semantics repair + per-mile rate analysis, 2026-06-11 late.
1. The £91 was a tank FILL, not the day's consumption: clone it into a proper
   fuel expense transaction (red ledger row) modelled on the 6 Jun one.
2. Clear fuel_cost on today's WorkDay row.
3. Compute fuel-per-mile from real data: fuel spend / miles, June window
   and tax-year-to-date, so the Fuel column can become rate x day-miles.
"""
import sqlite3

c = sqlite3.connect(r"D:\Orb\data\orb_memory.db")
cur = c.cursor()

# 1. Clone the 6 Jun fuel expense (id 560) as today's £91 fill.
already = cur.execute(
    "SELECT COUNT(*) FROM finance_transactions WHERE transaction_date='2026-06-11' "
    "AND amount=91.0 AND is_deleted=0").fetchone()[0]
if already == 0:
    cols = [d[1] for d in cur.execute("PRAGMA table_info(finance_transactions)").fetchall()]
    row = cur.execute("SELECT * FROM finance_transactions WHERE id=560").fetchone()
    rec = dict(zip(cols, row))
    rec.pop("id", None)
    rec["transaction_date"] = "2026-06-11"
    rec["amount"] = 91.0
    rec["created_at"] = "2026-06-11 22:45:00.000000"
    rec["updated_at"] = "2026-06-11 22:45:00.000000"
    rec["input_source"] = "repair"
    keys = ", ".join(rec.keys())
    ph = ", ".join("?" for _ in rec)
    cur.execute(f"INSERT INTO finance_transactions ({keys}) VALUES ({ph})", list(rec.values()))
    print("[fuel] GBP91 expense row created (cloned from 6 Jun fill)")
else:
    print("[fuel] expense already present, skipped")

# 2. Clear the misused per-day fuel_cost on today's work day.
cur.execute("UPDATE finance_work_days SET fuel_cost=NULL WHERE work_date='2026-06-11'")
print("[workday] fuel_cost cleared:", cur.rowcount)

c.commit()

# 3. Rate analysis (read-only from here).
fuels = cur.execute(
    "SELECT transaction_date, amount FROM finance_transactions "
    "WHERE is_deleted=0 AND transaction_type='expense' AND description LIKE '%Fuel%' "
    "ORDER BY transaction_date").fetchall()
print("\nfuel fills on record:")
total_fuel = 0.0
for d, a in fuels:
    print(f"   {d}  GBP{a}")
    total_fuel += a

days = cur.execute(
    "SELECT work_date, start_odometer, end_odometer, personal_miles FROM finance_work_days "
    "WHERE status='complete' AND end_odometer IS NOT NULL AND start_odometer IS NOT NULL "
    "ORDER BY work_date").fetchall()
total_miles = 0.0
june_miles = 0.0
june_fuel = sum(a for d, a in fuels if d >= "2026-05-29")
for wd, s, e, pm in days:
    dist = max(0.0, (e or 0) - (s or 0))
    if dist > 0 and dist < 400:
        total_miles += dist
        if wd >= "2026-05-29":
            june_miles += dist

print(f"\nwindow 29 May-today: GBP{june_fuel:.2f} fuel over {june_miles:.0f} work miles")
if june_miles:
    r = june_fuel / june_miles
    print(f"   -> {r*100:.1f}p per mile; typical 106-mile day ~ GBP{r*106:.2f}")
print(f"all-time logged: GBP{total_fuel:.2f} fuel over {total_miles:.0f} work miles")
if total_miles:
    r2 = total_fuel / total_miles
    print(f"   -> {r2*100:.1f}p per mile; typical 106-mile day ~ GBP{r2*106:.2f}")
c.close()
