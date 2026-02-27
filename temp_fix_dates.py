import sqlite3
import re
from datetime import date

conn = sqlite3.connect(r'D:\Orb\data\orb_memory.db')

# Get all processed screenshots with their log IDs and filenames
screenshots = conn.execute("""
    SELECT ps.work_log_id, ps.drive_filename, wl.work_date, wl.delivery_count, wl.collections
    FROM finance_processed_screenshots ps
    JOIN finance_daily_work_logs wl ON ps.work_log_id = wl.id
    ORDER BY ps.id
""").fetchall()

print(f"Found {len(screenshots)} screenshot-to-log mappings\n")

# Rate change: £1.85 before 20 Jan 2026, £2.35 from 20 Jan 2026 onward
RATE_CHANGE_DATE = date(2026, 1, 20)  # approximate: "third week of January"
OLD_RATE = 1.85
NEW_RATE = 2.35
DEFAULT_HOURS = 10.0

updated = 0
for log_id, filename, current_date, deliveries, collections in screenshots:
    # Extract real date from filename: Screenshot_YYYYMMDD-HHMMSS.png
    m = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if not m:
        print(f"  SKIP log#{log_id}: can't parse date from '{filename}'")
        continue
    
    real_year = int(m.group(1))
    real_month = int(m.group(2))
    real_day = int(m.group(3))
    
    try:
        real_date = date(real_year, real_month, real_day)
    except ValueError:
        print(f"  SKIP log#{log_id}: invalid date from '{filename}'")
        continue
    
    # Determine rate based on real date
    rate = NEW_RATE if real_date >= RATE_CHANGE_DATE else OLD_RATE
    
    # Earnings = deliveries only (not collections) × rate
    earnings = round(deliveries * rate, 2)
    
    # Compute HMRC tax year
    if real_date.month > 4 or (real_date.month == 4 and real_date.day >= 6):
        tax_year = f"{real_date.year}/{str(real_date.year + 1)[-2:]}"
    else:
        tax_year = f"{real_date.year - 1}/{str(real_date.year)[-2:]}"
    
    # Per-hour rate
    per_hour = round(earnings / DEFAULT_HOURS, 2) if DEFAULT_HOURS > 0 else 0
    per_delivery = rate
    
    # Food allowance: qualifies if working > 10 hours
    food_qual = DEFAULT_HOURS >= 10
    
    conn.execute("""
        UPDATE finance_daily_work_logs SET
            work_date = ?,
            total_hours = ?,
            rate_per_parcel = ?,
            gross_earnings = ?,
            per_hour = ?,
            per_delivery = ?,
            total_parcels = ?,
            tax_year = ?,
            qualifies_food_allowance = ?,
            food_allowance_claimed = ?
        WHERE id = ?
    """, (
        str(real_date), DEFAULT_HOURS, rate, earnings, per_hour, per_delivery,
        deliveries, tax_year, food_qual, food_qual, log_id,
    ))
    
    status = "RATE_NEW" if rate == NEW_RATE else "RATE_OLD"
    if str(real_date) != current_date:
        print(f"  [OK] log#{log_id}: {current_date} -> {real_date} | {deliveries} del × £{rate} = £{earnings} [{status}] [DATE FIXED]")
    else:
        print(f"  [OK] log#{log_id}: {real_date} | {deliveries} del × £{rate} = £{earnings} [{status}]")
    updated += 1

conn.commit()
print(f"\nUpdated {updated} work logs")

# Summary
print("\n=== Summary by tax year ===")
for r in conn.execute("""
    SELECT tax_year, COUNT(*), SUM(delivery_count), SUM(gross_earnings), 
           AVG(delivery_count), AVG(gross_earnings)
    FROM finance_daily_work_logs
    GROUP BY tax_year ORDER BY tax_year
""").fetchall():
    print(f"  {r[0]}: {r[1]} days | {r[2]} deliveries | £{r[3]:.2f} earnings | avg {r[4]:.0f} del/day £{r[5]:.2f}/day")

conn.close()

