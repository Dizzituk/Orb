import sqlite3
conn = sqlite3.connect(r'D:\Orb\data\orb_memory.db')

# Get all work logs sorted by date
print("=== All work logs ===")
rows = conn.execute("""
    SELECT id, work_date, tour_id, delivery_count, collections, 
           gross_earnings, total_hours, rate_per_parcel, tax_year
    FROM finance_daily_work_logs
    ORDER BY work_date DESC
""").fetchall()

for r in rows:
    print(f"  #{r[0]}: {r[1]} | tour={r[2]} | del={r[3]} col={r[4]} | £{r[5]:.2f} | {r[6]}h | rate=£{r[7]} | ty={r[8]}")

print(f"\nTotal: {len(rows)} work logs")

# Check the screenshot filenames for actual dates
print("\n=== Processed screenshots (filename has real date) ===")
for r in conn.execute("""
    SELECT id, drive_filename, processed_at, work_log_id
    FROM finance_processed_screenshots
    ORDER BY processed_at DESC
    LIMIT 20
""").fetchall():
    print(f"  #{r[0]}: {r[1]} -> log#{r[3]}")

conn.close()
