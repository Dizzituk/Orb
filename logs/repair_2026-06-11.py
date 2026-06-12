"""One-shot data repair, 2026-06-11 evening.
Backfills what the Health-panel chat confabulated (work day close + fuel),
deletes the phantom 0-kcal batch entry, logs tomorrow's batch half by
cloning today's real one, and corrects the doubled step count.
Runs the REAL executors so all business logic (gross, hours, recompute) applies.
"""
import asyncio
import sqlite3
import sys

sys.path.insert(0, r"D:\Orb")

from app.debug.gemini_finance_tools import _exec_finish_work_day  # noqa: E402


async def main():
    # 1. Close the work day properly: end 19:25, odometer 83376, 119 parcels,
    #    £91 fuel (the executor computes miles, gross at £2.35, per-hour).
    out = await _exec_finish_work_day({
        "end_time": "19:25",
        "end_odometer": 83376,
        "parcels": 119,
        "fuel_cost": 91,
    })
    print("[finish_work_day] ->", out[:300])


asyncio.run(main())

# 2. Nutrition + steps repairs (direct, surgical).
c = sqlite3.connect(r"D:\Orb\data\orb_memory.db")
cur = c.cursor()

# Delete the phantom 0-kcal "half of the batch-cooked meal" row (id 87).
cur.execute("DELETE FROM lifestyle_nutrition_logs WHERE id=87 AND calories=0")
print("[nutrition] phantom rows deleted:", cur.rowcount)

# Clone today's real batch half (id 86) onto tomorrow lunchtime.
cols = [d[1] for d in cur.execute("PRAGMA table_info(lifestyle_nutrition_logs)").fetchall()]
row = cur.execute("SELECT * FROM lifestyle_nutrition_logs WHERE id=86").fetchone()
if row:
    rec = dict(zip(cols, row))
    rec.pop("id", None)
    rec["logged_at"] = "2026-06-12 12:30:00.000000"
    if "date" in rec:
        rec["date"] = "2026-06-12"
    rec["description"] = "portion of chicken curry batch (second half, batch-cooked 11 Jun)"
    keys = ", ".join(rec.keys())
    ph = ", ".join("?" for _ in rec)
    cur.execute(f"INSERT INTO lifestyle_nutrition_logs ({keys}) VALUES ({ph})",
                list(rec.values()))
    print("[nutrition] tomorrow's batch half logged, kcal:", rec.get("calories"))
else:
    print("[nutrition] source row 86 missing — skipped clone")

# Correct the doubled step count (Garmin's true figure for today: 8,666).
cur.execute("UPDATE lifestyle_daily_summaries SET steps=8666 WHERE date='2026-06-11'")
print("[steps] corrected rows:", cur.rowcount)

c.commit()
c.close()
print("DONE")
