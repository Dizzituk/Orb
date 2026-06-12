import sqlite3, datetime

c = sqlite3.connect(r"file:D:\Orb\data\orb_memory.db?mode=ro", uri=True)
today = "2026-06-11"
tomorrow = "2026-06-12"

print("== today's daily summary ==")
r = c.execute("SELECT date, steps, floors, active_calories, total_calories_burned, resting_hr, sleep_minutes "
              "FROM lifestyle_daily_summaries WHERE date=?", (today,)).fetchone()
print("  ", r)

print("== nutrition entries today (id, meal, desc, kcal, P) ==")
for r in c.execute("SELECT id, meal_type, substr(description,1,48), calories, protein_g FROM lifestyle_nutrition_logs "
                   "WHERE date(logged_at)=? ORDER BY id", (today,)).fetchall():
    print("  ", r)

print("== nutrition entries tomorrow ==")
for r in c.execute("SELECT id, meal_type, substr(description,1,48), calories FROM lifestyle_nutrition_logs "
                   "WHERE date(logged_at)=?", (tomorrow,)).fetchall():
    print("  ", r)

print("== workday row today ==")
cols = [d[1] for d in c.execute("PRAGMA table_info(finance_work_days)").fetchall()]
row = c.execute("SELECT * FROM finance_work_days WHERE work_date=?", (today,)).fetchone()
if row:
    for k, v in zip(cols, row):
        if v not in (None, 0, 0.0, ""):
            print(f"   {k} = {v}")

print("== transactions today (dup check) ==")
for r in c.execute("SELECT id, transaction_date, transaction_type, amount, substr(description,1,40), created_at "
                   "FROM finance_transactions WHERE date(created_at)=? AND is_deleted=0 ORDER BY id", (today,)).fetchall():
    print("  ", r)
