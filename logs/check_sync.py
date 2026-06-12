import sqlite3

c = sqlite3.connect(r"file:D:\Orb\data\orb_memory.db?mode=ro", uri=True)

print("DAYS (newest 3):")
rows = c.execute(
    "SELECT date, steps, floors, active_calories, total_calories_burned, "
    "resting_hr, sleep_minutes FROM lifestyle_daily_summaries "
    "ORDER BY date DESC LIMIT 3"
).fetchall()
for r in rows:
    print("  date=%s steps=%s floors=%s active=%s total=%s rhr=%s sleep=%s" % r)

print("WORKOUTS (newest 3):")
rows = c.execute(
    "SELECT started_at, activity_type, duration_mins, calories_burned, "
    "avg_hr, max_hr, hr_zones_json FROM lifestyle_workout_sessions "
    "ORDER BY started_at DESC LIMIT 3"
).fetchall()
for r in rows:
    print("  %s | %s | %smin | kcal=%s | avg=%s max=%s zones=%s" % r)

n = c.execute("SELECT COUNT(*) FROM lifestyle_health_ingest_ledger").fetchone()
print("ingest ledger rows:", n[0])
