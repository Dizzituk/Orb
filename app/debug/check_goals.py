import sys
sys.path.append('D:/Orb')

from app.database import SessionLocal
from app.lifestyle.models import LifestyleGoal

db = SessionLocal()
goals = db.query(LifestyleGoal).filter(LifestyleGoal.is_active == True).all()
print("ACTIVE GOALS:")
for g in goals:
    print(f"- id={g.id}, type={g.goal_type}, value={g.target_value}, unit={g.unit}, notes={g.notes}")
db.close()
