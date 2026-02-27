# FILE: app/lifestyle/__init__.py
"""
Lifestyle Engine — personalised health, fitness, nutrition & surf tracking.

Designed as a portable engine: all logic lives here, frontend is just a skin.
Can be lifted out and repackaged as a standalone app.

Modules:
- models.py      — SQLAlchemy ORM tables
- schemas.py     — Pydantic request/response shapes
- router.py      — FastAPI endpoints
- service.py     — Core business logic (weight trends, macro calcs, streaks)
- nutrition.py   — Food logging, macro estimation, keto tracking
- fitness.py     — Workout plans, stretching routines, bodyboarding programmes
- biometrics.py  — Weight tracking, trend analysis, goal calculations
- seed.py        — Default data seeding (stretches, exercise library)
"""
