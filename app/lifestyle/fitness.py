# FILE: app/lifestyle/fitness.py
"""
Fitness planning engine.

Generates and manages:
- Daily stretching routines (delivery driver optimised)
- Bodyboarding performance programmes
- Gym workout plans
- Activity recommendations based on context

All plans are personalised for Taz's profile:
- 45 years old, ~118kg, bodyboarder
- Delivery driver (hip flexors, lower back, shoulders need attention)
- Keto diet preference
- Sugar addiction management
- Goal: optimise bodyboarding performance, maintain strength
"""
import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════
# STRETCHING ROUTINES
# ═══════════════════════════════════════════

DRIVER_STRETCH_ROUTINE: Dict = {
    "title": "Delivery Driver Morning Stretch",
    "description": "10-15 minute routine targeting the muscles that tighten from driving and lifting parcels all day.",
    "duration_mins": 12,
    "exercises": [
        {
            "name": "Hip Flexor Stretch (Kneeling Lunge)",
            "duration_secs": 60,
            "sets": 2,
            "notes": "30s each side. Push hips forward gently. This is the #1 priority — sitting in the van all day shortens these.",
            "targets": ["hip_flexors", "quads"],
        },
        {
            "name": "Cat-Cow Spinal Mobility",
            "duration_secs": 60,
            "sets": 1,
            "notes": "Slow and controlled. Great for waking up the spine after sleeping.",
            "targets": ["lower_back", "thoracic_spine"],
        },
        {
            "name": "Thread the Needle (Thoracic Rotation)",
            "duration_secs": 60,
            "sets": 2,
            "notes": "30s each side. Crucial for paddle power and shoulder health.",
            "targets": ["thoracic_spine", "shoulders"],
        },
        {
            "name": "Standing Hamstring Stretch",
            "duration_secs": 60,
            "sets": 2,
            "notes": "30s each leg. Foot on van step works perfectly.",
            "targets": ["hamstrings", "lower_back"],
        },
        {
            "name": "Doorway Chest Stretch",
            "duration_secs": 60,
            "sets": 1,
            "notes": "Opens up the chest and front delts. Counteracts the hunched driving position.",
            "targets": ["chest", "front_delts"],
        },
        {
            "name": "Neck Rolls & Lateral Flexion",
            "duration_secs": 45,
            "sets": 1,
            "notes": "Gentle circles then ear-to-shoulder each side. Don't force it.",
            "targets": ["neck", "upper_traps"],
        },
        {
            "name": "Ankle Circles & Calf Raises",
            "duration_secs": 45,
            "sets": 1,
            "notes": "Important for bodyboarding — ankle mobility helps with fin control.",
            "targets": ["ankles", "calves"],
        },
    ],
}


BODYBOARD_PREP_ROUTINE: Dict = {
    "title": "Pre-Surf Bodyboard Warm-Up",
    "description": "Quick warm-up to do in the car park before hitting the water. Focuses on paddle muscles, core activation, and shoulder mobility.",
    "duration_mins": 8,
    "exercises": [
        {
            "name": "Arm Circles (Progressive)",
            "duration_secs": 45,
            "sets": 1,
            "notes": "Start small, build to full range. Warms up the rotator cuff for paddling.",
            "targets": ["shoulders", "rotator_cuff"],
        },
        {
            "name": "Torso Twists",
            "duration_secs": 30,
            "sets": 1,
            "notes": "Feet planted, rotate through the core. Primes the turning muscles.",
            "targets": ["obliques", "thoracic_spine"],
        },
        {
            "name": "Bodyweight Squats",
            "duration_secs": 45,
            "sets": 1,
            "notes": "10-15 reps. Wakes up the legs for kicking and getting back out.",
            "targets": ["quads", "glutes", "hamstrings"],
        },
        {
            "name": "Prone Cobra (Lying Back Extension)",
            "duration_secs": 30,
            "sets": 2,
            "notes": "Mimics the paddling arch position. Activates the posterior chain.",
            "targets": ["lower_back", "glutes", "rear_delts"],
        },
        {
            "name": "Wrist Circles & Finger Flexion",
            "duration_secs": 30,
            "sets": 1,
            "notes": "Often forgotten but critical — cold water + gripping the board = stiff wrists.",
            "targets": ["wrists", "forearms"],
        },
    ],
}


# ═══════════════════════════════════════════
# GYM PROGRAMMES
# ═══════════════════════════════════════════

BODYBOARD_STRENGTH_PROGRAMME: Dict = {
    "title": "Bodyboard Performance Strength",
    "description": "3-day programme targeting paddle power, core stability, leg drive, and overall functional strength for a 118kg bodyboarder.",
    "days_per_week": 3,
    "target_weeks": 8,
    "schedule": {
        "day_1": {
            "name": "Push + Core",
            "exercises": [
                {"name": "Bench Press", "sets": 4, "reps": "8-10", "notes": "Paddle power base"},
                {"name": "Overhead Press", "sets": 3, "reps": "8-10", "notes": "Shoulder stability for duck-diving"},
                {"name": "Dips", "sets": 3, "reps": "8-12", "notes": "Tricep endurance for sustained paddling"},
                {"name": "Plank Variations", "sets": 3, "reps": "30-45s", "notes": "Prone stability = board control"},
                {"name": "Russian Twists", "sets": 3, "reps": "15 each side", "notes": "Rotational power for turns"},
            ],
        },
        "day_2": {
            "name": "Pull + Back",
            "exercises": [
                {"name": "Lat Pulldown / Pull-Ups", "sets": 4, "reps": "8-10", "notes": "Direct paddle muscle"},
                {"name": "Seated Row", "sets": 4, "reps": "10-12", "notes": "Endurance rowing = endurance paddling"},
                {"name": "Face Pulls", "sets": 3, "reps": "15", "notes": "Rotator cuff health — non-negotiable"},
                {"name": "Back Extension", "sets": 3, "reps": "12-15", "notes": "Prone arch endurance"},
                {"name": "Dead Hang", "sets": 3, "reps": "30-45s", "notes": "Grip endurance + shoulder decompression"},
            ],
        },
        "day_3": {
            "name": "Legs + Power",
            "exercises": [
                {"name": "Squat", "sets": 4, "reps": "8-10", "notes": "Leg drive for kicking through waves"},
                {"name": "Romanian Deadlift", "sets": 3, "reps": "10-12", "notes": "Posterior chain for prone paddling"},
                {"name": "Leg Press", "sets": 3, "reps": "12-15", "notes": "Quad endurance for fin work"},
                {"name": "Calf Raises", "sets": 4, "reps": "15-20", "notes": "Fin propulsion"},
                {"name": "Box Jumps", "sets": 3, "reps": "8", "notes": "Explosive power for getting to your feet / takeoffs"},
            ],
        },
    },
}


def get_stretch_routine(routine_type: str = "driver") -> Dict:
    """Get a stretching routine by type."""
    routines = {
        "driver": DRIVER_STRETCH_ROUTINE,
        "bodyboard_prep": BODYBOARD_PREP_ROUTINE,
    }
    return routines.get(routine_type, DRIVER_STRETCH_ROUTINE)


def get_gym_programme() -> Dict:
    """Get the current gym programme."""
    return BODYBOARD_STRENGTH_PROGRAMME


def get_all_routines() -> List[Dict]:
    """Get all available routines and programmes."""
    return [
        {**DRIVER_STRETCH_ROUTINE, "type": "stretch", "key": "driver"},
        {**BODYBOARD_PREP_ROUTINE, "type": "stretch", "key": "bodyboard_prep"},
        {**BODYBOARD_STRENGTH_PROGRAMME, "type": "gym", "key": "bodyboard_strength"},
    ]
