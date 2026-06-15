# FILE: app/lifestyle/recipe_models.py
# Purpose: Saved recipes / meal templates — remember repeated meals AND their volumes (Phase 6).
# Called-by: app.lifestyle.recipe_service, app.db (init_db import)
# Depends-on: app.db
# Last-renovated: 2026-06-14
"""
Saved-recipe memory (food overhaul Phase 6).

The capstone of the overhaul: Taz cooks the same dishes at the same volumes
over and over ("beef chilli", a meal-prep batch). A FoodRecipe stores a named
dish plus its ingredient list and typical grams, keyed by a signature (the
sorted set of normalised ingredient names) so a repeated meal — typed or
photographed — matches a prior recipe and auto-expands into one itemised
NutritionLog row per ingredient via the SAME writer everything else uses
(service.log_nutrition_items). Near-zero input on repeats.

food_product_id is a SOFT reference (plain Integer, no FK) to
lifestyle_food_products so the per-ingredient macros can come from the product
library when available; otherwise the ingredient carries its own macro override.
"""
from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, Float, String, Text, DateTime, ForeignKey,
)
from sqlalchemy.orm import relationship

from app.db import Base, engine


def _now() -> datetime:
    return datetime.now(timezone.utc)


class FoodRecipe(Base):
    """A named dish / meal template the user cooks repeatedly."""

    __tablename__ = "lifestyle_food_recipes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)            # display name, e.g. "Beef chilli"
    norm_name = Column(String, nullable=False, index=True)  # normalised dedupe key
    # Signature = sorted set of normalised ingredient names, joined by '|'.
    signature = Column(String, nullable=True, index=True)
    meal_type = Column(String, default="meal")
    notes = Column(Text, nullable=True)
    times_logged = Column(Integer, default=0)
    last_logged_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_now)
    updated_at = Column(DateTime, nullable=False, default=_now)

    ingredients = relationship(
        "RecipeIngredient",
        back_populates="recipe",
        cascade="all, delete-orphan",
        order_by="RecipeIngredient.id",
    )


class RecipeIngredient(Base):
    """One ingredient of a recipe, with its typical grams and macros."""

    __tablename__ = "lifestyle_recipe_ingredients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    recipe_id = Column(
        Integer,
        ForeignKey("lifestyle_food_recipes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    display_name = Column(String, nullable=False)        # e.g. "500g beef mince"
    food_product_id = Column(Integer, nullable=True)     # soft ref to lifestyle_food_products
    grams = Column(Float, nullable=True)                 # typical portion for this recipe
    # Per-ingredient macro override (used when there's no product to scale from).
    calories = Column(Float, nullable=True)
    protein_g = Column(Float, nullable=True)
    carbs_g = Column(Float, nullable=True)
    fat_g = Column(Float, nullable=True)
    sugar_g = Column(Float, nullable=True)

    recipe = relationship("FoodRecipe", back_populates="ingredients")


def ensure_recipe_tables() -> None:
    """Idempotent create — db.py also imports this module so create_all makes
    these tables; this guards any first-use path that runs earlier."""
    Base.metadata.create_all(
        bind=engine,
        tables=[FoodRecipe.__table__, RecipeIngredient.__table__],
    )
