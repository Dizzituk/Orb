# FILE: app/lifestyle/product_router.py
# Purpose: HTTP API for the personal food product library — Job 4 (2026-06-10).
# Called-by: main
# Depends-on: app.db, app.lifestyle, app.lifestyle.product_models, app.lifestyle.product_service
# Last-renovated: 2026-06-11
"""
HTTP API for the personal food product library — Job 4 (2026-06-10).

  GET  /lifestyle/products?q=            search the local library
  GET  /lifestyle/products/barcode/{c}   resolve a barcode (library → OFF, auto-saves)
  POST /lifestyle/products/resolve       resolve by name/brand/barcode (the
                                          "vague photo → look it up online" path)
  POST /lifestyle/products               manual upsert
  POST /lifestyle/products/{id}/log      log eating a known product (scaled portion)

Designed to be called by the AstraBridge photo/barcode flow and by chat.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.db import SessionLocal
from app.lifestyle.product_models import ensure_product_table
from app.lifestyle import product_service as svc

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/lifestyle/products", tags=["lifestyle-products"])

ensure_product_table()


class ResolveRequest(BaseModel):
    name: Optional[str] = None
    brand: Optional[str] = None
    barcode: Optional[str] = None


class UpsertRequest(BaseModel):
    display_name: str
    brand: Optional[str] = None
    barcode: Optional[str] = None
    calories: Optional[float] = None
    protein_g: Optional[float] = None
    carbs_g: Optional[float] = None
    fat_g: Optional[float] = None
    fibre_g: Optional[float] = None
    sugar_g: Optional[float] = None
    serving_size_g: Optional[float] = None
    serving_description: Optional[str] = None
    source: str = "manual"


class LogUseRequest(BaseModel):
    grams: Optional[float] = None
    meal_type: str = "meal"


@router.get("")
def search(q: str = Query(..., min_length=2), limit: int = Query(10, ge=1, le=50)) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        items = svc.search_products(db, q, limit=limit)
        return {"count": len(items), "items": [svc.product_to_dict(p) for p in items]}
    finally:
        db.close()


@router.get("/barcode/{code}")
def by_barcode(code: str) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        result = svc.resolve_and_save(db, barcode=code)
        if not result:
            raise HTTPException(status_code=404, detail=f"No product found for barcode {code}")
        return {"found_via": result["found_via"], "product": svc.product_to_dict(result["product"])}
    finally:
        db.close()


@router.post("/resolve")
def resolve(request: ResolveRequest) -> Dict[str, Any]:
    if not request.name and not request.barcode:
        raise HTTPException(status_code=400, detail="name or barcode required")
    db = SessionLocal()
    try:
        result = svc.resolve_and_save(
            db, name=request.name, brand=request.brand, barcode=request.barcode,
        )
        if not result:
            return {"found": False, "hint": "Not in library or Open Food Facts — fall back to LLM estimation."}
        return {
            "found": True,
            "found_via": result["found_via"],
            "product": svc.product_to_dict(result["product"]),
        }
    finally:
        db.close()


@router.post("")
def upsert(request: UpsertRequest) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        macros = {
            "calories": request.calories, "protein_g": request.protein_g,
            "carbs_g": request.carbs_g, "fat_g": request.fat_g,
            "fibre_g": request.fibre_g, "sugar_g": request.sugar_g,
        }
        product = svc.upsert_product(
            db,
            display_name=request.display_name,
            brand=request.brand,
            barcode=request.barcode,
            macros=macros,
            serving_size_g=request.serving_size_g,
            serving_description=request.serving_description,
            source=request.source,
        )
        return {"product": svc.product_to_dict(product)}
    finally:
        db.close()


@router.post("/{product_id}/log")
def log_use(product_id: int, request: LogUseRequest) -> Dict[str, Any]:
    from app.lifestyle.product_models import FoodProduct
    db = SessionLocal()
    try:
        product = db.query(FoodProduct).filter(FoodProduct.id == product_id).first()
        if not product:
            raise HTTPException(status_code=404, detail=f"Unknown product id {product_id}")
        log = svc.record_product_use(
            db, product, grams=request.grams, meal_type=request.meal_type,
        )
        return {
            "logged": True,
            "nutrition_log_id": log.id,
            "description": log.description,
            "calories": log.calories,
            "protein_g": log.protein_g,
            "product": svc.product_to_dict(product),
        }
    finally:
        db.close()
