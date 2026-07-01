# FILE: app/watchers/__init__.py
# Purpose: Watcher framework — observe -> record -> summarise -> alert, on the idle queue.
# Called-by: app.tools.registry (tool registration), app.idle.router (task registration)
# Depends-on: app.watchers.framework, app.watchers.models
# Last-renovated: 2026-07-01
"""
Generic daily watchers over external state (prices, availability). Observation
is deterministic scrape-and-store — no LLM per scrape; the local model is only
invoked to summarise trends on demand. Each watcher instance auto-registers a
chat tool and a daily idle-ledger task on creation.
"""
