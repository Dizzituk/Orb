# FILE: app/idle/__init__.py
# Purpose: Idle-time agent system — governor, persistent task ledger, task router.
# Called-by: main.py (startup), app.memory.service (activity hook), app.tools.registry
# Depends-on: app.idle.governor, app.idle.ledger, app.idle.router, app.idle.tasks
# Last-renovated: 2026-07-01
"""
ASTRA's idle-time background work system.

No user chat activity for IDLE_MINUTES -> the governor drains the persistent
task ledger (repo map refresh, watcher observations, deep research). A new
user message checkpoints in-flight work between resumable units; work resumes
in the next idle window. All LLM use inside idle tasks goes through
llm_call(execution_context="background_local") — cloud chat calls are refused
at the provider registry.
"""
