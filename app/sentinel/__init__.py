# FILE: app/sentinel/__init__.py
# Purpose: ASTRA Sentinel package — Phase 1 network security monitor (collector,
#          baseline, rules, LLM triage, alerts, ask-first firewall blocking).
# Called-by: main (router + scheduler), app.tools.registry, app.llm.routing.memory_injection
# Depends-on: app.sentinel submodules (imported lazily by consumers)
# Last-renovated: 2026-06-12
"""ASTRA Sentinel — Phase 1.

Watches every network connection on this PC via the elevated sentinel_agent
(127.0.0.1:8771), learns what is normal, flags anomalies through deterministic
rules, explains them with a cheap LLM, and alerts Taz on desktop + phone.
Blocking is ask-first ALWAYS: nothing is ever blocked without explicit
confirmation. Kept import-light deliberately — consumers import submodules.
"""
