# FILE: app/overwatcher/__init__.py
# Purpose: Overwatcher — legacy package, mostly retired.
# Called-by: app.llm.local_tools.zobie.streams.batch_ops, app.llm.routing.routing_persistence, tests.test_evidence, tests.test_ow_orchestrator (+4 more)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""Overwatcher — legacy package, mostly retired.

Only the architecture_executor subpackage remains (used by pipeline_v2
and orchestrator for code extraction and parsing utilities).

Cost tracking  -> app/cost/
Sandbox client -> app/sandbox/client.py
Det checker    -> app/pipeline_v2/checks/

Dead modules removed 2026-03-08 during codebase audit.
"""
