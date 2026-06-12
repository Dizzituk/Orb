# FILE: app/debug/__init__.py
# Purpose: Debug Assistant Module
# Called-by: app.debug.orchestrator.behaviour_verifier, app.debug.web_tool_definitions
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Debug Assistant Module

Conversational debug agent with full sandbox access.
Phase 1: Read-only assistant (context assembly + model routing + SSE chat).
Phase 2: Agentic write access via sandbox bridge.
Phase 3: Cost tracker + UI polish.
"""
