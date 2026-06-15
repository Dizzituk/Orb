# FILE: app/astra_presence/__init__.py
# Purpose: ASTRA presence package — in-memory orb-state broadcast so any surface
#          (Room orb, desktop, phone) can reflect ASTRA's live processing state.
# Called-by: main (router), app.scene_director.voice, tests
# Depends-on: app.astra_presence submodules (imported by consumers)
# Last-renovated: 2026-06-13
"""ASTRA presence (v2).

A tiny, DB-less broadcast of ASTRA's current "state" — one of the eight orb
states (idle / listening / thinking / speaking / deep research / message /
error / wake). Mirrors the scene_director subscriber-registry pattern. The
Room orb subscribes over WS /astra/ws and shifts its plasma look to match;
other backend code calls presence_state.set_state(...) at cheap seams.
"""
