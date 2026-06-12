# FILE: app/scene_director/__init__.py
# Purpose: ASTRA Room scene director package — composes renderer-agnostic SceneDocs
#          (LLM-directed 3D scenes) that the Unity Room app interprets.
# Called-by: main (router), tests
# Depends-on: app.scene_director submodules (imported lazily by consumers)
# Last-renovated: 2026-06-12
"""ASTRA Room — Scene Director (v1).

ASTRA acts as a SCENE DIRECTOR: it emits a structured SceneDoc (JSON contract,
see schemas.py) describing skybox, environment props, actors with waypoints,
and a narrated timeline. A renderer (Unity in v1, potentially a generative
world model later) interprets that document. The intelligence lives in the
contract, never the renderer. Kept import-light deliberately — consumers
import submodules directly.
"""
