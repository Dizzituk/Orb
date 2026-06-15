# FILE: app/media/__init__.py
# Purpose: Media module — display-manager client and media playback flows
#          (stream sites, YouTube search, film discovery land here).
# Called-by: app.tools.display_tools, app.tools (media tool files)
# Depends-on: app.media.display_client
# Last-renovated: 2026-06-12
"""
Media module.

Desktop-facing media plumbing: the display-manager client (named-display
windows like "the bed screen") and the playback flows built on top of it.
The desktop side lives in orb-desktop/displays/ and is driven through the
web_automation action channel (control session platform key: "displays").
"""
