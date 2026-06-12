# FILE: app/web_automation/__init__.py
# Purpose: Web Automation module.
# Called-by: app.content.distribution.browser_analytics.acquire, app.content.distribution.browser_analytics.popup_dismiss, app.content.distribution.browser_analytics.recon, app.content.distribution.browser_analytics.scrape (+5 more)
# Depends-on: app.web_automation.bridge, app.web_automation.register, app.web_automation.router, app.web_automation.seed
# Last-renovated: 2026-06-11
"""
Web Automation module.

Drives a Chromium session inside the Electron shell via a polling bridge.
Primary use: let ASTRA interact with sites that lack APIs (Coursera,
TikTok, IG/FB publishing walls) through human-speed actions, reusing
the logged-in browser session the user already has.

Public surface:
  router:             FastAPI routes (mount from main.py)
  register_web_tools: adds tool defs to app.tools.registry
  seed_sessions:      idempotent seed of default platform sessions
  bridge:             Python-side dispatch (execute_action, ensure_session_open)
"""
from app.web_automation.router import router
from app.web_automation.register import register_web_tools
from app.web_automation.seed import seed_sessions
from app.web_automation import bridge

__all__ = ["router", "register_web_tools", "seed_sessions", "bridge"]
