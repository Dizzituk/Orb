# FILE: app/content/engagement/__init__.py
"""
Engagement Management System.

Automated comment classification, response, and flagging.
Deterministic by default — AI used only for classification.

Modules:
- models: DB models for comments, responses, and flags
- classifier: Sentiment/intent classification (tiered)
- responder: Template-based auto-response engine
- scanner: Polls platform APIs for new comments
- router: API endpoints for engagement dashboard
"""
