# FILE: app/documents/__init__.py
# Purpose: Documents module — file ⇄ Univer-snapshot conversion seam, the
#          editor action queue, and atomic save plumbing for the desktop
#          editor pane (command centre).
# Called-by: main.py (router), app.tools.document_tools
# Depends-on: app.documents.router, app.documents.editor_actions, converters
# Last-renovated: 2026-06-12
"""
Documents module.

Univer's native xlsx/docx import-export is pro-tier, so conversion happens
HERE in Python: openpyxl for xlsx, python-docx for docx, stdlib for csv/md.
The desktop's Univer pane fetches snapshots from /documents/open, edits in
the renderer, and posts snapshots back to /documents/save (atomic write,
.bak of the original kept on first save).

Agent addressability: tools enqueue editor actions (editor_actions.py);
the OPEN editor pane long-polls /documents/editor/pending-action, executes
them through Univer's Facade API, and posts results back — the same
shape as the web_automation action loop, scoped to one editor session.
"""
from app.documents.router import router  # noqa: F401

__all__ = ["router"]
