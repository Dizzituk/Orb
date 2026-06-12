# Purpose: RAG content capture.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""RAG content capture."""
from .content_capture import (
    capture_file_content,
    is_binary_content,
    is_capturable_file,
)
