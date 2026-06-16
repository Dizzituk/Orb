# FILE: tests/test_capability_guard.py
# Purpose: Lock the J1 capability-probe boundary — bare probes -> chat, concrete
#          requests -> image. Guards the P0 cost bug from regressing.
# Called-by: pytest
# Depends-on: app.llm.routing._capability_guard
# Last-renovated: 2026-06-16
"""Tests for is_capability_question() — the J1 image-route cost guard."""

import pytest

from app.llm.routing._capability_guard import is_capability_question

# The EXACT failing message from the field report.
FAILING_MESSAGE = (
    "can you make images, can you make 3d worlds, "
    "can you edit social media posts, can you edit videos"
)


@pytest.mark.parametrize(
    "message",
    [
        FAILING_MESSAGE,
        "can you make images",
        "are you able to generate pictures",
        "do you support image generation",
        "what can you do",
        "could you ever make a logo",
    ],
)
def test_bare_capability_probe_is_true(message):
    """Bare capability probes must route to chat (True), not image gen."""
    assert is_capability_question(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "make me an image of a red surfboard",
        "generate a picture of a sunset",
        "make me a bar chart",
        "can you make me an image of a red surfboard",
        "draw me a logo of a wave",
    ],
)
def test_concrete_request_is_false(message):
    """Requests naming a concrete subject must still route to image (False)."""
    assert is_capability_question(message) is False
