# FILE: app/llm/text_cards/_gradient.py
"""
Background painter for text cards.

Two background types are supported:
    - flat: single solid colour
    - linear_gradient: multi-stop diagonal gradient at any angle

Optional radial highlights overlay on top of either background type to
add subtle depth (the trick that makes "premium" feel less flat).

Gradient generation walks every pixel in pure Python. At 1080x1080 this
takes ~30-50ms which is fine for a single render. NumPy would speed this
up but adds a dependency we don't currently need.

v1.0 (2026-05-01): Initial implementation.
"""
from __future__ import annotations

import math
from typing import Any


def paint_background(canvas, spec: dict[str, Any]) -> None:
    """Paint the background onto `canvas` in place, per spec.

    `spec` keys:
        type: "flat" | "linear_gradient"
        colour: (r,g,b)              [flat only]
        angle_deg: float             [linear_gradient only — 0=horizontal, 90=vertical]
        stops: list[(pos_0_1, (r,g,b))]   [linear_gradient only]
        radial_highlights: list[(cx_frac, cy_frac, radius_frac, alpha_0_255)]
    """
    bg_type = spec.get("type", "flat")

    if bg_type == "flat":
        _paint_flat(canvas, spec.get("colour", (255, 255, 255)))
    elif bg_type == "linear_gradient":
        _paint_linear_gradient(
            canvas,
            angle_deg=float(spec.get("angle_deg", 135)),
            stops=spec["stops"],
        )
    else:
        # Unknown type — fall back to white. Don't crash a render over
        # a typo in a style preset.
        _paint_flat(canvas, (255, 255, 255))

    for highlight in spec.get("radial_highlights", []) or []:
        _apply_radial_highlight(canvas, highlight)


# ---------------------------------------------------------------------------
# Flat background
# ---------------------------------------------------------------------------

def _paint_flat(canvas, colour: tuple[int, int, int]) -> None:
    """Fill canvas with a single colour."""
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([(0, 0), canvas.size], fill=colour)


# ---------------------------------------------------------------------------
# Linear gradient
# ---------------------------------------------------------------------------

def _paint_linear_gradient(
    canvas,
    angle_deg: float,
    stops: list[tuple[float, tuple[int, int, int]]],
) -> None:
    """Paint a multi-stop linear gradient along an arbitrary angle.

    Algorithm: for each pixel, project onto the gradient axis (a unit
    vector at `angle_deg` clockwise from east). The projection, normalised
    to [0, 1] across the canvas, picks the colour by interpolating between
    the two nearest stops.
    """
    width, height = canvas.size
    pixels = canvas.load()

    # Normalise stops: sorted, with 0.0 and 1.0 endpoints anchored.
    stops = sorted(stops, key=lambda s: s[0])
    if stops[0][0] > 0.0:
        stops = [(0.0, stops[0][1])] + stops
    if stops[-1][0] < 1.0:
        stops = stops + [(1.0, stops[-1][1])]

    # Gradient axis as a unit vector. CSS-style angles: 0deg points "to top",
    # 90deg points "to right". We use mathematical convention internally:
    # angle 0 = +x axis, angle 90 = +y (downward in image coords).
    theta = math.radians(angle_deg - 90.0)
    ux, uy = math.cos(theta), math.sin(theta)

    # Canvas extent along the axis: project the four corners and take min/max.
    corners = [(0, 0), (width, 0), (0, height), (width, height)]
    projections = [px * ux + py * uy for (px, py) in corners]
    p_min, p_max = min(projections), max(projections)
    p_range = p_max - p_min if p_max != p_min else 1.0

    # Pre-compute stop boundaries for fast lookup
    stop_positions = [s[0] for s in stops]
    stop_colours = [s[1] for s in stops]

    for y in range(height):
        for x in range(width):
            t = ((x * ux + y * uy) - p_min) / p_range
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            colour = _sample_stops(t, stop_positions, stop_colours)
            pixels[x, y] = colour


def _sample_stops(
    t: float,
    positions: list[float],
    colours: list[tuple[int, int, int]],
) -> tuple[int, int, int]:
    """Linearly interpolate a colour at position `t` (in [0,1]) across stops."""
    # Find the bracketing stops. Stops are sorted ascending.
    for i in range(len(positions) - 1):
        if positions[i] <= t <= positions[i + 1]:
            span = positions[i + 1] - positions[i]
            if span <= 0:
                return colours[i]
            local = (t - positions[i]) / span
            c0, c1 = colours[i], colours[i + 1]
            return (
                int(c0[0] + (c1[0] - c0[0]) * local),
                int(c0[1] + (c1[1] - c0[1]) * local),
                int(c0[2] + (c1[2] - c0[2]) * local),
            )
    # t falls outside all brackets — return nearest endpoint.
    return colours[0] if t <= positions[0] else colours[-1]


# ---------------------------------------------------------------------------
# Radial highlight overlay
# ---------------------------------------------------------------------------

def _apply_radial_highlight(canvas, highlight: tuple[float, float, float, int]) -> None:
    """Overlay a soft white radial glow on the canvas in place.

    `highlight` is (cx_frac, cy_frac, radius_frac, alpha_0_255).
    Implementation: build an L-mode (luminance) mask via radial falloff,
    then composite a white image through it onto the canvas.
    """
    from PIL import Image, ImageDraw

    cx_frac, cy_frac, radius_frac, alpha = highlight
    width, height = canvas.size
    cx = int(width * cx_frac)
    cy = int(height * cy_frac)
    radius = int(min(width, height) * radius_frac)

    if radius <= 0 or alpha <= 0:
        return

    # Build a radial falloff mask. Drawing many concentric circles with
    # decreasing alpha gives a smooth glow without requiring NumPy.
    mask = Image.new("L", (width, height), 0)
    mask_draw = ImageDraw.Draw(mask)
    steps = 32
    for i in range(steps, 0, -1):
        r = int(radius * (i / steps))
        # Soft falloff — alpha drops with the square of the normalised distance
        # from centre, which looks more natural than a linear ramp.
        local_alpha = int(alpha * ((1 - i / steps) ** 1.6))
        if local_alpha <= 0:
            continue
        mask_draw.ellipse(
            [(cx - r, cy - r), (cx + r, cy + r)],
            fill=local_alpha,
        )

    overlay = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(overlay, (0, 0), mask)


__all__ = ["paint_background"]
