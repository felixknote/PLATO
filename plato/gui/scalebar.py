"""Draws a scale bar onto a PIL image, for baking into JPEG exports.

Shares the same sizing rule as the live viewer overlay (see viewer.py):
bar length is a fraction of the image width (settings.get_scale_bar_fraction),
rounded to a "nice" 1/2/5 x 10^n micrometre value, converted back to pixels
using the configured nm/px.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .settings import get_nm_per_pixel, get_scale_bar_fraction

MARGIN_PX = 20
BAR_HEIGHT_PX = 6


def bar_length_um(width_px: int, nm_per_pixel: float, fraction: float) -> float:
    um_per_pixel = nm_per_pixel / 1000.0
    target_um = width_px * um_per_pixel * fraction
    magnitude = 10 ** np.floor(np.log10(max(target_um, 1e-9)))
    for factor in (1, 2, 5, 10):
        um_length = factor * magnitude
        if um_length >= target_um:
            return um_length
    return um_length


def draw_scale_bar(image: Image.Image) -> Image.Image:
    """Return a copy of ``image`` (converted to RGB) with a scale bar burned in."""
    nm_per_pixel = get_nm_per_pixel()
    if nm_per_pixel <= 0:
        return image

    width_px, height_px = image.size
    um_length = bar_length_um(width_px, nm_per_pixel, get_scale_bar_fraction())
    px_length = round(um_length * 1000.0 / nm_per_pixel)
    px_length = max(1, min(px_length, width_px - 2 * MARGIN_PX))

    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)

    x1 = width_px - MARGIN_PX
    x0 = x1 - px_length
    y1 = height_px - MARGIN_PX
    y0 = y1 - BAR_HEIGHT_PX
    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))

    label = f"{um_length:g} µm"
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    draw.text(
        (x1 - text_w, y0 - text_h - 4),
        label,
        fill=(255, 255, 255),
        font=font,
    )
    return out
