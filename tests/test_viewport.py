"""The shared zoom viewport of the comparison view.

The invariant that matters: whatever the zoom and pan, the visible region
stays inside the image and is the same region for every column -- columns
showing different regions would silently break the comparison the view exists
to make.
"""

from __future__ import annotations

import pytest

from plato.views.compare_view import Viewport


def test_starts_showing_the_whole_image() -> None:
    v = Viewport()
    assert v.is_identity
    assert v.crop_box(100, 100) == (0, 0, 100, 100)


def test_does_not_zoom_out_past_the_whole_image() -> None:
    v = Viewport()
    v.zoom_at(0.01, 0.5, 0.5)
    assert v.zoom == 1.0


def test_zoom_is_capped() -> None:
    v = Viewport()
    for _ in range(50):
        v.zoom_at(10.0, 0.5, 0.5)
    assert v.zoom == pytest.approx(20.0)


def test_zooming_keeps_the_anchor_point_under_the_cursor() -> None:
    v = Viewport()
    fx, fy = 0.3, 0.7

    def image_coord_under_cursor() -> tuple[float, float]:
        return (v.cx + (fx - 0.5) / v.zoom, v.cy + (fy - 0.5) / v.zoom)

    before = image_coord_under_cursor()
    v.zoom_at(2.0, fx, fy)
    assert image_coord_under_cursor() == pytest.approx(before)


@pytest.mark.parametrize("zoom", [1.0, 1.5, 2.0, 4.0, 20.0])
def test_crop_box_stays_inside_the_image(zoom) -> None:
    v = Viewport(zoom=zoom)
    v.clamp()
    left, top, right, bottom = v.crop_box(2720, 2720)
    assert 0 <= left < right <= 2720
    assert 0 <= top < bottom <= 2720


def test_panning_cannot_leave_the_image() -> None:
    v = Viewport(zoom=4.0)
    v.clamp()
    v.pan_by(-10.0, -10.0)
    left, top, right, bottom = v.crop_box(1000, 1000)
    assert left >= 0 and top >= 0
    v.pan_by(10.0, 10.0)
    left, top, right, bottom = v.crop_box(1000, 1000)
    assert right <= 1000 and bottom <= 1000


def test_crop_box_is_never_degenerate_at_extreme_zoom() -> None:
    # A small image at maximum zoom must still yield at least one pixel.
    v = Viewport(zoom=20.0)
    v.clamp()
    left, top, right, bottom = v.crop_box(8, 8)
    assert right > left and bottom > top


def test_reset_returns_to_the_whole_image() -> None:
    v = Viewport()
    v.zoom_at(4.0, 0.2, 0.8)
    v.reset()
    assert v.is_identity
    assert (v.cx, v.cy) == (0.5, 0.5)
