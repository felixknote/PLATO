"""Lasso selection and density rendering.

Both need a QApplication, so they are skipped where Qt cannot start a
(possibly offscreen) platform plugin.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from plato.views.scatter import EmbeddingScatter  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def scatter(app):
    widget = EmbeddingScatter()
    # Two separated blobs, so a lasso round one has an unambiguous answer.
    rng = np.random.default_rng(0)
    left = rng.normal(loc=(-5, 0), scale=0.5, size=(300, 2))
    right = rng.normal(loc=(5, 0), scale=0.5, size=(200, 2))
    coords = np.vstack([left, right]).astype(np.float32)
    rows = np.arange(len(coords), dtype=np.int64)
    widget.set_points(coords, rows, ["#4a90d9"] * len(coords))
    return widget


def _lasso(widget, x0, x1, y0, y1):
    widget._lasso_points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    widget._lasso_finish()


def test_lasso_selects_only_the_enclosed_blob(scatter):
    received = []
    scatter.points_selected.connect(lambda rows: received.append(rows))
    _lasso(scatter, -8, -2, -3, 3)

    assert len(scatter.selected_rows) == 300
    # The left blob is rows 0-299; nothing from the right one may appear.
    assert scatter.selected_rows.max() < 300
    assert received and len(received[-1]) == 300


def test_lasso_ignores_a_stray_click(scatter):
    scatter._lasso_points = [(0.0, 0.0), (0.1, 0.1)]
    scatter._lasso_finish()
    assert len(scatter.selected_rows) == 0


def test_lasso_over_empty_space_selects_nothing(scatter):
    _lasso(scatter, 100, 110, 100, 110)
    assert len(scatter.selected_rows) == 0


def test_selection_is_dropped_when_the_points_change(scatter):
    _lasso(scatter, -8, -2, -3, 3)
    assert len(scatter.selected_rows) == 300
    # New coordinates mean the old row indices are meaningless.
    coords = np.zeros((10, 2), dtype=np.float32)
    scatter.set_points(coords, np.arange(10, dtype=np.int64), ["#4a90d9"] * 10)
    assert len(scatter.selected_rows) == 0


def test_density_replaces_the_marks(scatter):
    scatter.set_density(True)
    assert scatter._density_image.isVisible()
    assert not any(item.isVisible() for item in scatter._items)

    scatter.set_density(False)
    assert not scatter._density_image.isVisible()
    assert all(item.isVisible() for item in scatter._items)


def test_density_is_oriented_like_the_points(scatter):
    """The bright regions must land where the points are, not transposed."""
    scatter.set_density(True)
    image = scatter._density_image.image
    assert image is not None

    # Rows are y, columns are x. The blobs differ in x only, so the column
    # profile must be bimodal while the row profile stays single-peaked.
    column_profile = image.sum(axis=0)
    row_profile = image.sum(axis=1)

    def peaks(profile):
        threshold = profile.max() * 0.5
        above = profile > threshold
        return int(np.sum(above[1:] & ~above[:-1])) + int(above[0])

    assert peaks(column_profile) == 2, "x profile should show two blobs"
    assert peaks(row_profile) == 1, "y profile should show one band"


def test_density_survives_a_degenerate_extent(app):
    """Every point identical would make a zero-width bin range."""
    widget = EmbeddingScatter()
    coords = np.zeros((20, 2), dtype=np.float32)
    widget.set_points(coords, np.arange(20, dtype=np.int64), ["#4a90d9"] * 20)
    widget.set_density(True)  # must not raise
    assert widget._density_image.isVisible()
