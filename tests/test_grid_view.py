"""GridView: faceted rendering, its shared legend, and page export.

Needs a QApplication, so it is skipped where Qt cannot start a (possibly
offscreen) platform plugin -- same pattern as test_scatter_tools.py.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from plato.views.grid_view import GridView, build_groups  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def grid(app):
    return GridView()


def _two_groups(n=40):
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(n, 2)).astype(np.float32)
    rows = np.arange(n, dtype=np.int64)
    colours = ["#4a90d9" if i % 2 == 0 else "#e8a33d" for i in range(n)]
    values = np.array(["A" if i % 2 == 0 else "B" for i in range(n)])
    return coords, rows, colours, values


def test_legend_hidden_with_no_entries(grid):
    coords, rows, colours, values = _two_groups()
    groups, _ = build_groups(values)
    grid.set_groups(groups)
    grid.render(coords, rows, colours, legend_entries=None)
    assert not grid.legend.has_entries


def test_legend_shown_with_entries(grid):
    coords, rows, colours, values = _two_groups()
    groups, _ = build_groups(values)
    grid.set_groups(groups)
    grid.render(
        coords,
        rows,
        colours,
        legend_entries=[("A", "#4a90d9"), ("B", "#e8a33d")],
    )
    assert grid.legend.has_entries
    # One entry widget per (label, colour) pair, not one per facet -- the
    # whole point is a single shared strip rather than a repeat per panel.
    assert grid.legend._layout.count() >= 2


def test_legend_cleared_alongside_facets(grid):
    coords, rows, colours, values = _two_groups()
    groups, _ = build_groups(values)
    grid.set_groups(groups)
    grid.render(coords, rows, colours, legend_entries=[("A", "#4a90d9")])
    assert grid.legend.has_entries
    grid.clear()
    assert not grid.legend.has_entries


def test_export_png_produces_a_nonempty_file(grid, tmp_path):
    coords, rows, colours, values = _two_groups()
    groups, _ = build_groups(values)
    grid.set_groups(groups)
    grid.render(
        coords,
        rows,
        colours,
        legend_entries=[("A", "#4a90d9"), ("B", "#e8a33d")],
    )
    target = tmp_path / "grid.png"
    grid.export(str(target))
    assert target.exists()
    assert target.stat().st_size > 0


def test_export_svg_produces_a_nonempty_file(grid, tmp_path):
    coords, rows, colours, values = _two_groups()
    groups, _ = build_groups(values)
    grid.set_groups(groups)
    grid.render(coords, rows, colours, legend_entries=None)
    target = tmp_path / "grid.svg"
    grid.export(str(target))
    assert target.exists()
    assert target.stat().st_size > 0


def test_export_matches_the_actual_on_screen_size(app, tmp_path):
    """Regression: export used to size the canvas from the container's
    sizeHint() unconditionally, but a QGridLayout's sizeHint() is its
    PREFERRED size, not its current on-screen stretched size -- a grid shown
    in a 900px-wide window stretches its facets to fill that width, which
    sizeHint() does not reflect. Exporting at sizeHint() produced a file
    proportioned differently to whatever was actually on screen. When the
    view genuinely is on screen, its live geometry must be trusted outright,
    not blended with sizeHint() via max().
    """
    grid = GridView()
    grid.resize(900, 700)
    grid.show()
    app.processEvents()

    coords, rows, colours, values = _two_groups()
    groups, _ = build_groups(values)
    grid.set_groups(groups)
    grid.render(coords, rows, colours, legend_entries=[("A", "#4a90d9"), ("B", "#e8a33d")])
    app.processEvents()

    target = tmp_path / "shown_grid.png"
    grid.export(str(target))

    from PySide6.QtGui import QImage

    saved = QImage(str(target))
    assert saved.width() == grid._container.width()
    grid.close()


def test_export_covers_every_facet_on_the_page(grid, tmp_path):
    """Exporting must not silently crop to the scroll viewport.

    Regression guard: the container widget is what gets rendered, not
    grid.scroll (which clips to whatever fits on screen). A grid several
    panels tall would otherwise lose its bottom rows in the exported file
    while looking complete on screen (only the visible slice would export).
    """
    rng = np.random.default_rng(1)
    n = 200
    coords = rng.normal(size=(n, 2)).astype(np.float32)
    rows = np.arange(n, dtype=np.int64)
    colours = ["#4a90d9"] * n
    # Ten groups forces multiple grid rows well past one screen's height.
    values = np.array([f"g{i % 10}" for i in range(n)])
    groups, _ = build_groups(values)
    grid.set_groups(groups)
    grid.render(coords, rows, colours, legend_entries=None)
    assert len(grid.facets) == 10

    target = tmp_path / "tall_grid.png"
    grid.export(str(target))

    from PySide6.QtGui import QPixmap

    saved = QPixmap(str(target))
    # The full container's laid-out height must be reflected in the file,
    # not the (typically much shorter) on-screen scroll viewport.
    assert saved.height() >= grid._container.sizeHint().height()
