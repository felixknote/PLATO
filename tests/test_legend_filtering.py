"""Clicking a legend swatch filters the plot to that value.

The Filters section already has a multi-select list per field, but reaching
"just gyrA" meant opening that section, finding Gene in a list of fields, and
selecting it there -- three steps to ask a question the plot is already
answering visually. A click on the swatch itself is the fast path: the
legend already names every value and colours it, so it is also the natural
place to act on one.

Plain click means "just this value" (replacing whatever was filtered on this
column); shift/ctrl-click means "add or remove this value", matching the
modifier convention lasso selection already uses for additive selection.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from plato.data.embeddings import EmbeddingDataset
from plato.data.explorer_model import build_frame
from plato.data.workspace import EmbeddingEntry


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


GENES = ["gyrA", "ftsZ", "murA", "rpoB"]


@pytest.fixture
def explorer(app):
    from plato.data.projection import UMAP, ProjectionParams, ProjectionResult
    from plato.views.explorer import EmbeddingExplorer

    class _Session:
        pass

    n = 200
    wells = [f"{chr(65 + r)}{c:02d}" for r in range(8) for c in range(1, 13)]
    raw = pd.DataFrame(
        {
            "plate": [f"P{i % 3 + 1}" for i in range(n)],
            "well": [wells[i % len(wells)] for i in range(n)],
            "condition": [f"{GENES[i % len(GENES)]}_1" for i in range(n)],
            "image_name": [f"i{i}" for i in range(n)],
        }
    )
    dataset = EmbeddingDataset(
        name="d",
        directory=Path("/x"),
        vectors=np.zeros((n, 8), dtype=np.float32),
        frame=raw,
        run_info={},
    )
    frame, _ = build_frame(dataset)

    widget = EmbeddingExplorer(_Session(), Path(tempfile.mkdtemp()))
    widget.set_image_mode(False)
    widget._add_entry(EmbeddingEntry(name="d", dataset=dataset, frame=frame))
    rng = np.random.default_rng(0)
    widget.result = ProjectionResult(
        coords=rng.normal(size=(n, 2)).astype(np.float32),
        row_indices=np.arange(n),
        params=ProjectionParams(method=UMAP),
    )
    widget._set_message("")
    index = widget.colour_box.findData("gene")
    widget.colour_box.setCurrentIndex(index)
    widget._redraw(reset_view=False)
    return widget


def test_the_legend_remembers_its_own_values(explorer):
    """The precondition for a click to mean anything at all."""
    values = explorer.scatter._legend_values
    assert values is not None
    assert set(values) == set(GENES)


def test_a_plain_click_filters_to_exactly_that_value(explorer):
    explorer._on_legend_value_clicked("gyrA", False)
    assert explorer.current_filters() == {"gene": ["gyrA"]}


def test_the_plot_actually_narrows_after_the_click(explorer):
    """Not just a filter box ticked -- the redraw has to follow it."""
    total_before = len(explorer._visible_rows)
    explorer._on_legend_value_clicked("gyrA", False)
    explorer._redraw(reset_view=False)
    assert 0 < len(explorer._visible_rows) < total_before


def test_a_second_plain_click_replaces_rather_than_adds(explorer):
    explorer._on_legend_value_clicked("gyrA", False)
    explorer._on_legend_value_clicked("ftsZ", False)
    assert explorer.current_filters() == {"gene": ["ftsZ"]}


def test_a_shift_click_adds_to_the_filter(explorer):
    explorer._on_legend_value_clicked("gyrA", False)
    explorer._on_legend_value_clicked("ftsZ", True)
    assert set(explorer.current_filters()["gene"]) == {"gyrA", "ftsZ"}


def test_a_shift_click_on_an_already_filtered_value_removes_it(explorer):
    """Toggle, not just add -- the same modifier undoes what it did."""
    explorer._on_legend_value_clicked("gyrA", True)
    explorer._on_legend_value_clicked("ftsZ", True)
    explorer._on_legend_value_clicked("gyrA", True)
    assert explorer.current_filters()["gene"] == ["ftsZ"]


def test_clicking_the_same_value_again_does_not_clear_the_filter(explorer):
    """A repeated plain click must not be treated as 'undo'.

    Nothing else in the app treats a second identical action as a reset, and
    surprising that convention here would be worse than doing nothing.
    """
    explorer._on_legend_value_clicked("gyrA", False)
    explorer._on_legend_value_clicked("gyrA", False)
    assert explorer.current_filters() == {"gene": ["gyrA"]}


def test_a_column_with_no_existing_filter_box_gets_one_on_demand(explorer):
    """Plate location, Dataset and a custom grouping are not in
    FILTER_FIELDS, so a legend click on one of those would otherwise have
    nowhere to land."""
    from plato.data import plate_location as pl

    index = explorer.colour_box.findData(pl.WELL_ROW)
    explorer.colour_box.setCurrentIndex(index)
    explorer._redraw(reset_view=False)
    assert not any(box.column == pl.WELL_ROW for box in explorer.filters)

    explorer._on_legend_value_clicked("A", False)
    assert any(box.column == pl.WELL_ROW for box in explorer.filters)
    assert explorer.current_filters() == {pl.WELL_ROW: ["A"]}


def test_an_on_demand_filter_box_behaves_like_an_ordinary_one(explorer):
    """Once created it is not a special case -- clear_filters must reach it."""
    from plato.data import plate_location as pl

    index = explorer.colour_box.findData(pl.WELL_ROW)
    explorer.colour_box.setCurrentIndex(index)
    explorer._redraw(reset_view=False)
    explorer._on_legend_value_clicked("A", False)
    explorer.clear_filters()
    assert explorer.current_filters() == {}


def test_the_blank_placeholder_maps_back_to_an_empty_string():
    """The legend shows "(blank)" for an empty value; the frame stores "".

    A click on that swatch has to filter on the real stored value, not on
    the literal display text, or the filter would match nothing.
    """
    import tempfile as _tempfile

    from plato.data.embeddings import EmbeddingDataset as _Dataset
    from plato.data.explorer_model import build_frame as _build_frame
    from plato.data.projection import UMAP, ProjectionParams, ProjectionResult
    from plato.data.workspace import EmbeddingEntry as _Entry
    from plato.views.explorer import EmbeddingExplorer

    class _Session:
        pass

    n = 30
    raw = pd.DataFrame(
        {
            "plate": ["P1"] * n,
            "well": [f"A{i:02d}" for i in range(n)],
            "condition": ["" if i % 3 == 0 else f"gyrA_{i}" for i in range(n)],
            "image_name": [f"i{i}" for i in range(n)],
        }
    )
    dataset = _Dataset(
        name="d",
        directory=Path("/x"),
        vectors=np.zeros((n, 8), dtype=np.float32),
        frame=raw,
        run_info={},
    )
    frame, _ = _build_frame(dataset)
    widget = EmbeddingExplorer(_Session(), Path(_tempfile.mkdtemp()))
    widget.set_image_mode(False)
    widget._add_entry(_Entry(name="d", dataset=dataset, frame=frame))
    widget.result = ProjectionResult(
        coords=np.zeros((n, 2), dtype=np.float32),
        row_indices=np.arange(n),
        params=ProjectionParams(method=UMAP),
    )
    widget._set_message("")
    index = widget.colour_box.findData("condition")
    widget.colour_box.setCurrentIndex(index)
    widget._redraw(reset_view=False)

    widget._on_legend_value_clicked("(blank)", False)
    assert widget.current_filters() == {"condition": [""]}


def test_opening_the_filters_section_on_a_click(explorer):
    """The result is otherwise invisible: nothing on the plot itself says a
    filter box now exists, so the section has to be revealed."""
    explorer._on_legend_value_clicked("gyrA", False)
    assert explorer.accordion.open_key() == "filters"
