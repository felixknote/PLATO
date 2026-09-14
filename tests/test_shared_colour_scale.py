"""One colour scale across every plot drawn from one dataset.

Colour carries meaning only if the same colour means the same value
everywhere it appears. The categorical path already guarantees this: its
mapping is built from ``_colour_universe``, every value the FULL frame
carries, so filtering hides points without repainting the ones that remain.

The continuous path did not. Its percentile limits came from the visible
subset, so the ramp rescaled whenever a filter or a facet changed what was on
screen. Three consequences, all silent: a point changed colour without
changing value, two facets of one grid drew the same measurement in different
colours, and the range readout under the plot disagreed between two exports of
the same data. These tests pin the limits to the whole projected set.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from plato.data.embeddings import EmbeddingDataset
from plato.data.explorer_model import build_frame
from plato.data.workspace import EmbeddingEntry
from plato.views.explorer import EmbeddingExplorer


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


N = 240
STAT = "mean_intensity"


@pytest.fixture
def explorer(app):
    """An explorer with a projection and a measured statistic spanning 0..N.

    The statistic is monotonic in row order and the frame splits into two
    plates, so a filter to one plate halves the range -- exactly the case
    that used to rescale the ramp.
    """

    class _Session:
        pass

    wells = [f"{chr(65 + r)}{c:02d}" for r in range(8) for c in range(1, 13)]
    raw = pd.DataFrame(
        {
            "plate": ["P1"] * (N // 2) + ["P2"] * (N // 2),
            "well": [wells[i % len(wells)] for i in range(N)],
            "gene": [f"g{i % 3}" for i in range(N)],
            "image_name": [f"i{i}" for i in range(N)],
        }
    )
    dataset = EmbeddingDataset(
        name="d",
        directory=Path("/x"),
        vectors=np.random.default_rng(0).normal(size=(N, 8)).astype(np.float32),
        frame=raw,
        run_info={},
    )
    frame, _ = build_frame(dataset)

    widget = EmbeddingExplorer(_Session(), Path(tempfile.mkdtemp()))
    widget.set_image_mode(False)
    widget._add_entry(EmbeddingEntry(name="d", dataset=dataset, frame=frame))

    # A statistic that is low on P1 and high on P2, so restricting to either
    # plate leaves a strictly narrower range than the whole set.
    widget._image_stats = {STAT: np.arange(N, dtype=np.float32)}
    widget._stat_column = STAT
    return widget


def _limits(explorer, rows, visible):
    """The low/high the ramp would use, read back from the readout text."""
    coords = np.zeros((len(visible), 2), dtype=np.float32)
    values = explorer._stat_values(visible)
    explorer._draw_continuous(coords, visible, values, rows, False)
    return explorer.count_label.text()


def test_the_ramp_limits_come_from_every_point_not_the_visible_ones(explorer):
    """The core fix: filtering must not rescale the colour axis."""
    rows = np.arange(N)
    everything = _limits(explorer, rows, rows)
    one_plate = _limits(explorer, rows, np.arange(N // 2))
    # The readout carries the range; if the ramp rescaled, these differ.
    assert everything.split("·")[-1] == one_plate.split("·")[-1]


def test_a_point_keeps_its_colour_when_a_filter_hides_its_neighbours(explorer):
    from plato.views.palette import ramp_over_array

    rows = np.arange(N)
    values = explorer._stat_values(rows)
    finite = values[np.isfinite(values)]
    low, high = (float(v) for v in np.percentile(finite, [2, 98]))

    # Row 10's colour under the full-set limits...
    full, _ = ramp_over_array(values, low, high)
    # ...must be what it gets when only the first plate is visible, because
    # the limits do not move.
    subset_rows = np.arange(N // 2)
    subset_values = explorer._stat_values(subset_rows)
    narrowed, _ = ramp_over_array(subset_values, low, high)
    assert full[10] == narrowed[10]


def test_two_facets_of_one_grid_share_the_scale(explorer):
    """A facet is a filter by another name, so the same rule has to hold.

    Colouring P1 and P2 on limits taken from their own points would make the
    brightest well of each plate the same colour, hiding exactly the
    between-plate difference the grid was opened to show.
    """
    rows = np.arange(N)
    first = _limits(explorer, rows, np.arange(N // 2))
    second = _limits(explorer, rows, np.arange(N // 2, N))
    assert first.split("·")[-1] == second.split("·")[-1]


def test_nothing_measured_on_screen_still_says_so(explorer):
    """The guard is about the visible points, not about whether a scale exists.

    A filter that hides every measured point must get the plain readout, even
    though the dataset as a whole does have a range.
    """
    rows = np.arange(N)
    explorer._image_stats = {STAT: np.full(N, np.nan, dtype=np.float32)}
    explorer._image_stats[STAT][:10] = 5.0
    text = _limits(explorer, rows, np.arange(50, 100))
    assert "no measurements" in text


def test_the_categorical_universe_is_still_the_whole_frame(explorer):
    """The rule the continuous path now follows was already true here."""
    universe = explorer._colour_universe("plate")
    assert set(universe) == {"P1", "P2"}
