"""Selection semantics, the preview column, and image statistics.

The behaviours here are the ones a user notices immediately when they break:
a single click that opens a window it should not, a shift-click that discards
what was already chosen, a preview column that empties when the plot is
filtered. All of them are cheap to test and were specified explicitly, so
they are pinned rather than left to manual checking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from plato.views.scatter import EmbeddingScatter  # noqa: E402
from plato.views.selection_panel import SelectionPanel, _centre_crop  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def scatter(app):
    widget = EmbeddingScatter()
    coords = np.random.default_rng(0).normal(0, 1, (50, 2)).astype(np.float32)
    widget.set_points(coords, np.arange(50, dtype=np.int64), ["#4a90d9"] * 50)
    return widget


class FakeClick:
    """A pyqtgraph mouse-click event, enough of one for the handler."""

    def __init__(self, scatter, index, *, double=False, shift=False, miss=False):
        self._double = double
        self._shift = shift
        view = scatter.plot.getViewBox()
        if miss:
            # Far outside the data, so _nearest returns -1.
            self._pos = view.mapViewToScene(QPointF(1e6, 1e6))
        else:
            x, y = scatter._coords[index]
            self._pos = view.mapViewToScene(QPointF(float(x), float(y)))

    def button(self):
        return Qt.MouseButton.LeftButton

    def double(self):
        return self._double

    def modifiers(self):
        return (
            Qt.KeyboardModifier.ShiftModifier
            if self._shift
            else Qt.KeyboardModifier.NoModifier
        )

    def scenePos(self):
        return self._pos

    def accept(self):
        pass


# -- click semantics ---------------------------------------------------------


def test_single_click_selects_without_opening(scatter):
    """The explicit requirement: no accidental detail viewer on one click."""
    opened, clicked = [], []
    scatter.point_activated.connect(opened.append)
    scatter.point_clicked.connect(clicked.append)

    scatter._on_mouse_clicked(FakeClick(scatter, 5))

    assert opened == [], "a single click must not open the viewer"
    assert clicked == [5]
    assert scatter.selected_rows.tolist() == [5]


def test_double_click_opens(scatter):
    opened = []
    scatter.point_activated.connect(opened.append)
    scatter._on_mouse_clicked(FakeClick(scatter, 7, double=True))
    assert opened == [7]


def test_plain_click_replaces_the_selection(scatter):
    scatter.set_selection(np.asarray([1, 2, 3]))
    scatter._on_mouse_clicked(FakeClick(scatter, 9))
    assert scatter.selected_rows.tolist() == [9]


def test_shift_click_adds_without_replacing(scatter):
    scatter.set_selection(np.asarray([1]))
    scatter._on_mouse_clicked(FakeClick(scatter, 9, shift=True))
    assert sorted(scatter.selected_rows.tolist()) == [1, 9]

    scatter._on_mouse_clicked(FakeClick(scatter, 20, shift=True))
    assert sorted(scatter.selected_rows.tolist()) == [1, 9, 20]


def test_shift_click_toggles_a_selected_point_off(scatter):
    """Removing an individual point, as specified."""
    scatter.set_selection(np.asarray([1, 9]))
    scatter._on_mouse_clicked(FakeClick(scatter, 9, shift=True))
    assert scatter.selected_rows.tolist() == [1]


def test_clicking_empty_space_clears(scatter):
    scatter.set_selection(np.asarray([1, 2]))
    scatter._on_mouse_clicked(FakeClick(scatter, 0, miss=True))
    assert len(scatter.selected_rows) == 0


def test_shift_clicking_empty_space_keeps_the_selection(scatter):
    """Missing a point while extending must not throw the work away."""
    scatter.set_selection(np.asarray([1, 2]))
    scatter._on_mouse_clicked(FakeClick(scatter, 0, miss=True, shift=True))
    assert scatter.selected_rows.tolist() == [1, 2]


def test_selection_emits_once_per_change(scatter):
    seen = []
    scatter.points_selected.connect(lambda r: seen.append(len(r)))
    scatter.set_selection(np.asarray([1]))
    scatter.toggle_selection(2)
    assert seen == [1, 2]


def test_duplicate_rows_are_collapsed(scatter):
    scatter.set_selection(np.asarray([3, 3, 3, 1]))
    assert scatter.selected_rows.tolist() == [1, 3]


# -- the preview column ------------------------------------------------------


@pytest.fixture
def panel(app):
    frame = pd.DataFrame(
        {
            "condition": [f"cond_{i}" for i in range(100)],
            "gene": ["acrB"] * 50 + ["tolC"] * 50,
        }
    )
    widget = SelectionPanel()
    widget.set_providers(
        lambda row: f"<b>{frame.iloc[row]['condition']}</b><br>gene",
        lambda row: None,
    )
    return widget


def test_every_selected_point_gets_an_entry(panel):
    panel.set_rows([1, 2, 3])
    assert len(panel._entries) == 3
    assert panel.compare_button.isEnabled()


def test_compare_needs_at_least_two(panel):
    panel.set_rows([1])
    assert not panel.compare_button.isEnabled()
    panel.set_rows([1, 2])
    assert panel.compare_button.isEnabled()


def test_large_selections_are_paged(panel):
    """5,000 widgets would stall the GUI thread; a page does not."""
    from plato.views.selection_panel import PAGE_SIZE

    panel.set_rows(list(range(500)))
    assert len(panel._entries) == PAGE_SIZE
    assert panel.rows == list(range(500)), "the full selection is still held"


def test_removing_an_entry_reports_the_remainder(panel):
    reported = []
    panel.selection_changed.connect(lambda rows: reported.append(list(rows)))
    panel.set_rows([4, 5, 6])
    panel._on_entry_removed(5)
    assert reported == [[4, 6]]


def test_clearing_reports_an_empty_selection(panel):
    reported = []
    panel.selection_changed.connect(lambda rows: reported.append(list(rows)))
    panel.set_rows([1, 2])
    panel.clear_button.click()
    assert reported == [[]]


def test_widening_the_column_enlarges_the_images(panel):
    """Dragging the splitter is the gesture for 'show me these bigger'."""
    panel.set_rows([1, 2])
    small = panel._image_px
    panel.set_image_width(700)
    assert panel._image_px > small
    panel.set_image_width(220)
    assert panel._image_px < 700


def test_image_size_stays_within_bounds(panel):
    from plato.views.selection_panel import MAX_IMAGE_PX, MIN_IMAGE_PX

    panel.set_image_width(5000)
    assert panel._image_px <= MAX_IMAGE_PX
    panel.set_image_width(10)
    assert panel._image_px >= MIN_IMAGE_PX


def test_setting_the_same_rows_does_not_rebuild(panel):
    """Avoiding needless recomputation when a selection has not changed."""
    panel.set_rows([1, 2, 3])
    entries = dict(panel._entries)
    panel.set_rows([1, 2, 3])
    assert panel._entries == entries


def test_centre_crop_keeps_the_middle():
    plane = np.arange(100 * 100, dtype=np.uint16).reshape(100, 100)
    cropped = _centre_crop(plane, 0.5)
    assert cropped.shape == (50, 50)
    assert cropped[0, 0] == plane[25, 25]
    # A full crop is the identity, not a copy of a smaller region.
    assert _centre_crop(plane, 1.0).shape == plane.shape


# -- image statistics --------------------------------------------------------


def test_statistics_move_in_the_expected_direction():
    from plato.data.image_stats import describe_raw

    flat = np.full((256, 256), 1000, dtype=np.uint16)
    brighter = np.full((256, 256), 1500, dtype=np.uint16)
    assert describe_raw(brighter)["brightness"] > describe_raw(flat)["brightness"]

    noisy = (
        flat + np.random.default_rng(0).normal(0, 300, flat.shape)
    ).astype(np.uint16)
    assert describe_raw(noisy)["contrast"] > describe_raw(flat)["contrast"]

    sharp = flat.copy()
    sharp[::8, :] = 4000
    blurred = flat.copy()
    blurred[::8, :] = 1200
    assert describe_raw(sharp)["focus"] > describe_raw(blurred)["focus"]

    saturated = np.full((256, 256), 65535, dtype=np.uint16)
    assert describe_raw(saturated)["saturation"] > 99
    assert describe_raw(flat)["saturation"] < 1


def test_statistics_are_raw_not_per_image_normalised():
    """The whole point: brightness must survive, unlike in image_features.

    describe() normalises each image to its own range before measuring, which
    is correct when the descriptors ARE the embedding and exactly wrong when
    the question is 'is this cluster just a brightness difference'.
    """
    from plato.data.image_features import describe
    from plato.data.image_stats import describe_raw

    dim = np.full((128, 128), 500, dtype=np.uint16)
    dim[::4, :] = 800
    bright = dim.astype(np.uint32) * 4
    bright = bright.astype(np.uint16)

    assert describe_raw(bright)["brightness"] > describe_raw(dim)["brightness"] * 3
    # The normalised descriptor sees essentially the same image.
    assert describe(dim)[0] == pytest.approx(describe(bright)[0], abs=0.02)


def test_empty_and_odd_planes_do_not_raise():
    from plato.data.image_stats import STAT_NAMES, describe_raw

    empty = np.zeros((0, 0), dtype=np.uint16)
    assert set(describe_raw(empty)) == set(STAT_NAMES)

    stack = np.ones((3, 32, 32), dtype=np.uint16)
    assert np.isfinite(describe_raw(stack)["brightness"])

    floats = np.full((16, 16), np.nan, dtype=np.float32)
    assert set(describe_raw(floats)) == set(STAT_NAMES)


def test_stats_cache_rejects_a_mismatched_row_count(tmp_path):
    """A cache written for another frame must not be silently misaligned."""
    from plato.data.image_stats import StatsCache, empty

    cache = StatsCache(tmp_path)
    cache.save("fp", empty(10))
    assert cache.load("fp", 10) is not None
    assert cache.load("fp", 11) is None
    assert cache.load("other", 10) is None


def test_ramp_colours_are_batched_into_groups():
    """One draw call per band, not per point."""
    from plato.views.palette import UNKNOWN_COLOUR, ramp_over_array

    values = np.linspace(0.0, 1.0, 5000, dtype=np.float32)
    colours, groups = ramp_over_array(values, 0.0, 1.0, steps=24)
    assert len(colours) == 5000
    assert len(groups) <= 24, "colours must be quantised for batched drawing"
    assert sum(len(mask) for mask in groups.values()) == 5000


def test_unmeasured_points_are_grey_not_low():
    """NaN means 'not measured', which is not the same as 'dark'."""
    from plato.views.palette import UNKNOWN_COLOUR, ramp_over_array

    values = np.asarray([0.0, np.nan, 1.0], dtype=np.float32)
    colours, _ = ramp_over_array(values, 0.0, 1.0)
    assert colours[1] == UNKNOWN_COLOUR
    assert colours[0] != UNKNOWN_COLOUR


def test_progress_is_chunked_finely_enough_to_move():
    """A bar that jumps 0 -> 100%% reads as a hang, not as speed.

    A fixed chunk size put a 30-image dataset in a single task, so the only
    progress event was the last one. Chunks are sized to the workload instead,
    with a floor on the number of them.
    """
    from plato.views.stats_worker import MAX_CHUNK, MIN_CHUNKS, chunk_size

    for n_rows in (30, 120, 600, 24_000):
        size = chunk_size(n_rows)
        chunks = -(-n_rows // size)
        assert size <= MAX_CHUNK
        assert chunks >= MIN_CHUNKS, f"{n_rows} rows gave only {chunks} chunks"

    # Tiny inputs stay sane rather than producing a zero-sized chunk.
    assert chunk_size(1) == 1
    assert chunk_size(0) > 0


def test_a_cancelled_stats_run_delivers_nothing():
    """Cancelling must not leave a half-measured array looking complete."""
    from plato.views.stats_worker import StatsRun, StatsSignals

    signals = StatsSignals()
    delivered = []
    signals.finished.connect(delivered.append)

    run = StatsRun(10, signals)
    run.expect(1)
    run.cancel()
    run.chunk_finished(5)
    assert delivered == []


# -- image viewer on/off -----------------------------------------------------


class _CountingResolver:
    """Records how often a path is actually resolved."""

    def __init__(self):
        self.calls = 0
        from pathlib import Path

        self.root = Path("/fake")
        self.roots = [self.root]

    def path_for(self, record):
        from pathlib import Path

        self.calls += 1
        return Path("/fake/img.tif")


@pytest.fixture
def explorer(app, tmp_path):
    from plato.views.explorer import EmbeddingExplorer

    class _Session:
        pass

    widget = EmbeddingExplorer(_Session(), tmp_path)
    frame = pd.DataFrame(
        {
            "condition": [f"c{i % 4}" for i in range(50)],
            "gene": ["acrB"] * 25 + ["tolC"] * 25,
            "image_name": [f"img_{i}.tif" for i in range(50)],
        }
    )
    widget.frame = frame
    widget.metadata_panel.set_frame(frame)
    widget.resolver = _CountingResolver()
    return widget


def test_image_mode_off_resolves_nothing(explorer):
    """The requirement: OFF must be a real optimisation, not hidden UI."""
    explorer.set_image_mode(False)
    explorer.resolver.calls = 0

    explorer.scatter.set_selection(np.arange(10))
    assert explorer.resolver.calls == 0
    # However it is reached, not just through the panel.
    assert explorer._path_for(3) is None
    assert explorer.resolver.calls == 0


def test_image_mode_off_starts_no_root_hunt(explorer):
    """The hunt is ~0.3s per candidate directory on a share."""
    explorer.set_image_mode(False)
    assert explorer._start_resolving_images(explorer.frame) is None


def test_image_mode_on_does_resolve(explorer):
    explorer.set_image_mode(True)
    explorer.resolver.calls = 0
    explorer._path_for(1)
    assert explorer.resolver.calls == 1


def test_paths_are_resolved_once_per_row(explorer):
    """Several panels ask for the same row; the filesystem should not."""
    explorer.set_image_mode(True)
    explorer.resolver.calls = 0
    for _ in range(5):
        explorer._path_for(7)
    assert explorer.resolver.calls == 1


def test_changing_resolver_invalidates_cached_paths(explorer):
    explorer.set_image_mode(True)
    explorer._path_for(2)
    explorer.resolver = _CountingResolver()
    explorer._invalidate_paths()
    explorer._path_for(2)
    assert explorer.resolver.calls == 1, "a new resolver must be consulted"


def test_selection_survives_toggling_image_mode(explorer):
    explorer.set_image_mode(True)
    explorer.scatter.set_selection(np.arange(8))
    before = explorer.scatter.selected_rows.copy()

    explorer.set_image_mode(False)
    assert np.array_equal(explorer.scatter.selected_rows, before)
    assert explorer.metadata_panel.table.rowCount() == 8

    explorer.set_image_mode(True)
    assert np.array_equal(explorer.scatter.selected_rows, before)
    assert explorer.selection_panel.rows == before.tolist()


def test_analysis_still_works_with_images_off(explorer):
    explorer.set_image_mode(False)
    explorer.scatter.set_lasso(True)
    explorer.scatter.set_selection(np.arange(25))
    blocks = [b.summary.label for b in explorer.cluster_panel._blocks]
    assert blocks, "lasso analysis must not depend on images"


def test_clicking_does_not_trigger_cluster_analysis(explorer):
    """A click/shift-click selection must not run the composition breakdown.

    points_selected fires for both a plain click and a lasso; without gating
    on lasso_enabled, selecting 1-3 points by clicking produced a "composition"
    like "60% of the selection" for 3 points -- a comparison the panel exists
    to make about a drawn REGION, not whichever points happen to be clicked.
    """
    assert not explorer.scatter.lasso_enabled
    before = explorer.cluster_panel.summary_label.text()
    explorer.scatter.set_selection(np.array([1, 2, 3]))
    assert explorer.cluster_panel.summary_label.text() == before


def test_turning_lasso_off_freezes_the_last_analysis(explorer):
    """The last lasso result stays visible; it is not cleared by a click."""
    explorer.scatter.set_lasso(True)
    explorer.scatter.set_selection(np.arange(25))
    snapshot = explorer.cluster_panel.summary_label.text()

    explorer.scatter.set_lasso(False)
    explorer.scatter.set_selection(np.array([26]))
    assert explorer.cluster_panel.summary_label.text() == snapshot
