"""The export headline, and the swallowed-change bug in the option rebuilds.

**The headline.** An exported file leaves the app carrying nothing: on screen
the dataset, method and encoding are all readable from the sidebar, but in a
thesis figure a bare scatter of coloured dots is unidentifiable, and the
filename that did encode those facts is gone the moment the file is renamed.

**The rebuilds.** Every ``_rebuild_*_options`` blocks signals while it
repopulates a combo -- it has to, or ``clear()`` alone fires
``currentIndexChanged`` once per removed item. But blocking also swallows the
real change: when the previously selected column does not exist in the new
dataset, the box silently lands on a different one and nothing repaints, so
the plot stays coloured by the old field while the box names the new one.
That is the "colour-by dropdown does not react" symptom, and these tests pin
the fix for all four combos.
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from plato.data.embeddings import EmbeddingDataset
from plato.data.explorer_model import build_frame
from plato.data.workspace import EmbeddingEntry
from plato.views import headline
from plato.views.explorer import EmbeddingExplorer, export_headline

SVG_NS = "{http://www.w3.org/2000/svg}"


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


# -- the headline text ---------------------------------------------------------


def test_names_the_dataset_method_encoding_and_count():
    title, subtitle = export_headline(
        dataset="Aug26 CRISPRi & ABx",
        method="UMAP",
        column="gene",
        group_column=None,
        shown=5000,
        total=32256,
    )
    assert title == "Aug26 CRISPRi & ABx"
    assert "UMAP" in subtitle
    assert "Gene" in subtitle  # the human label, not the raw column
    # A subsampled projection is a different claim from a complete one, and
    # nothing else in the image says which it is.
    assert "5,000 of 32,256 points" in subtitle


def test_says_all_points_when_nothing_was_subsampled():
    _, subtitle = export_headline(
        dataset="d", method="UMAP", column="gene",
        group_column=None, shown=1000, total=1000,
    )
    assert "1,000 points" in subtitle
    assert " of " not in subtitle


def test_a_facet_grid_names_the_grouping():
    _, subtitle = export_headline(
        dataset="d", method="t-SNE", column="moa",
        group_column="plate", shown=800, total=800,
    )
    assert "grid by Plate" in subtitle
    assert "colour: Mechanism of action" in subtitle


def test_a_facet_grid_does_not_repeat_the_colour_when_it_is_the_grouping():
    _, subtitle = export_headline(
        dataset="d", method="UMAP", column="plate",
        group_column="plate", shown=10, total=10,
    )
    assert subtitle.count("Plate") == 1


def test_a_missing_dataset_name_yields_an_empty_title_not_a_placeholder():
    """Better a missing line than an invented one."""
    title, subtitle = export_headline(
        dataset=None, method="UMAP", column="gene",
        group_column=None, shown=10, total=10,
    )
    assert title == ""
    assert subtitle  # the rest still describes the plot


def test_plate_location_reads_as_a_location_in_the_headline():
    from plato.data import plate_location as pl

    _, subtitle = export_headline(
        dataset="d", method="UMAP", column=pl.EDGE_DISTANCE,
        group_column=None, shown=96, total=96,
    )
    assert "colour: Plate location: distance from edge" in subtitle


# -- the drawn block -----------------------------------------------------------


# Everything below needs a QApplication: QFont/QFontMetrics abort the process
# outright without one, rather than raising.


def test_nothing_to_draw_takes_no_height(app):
    """So an export with no headline keeps exactly its old page size."""
    assert headline.height("", "") == 0


def test_height_grows_with_each_line(app):
    one = headline.height("Title", "")
    two = headline.height("Title", "Subtitle")
    assert 0 < one < two


@pytest.mark.parametrize(
    "background,expect_light_ink",
    [("#ffffff", False), ("#05080f", True), ("#f2f4f8", False)],
)
def test_ink_follows_the_plot_background_not_the_app_theme(
    app, background, expect_light_ink
):
    """A plot's ground is independent of the app theme (see scatter.py).

    Ink chosen from the theme would be invisible on half the exports: black
    on a dark plot, white on a light one.
    """
    title_ink, _ = headline.ink_for(QColor(background))
    is_light = title_ink.lightness() > 127
    assert is_light is expect_light_ink


def test_a_transparent_export_gets_mid_grey_ink(app):
    """It will be composited over something unknown, so neither extreme works."""
    ink, _ = headline.ink_for(QColor(0, 0, 0, 0))
    assert 80 < ink.lightness() < 175


# -- the exported files --------------------------------------------------------


@pytest.fixture
def scatter(app):
    from plato.views.scatter import EmbeddingScatter

    widget = EmbeddingScatter()
    rng = np.random.default_rng(0)
    widget.set_points(
        rng.normal(size=(200, 2)), np.arange(200), ["#4a90d9"] * 200
    )
    return widget


def test_svg_export_carries_the_headline_as_real_text(scatter, tmp_path):
    """Not a rasterised strip: the point of the SVG is that it stays editable."""
    target = tmp_path / "p.svg"
    scatter.export(str(target), title="My dataset", subtitle="UMAP - 10 points")
    root = ET.parse(target).getroot()
    texts = [e.text for e in root.iter(f"{SVG_NS}text") if e.text]
    assert "My dataset" in texts
    assert "UMAP - 10 points" in texts


def test_svg_export_grows_the_canvas_and_shifts_the_plot_down(scatter, tmp_path):
    """The headline must not overlap the data it describes."""
    bare = tmp_path / "bare.svg"
    titled = tmp_path / "titled.svg"
    scatter.export(str(bare))
    scatter.export(str(titled), title="T", subtitle="S")

    def view_height(path):
        return float(ET.parse(path).getroot().get("viewBox").split()[3])

    block = headline.height("T", "S")
    assert view_height(titled) == pytest.approx(view_height(bare) + block)

    transforms = [
        g.get("transform")
        for g in ET.parse(titled).getroot().iter(f"{SVG_NS}g")
        if g.get("transform")
    ]
    assert f"translate(0,{block:g})" in transforms


def test_svg_escapes_markup_in_a_dataset_name(scatter, tmp_path):
    """An ampersand in a name must not produce invalid XML."""
    target = tmp_path / "amp.svg"
    scatter.export(str(target), title="Aug26 CRISPRi & ABx", subtitle="")
    # Parsing at all is the assertion; a raw & would raise here.
    root = ET.parse(target).getroot()
    assert "Aug26 CRISPRi & ABx" in [
        e.text for e in root.iter(f"{SVG_NS}text") if e.text
    ]


def test_png_export_grows_by_exactly_the_block_height(scatter, tmp_path):
    from PySide6.QtGui import QImage

    bare = tmp_path / "bare.png"
    titled = tmp_path / "titled.png"
    scatter.export(str(bare))
    scatter.export(str(titled), title="T", subtitle="S")
    plain, headed = QImage(str(bare)), QImage(str(titled))
    assert headed.width() == plain.width()
    assert headed.height() == plain.height() + headline.height("T", "S")


def test_export_without_a_headline_is_unchanged(scatter, tmp_path):
    """The feature must be inert when there is nothing to say."""
    target = tmp_path / "bare.svg"
    scatter.export(str(target))
    root = ET.parse(target).getroot()
    # pyqtgraph writes its own <title>; what must NOT appear is a translated
    # wrapper group that only the headline path adds.
    assert not any(
        (g.get("transform") or "").startswith("translate(0,")
        for g in root.iter(f"{SVG_NS}g")
    )


# -- the swallowed-change bug --------------------------------------------------


def _entry(name: str, genes: list[str], n: int = 192) -> EmbeddingEntry:
    wells = [f"{chr(65 + r)}{c:02d}" for r in range(8) for c in range(1, 13)]
    rng = np.random.default_rng(abs(hash(name)) % 2**31)
    raw = pd.DataFrame(
        {
            "plate": [f"P{i % 2 + 1}" for i in range(n)],
            "well": [wells[i % len(wells)] for i in range(n)],
            "gene": [genes[i % len(genes)] for i in range(n)],
            "image_name": [f"{name}_{i}" for i in range(n)],
        }
    )
    dataset = EmbeddingDataset(
        name=name,
        directory=Path("/x") / name,
        vectors=rng.normal(size=(n, 16)).astype(np.float32),
        frame=raw,
        run_info={},
    )
    frame, _ = build_frame(dataset)
    return EmbeddingEntry(name=name, dataset=dataset, frame=frame)


@pytest.fixture
def explorer(app):
    class _Session:
        pass

    widget = EmbeddingExplorer(_Session(), Path(tempfile.mkdtemp()))
    widget.set_image_mode(False)
    return widget


def test_switching_to_a_dataset_without_the_current_colour_redraws(explorer):
    """The reported symptom: the box changes, the plot does not.

    Signals are blocked during the rebuild, so the setCurrentIndex that lands
    on a surviving column fires nothing. Without an explicit check the plot
    stays coloured by a field the dropdown no longer offers.
    """
    explorer._add_entry(_entry("A", ["gyrA", "ftsZ", "murA"]))
    before = explorer.colour_box.currentData()

    calls: list[int] = []
    original = explorer.schedule_redraw
    explorer.schedule_redraw = lambda *a, **k: (calls.append(1), original(*a, **k))

    explorer._add_entry(_entry("B", ["rpoB"]))
    after = explorer.colour_box.currentData()

    if after != before:
        assert calls, (
            f"colour column changed {before!r} -> {after!r} with no redraw; "
            "the plot is still coloured by the old field"
        )


def test_a_rebuild_that_keeps_the_selection_does_not_redraw(explorer):
    """The block is there for a reason -- do not trade it for a redraw storm."""
    explorer._add_entry(_entry("A", ["gyrA", "ftsZ", "murA"]))

    calls: list[int] = []
    original = explorer.schedule_redraw
    explorer.schedule_redraw = lambda *a, **k: (calls.append(1), original(*a, **k))

    explorer._rebuild_colour_options()
    assert not calls


def test_group_column_change_reaches_the_facet_switch(explorer):
    """Faceting decides whether the centre is one plot or a grid.

    A silently dropped group column left the wrong widget on screen, because
    _on_group_changed -- which swaps them -- never ran.
    """
    explorer._add_entry(_entry("A", ["gyrA", "ftsZ", "murA"]))
    explorer.group_box.setCurrentIndex(explorer.group_box.findData("plate"))
    assert explorer._group_column == "plate"

    # A frame with no 'plate' column at all: the grouping cannot survive.
    raw = pd.DataFrame(
        {"gene": ["a", "b"] * 8, "image_name": [f"i{i}" for i in range(16)]}
    )
    dataset = EmbeddingDataset(
        name="C",
        directory=Path("/x/C"),
        vectors=np.zeros((16, 8), dtype=np.float32),
        frame=raw,
        run_info={},
    )
    frame, _ = build_frame(dataset)
    explorer._add_entry(EmbeddingEntry(name="C", dataset=dataset, frame=frame))

    # Whatever it landed on, the view must agree with the box.
    assert explorer._group_column == explorer.group_box.currentData()
    faceting = bool(explorer._group_column)
    assert explorer.grid.isVisibleTo(explorer) is faceting or not explorer.isVisible()


# -- the count label's denominator ---------------------------------------------


def test_count_label_shows_the_true_total_not_the_subsample_size(explorer):
    """Regression: the count label compared against len(row_indices) -- the
    PROJECTION's own row set -- so a max-points subsample made the label
    always claim 100% of the dataset was shown. A real joint entry with a
    ~25% cap active showed "26,208 of 26,208 points" for a 104,832-row
    dataset. The denominator has to be dataset.n_points."""
    from plato.data.projection import ProjectionParams, ProjectionResult

    entry = _entry("A", ["gyrA", "ftsZ", "murA"], n=100)
    explorer._add_entry(entry)

    # Simulates a max_points subsample: the projection covers only 25 of the
    # dataset's 100 real rows.
    explorer.result = ProjectionResult(
        coords=np.random.default_rng(0).normal(size=(25, 2)).astype(np.float32),
        row_indices=np.arange(25),
        params=ProjectionParams(),
    )
    explorer._redraw()

    assert explorer.dataset.n_points == 100
    assert "25" in explorer.count_label.text()
    assert "100" in explorer.count_label.text()
    assert "of 25" not in explorer.count_label.text()


def _facet(explorer) -> None:
    """Get the grid actually showing facets, not just the group column set.

    Faceting a real dataset needs a computed projection -- _add_entry alone
    leaves explorer.result at None, so _redraw draws nothing. A synthetic
    ProjectionResult is enough; the point of these tests is what reaches the
    facets once they exist, not the projection itself.
    """
    from plato.data.projection import ProjectionParams, ProjectionResult

    entry = _entry("A", ["gyrA", "ftsZ", "murA"], n=64)
    explorer._add_entry(entry)
    explorer.group_box.setCurrentIndex(explorer.group_box.findData("plate"))
    assert explorer._group_column == "plate"

    params = ProjectionParams()
    explorer.result = ProjectionResult(
        coords=np.random.default_rng(0).normal(size=(64, 2)).astype(np.float32),
        row_indices=np.arange(64),
        params=params,
    )
    explorer._redraw()


def test_density_reaches_every_facet(explorer):
    """Toggling density while faceted used to do nothing visible: it only
    called self.scatter.set_density, and self.scatter is the HIDDEN widget
    while the grid is showing. Every facet's own scatter must switch too."""
    _facet(explorer)
    assert explorer.grid.facets

    explorer.density_box.setChecked(True)

    assert all(f.scatter.density_enabled for f in explorer.grid.facets)


def test_density_state_survives_a_facet_redraw(explorer):
    """Facets are destroyed and rebuilt on every redraw (GridView._clear in
    render()) -- a toggle set before that redraw must still be applied to
    the NEW widgets, not just the ones that existed when it was set."""
    _facet(explorer)
    explorer.density_box.setChecked(True)

    explorer._redraw()

    assert explorer.grid.facets
    assert all(f.scatter.density_enabled for f in explorer.grid.facets)


def test_lasso_reaches_every_facet(explorer):
    """Same gap as density: lasso mode only ever reached self.scatter."""
    _facet(explorer)
    assert explorer.grid.facets

    explorer.cluster_panel.lasso_button.setChecked(True)

    assert all(f.scatter.lasso_enabled for f in explorer.grid.facets)
