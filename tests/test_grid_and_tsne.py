"""Grid/facet view, t-SNE parameters, and the bugs the adversarial pass found.

Three real bugs are pinned here, each of which passed every existing test:

* facet variables came from a fixed whitelist, so a dataset carrying its own
  categorical column could never be grouped by it -- contrary to the
  requirement that options be derived from the loaded metadata;
* leaving the grid hid the facets instead of destroying them, leaking a
  pyqtgraph scene per facet on every switch;
* nothing verified that t-SNE's exposed parameters reached openTSNE at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plato.data.projection import (
    TSNE,
    TSNE_PRESETS,
    ProjectionParams,
    tsne_preset,
)

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from plato.data.explorer_model import categorical_fields  # noqa: E402
from plato.views.grid_view import GridView, build_groups  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


# -- discovering what to group by --------------------------------------------


def test_unknown_columns_can_be_grouped_by():
    """The bug: only whitelisted columns were ever offered.

    A dataset carrying a timepoint, a donor or an instrument column must be
    groupable by it without this module having heard of the name.
    """
    frame = pd.DataFrame(
        {
            "gene": ["a"] * 50 + ["b"] * 50,
            "instrument_serial": ["X"] * 60 + ["Y"] * 40,
            "timepoint": [f"t{i % 4}" for i in range(100)],
        }
    )
    fields = categorical_fields(frame)
    assert "instrument_serial" in fields
    assert "timepoint" in fields
    # Known fields still come first, in their established order.
    assert fields[0] == "gene"


def test_identifier_and_constant_columns_are_not_offered():
    frame = pd.DataFrame(
        {
            "image_name": [f"img_{i}.tif" for i in range(100)],
            "constant": ["same"] * 100,
            "gene": ["a"] * 50 + ["b"] * 50,
        }
    )
    fields = categorical_fields(frame)
    assert "image_name" not in fields, "an identifier is not a category"
    assert "constant" not in fields, "one value is not a grouping"
    assert "gene" in fields


def test_high_cardinality_is_capped():
    frame = pd.DataFrame({"wide": [f"v{i}" for i in range(500)]})
    assert categorical_fields(frame, max_values=60) == []


# -- grouping ----------------------------------------------------------------


def test_blank_values_become_their_own_group():
    """"The points with no value here" is a real, often informative group."""
    values = np.asarray([""] * 30 + ["v"] * 70, dtype=object)
    groups, skipped = build_groups(values)
    labels = {label for label, _ in groups}
    assert labels == {"(blank)", "v"}
    assert skipped == 0


def test_group_cap_keeps_the_largest():
    values = np.asarray([f"g{i % 30}" for i in range(300)], dtype=object)
    groups, skipped = build_groups(values, max_groups=10)
    assert len(groups) == 10
    assert skipped == 20


# -- the grid ----------------------------------------------------------------


@pytest.fixture
def grid(app):
    widget = GridView()
    widget.resize(800, 600)
    return widget


def _points(n=200):
    rng = np.random.default_rng(0)
    return (
        rng.normal(0, 1, (n, 2)).astype(np.float32),
        np.arange(n),
        ["#4a90d9"] * n,
    )


def test_shared_axes_give_identical_extents(grid):
    """The default, and the reason faceting is not misleading.

    Independent axes make every group fill its panel, so two groups occupying
    different regions look identical and a tight cluster looks as spread as a
    diffuse one.
    """
    coords, rows, colours = _points()
    values = np.asarray(["A"] * 80 + ["B"] * 120, dtype=object)
    groups, _ = build_groups(values)
    grid.set_groups(groups)

    grid.shared_axes = True
    grid.render(coords, rows, colours)
    ranges = [f.scatter.plot.getViewBox().viewRange() for f in grid.facets]
    assert len(ranges) == 2
    assert ranges[0][0] == pytest.approx(ranges[1][0])
    assert ranges[0][1] == pytest.approx(ranges[1][1])


def test_independent_axes_differ(grid):
    coords, rows, colours = _points()
    # Two groups deliberately placed in different regions.
    coords[:80] += 20
    values = np.asarray(["A"] * 80 + ["B"] * 120, dtype=object)
    groups, _ = build_groups(values)
    grid.set_groups(groups)

    grid.shared_axes = False
    grid.render(coords, rows, colours)
    ranges = [f.scatter.plot.getViewBox().viewRange() for f in grid.facets]
    assert ranges[0][0] != pytest.approx(ranges[1][0])


def test_many_groups_page_rather_than_shrink(grid):
    coords, rows, colours = _points(400)
    values = np.asarray([f"g{i % 40}" for i in range(400)], dtype=object)
    groups, _ = build_groups(values)
    grid.set_groups(groups)
    grid.page_size = 12

    assert grid.n_groups == 40
    assert grid.n_pages == 4
    grid.render(coords, rows, colours)
    assert len(grid.facets) == 12

    grid.set_page(3)
    grid.render(coords, rows, colours)
    assert len(grid.facets) == 4, "the last page holds the remainder"

    grid.set_page(99)
    assert grid.page == 3, "paging must clamp"


def test_leaving_the_grid_destroys_the_facets(grid):
    """The bug: facets were hidden, not destroyed.

    Each facet owns a pyqtgraph PlotWidget with its own scene and cached
    arrays, so keeping them behind a hidden widget leaks one scene per facet
    on every switch back and forth.
    """
    coords, rows, colours = _points()
    values = np.asarray(["A"] * 100 + ["B"] * 100, dtype=object)
    groups, _ = build_groups(values)
    grid.set_groups(groups)
    grid.render(coords, rows, colours)
    assert len(grid.facets) == 2

    grid.clear()
    assert grid.facets == []
    assert grid.n_groups == 0


def test_column_count_is_squarish(grid):
    assert grid.column_count(1) == 1
    assert grid.column_count(4) == 2
    assert grid.column_count(9) == 3
    grid.columns = 5
    assert grid.column_count(9) == 5, "an explicit choice wins"


# -- t-SNE -------------------------------------------------------------------


@pytest.fixture(scope="module")
def blobs():
    rng = np.random.default_rng(0)
    return np.vstack(
        [rng.normal(m, 0.6, (60, 12)) for m in (-6, 0, 6)]
    ).astype(np.float32)


def _tsne(vectors, **kwargs):
    from plato.data.projection import project

    params = ProjectionParams(method=TSNE, perplexity=15.0, n_iter=250, seed=42)
    for key, value in kwargs.items():
        setattr(params, key, value)
    return project(vectors, params)


def test_the_same_seed_reproduces_exactly(blobs):
    pytest.importorskip("openTSNE")
    first = _tsne(blobs)
    second = _tsne(blobs)
    assert np.allclose(first.coords, second.coords)


def test_a_different_seed_gives_a_different_layout(blobs):
    pytest.importorskip("openTSNE")
    assert not np.allclose(_tsne(blobs).coords, _tsne(blobs, seed=7).coords)


def test_perplexity_changes_the_result(blobs):
    """If it did not, exposing the control would be theatre."""
    pytest.importorskip("openTSNE")
    low = _tsne(blobs, perplexity=5.0)
    high = _tsne(blobs, perplexity=40.0)
    assert not np.allclose(low.coords, high.coords)


def test_initialisation_changes_the_result(blobs):
    pytest.importorskip("openTSNE")
    pca = _tsne(blobs, initialization="pca")
    random = _tsne(blobs, initialization="random")
    assert not np.allclose(pca.coords, random.coords)


def test_late_exaggeration_tightens_without_regrouping(blobs):
    """It changes the picture, not the grouping -- so say so, not 'no effect'."""
    pytest.importorskip("openTSNE")
    pytest.importorskip("sklearn")
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    truth = np.repeat([0, 1, 2], 60)
    plain = _tsne(blobs)
    exaggerated = _tsne(blobs, late_exaggeration=4.0)

    assert np.ptp(exaggerated.coords) < np.ptp(plain.coords), "should contract"
    labels = KMeans(3, n_init=10, random_state=0).fit_predict(exaggerated.coords)
    assert adjusted_rand_score(truth, labels) > 0.85, "grouping must survive"


def test_extreme_parameters_are_clamped_not_crashes(blobs):
    pytest.importorskip("openTSNE")
    small = blobs[:40]
    for kwargs in (
        {"perplexity": 10_000.0},
        {"perplexity": 0.0},
        {"n_iter": 1},
        {"early_exaggeration_iter": 0},
    ):
        result = _tsne(small, **kwargs)
        assert result.coords.shape == (40, 2)
        assert np.isfinite(result.coords).all()


def test_tsne_parameters_key_the_cache_but_do_not_disturb_umap():
    """Changing iterations must not silently reuse an old layout..."""
    assert (
        ProjectionParams(method=TSNE, n_iter=250).key()
        != ProjectionParams(method=TSNE, n_iter=1000).key()
    )
    # ...and must not invalidate a UMAP cache entry it has no bearing on.
    assert (
        ProjectionParams(method="UMAP", n_iter=250).key()
        == ProjectionParams(method="UMAP", n_iter=1000).key()
    )


def test_presets_are_ordered_by_cost():
    iterations = [n for _name, n, _early in TSNE_PRESETS]
    assert iterations == sorted(iterations)
    assert tsne_preset("Standard") == (500, 250)
    assert tsne_preset("nonsense") is None
