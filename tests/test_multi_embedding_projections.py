"""Switching between open embeddings must restore an already-computed
projection instead of silently discarding it.

EmbeddingEntry.projections has existed since the workspace was introduced
specifically for this, but the explorer never wrote through to it: switching
entries unconditionally set self.result = None, so comparing two embeddings
meant recomputing UMAP/t-SNE every single time you looked back and forth --
exactly the workflow multi-embedding support exists to make cheap. This pins
the fix so it cannot regress silently again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from plato.data.embeddings import EmbeddingDataset  # noqa: E402
from plato.data.projection import ProjectionParams, ProjectionResult, UMAP, TSNE  # noqa: E402
from plato.data.workspace import EmbeddingEntry  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _entry(name: str, n: int = 20) -> EmbeddingEntry:
    frame = pd.DataFrame(
        {"condition": [f"c{i % 3}" for i in range(n)], "gene": ["a"] * n}
    )
    dataset = EmbeddingDataset(
        name=name,
        directory=__import__("pathlib").Path(f"/data/{name}"),
        vectors=np.random.default_rng(0).normal(size=(n, 8)).astype(np.float32),
        frame=frame,
        run_info={},
    )
    return EmbeddingEntry(name=name, dataset=dataset, frame=frame)


def _result(params: ProjectionParams, n: int = 20) -> ProjectionResult:
    return ProjectionResult(
        coords=np.random.default_rng(1).normal(size=(n, 2)).astype(np.float32),
        row_indices=np.arange(n),
        params=params,
    )


@pytest.fixture
def explorer(app, tmp_path):
    from plato.views.explorer import EmbeddingExplorer

    class _Session:
        pass

    widget = EmbeddingExplorer(_Session(), tmp_path)
    widget.set_image_mode(False)
    return widget


def test_on_projection_stores_result_on_the_current_entry(explorer):
    entry = _entry("A")
    explorer.workspace.add(entry)
    explorer._active_key = entry.key
    explorer.dataset = entry.dataset
    explorer.frame = entry.frame

    params = ProjectionParams(method=UMAP)
    result = _result(params)
    explorer._on_projection(result)

    assert entry.projections.get(params.key()) is result
    assert entry.last_result_key == params.key()


def test_switching_back_restores_the_projection_without_recomputing(explorer):
    """The bug, reproduced and pinned: switch away, switch back, no recompute."""
    entry_a = _entry("A")
    entry_b = _entry("B")
    explorer.workspace.add(entry_a)
    explorer.workspace.add(entry_b)
    explorer._sync_open_box()

    # Land on A and give it a computed result, the way _on_projection does.
    explorer._switch_to(entry_a.key)
    explorer._active_key = entry_a.key
    explorer.dataset = entry_a.dataset
    explorer.frame = entry_a.frame
    params = ProjectionParams(method=UMAP)
    result = _result(params)
    explorer._on_projection(result)
    assert explorer.result is result

    # Switch to B, which has never been projected.
    explorer._switch_to(entry_b.key)
    assert explorer.result is None, "B has no projection yet"

    # Switch back to A: must come back without recomputing.
    explorer._switch_to(entry_a.key)
    assert explorer.result is result
    np.testing.assert_array_equal(explorer.result.coords, result.coords)


def test_switching_back_shows_the_plot_not_a_stale_message(explorer):
    """Restoring a result must also make it visible, not hide it behind text."""
    entry_a = _entry("A")
    entry_b = _entry("B")
    explorer.workspace.add(entry_a)
    explorer.workspace.add(entry_b)
    explorer._sync_open_box()

    explorer._switch_to(entry_a.key)
    explorer._active_key = entry_a.key
    explorer.dataset = entry_a.dataset
    explorer.frame = entry_a.frame
    explorer._on_projection(_result(ProjectionParams(method=UMAP)))

    explorer._switch_to(entry_b.key)
    assert explorer.message.isVisibleTo(explorer), "B has nothing to show yet"
    assert not explorer.scatter.isVisibleTo(explorer)

    explorer._switch_to(entry_a.key)
    assert not explorer.message.isVisibleTo(explorer), "A's restored result must be shown"
    assert explorer.scatter.isVisibleTo(explorer)


def test_two_parameter_sets_on_one_entry_do_not_evict_each_other(explorer):
    """UMAP and t-SNE (or two parameter sets) can coexist on one entry."""
    entry = _entry("A")
    explorer.workspace.add(entry)
    explorer._active_key = entry.key
    explorer.dataset = entry.dataset
    explorer.frame = entry.frame

    umap_result = _result(ProjectionParams(method=UMAP))
    explorer._on_projection(umap_result)
    tsne_result = _result(ProjectionParams(method=TSNE, perplexity=10.0))
    explorer._on_projection(tsne_result)

    assert len(entry.projections) == 2
    assert entry.projections[ProjectionParams(method=UMAP).key()] is umap_result
    assert (
        entry.projections[ProjectionParams(method=TSNE, perplexity=10.0).key()]
        is tsne_result
    )
    # The most recently landed result is what "last shown" points at.
    assert entry.last_result_key == ProjectionParams(method=TSNE, perplexity=10.0).key()


def test_compute_reuses_in_memory_result_before_touching_disk_cache(explorer, monkeypatch):
    """The in-memory hit must short-circuit before the disk cache is even asked."""
    entry = _entry("A")
    explorer.workspace.add(entry)
    explorer._active_key = entry.key
    explorer.dataset = entry.dataset
    explorer.frame = entry.frame

    params = ProjectionParams(method=UMAP)
    result = _result(params)
    entry.projections[params.key()] = result

    def _fail(*_a, **_k):
        raise AssertionError("disk cache must not be consulted on an in-memory hit")

    monkeypatch.setattr(explorer.cache, "load", _fail)
    monkeypatch.setattr(explorer, "_params", lambda: params)

    explorer.compute()
    assert explorer.result is result


def test_a_cancelled_run_does_not_populate_the_entry_cache(explorer):
    """Cancel must not leave a half-trusted result sitting on the entry either."""
    entry = _entry("A")
    explorer.workspace.add(entry)
    explorer._active_key = entry.key
    explorer.dataset = entry.dataset
    explorer.frame = entry.frame

    explorer._cancelled = True
    explorer._on_projection(_result(ProjectionParams(method=UMAP)))

    assert entry.projections == {}
    assert entry.last_result_key is None
    assert explorer.result is None
