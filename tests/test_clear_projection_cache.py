"""EmbeddingExplorer.clear_projection_cache: the Help-menu action that empties
the on-disk t-SNE/UMAP cache and every open embedding's in-memory results.

One ProjectionCache backs the whole workspace (see EmbeddingExplorer.__init__:
self.cache = ProjectionCache(self.work_dir / "projections")), so this is a
single whole-cache operation reached from Help, not a per-embedding one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from plato.data.embeddings import EmbeddingDataset  # noqa: E402
from plato.data.projection import (  # noqa: E402
    TSNE,
    ProjectionCache,
    ProjectionParams,
    project,
)
from plato.data.workspace import EmbeddingEntry, Workspace  # noqa: E402
from plato.views.explorer import EmbeddingExplorer  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _make_explorer(app, tmp_path) -> EmbeddingExplorer:
    """Just enough state for clear_projection_cache to run for real, without
    constructing the whole widget (which needs a live Session)."""
    from PySide6.QtWidgets import QLabel

    explorer = EmbeddingExplorer.__new__(EmbeddingExplorer)
    # Qt signals (status = Signal(str)) only work once QObject.__init__ has
    # run; __new__ alone skips it, which is fine for plain-Python attributes
    # but not for .connect().
    from PySide6.QtWidgets import QWidget

    QWidget.__init__(explorer)
    from plato.views.grid_view import GridView
    from plato.views.scatter import EmbeddingScatter

    explorer.cache = ProjectionCache(tmp_path / "projections")
    explorer.workspace = Workspace()
    explorer.result = None
    explorer._group_column = None
    explorer.message = QLabel()
    explorer.scatter = EmbeddingScatter()
    explorer.grid = GridView()
    return explorer


def _entry_with_projection(cache: ProjectionCache, name: str, tmp_path) -> EmbeddingEntry:
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(80, 8)).astype(np.float32)
    frame = pd.DataFrame(index=range(80))
    dataset = EmbeddingDataset(
        name=name, directory=tmp_path / name, vectors=vectors, frame=frame, run_info={}
    )
    entry = EmbeddingEntry(name=name, dataset=dataset, frame=frame)
    params = ProjectionParams(method=TSNE, pca_components=4)
    result = project(vectors, params, fingerprint=entry.fingerprint(), cache=cache)
    entry.projections[result.params.key()] = result
    entry.last_result_key = result.params.key()
    return entry


def test_clears_disk_and_in_memory_and_message_on_confirm(app, tmp_path, monkeypatch):
    explorer = _make_explorer(app, tmp_path)
    entry = _entry_with_projection(explorer.cache, "demo", tmp_path)
    explorer.workspace.add(entry)
    explorer.result = entry.projections[entry.last_result_key]

    assert len(explorer.cache.entries()) == 1
    assert len(entry.projections) == 1

    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    statuses = []
    explorer.status.connect(statuses.append)

    explorer.clear_projection_cache()

    assert explorer.cache.entries() == []
    assert entry.projections == {}
    assert entry.last_result_key is None
    assert explorer.result is None
    assert "1" in statuses[-1]


def test_declining_the_confirmation_changes_nothing(app, tmp_path, monkeypatch):
    explorer = _make_explorer(app, tmp_path)
    entry = _entry_with_projection(explorer.cache, "demo", tmp_path)
    explorer.workspace.add(entry)
    explorer.result = entry.projections[entry.last_result_key]

    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No
    )

    explorer.clear_projection_cache()

    assert len(explorer.cache.entries()) == 1
    assert len(entry.projections) == 1
    assert explorer.result is not None


def test_nothing_to_clear_informs_rather_than_asks(app, tmp_path, monkeypatch):
    explorer = _make_explorer(app, tmp_path)

    asked = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: asked.append(True) or QMessageBox.StandardButton.Yes,
    )
    informed = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *a, **k: informed.append(True),
    )

    explorer.clear_projection_cache()

    assert not asked, "an empty cache has nothing to confirm deleting"
    assert informed
