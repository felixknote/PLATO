"""The explorer's "Combine..." flow: turning several open embeddings into
one joint entry, and the edge cases around it.

The bug worth remembering here: setCurrentIndex(index) on a combobox is a
no-op, SIGNAL INCLUDED, when the box already shows that index -- and
_sync_open_box moves the box to a newly-added entry while signals are
blocked. So _combine_embeddings switching by calling setCurrentIndex alone
silently left self.frame pointing at the PREVIOUS entry; the fix is to call
_switch_to(key) directly rather than rely on the combobox's own signal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from plato.data.embeddings import EmbeddingDataset  # noqa: E402
from plato.data.projection import TSNE  # noqa: E402
from plato.data.workspace import DATASET_COLUMN, SOURCE_JOINT, EmbeddingEntry  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _entry(name: str, n: int, dims: int) -> EmbeddingEntry:
    frame = pd.DataFrame({"gene": [f"g{i % 3}" for i in range(n)]})
    dataset = EmbeddingDataset(
        name=name,
        directory=Path(f"/data/{name}"),
        vectors=np.random.default_rng(abs(hash(name)) % 2**31)
        .normal(size=(n, dims))
        .astype(np.float32),
        frame=frame,
        run_info={},
    )
    return EmbeddingEntry(name=name, dataset=dataset, frame=frame)


@pytest.fixture
def explorer(app, tmp_path, monkeypatch):
    from plato.views.explorer import EmbeddingExplorer

    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))

    class _Session:
        pass

    widget = EmbeddingExplorer(_Session(), tmp_path)
    widget.set_image_mode(False)
    return widget


def _accept_with(monkeypatch, entries):
    """Patch CombineEmbeddingsDialog to act as if the user checked `entries`."""
    from plato.views import combine_dialog as cd

    class _FakeDialog:
        DialogCode = cd.CombineEmbeddingsDialog.DialogCode

        def __init__(self, _entries, parent=None):
            pass

        def exec(self):
            return self.DialogCode.Accepted

        def selected_entries(self):
            return entries

    monkeypatch.setattr(cd, "CombineEmbeddingsDialog", _FakeDialog)


def _reject(monkeypatch):
    from plato.views import combine_dialog as cd

    class _FakeDialog:
        DialogCode = cd.CombineEmbeddingsDialog.DialogCode

        def __init__(self, _entries, parent=None):
            pass

        def exec(self):
            return self.DialogCode.Rejected

        def selected_entries(self):
            return []

    monkeypatch.setattr(cd, "CombineEmbeddingsDialog", _FakeDialog)


# -- the fix itself -----------------------------------------------------------


def test_combining_actually_switches_the_active_frame(explorer, monkeypatch):
    """The bug: the combobox signal did not fire, so self.frame stayed stale."""
    a, b = _entry("A", 20, 8), _entry("B", 30, 8)
    explorer._add_entry(a)
    explorer._add_entry(b)
    _accept_with(monkeypatch, [a, b])

    explorer._combine_embeddings()

    assert explorer.workspace.current.source == SOURCE_JOINT
    assert explorer._active_key == explorer.workspace.current.key
    assert len(explorer.frame) == 50, "frame must be the JOINT total, not one source"


def test_combining_selects_tsne_and_colour_by_dataset(explorer, monkeypatch):
    a, b = _entry("A", 10, 8), _entry("B", 10, 8)
    explorer._add_entry(a)
    explorer._add_entry(b)
    _accept_with(monkeypatch, [a, b])

    explorer._combine_embeddings()

    assert explorer.method_box.currentText() == TSNE
    assert explorer.colour_box.currentData() == DATASET_COLUMN


def test_dataset_colour_option_only_appears_for_a_joint_entry(explorer, monkeypatch):
    a, b = _entry("A", 10, 8), _entry("B", 10, 8)
    explorer._add_entry(a)
    options = [
        explorer.colour_box.itemData(i) for i in range(explorer.colour_box.count())
    ]
    assert DATASET_COLUMN not in options, "a lone entry has no Dataset column"

    explorer._add_entry(b)
    _accept_with(monkeypatch, [a, b])
    explorer._combine_embeddings()
    options = [
        explorer.colour_box.itemData(i) for i in range(explorer.colour_box.count())
    ]
    assert DATASET_COLUMN in options


# -- gating / visibility -------------------------------------------------------


def test_combine_button_hidden_with_fewer_than_two_entries(explorer):
    assert not explorer.combine_button.isVisibleTo(explorer)
    explorer._add_entry(_entry("A", 10, 8))
    assert not explorer.combine_button.isVisibleTo(explorer)
    explorer._add_entry(_entry("B", 10, 8))
    assert explorer.combine_button.isVisibleTo(explorer)


def test_combine_button_hidden_once_only_the_joint_entry_remains(explorer, monkeypatch):
    a, b = _entry("A", 10, 8), _entry("B", 10, 8)
    explorer._add_entry(a)
    explorer._add_entry(b)
    _accept_with(monkeypatch, [a, b])
    explorer._combine_embeddings()

    explorer._switch_to(a.key)
    explorer._close_current_embedding()
    explorer._switch_to(b.key)
    explorer._close_current_embedding()

    assert len(explorer.workspace) == 1
    assert not explorer.combine_button.isVisibleTo(explorer)


# -- rejection / no-op paths ---------------------------------------------------


def test_cancelling_the_dialog_adds_nothing(explorer, monkeypatch):
    explorer._add_entry(_entry("A", 10, 8))
    explorer._add_entry(_entry("B", 10, 8))
    _reject(monkeypatch)

    before = len(explorer.workspace)
    explorer._combine_embeddings()
    assert len(explorer.workspace) == before


def test_accepting_with_fewer_than_two_checked_adds_nothing(explorer, monkeypatch):
    a, b = _entry("A", 10, 8), _entry("B", 10, 8)
    explorer._add_entry(a)
    explorer._add_entry(b)
    _accept_with(monkeypatch, [])  # accepted, but nothing checked

    before = len(explorer.workspace)
    explorer._combine_embeddings()
    assert len(explorer.workspace) == before


def test_mismatched_dimensionality_warns_and_adds_nothing(explorer, monkeypatch):
    a, b = _entry("A", 10, 64), _entry("B", 10, 32)
    explorer._add_entry(a)
    explorer._add_entry(b)
    _accept_with(monkeypatch, [a, b])

    warned = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a))
    )

    before = len(explorer.workspace)
    explorer._combine_embeddings()
    assert len(explorer.workspace) == before
    assert warned, "the dimension mismatch must be explained, not silently dropped"


# -- a joint entry is a first-class, independent entry -------------------------


def test_a_joint_entry_cannot_be_combined_again(explorer, monkeypatch):
    """Combining an already-joint entry would double-count its source vectors."""
    a, b, c = _entry("A", 10, 8), _entry("B", 10, 8), _entry("C", 10, 8)
    explorer._add_entry(a)
    explorer._add_entry(b)
    _accept_with(monkeypatch, [a, b])
    explorer._combine_embeddings()
    explorer._add_entry(c)

    candidates = [e for e in explorer.workspace.entries if e.source != SOURCE_JOINT]
    names = {e.name for e in candidates}
    assert "A" in names and "B" in names and "C" in names
    assert not any(e.source == SOURCE_JOINT for e in candidates)


def test_joint_entry_survives_closing_one_of_its_source_entries(explorer, monkeypatch):
    """The joint entry owns its own concatenated vectors; closing a source
    entry it was BUILT FROM does not touch it."""
    a, b = _entry("A", 10, 8), _entry("B", 10, 8)
    explorer._add_entry(a)
    explorer._add_entry(b)
    _accept_with(monkeypatch, [a, b])
    explorer._combine_embeddings()
    joint = explorer.workspace.current

    explorer._switch_to(a.key)
    explorer._close_current_embedding()

    remaining = {e.key: e for e in explorer.workspace.entries}
    assert joint.key in remaining
    assert remaining[joint.key].n_points == 20


def test_three_way_combine(explorer, monkeypatch):
    a, b, c = _entry("A", 10, 8), _entry("B", 12, 8), _entry("C", 8, 8)
    explorer._add_entry(a)
    explorer._add_entry(b)
    explorer._add_entry(c)
    _accept_with(monkeypatch, [a, b, c])

    explorer._combine_embeddings()
    assert explorer.workspace.current.n_points == 30
