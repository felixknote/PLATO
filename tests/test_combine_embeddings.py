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


def _accept_with(monkeypatch, entries, align="none"):
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

        def selected_align(self):
            return align

        def selected_aligns(self):
            return [align]

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

        def selected_align(self):
            from plato.data.joint_projection import ALIGN_NONE

            return ALIGN_NONE

        def selected_aligns(self):
            from plato.data.joint_projection import ALIGN_NONE

            return [ALIGN_NONE]

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
    # isHidden(), not isVisibleTo(explorer): the button lives inside a
    # collapsible Data section that starts closed, and isVisibleTo composes
    # through that closed ancestor regardless of what this widget's own
    # setVisible(...) decided. isHidden() reports only this widget's flag.
    assert explorer.combine_button.isHidden()
    explorer._add_entry(_entry("A", 10, 8))
    assert explorer.combine_button.isHidden()
    explorer._add_entry(_entry("B", 10, 8))
    assert not explorer.combine_button.isHidden()


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
    assert explorer.combine_button.isHidden()


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


# -- alignment reaching the explorer -------------------------------------------


def test_the_dialogs_alignment_choice_reaches_the_joint_entry(explorer, monkeypatch):
    """The dialog decides; the entry must actually be built that way.

    Alignment that is offered but silently dropped is worse than not offering
    it: the user believes the offset was removed and reads the layout
    accordingly.
    """
    from plato.data.joint_projection import ALIGN_CENTRE

    a, b = _entry("A", 20, 8), _entry("B", 20, 8)
    explorer._add_entry(a)
    explorer._add_entry(b)
    _accept_with(monkeypatch, [a, b], align=ALIGN_CENTRE)
    explorer._combine_embeddings()

    joint = explorer.workspace.current
    assert joint.info["align"] == ALIGN_CENTRE
    assert "centred" in joint.name


def test_an_unaligned_combine_is_unchanged(explorer, monkeypatch):
    """The default path must stay exactly what it was."""
    from plato.data.joint_projection import ALIGN_NONE

    a, b = _entry("A", 20, 8), _entry("B", 20, 8)
    explorer._add_entry(a)
    explorer._add_entry(b)
    _accept_with(monkeypatch, [a, b])
    explorer._combine_embeddings()

    joint = explorer.workspace.current
    assert joint.info["align"] == ALIGN_NONE
    assert "centred" not in joint.name


# -- working with several datasets at once -------------------------------------


def test_the_compare_option_builds_both_fits_at_once(explorer, monkeypatch):
    """Comparing aligned against raw used to mean combining twice by hand.

    The real use of centring is not "which is right" but "what survives it",
    which needs both entries present to flick between.
    """
    from plato.data.joint_projection import ALIGN_CENTRE, ALIGN_NONE
    from plato.views import combine_dialog as cd

    a, b = _entry("A", 20, 8), _entry("B", 20, 8)
    explorer._add_entry(a)
    explorer._add_entry(b)

    class _Pair:
        DialogCode = cd.CombineEmbeddingsDialog.DialogCode

        def __init__(self, _entries, parent=None):
            pass

        def exec(self):
            return self.DialogCode.Accepted

        def selected_entries(self):
            return [a, b]

        def selected_align(self):
            return ALIGN_NONE

        def selected_aligns(self):
            return [ALIGN_NONE, ALIGN_CENTRE]

    monkeypatch.setattr(cd, "CombineEmbeddingsDialog", _Pair)
    explorer._combine_embeddings()

    joints = [e for e in explorer.workspace.entries if e.source == SOURCE_JOINT]
    assert len(joints) == 2
    assert {e.info["align"] for e in joints} == {ALIGN_NONE, ALIGN_CENTRE}
    # The unaligned one is what you land on: it is the normal view.
    assert explorer.workspace.current.info["align"] == ALIGN_NONE


def test_cycling_moves_between_open_embeddings(explorer):
    """Ctrl+Tab. Holding one layout in your eye while flicking to the other
    is how you see what moved; clicking a row each way breaks the rhythm."""
    a, b = _entry("A", 20, 8), _entry("B", 20, 8)
    explorer._add_entry(a)
    explorer._add_entry(b)
    first = explorer._active_key

    explorer._cycle_embedding(1)
    assert explorer._active_key != first
    # Wraps, so two entries are a toggle.
    explorer._cycle_embedding(1)
    assert explorer._active_key == first


def test_cycling_backwards_is_the_inverse(explorer):
    a, b, c = _entry("A", 20, 8), _entry("B", 20, 8), _entry("C", 20, 8)
    for entry in (a, b, c):
        explorer._add_entry(entry)
    start = explorer._active_key
    explorer._cycle_embedding(1)
    explorer._cycle_embedding(-1)
    assert explorer._active_key == start


def test_cycling_with_one_embedding_does_nothing(explorer):
    """Not an error, and not a switch to itself that would clear state."""
    explorer._add_entry(_entry("A", 20, 8))
    only = explorer._active_key
    explorer._cycle_embedding(1)
    assert explorer._active_key == only


def test_a_joint_entry_describes_what_went_into_it(explorer, monkeypatch):
    """An entry reopened later must say its sources, their sizes and whether
    it was aligned -- otherwise two joint entries are indistinguishable."""
    from plato.data.joint_projection import ALIGN_CENTRE

    a, b = _entry("CRISPRi", 120, 8), _entry("ABx", 90, 8)
    explorer._add_entry(a)
    explorer._add_entry(b)
    _accept_with(monkeypatch, [a, b], align=ALIGN_CENTRE)
    explorer._combine_embeddings()

    text = explorer.workspace.current.describe()
    assert "CRISPRi (120)" in text
    assert "ABx (90)" in text
    assert "centred" in text


def test_an_unaligned_joint_entry_does_not_claim_alignment(explorer, monkeypatch):
    a, b = _entry("A", 20, 8), _entry("B", 20, 8)
    explorer._add_entry(a)
    explorer._add_entry(b)
    _accept_with(monkeypatch, [a, b])
    explorer._combine_embeddings()
    assert "centred" not in explorer.workspace.current.describe()
