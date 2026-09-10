"""The open-embeddings list: every loaded embedding visible at once, with a
row-level close button and a click-to-activate row, replacing the old single
dropdown-plus-close-button pair.

The bug worth remembering here: Workspace.add() makes a newly added entry
current IN THE WORKSPACE, but self._active_key is the explorer's own separate
mirror of that and does not follow automatically. _add_entry only called
_sync_open_box() (which correctly moves the VISUAL highlight to the new
entry), so self.frame/self.dataset/self._active_key silently kept pointing at
whatever was active before -- loading a third embedding while viewing a
different one left the explorer's own idea of "active" one entry behind the
workspace's. The visible symptom: closing a row that LOOKED inactive in the
list actually closed the one that was truly active underneath.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from plato.data.embeddings import EmbeddingDataset  # noqa: E402
from plato.data.workspace import EmbeddingEntry  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _entry(name: str, n: int = 10, dims: int = 8) -> EmbeddingEntry:
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
def explorer(app, tmp_path):
    from plato.views.explorer import EmbeddingExplorer

    class _Session:
        pass

    widget = EmbeddingExplorer(_Session(), tmp_path)
    widget.set_image_mode(False)
    return widget


# -- visibility ----------------------------------------------------------------


def test_list_hidden_with_zero_or_one_entry(explorer):
    assert not explorer.open_list.isVisibleTo(explorer)
    explorer._add_entry(_entry("A"))
    assert not explorer.open_list.isVisibleTo(explorer)


def test_list_visible_with_two_or_more(explorer):
    explorer._add_entry(_entry("A"))
    explorer._add_entry(_entry("B"))
    assert explorer.open_list.isVisibleTo(explorer)
    assert len(explorer.open_list._rows) == 2


# -- the bug: _active_key must track a THIRD entry added while viewing another --


def test_adding_an_entry_while_viewing_a_different_one_still_makes_it_active(explorer):
    """The bug, reproduced directly: load A, B, switch to A, then load C.

    C becomes current in the Workspace immediately on add. Before the fix,
    self._active_key stayed on A (the one being viewed) instead of following
    to C, even though the on-screen list correctly highlighted C.
    """
    a, b = _entry("A"), _entry("B")
    explorer._add_entry(a)
    explorer._add_entry(b)
    explorer._switch_to(a.key)
    assert explorer._active_key == a.key

    c = _entry("C")
    explorer._add_entry(c)

    assert explorer.workspace.current_key == c.key
    assert explorer._active_key == c.key, "the explorer's mirror must follow"
    assert len(explorer.frame) == c.n_points


def test_the_highlighted_row_matches_active_key_after_adding(explorer):
    a, b = _entry("A"), _entry("B")
    explorer._add_entry(a)
    explorer._add_entry(b)
    explorer._switch_to(a.key)

    c = _entry("C")
    explorer._add_entry(c)

    assert explorer.open_list._rows[c.key]._active
    assert not explorer.open_list._rows[a.key]._active
    assert not explorer.open_list._rows[b.key]._active


# -- row-level close: closing a row that is NOT active must not disturb the view


def test_closing_a_non_active_row_leaves_the_active_view_untouched(explorer):
    a, b, c = _entry("A"), _entry("B"), _entry("C")
    explorer._add_entry(a)
    explorer._add_entry(b)
    explorer._add_entry(c)
    explorer._switch_to(b.key)

    active_before = explorer._active_key
    frame_len_before = len(explorer.frame)
    assert active_before == b.key

    explorer.open_list.close_requested.emit(a.key)

    assert explorer._active_key == active_before
    assert len(explorer.frame) == frame_len_before
    assert a.key not in explorer.open_list._rows
    assert b.key in explorer.open_list._rows and c.key in explorer.open_list._rows


def test_closing_the_active_row_switches_to_another(explorer):
    a, b = _entry("A"), _entry("B")
    explorer._add_entry(a)
    explorer._add_entry(b)
    explorer._switch_to(a.key)

    explorer.open_list.close_requested.emit(a.key)

    assert explorer._active_key == b.key
    assert len(explorer.workspace) == 1


def test_cannot_close_the_last_remaining_entry(explorer):
    a = _entry("A")
    explorer._add_entry(a)
    explorer.open_list.close_requested.emit(a.key)
    assert len(explorer.workspace) == 1


# -- clicking a row switches -----------------------------------------------------


def test_clicking_a_row_activates_it(explorer):
    a, b = _entry("A"), _entry("B")
    explorer._add_entry(a)
    explorer._add_entry(b)
    explorer._switch_to(a.key)

    explorer.open_list.activated.emit(b.key)

    assert explorer._active_key == b.key
    assert len(explorer.frame) == b.n_points
