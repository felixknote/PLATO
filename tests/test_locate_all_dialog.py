"""Locate images for every open embedding in one popup.

The property that matters here is isolation: choosing a folder for one
embedding's row must never touch another entry's resolver, and cancelling
the whole dialog must leave every entry exactly as it was.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from plato.data.embeddings import EmbeddingDataset  # noqa: E402
from plato.data.workspace import EmbeddingEntry  # noqa: E402
from plato.views.locate_all_dialog import LocateAllDialog, _EntryRow  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@dataclass
class _FakeReport:
    n_files: int
    ok: bool
    fraction: float = 1.0

    def describe(self) -> str:
        return "no match"


@dataclass
class _FakeResolver:
    root: Path
    report: _FakeReport
    roots: list = None

    def __post_init__(self):
        if self.roots is None:
            self.roots = [self.root]


def _entry(name: str, n: int = 5) -> EmbeddingEntry:
    frame = pd.DataFrame({"image_name": [f"img_{i}" for i in range(n)]})
    dataset = EmbeddingDataset(
        name=name,
        directory=Path(f"/data/{name}"),
        vectors=np.zeros((n, 4), dtype=np.float32),
        frame=frame,
        run_info={},
    )
    return EmbeddingEntry(name=name, dataset=dataset, frame=frame)


def test_row_starts_unresolved_when_entry_has_no_resolver(app):
    row = _EntryRow(_entry("A"))
    assert row.status_label.text() == "Not located yet."
    assert row.pending_resolver is None


def test_row_shows_existing_resolver_on_open(app):
    """Reopening the dialog must show what is already resolved, not blank it."""
    entry = _entry("A")
    entry.resolver = _FakeResolver(
        root=Path(r"E:\Data\mydata"), report=_FakeReport(n_files=24192, ok=True)
    )
    row = _EntryRow(entry)
    assert row.pending_resolver is entry.resolver
    text = row.status_label.text()
    assert "24,192" in text
    assert "mydata" in text


def test_dialog_builds_one_row_per_entry(app):
    entries = [_entry("A"), _entry("B"), _entry("C")]
    dialog = LocateAllDialog(entries)
    assert len(dialog._rows) == 3
    assert set(dialog._rows) == {e.key for e in entries}


def test_resolving_one_row_does_not_affect_another(app):
    """The core requirement: independence between rows."""
    entries = [_entry("A"), _entry("B")]
    dialog = LocateAllDialog(entries)
    rows = list(dialog._rows.values())

    rows[0].set_result(
        _FakeResolver(root=Path(r"X:\rootA"), report=_FakeReport(n_files=100, ok=True))
    )
    assert rows[0].pending_resolver.root == Path(r"X:\rootA")
    assert rows[1].pending_resolver is None, "resolving row 0 must not touch row 1"

    rows[1].set_result(
        _FakeResolver(root=Path(r"Y:\rootB"), report=_FakeReport(n_files=200, ok=True))
    )
    assert rows[0].pending_resolver.root == Path(r"X:\rootA"), "row 0 must be unchanged"
    assert rows[1].pending_resolver.root == Path(r"Y:\rootB")


def test_results_maps_each_key_to_its_own_resolver(app):
    entries = [_entry("A"), _entry("B")]
    dialog = LocateAllDialog(entries)
    rows = list(dialog._rows.values())
    rows[0].set_result(
        _FakeResolver(root=Path(r"X:\rootA"), report=_FakeReport(n_files=1, ok=True))
    )
    rows[1].set_result(
        _FakeResolver(root=Path(r"Y:\rootB"), report=_FakeReport(n_files=1, ok=True))
    )

    results = dialog.results()
    assert len(results) == 2
    assert results[entries[0].key].root == Path(r"X:\rootA")
    assert results[entries[1].key].root == Path(r"Y:\rootB")


def test_a_failed_match_does_not_populate_pending_resolver(app):
    """A folder that resolves nothing must not become the accepted answer."""
    entry = _entry("A")
    dialog = LocateAllDialog([entry])
    row = list(dialog._rows.values())[0]
    row.set_result(
        _FakeResolver(root=Path(r"X:\wrong"), report=_FakeReport(n_files=0, ok=False))
    )
    assert row.pending_resolver is None
    assert dialog.results() == {}


def test_cancelling_the_dialog_leaves_entries_untouched(app):
    """Entries are only mutated by the caller after exec() == Accepted."""
    entry = _entry("A")
    assert entry.resolver is None
    dialog = LocateAllDialog([entry])
    row = list(dialog._rows.values())[0]
    row.set_result(
        _FakeResolver(root=Path(r"X:\rootA"), report=_FakeReport(n_files=1, ok=True))
    )
    # The row has a pending choice, but nothing has been applied back onto
    # the entry -- that only happens in explorer._browse_for_source_data
    # after the dialog is accepted.
    assert entry.resolver is None


# -- images split across more than one folder --------------------------------


def test_add_folder_button_hidden_until_a_folder_is_chosen(app):
    # isVisible() reflects the whole ancestor chain, which is always False
    # for a row that is never actually shown on screen; isVisibleTo(row)
    # checks the widget's own flag relative to its parent instead.
    row = _EntryRow(_entry("A"))
    assert not row.add_folder_button.isVisibleTo(row)

    row.set_result(
        _FakeResolver(root=Path(r"X:\rootA"), report=_FakeReport(n_files=1, ok=True))
    )
    assert row.add_folder_button.isVisibleTo(row)


def test_add_folder_requested_carries_the_row_key(app):
    """A standalone row, NOT one built by LocateAllDialog: the dialog already
    connects every row's add_folder_requested to its own _add_folder_for,
    which opens a real (blocking) QFileDialog -- adding a second listener
    onto a dialog-owned row fires that real dialog too and hangs headlessly.
    A bare _EntryRow has no such connection, so this checks the row's own
    wiring in isolation instead.
    """
    row = _EntryRow(_entry("A"))
    row.set_result(
        _FakeResolver(root=Path(r"X:\rootA"), report=_FakeReport(n_files=1, ok=True))
    )

    seen = []
    row.add_folder_requested.connect(seen.append)
    row.add_folder_button.clicked.emit()
    assert seen == [row.key]


def test_scan_with_a_base_resolver_passes_it_through_to_the_task(app):
    """"Add another folder..." must merge, so _scan's base must reach the task.

    The merge logic itself (ImageResolver.combined_with) has direct coverage
    in test_explorer.py; this only checks the dialog wires the base resolver
    through to _ScanTask rather than silently discarding it and scanning the
    new folder as a plain replacement.
    """
    import plato.views.locate_all_dialog as mod

    entry = _entry("A")
    dialog = LocateAllDialog([entry])
    base = _FakeResolver(root=Path(r"X:\first"), report=_FakeReport(n_files=1, ok=True))

    captured = {}

    class _RecordingTask:
        def __init__(self, key, root, frame, signals, *, base=None):
            captured["root"] = root
            captured["base"] = base

        def run(self):
            pass

    original = mod._ScanTask
    mod._ScanTask = _RecordingTask
    dialog._pool = type("P", (), {"start": staticmethod(lambda task: task.run())})()
    try:
        dialog._scan(entry.key, r"X:\second", base=base)
    finally:
        mod._ScanTask = original

    assert captured["base"] is base
    assert captured["root"] == Path(r"X:\second")
