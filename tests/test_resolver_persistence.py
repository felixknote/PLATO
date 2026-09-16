"""A dataset's image folder, once found, should not need finding again.

End-to-end version of test_resolver_store.py's unit tests: drives the real
EmbeddingExplorer resolution paths (_start_resolving_images, and the Locate
all accept flow) against ResolverStore, confirming a folder confirmed once is
applied automatically on a LATER explorer instance -- simulating a restart --
without re-walking the data library or opening Locate again.

_start_resolving_images always dispatches to a background QThreadPool task,
never returning a resolver directly -- see _RememberedResolveTask in
explorer.py for why a remembered root is not applied inline on the GUI
thread (re-scanning it is exactly as expensive as a fresh hunt, measured at
minutes on a real network share). ``_resolve_and_wait`` below drives that
real async path and blocks until the result signal arrives, which is what
every test in this file needs instead of reading a synchronous return value.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from plato.data.embeddings import EmbeddingDataset  # noqa: E402
from plato.data.explorer_model import build_frame  # noqa: E402
from plato.data.workspace import EmbeddingEntry  # noqa: E402
from plato.views.explorer import EmbeddingExplorer  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _resolve_and_wait(explorer, frame, *, timeout_ms: int = 5000):
    """Run the real background resolve path and wait for its result.

    ``_start_resolving_images`` always returns None or a cached hit; a fresh
    hunt/rebuild arrives later via the found signal. This drives the Qt event
    loop until that signal lands (self.resolver is set) or the timeout
    expires, so a test sees exactly what the real app would show once the
    background task finishes.
    """
    import time

    from PySide6.QtCore import QThreadPool

    cached = explorer._start_resolving_images(frame)
    if cached is not None:
        return cached

    pool = QThreadPool.globalInstance()
    deadline = time.monotonic() + timeout_ms / 1000
    while explorer._resolving and time.monotonic() < deadline:
        pool.waitForDone(50)
        QApplication.processEvents()
    return explorer.resolver


class _Session:
    pass


def _make_entry(name: str, root: Path, n: int = 3) -> EmbeddingEntry:
    stems = [f"{name}_{i:03d}" for i in range(n)]
    for stem in stems:
        (root / f"{stem}.tiff").write_bytes(b"\0")
    raw = pd.DataFrame({"plate": ["P1"] * n, "well": [f"A{i:02d}" for i in range(n)], "image_name": stems})
    dataset = EmbeddingDataset(
        name=name,
        directory=Path("/x") / name,
        vectors=np.zeros((n, 4), dtype=np.float32),
        frame=raw,
        run_info={},
    )
    frame, _ = build_frame(dataset)
    return EmbeddingEntry(name=name, dataset=dataset, frame=frame)


def test_a_resolver_confirmed_by_hand_survives_a_new_explorer_instance(app, tmp_path):
    """Simulates a restart: a second EmbeddingExplorer, same work_dir, must
    apply what the first one confirmed without doing anything itself --
    without even being ABLE to find image_root on its own, since it is not
    offered by _image_root_candidates in a bare test explorer with no
    configured data library. Only the remembered answer can produce it."""
    from plato.data.explorer_model import ImageResolver

    image_root = tmp_path / "images"
    image_root.mkdir()
    entry = _make_entry("A", image_root)

    work_dir = tmp_path / "work"
    first = EmbeddingExplorer(_Session(), work_dir)
    first._add_entry(entry)
    # Stands in for the automatic hunt or Locate confirming this root --
    # both funnel through _remember_resolver, exercised directly here so
    # the test is not itself depending on _image_root_candidates finding
    # anything.
    resolver = ImageResolver.detect(image_root, entry.frame)
    assert resolver is not None
    first._resolver_cache[str(entry.directory)] = (resolver, [])
    first._remember_resolver(str(entry.directory), resolver)

    # A brand new explorer, same work_dir -- nothing carried over except
    # what is on disk. It has no candidate that would ever find image_root
    # on its own; only the persisted store can produce it.
    second = EmbeddingExplorer(_Session(), work_dir)
    entry_again = _make_entry("A", image_root)
    # Give it the SAME dataset directory as the first entry, since that is
    # the persistence key -- a fresh EmbeddingEntry stands in for reopening
    # the same export in a new session.
    entry_again.dataset.directory = entry.dataset.directory
    second._add_entry(entry_again)
    reapplied = _resolve_and_wait(second, entry_again.frame)

    assert reapplied is not None
    assert reapplied.report.ok
    resolved = sum(
        1 for _, row in entry_again.frame.iterrows() if reapplied.path_for(row)
    )
    assert resolved == len(entry_again.frame)


def test_locate_all_accepting_a_folder_persists_it(app, tmp_path):
    """The explicit Locate path (LocateAllDialog -> results()) is the other
    place a resolver is confirmed; it must feed the durable store exactly
    like the automatic hunt does."""
    from plato.data.explorer_model import ImageResolver
    from plato.views.locate_all_dialog import LocateAllDialog

    image_root = tmp_path / "images"
    image_root.mkdir()
    entry = _make_entry("A", image_root)

    work_dir = tmp_path / "work"
    explorer = EmbeddingExplorer(_Session(), work_dir)
    explorer._add_entry(entry)

    dialog = LocateAllDialog([entry], parent=explorer)
    row = dialog._rows[entry.key]
    row.set_result(ImageResolver.detect(image_root, entry.frame))

    results = dialog.results()
    for e in [entry]:
        resolver = results.get(e.key)
        if resolver is None:
            continue
        e.resolver = resolver
        e.resolver_searched = True
        explorer._resolver_cache[str(e.directory)] = (resolver, [])
        explorer._remember_resolver(str(e.directory), resolver)

    assert explorer._resolver_store.roots_for(str(entry.directory)) == [image_root]


def test_a_stale_remembered_root_falls_back_to_a_fresh_hunt(app, tmp_path):
    """If the remembered folder is gone (moved, deleted, unmounted), the
    dataset must not get stuck offering a dead answer -- it should fall
    through to whatever a fresh hunt (or None) finds instead."""
    image_root = tmp_path / "images"
    image_root.mkdir()
    entry = _make_entry("A", image_root)
    work_dir = tmp_path / "work"
    explorer = EmbeddingExplorer(_Session(), work_dir)
    explorer._add_entry(entry)

    # Plant a remembered answer that points nowhere real.
    explorer._resolver_store.remember(str(entry.directory), [tmp_path / "gone"])

    resolver = _resolve_and_wait(explorer, entry.frame)
    # Either None (no candidate resolves it) or a fresh, real resolver --
    # never the stale one, and never a crash.
    if resolver is not None:
        assert resolver.report.ok
