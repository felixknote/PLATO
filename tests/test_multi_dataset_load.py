"""Loading several embedding exports at once, off the GUI thread.

_browse_for_dataset used to call load_dataset + build_frame synchronously,
once per selected directory, on the GUI thread -- measured at 19.4 s
combined for six real DINO exports, almost all of it numpy unzipping the
.npz, with the window frozen and no progress shown for the whole batch.
_LoadTask moves that work to a QThreadPool so several loads overlap and the
window stays responsive; these tests exercise the task itself and the
GUI-thread orchestration that applies results as they arrive, in whatever
order the disk hands them back.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QThreadPool  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from plato.views.explorer import EmbeddingExplorer, _LoadSignals, _LoadTask  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _write_dataset(directory: Path, *, n: int = 30, dim: int = 8, valid: bool = True) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(directory.name)) % 2**31)
    rows = [
        {"plate": "P1", "well": f"A{i:02d}", "label": f"g{i % 3}", "image_name": f"img_{i}"}
        for i in range(n)
    ]
    pd.DataFrame(rows).to_csv(directory / "features_metadata.csv", index=False)
    if valid:
        np.savez_compressed(
            directory / "features_all.npz",
            embeddings=rng.normal(size=(n, dim)).astype(np.float32),
        )
    (directory / "metadata.json").write_text(json.dumps({"model": "test"}))
    return directory


@pytest.fixture
def explorer(app, tmp_path, monkeypatch):
    class _Session:
        pass

    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    return EmbeddingExplorer(_Session(), tmp_path)


def _run_task_inline(task: _LoadTask) -> None:
    """Run a _LoadTask synchronously, bypassing the thread pool -- deterministic
    for a test, since ordering across real threads is not something to assert on."""
    task.run()


# -- _LoadTask itself ----------------------------------------------------------


def test_load_task_emits_loaded_for_a_real_export(app, tmp_path):
    directory = _write_dataset(tmp_path / "A", n=30, dim=8)
    signals = _LoadSignals()
    results = []
    signals.loaded.connect(lambda d, ds, fr: results.append((d, ds, fr)))
    signals.failed.connect(lambda d, msg: pytest.fail(f"unexpected failure: {msg}"))

    _run_task_inline(_LoadTask(directory, None, None, signals))

    assert len(results) == 1
    got_dir, dataset, frame = results[0]
    assert got_dir == directory
    assert dataset.n_points == 30
    assert dataset.n_dimensions == 8
    assert len(frame) == 30


def test_load_task_emits_failed_for_missing_vectors(app, tmp_path):
    directory = _write_dataset(tmp_path / "B", valid=False)
    signals = _LoadSignals()
    failures = []
    signals.loaded.connect(lambda *a: pytest.fail("should not have loaded"))
    signals.failed.connect(lambda d, msg: failures.append((d, msg)))

    _run_task_inline(_LoadTask(directory, None, None, signals))

    assert len(failures) == 1
    assert failures[0][0] == directory
    assert "features_all.npz" in failures[0][1]


def test_load_task_emits_failed_for_a_nonexistent_directory(app, tmp_path):
    directory = tmp_path / "does_not_exist"
    signals = _LoadSignals()
    failures = []
    signals.loaded.connect(lambda *a: pytest.fail("should not have loaded"))
    signals.failed.connect(lambda d, msg: failures.append((d, msg)))

    _run_task_inline(_LoadTask(directory, None, None, signals))

    assert len(failures) == 1


# -- the real background path, through QThreadPool -----------------------------


def _wait_for(predicate, *, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        QApplication.processEvents()
        QThreadPool.globalInstance().waitForDone(20)
    raise AssertionError("timed out waiting for background loads to settle")


def test_multiple_datasets_load_concurrently_and_all_become_entries(app, explorer, tmp_path):
    """The real path: several _LoadTasks on the shared thread pool, results
    applied to the workspace as each arrives -- not all-or-nothing at the end."""
    dirs = [_write_dataset(tmp_path / name, n=20, dim=6) for name in ("A", "B", "C")]

    explorer._load_batch_plates = []
    explorer._load_batch_loaded = []
    explorer._load_batch_failed = []
    explorer._load_batch_pending = len(dirs)
    explorer._load_batch_last_directory = dirs[-1]

    signals = _LoadSignals()
    signals.loaded.connect(explorer._on_load_task_finished)
    signals.failed.connect(explorer._on_load_task_failed)
    explorer._load_signals = signals
    pool = QThreadPool.globalInstance()
    for directory in dirs:
        pool.start(_LoadTask(directory, explorer.moa_table, explorer.pathway_table, signals))

    _wait_for(lambda: explorer._load_batch_pending <= 0)

    assert len(explorer._load_batch_loaded) == 3
    assert explorer._load_batch_failed == []
    loaded_names = {d.name for d in explorer._load_batch_loaded}
    assert loaded_names == {"A", "B", "C"}
    # Every one of them actually joined the workspace, not just the batch list.
    assert {e.name for e in explorer.workspace.entries} == {"A", "B", "C"}


def test_a_failed_load_does_not_block_the_others_from_completing(app, explorer, tmp_path):
    good_a = _write_dataset(tmp_path / "Good1", n=15, dim=4)
    bad = _write_dataset(tmp_path / "Bad", valid=False)
    good_b = _write_dataset(tmp_path / "Good2", n=15, dim=4)
    dirs = [good_a, bad, good_b]

    explorer._load_batch_plates = []
    explorer._load_batch_loaded = []
    explorer._load_batch_failed = []
    explorer._load_batch_pending = len(dirs)
    explorer._load_batch_last_directory = dirs[-1]

    signals = _LoadSignals()
    signals.loaded.connect(explorer._on_load_task_finished)
    signals.failed.connect(explorer._on_load_task_failed)
    explorer._load_signals = signals
    pool = QThreadPool.globalInstance()
    for directory in dirs:
        pool.start(_LoadTask(directory, explorer.moa_table, explorer.pathway_table, signals))

    _wait_for(lambda: explorer._load_batch_pending <= 0)

    assert len(explorer._load_batch_loaded) == 2
    assert explorer._load_batch_failed == ["Bad"]
    assert {e.name for e in explorer.workspace.entries} == {"Good1", "Good2"}


def test_finish_lands_on_the_last_submitted_directory_not_last_finished(app, explorer, tmp_path):
    """Background tasks can complete in any order -- "land on the last one
    loaded" has to mean the order the user picked them in (submission
    order), not whichever happened to finish first on the thread pool."""
    dirs = [_write_dataset(tmp_path / name, n=10, dim=4) for name in ("First", "Second", "Third")]

    explorer._load_batch_plates = []
    explorer._load_batch_loaded = []
    explorer._load_batch_failed = []
    explorer._load_batch_pending = len(dirs)
    explorer._load_batch_last_directory = dirs[-1]  # "Third"

    signals = _LoadSignals()
    signals.loaded.connect(explorer._on_load_task_finished)
    signals.failed.connect(explorer._on_load_task_failed)
    explorer._load_signals = signals

    # Simulate results arriving OUT of submission order: Third finishes
    # first, then First, then Second.
    for directory in [dirs[2], dirs[0], dirs[1]]:
        task = _LoadTask(directory, explorer.moa_table, explorer.pathway_table, signals)
        task.run()
        QApplication.processEvents()

    explorer._finish_dataset_load(
        explorer._load_batch_loaded, explorer._load_batch_failed, explorer._load_batch_plates
    )

    active = explorer.workspace.current
    assert active is not None
    assert active.name == "Third"
