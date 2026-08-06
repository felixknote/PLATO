"""Offscreen smoke test. Run with QT_QPA_PLATFORM=offscreen.

This does not test that the app *looks* right — it tests that every widget
constructs, the model fills, thumbnails arrive from the worker pool, and the
annotation round-trip reaches SQLite. It also exercises the empty-state /
Load Data / Add Plate startup workflow via Session directly (headless, so the
native file pickers are bypassed and Session.add_plate is called with an
already-built Config, same as LoadPlateDialog would produce on accept).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PySide6.QtWidgets import QApplication

from plato.config import load_config
from plato.gui.main_window import MainWindow
from plato.gui.session import Session
from plato.index.db import IndexDB


def main(config_path: str) -> int:
    cfg = load_config(config_path)
    app = QApplication.instance() or QApplication(sys.argv)

    # -- empty-state startup ------------------------------------------------
    empty_session = Session()
    empty_window = MainWindow(empty_session)
    assert empty_window.primary is None, "empty session should show no browser panel"
    empty_window.close()
    print("empty state: no browser panel until data is loaded")

    session = Session()
    session.add_plate(cfg)
    window = MainWindow(session)
    window.show()

    panel = window.primary
    assert panel.model.rowCount() > 0, "model is empty"
    print(f"rows in grid: {panel.model.rowCount()}")

    # Let the thumbnail worker pool deliver.
    deadline = time.time() + 10
    while time.time() < deadline and not panel.model._pixmaps:
        panel.model.data(panel.model.index(0, 0), 6)  # DecorationRole
        app.processEvents()
        time.sleep(0.05)
    assert panel.model._pixmaps, "no thumbnails were loaded"
    print(f"thumbnails in memory: {len(panel.model._pixmaps)}")

    # Filtering
    assert not any(b.column == "field" for b in panel.filters), "field filter should be removed"
    gene_box = next(b for b in panel.filters if b.column == "gene")
    gene_box.list.setCurrentRow(1)
    app.processEvents()
    filtered = panel.model.rowCount()
    print(f"after gene filter: {filtered}")
    assert 0 < filtered < session.count()

    # Search
    panel.search.setText("Polymyxin")
    app.processEvents()
    print(f"after search: {panel.model.rowCount()}")
    panel.search.clear()
    gene_box.clear()
    app.processEvents()

    # Annotation round-trip
    panel.view.setCurrentIndex(panel.model.index(0, 0))
    panel.view.selectionModel().select(
        panel.model.index(0, 0),
        panel.view.selectionModel().SelectionFlag.Select,
    )
    target = panel.model.row_at(0)
    panel.toggle_flag()
    panel.set_rating(4)
    app.processEvents()
    fresh = IndexDB(cfg.db_path)
    stored = fresh.query({"well": [target.well]}, flagged_only=True)
    assert any(r.image_id == target.image_id and r.rating == 4 for r in stored), stored
    print(f"annotation persisted for {target.image_id}")
    fresh.close()

    # Full-resolution window
    panel.open_viewer(0)
    app.processEvents()
    viewer = panel._windows[-1]
    viewer.step(1)
    viewer.step(-1)
    app.processEvents()
    print(f"viewer title: {viewer.windowTitle()}")
    viewer.close()

    # Compare + blind
    window.compare_action.setChecked(True)
    app.processEvents()
    assert window.secondary is not None
    print(f"compare panel rows: {window.secondary.model.rowCount()}")

    window.blind_action.setChecked(True)
    app.processEvents()
    assert panel.model.caption(panel.model.row_at(0)) == ""
    assert not panel.side_container.isVisible()
    print("blind mode hides captions and metadata")
    window.blind_action.setChecked(False)
    window.compare_action.setChecked(False)
    app.processEvents()

    # Add Plate: load the same plate again as a second plate and confirm the
    # combined grid grows (separate DB per plate, unioned at query time).
    before = session.count()
    second_cfg = load_config(config_path)
    second_cfg.project.name = "second"
    session.add_plate(second_cfg)
    panel.refresh()
    app.processEvents()
    assert session.count() == before * 2
    assert panel.model.rowCount() == before * 2
    print(f"after add plate: {panel.model.rowCount()} rows across {len(session.plates)} plates")

    window.close()
    print("\nGUI smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "plato.toml"))
