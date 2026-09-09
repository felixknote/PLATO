"""Offscreen smoke test. Run with QT_QPA_PLATFORM=offscreen.

This does not test that the app *looks* right — it tests that every widget
constructs, the model fills, thumbnails arrive from the worker pool, and the
annotation round-trip reaches SQLite. It also exercises the empty-state /
Load Data / Add Plate startup workflow via Session directly (headless, so the
native file pickers are bypassed and Session.add_plate is called with an
already-built Config, same as LoadPlateDialog would produce on accept).
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

from PySide6.QtWidgets import QApplication

from plato.config import load_config
from plato.gui.main_window import MainWindow
from plato.data.session import SOURCE_PLATE, DuplicatePlateError, Session
from plato.data.index.db import IndexDB


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
    # toggle_flag flips, so a rerun over a database that kept the last run's
    # flag would clear it and the assertion below would fail on state rather
    # than on behaviour. Start from unflagged.
    if target.flagged:
        panel.toggle_flag()
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

    # Re-adding a loaded plate is refused rather than doubling the grid.
    before = session.count()
    try:
        session.add_plate(load_config(config_path))
    except DuplicatePlateError as exc:
        print(f"re-adding the same plate refused: {exc}")
    else:
        raise AssertionError("adding an already-loaded plate should raise")
    assert session.count() == before, "a refused add must not change the session"

    # Add Plate: a genuinely second plate, with its own index database. Built
    # by copying the first plate's .plato/ so the test needs no second dataset;
    # a different db_path is what makes it a different plate to the session.
    second_root = Path(config_path).parent / "second_plate"
    if second_root.exists():
        shutil.rmtree(second_root)
    shutil.copytree(Path(config_path).parent / ".plato", second_root)
    second_cfg = load_config(config_path)
    second_cfg.project.work_dir = second_root
    second_cfg.project.name = "second"
    session.add_plate(second_cfg)
    window._plates_changed()
    app.processEvents()
    assert session.count() == before * 2
    assert panel.model.rowCount() == before * 2
    print(f"after add plate: {panel.model.rowCount()} rows across {len(session.plates)} plates")

    # The sidebar must now offer the loaded-plate filter, or the plate just
    # added cannot be filtered to. The `plate` column cannot serve: both
    # plates were indexed with the same config, so both label every row
    # "Plate1" and selecting it selects both.
    source_box = next((b for b in panel.filters if b.column == SOURCE_PLATE), None)
    assert source_box is not None, "loaded plate should be offered as a filter"
    offered = [source_box.list.item(i).text() for i in range(source_box.list.count())]
    assert offered == session.plate_names, offered
    print(f"loaded-plate filter offers: {offered}")

    # set_selected is the silent restore used during a rebuild, so a
    # programmatic selection has to ask for the requery itself.
    source_box.set_selected([offered[1]])
    panel.refresh()
    app.processEvents()
    assert panel.model.rowCount() == before, panel.model.rowCount()
    assert all(r.session_index == 1 for r in panel.model.rows())
    print(
        f"filtering to '{offered[1]}' isolates {panel.model.rowCount()} "
        "rows from that plate alone"
    )
    source_box.clear()
    panel.refresh()
    app.processEvents()

    # A name collision is disambiguated rather than left ambiguous on screen.
    third_root = Path(config_path).parent / "third_plate"
    if third_root.exists():
        shutil.rmtree(third_root)
    shutil.copytree(Path(config_path).parent / ".plato", third_root)
    third_cfg = load_config(config_path)
    third_cfg.project.work_dir = third_root
    third_cfg.project.name = "second"
    third = session.add_plate(third_cfg)
    assert third.name == "second (2)", third.name
    print(f"name collision disambiguated to: {third.name}")

    # Removal unloads exactly one plate and leaves the rest queryable.
    session.remove_plate(2)
    window._plates_changed()
    app.processEvents()
    assert len(session.plates) == 2
    assert panel.model.rowCount() == before * 2
    print(f"after remove: {panel.model.rowCount()} rows across {len(session.plates)} plates")

    # Annotations from a multi-plate session name the plate they came from.
    exported = session.export_annotations()
    assert exported, "the earlier flag should still be exported"
    assert all("session_plate" in row and "uid" in row for row in exported)
    print(f"annotation export tags {len(exported)} row(s) with their plate")

    window.close()
    print("\nGUI smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "plato.toml"))
