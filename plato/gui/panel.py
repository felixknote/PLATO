"""One self-contained browser panel: filters + grid + metadata.

The main window holds either one panel or two side by side, which is all
"compare two conditions" needs to be.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..cache import read_plane, scale_to_uint8
from ..index.db import ImageRow
from .delegate import ThumbnailDelegate
from .model import ROW_ROLE, ThumbnailModel
from .session import Session
from .viewer import ImageWindow

MAX_DISTINCT_FOR_FILTER = 60
EXPORT_FORMATS = ["TIFF (original)", "JPEG (converted)"]


class FilterBox(QGroupBox):
    """Multi-select list for one column. Nothing selected = no constraint."""

    changed = Signal()

    def __init__(self, column: str, label: str, values: list[str]) -> None:
        super().__init__(label)
        self.column = column
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.setMaximumHeight(130)
        for value in values:
            self.list.addItem(QListWidgetItem(value))
        self.list.itemSelectionChanged.connect(self.changed.emit)

        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 4, 6, 4)
        layout.addWidget(self.list)
        layout.addWidget(clear)
        self.setLayout(layout)

    def selected(self) -> list[str]:
        return [item.text() for item in self.list.selectedItems()]

    def clear(self) -> None:
        self.list.clearSelection()


class BrowserPanel(QWidget):
    """Filters on the left, thumbnail grid in the middle, metadata on the right."""

    status = Signal(str)

    def __init__(self, session: Session, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        self.db = session  # Session exposes the same query/label/etc. API as IndexDB
        gui_cfg = session.plates[0].cfg.gui
        self.levels = session.display_limits()
        self.blind = False
        self._windows: list[ImageWindow] = []

        self.model = ThumbnailModel(
            session,
            caption_fields=gui_cfg.caption_fields or session.metadata_columns[:2],
            tile=gui_cfg.thumbnail_size,
            parent=self,
        )

        self.view = QListView()
        self.view.setModel(self.model)
        self.view.setViewMode(QListView.ViewMode.IconMode)
        self.view.setResizeMode(QListView.ResizeMode.Adjust)
        self.view.setUniformItemSizes(True)
        self.view.setSpacing(4)
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.view.setItemDelegate(ThumbnailDelegate(gui_cfg.thumbnail_size, self))
        self.view.doubleClicked.connect(lambda idx: self.open_viewer(idx.row()))
        self.view.selectionModel().currentChanged.connect(
            lambda cur, _prev: self._show_metadata(cur.row())
        )

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search genes, treatments, wells…")
        self.search.textChanged.connect(self.refresh)

        self.flagged_only = QCheckBox("Flagged only")
        self.flagged_only.toggled.connect(self.refresh)

        self.filters: list[FilterBox] = []
        filter_columns = gui_cfg.filter_fields or session.filter_columns()
        filter_form = QVBoxLayout()
        for column in filter_columns:
            values = session.distinct(column)
            if not values or len(values) > MAX_DISTINCT_FOR_FILTER:
                continue
            box = FilterBox(column, session.label(column), values)
            box.changed.connect(self.refresh)
            self.filters.append(box)
            filter_form.addWidget(box)
        filter_form.addStretch(1)

        clear_button = QPushButton("Clear all filters")
        clear_button.clicked.connect(self.clear_filters)
        filter_form.addWidget(clear_button)

        filter_container = QWidget()
        filter_container.setLayout(filter_form)
        filter_scroll = QScrollArea()
        filter_scroll.setWidget(filter_container)
        filter_scroll.setWidgetResizable(True)
        filter_scroll.setMinimumWidth(220)

        self.count_label = QLabel("—")
        self.metadata = QLabel("Select an image.")
        self.metadata.setWordWrap(True)
        self.metadata.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.metadata.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        random_button = QPushButton("Random image (R)")
        random_button.clicked.connect(self.jump_random)
        export_button = QPushButton("Export selected…")
        export_button.clicked.connect(self.export_selected)

        side = QVBoxLayout()
        side.addWidget(self.count_label)
        side.addWidget(self.metadata, 1)
        side.addWidget(random_button)
        side.addWidget(export_button)
        side_container = QWidget()
        side_container.setLayout(side)
        side_container.setMinimumWidth(240)
        self.side_container = side_container

        top = QHBoxLayout()
        top.addWidget(QLabel("Search"))
        top.addWidget(self.search, 1)
        top.addWidget(self.flagged_only)

        splitter = QSplitter()
        splitter.addWidget(filter_scroll)
        splitter.addWidget(self.view)
        splitter.addWidget(side_container)
        splitter.setStretchFactor(1, 1)
        self.filter_scroll = filter_scroll

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(splitter, 1)
        self.setLayout(layout)

        self.refresh()

    # -- querying ---------------------------------------------------------

    def current_filters(self) -> dict[str, list[str]]:
        return {box.column: box.selected() for box in self.filters if box.selected()}

    def refresh(self) -> None:
        order = "RANDOM()" if self.blind else "plate, well_row, well_col, field, channel"
        rows = self.session.query(
            self.current_filters(),
            self.search.text(),
            flagged_only=self.flagged_only.isChecked(),
            order=order,
        )
        self.model.set_rows(rows)
        self.count_label.setText(f"<b>{len(rows)}</b> images")
        self.status.emit(f"{len(rows)} images match")
        if rows:
            self.view.setCurrentIndex(self.model.index(0, 0))

    def clear_filters(self) -> None:
        for box in self.filters:
            box.clear()
        self.search.clear()
        self.flagged_only.setChecked(False)
        self.refresh()

    def set_blind(self, blind: bool) -> None:
        """Hide metadata and randomise order.

        Seeing the condition while scoring biases the scoring. If flags feed
        anything downstream, score blind and reveal afterwards.
        """
        self.blind = blind
        self.model.set_blind(blind)
        self.filter_scroll.setVisible(not blind)
        self.side_container.setVisible(not blind)
        self.refresh()

    # -- actions ----------------------------------------------------------

    def selected_rows(self) -> list[ImageRow]:
        return [
            idx.data(ROW_ROLE)
            for idx in self.view.selectionModel().selectedIndexes()
        ]

    def current_index(self) -> int:
        return self.view.currentIndex().row()

    def open_viewer(self, position: int) -> None:
        rows = self.model.rows()
        if not rows or position < 0:
            return
        cfg = self.session.plates[rows[position].session_index].cfg
        window = ImageWindow(
            cfg, rows, position, self.levels, blind=self.blind, parent=self
        )
        window.setWindowFlag(Qt.WindowType.Window, True)
        window.show()
        self._windows.append(window)

    def jump_random(self) -> None:
        count = self.model.rowCount()
        if count:
            self.view.setCurrentIndex(self.model.index(random.randrange(count), 0))

    def toggle_flag(self) -> None:
        for row in self.selected_rows() or []:
            self.session.set_annotation(row, flagged=not row.flagged)
            row.flagged = not row.flagged
            self.model.refresh_row(row)
        self.status.emit("flag toggled")

    def set_rating(self, rating: int | None) -> None:
        for row in self.selected_rows() or []:
            self.session.set_annotation(row, rating=rating)
            row.rating = rating
            self.model.refresh_row(row)
        self.status.emit(f"rating set to {rating}")

    def export_selected(self) -> None:
        rows = self.selected_rows()
        if not rows:
            QMessageBox.information(self, "Export", "Select some images first.")
            return
        target = QFileDialog.getExistingDirectory(self, "Export selected images to…")
        if not target:
            return
        fmt, ok = QInputDialog.getItem(
            self, "Export format", "Format:", EXPORT_FORMATS, 0, False
        )
        if not ok:
            return
        destination = Path(target)
        if fmt == EXPORT_FORMATS[0]:
            self._export_tiff(rows, destination)
        else:
            self._export_jpeg(rows, destination)
        self.status.emit(f"exported {len(rows)} images to {destination}")

    def _export_tiff(self, rows: list[ImageRow], destination: Path) -> None:
        for row in rows:
            shutil.copy2(row.path, destination / Path(row.path).name)

    def _export_jpeg(self, rows: list[ImageRow], destination: Path) -> None:
        for row in rows:
            path = Path(row.path)
            plane = read_plane(path)
            limits = self.levels.get(row.channel or "_", (float(plane.min()), float(plane.max())))
            eight_bit = scale_to_uint8(plane, limits)
            Image.fromarray(eight_bit, mode="L").save(
                destination / f"{path.stem}.jpg", format="JPEG", quality=92
            )

    # -- metadata panel ---------------------------------------------------

    def _show_metadata(self, position: int) -> None:
        row = self.model.row_at(position)
        if row is None:
            return
        if self.blind:
            self.metadata.setText("<b>Blinded review</b><br>Metadata hidden.")
            return
        lines = [
            f"<b>{row.plate} {row.well}</b>",
            f"field {row.field or '-'} · channel {row.channel or '-'}",
            "",
        ]
        lines += [
            f"{self.session.label(key)}: <b>{value}</b>"
            for key, value in row.metadata.items()
            if value is not None
        ]
        if row.rating:
            lines.append(f"rating: {'★' * row.rating}")
        lines += ["", f"<i>{row.image_id}</i>"]
        self.metadata.setText("<br>".join(lines))
