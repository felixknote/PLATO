"""One self-contained browser panel: filters + grid + metadata.

The main window holds either one panel or two side by side, which is all
"compare two conditions" needs to be.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
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
from ..data.index.db import ImageRow
from .delegate import ThumbnailDelegate
from .model import ROW_ROLE, ThumbnailModel
from ..gui.scalebar import draw_scale_bar
from ..data.session import Session
from ..gui.viewer import ImageWindow

MAX_DISTINCT_FOR_FILTER = 60
EXPORT_FORMATS = ["TIFF (original)", "JPEG (converted)"]


class ExportOptionsDialog(QDialog):
    """Format choice plus JPEG-only options (scale bar, shared contrast)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export format")

        self.format_box = QComboBox()
        self.format_box.addItems(EXPORT_FORMATS)
        self.format_box.currentIndexChanged.connect(self._sync_jpeg_options)

        self.scale_bar_box = QCheckBox("Bake in scale bar")
        self.shared_contrast_box = QCheckBox("Use one shared brightness/contrast for all images")
        self.shared_contrast_box.setToolTip(
            "Off (default) = each image uses the fixed per-channel display limits "
            "shown on screen. On = compute one min/max contrast stretch across all "
            "selected images and apply it to every one of them."
        )

        form = QFormLayout()
        form.addRow("Format", self.format_box)
        form.addRow(self.scale_bar_box)
        form.addRow(self.shared_contrast_box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.setLayout(form)
        self._sync_jpeg_options()

    def _sync_jpeg_options(self) -> None:
        is_jpeg = self.format_box.currentText() == EXPORT_FORMATS[1]
        self.scale_bar_box.setEnabled(is_jpeg)
        self.shared_contrast_box.setEnabled(is_jpeg)
        if not is_jpeg:
            self.scale_bar_box.setChecked(False)
            self.shared_contrast_box.setChecked(False)

    @property
    def format(self) -> str:
        return self.format_box.currentText()

    @property
    def bake_scale_bar(self) -> bool:
        return self.scale_bar_box.isChecked()

    @property
    def shared_contrast(self) -> bool:
        return self.shared_contrast_box.isChecked()


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

    def set_selected(self, values: list[str]) -> None:
        """Select by value. Values this box does not offer are ignored."""
        if not values:
            return
        wanted = set(values)
        with QSignalBlocker(self.list):
            for position in range(self.list.count()):
                item = self.list.item(position)
                item.setSelected(item.text() in wanted)

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
        self.view.setObjectName("thumbnailGrid")
        self.view.setModel(self.model)
        self.view.setViewMode(QListView.ViewMode.IconMode)
        self.view.setResizeMode(QListView.ResizeMode.Adjust)
        self.view.setUniformItemSizes(True)
        self.view.setSpacing(6)
        # The delegate paints a hover tint, which needs hover events tracked.
        self.view.setMouseTracking(True)
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.view.setItemDelegate(ThumbnailDelegate(gui_cfg.thumbnail_size, self))
        self.view.doubleClicked.connect(lambda idx: self.open_viewer(idx.row()))
        self.view.selectionModel().currentChanged.connect(
            lambda cur, _prev: self._show_metadata(cur.row())
        )

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search genes, treatments, wells…")
        # Debounced, not immediate. Each query is a LEFT JOIN over every loaded
        # plate's index -- ~130 ms across sixteen 2016-image plates -- and it
        # runs on the GUI thread, so firing one per keystroke makes typing in
        # this box feel broken on a large session. One query once typing pauses
        # returns the same rows.
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(180)
        self._search_timer.timeout.connect(self.refresh)
        self.search.textChanged.connect(self._search_timer.start)

        self.flagged_only = QCheckBox("Flagged only")
        self.flagged_only.toggled.connect(self.refresh)

        self.autoscale_previews = QCheckBox("Autoscale previews")
        self.autoscale_previews.setToolTip(
            "Off (default) = fixed per-channel contrast shared across the whole "
            "screen, so wells stay comparable. On = each thumbnail is contrast-"
            "stretched to its own min/max."
        )
        self.autoscale_previews.toggled.connect(self.model.set_autoscale)

        self.filters: list[FilterBox] = []
        self._filter_fields = gui_cfg.filter_fields
        self.filter_layout = QVBoxLayout()
        self.filter_layout.setContentsMargins(10, 10, 10, 10)
        self.filter_layout.setSpacing(10)

        self._clear_button = QPushButton("Clear all filters")
        self._clear_button.clicked.connect(self.clear_filters)
        self._build_filters()

        filter_container = QWidget()
        filter_container.setLayout(self.filter_layout)
        filter_scroll = QScrollArea()
        filter_scroll.setWidget(filter_container)
        filter_scroll.setWidgetResizable(True)
        # Bounded rather than just a minimum: the sidebar is chrome, and left
        # free to grow it takes width from the thumbnails, which are the point.
        filter_scroll.setMinimumWidth(200)
        filter_scroll.setMaximumWidth(280)

        self.count_label = QLabel("—")

        details_heading = QLabel("Details")
        details_heading.setObjectName("panelHeading")

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
        side.setContentsMargins(12, 10, 12, 10)
        side.setSpacing(8)
        side.addWidget(self.count_label)
        side.addWidget(details_heading)
        side.addWidget(self.metadata, 1)
        side.addWidget(random_button)
        side.addWidget(export_button)
        side_container = QWidget()
        side_container.setLayout(side)
        side_container.setMinimumWidth(230)
        side_container.setMaximumWidth(320)
        self.side_container = side_container

        top = QHBoxLayout()
        top.setContentsMargins(12, 10, 12, 6)
        top.setSpacing(12)
        top.addWidget(self.search, 1)
        top.addWidget(self.flagged_only)
        top.addWidget(self.autoscale_previews)
        # Wrapped in a widget so compact mode can hide the whole row at once.
        self.search_row = QWidget()
        self.search_row.setLayout(top)

        splitter = QSplitter()
        splitter.addWidget(filter_scroll)
        splitter.addWidget(self.view)
        splitter.addWidget(side_container)
        # The grid is what the window is for; the two sidebars are fixed-width
        # chrome around it and should not take space from it as the window
        # grows. Without the explicit sizes they start out sharing it evenly
        # and the thumbnails render in a narrow strip.
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([240, 1100, 260])
        splitter.setChildrenCollapsible(False)
        self.filter_scroll = filter_scroll

        layout = QVBoxLayout()
        layout.addWidget(self.search_row)
        layout.addWidget(splitter, 1)
        self.setLayout(layout)

        self.refresh()

    # -- filters ----------------------------------------------------------

    def _build_filters(self, keep: dict[str, list[str]] | None = None) -> None:
        """(Re)populate the filter sidebar from the session's current plates.

        Rebuilt rather than built once, because adding or removing a plate
        changes what there is to filter by: a new plate brings its own plate
        name, possibly a new timepoint, and metadata values the first plate
        never had. Leaving the sidebar as it was built at construction meant
        an added plate could not be filtered to at all -- the one thing you
        want to do straight after adding it.

        ``keep`` re-applies selections by value, so a rebuild does not silently
        drop the filter the grid is currently showing. Values that the new set
        of plates no longer offers are dropped, which is the honest outcome:
        the rows behind them are gone too.
        """
        keep = keep if keep is not None else self.current_filters()
        self.filters.clear()
        # Taken out first so the teardown below cannot reparent it: it is
        # reused across rebuilds, unlike the boxes, which are discarded.
        self.filter_layout.removeWidget(self._clear_button)
        while self.filter_layout.count():
            item = self.filter_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        columns = self._filter_fields or self.session.filter_columns()
        for column in columns:
            values = self.session.distinct(column)
            if not values or len(values) > MAX_DISTINCT_FOR_FILTER:
                continue
            box = FilterBox(column, self.session.label(column), values)
            box.set_selected(keep.get(column, []))
            box.changed.connect(self.refresh)
            self.filters.append(box)
            self.filter_layout.addWidget(box)
        self.filter_layout.addStretch(1)
        self.filter_layout.addWidget(self._clear_button)

    def plates_changed(self) -> None:
        """Re-read everything derived from the set of loaded plates.

        Called after Add Plate / Remove Plate. Display limits are pooled across
        plates, so they change when the set does; recomputing them here is what
        stops a newly added plate's channels from having no limits at all in
        the viewer and in JPEG export, which rendered them by each image's own
        min/max -- per-image autoscaling by accident, in the one place the app
        is emphatic about not doing that.
        """
        self.levels = self.session.display_limits()
        self._build_filters()
        self.refresh()

    # -- querying ---------------------------------------------------------

    def current_filters(self) -> dict[str, list[str]]:
        return {box.column: box.selected() for box in self.filters if box.selected()}

    def refresh(self) -> None:
        # Cancel a queued keystroke-driven refresh: this call supersedes it.
        self._search_timer.stop()
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
        window = ImageWindow(
            self.session, rows, position, self.levels, blind=self.blind, parent=self
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
        self.export_rows(rows)

    def export_rows(self, rows: list[ImageRow]) -> None:
        """Export an explicit set of rows. Shared by the grid's "Export
        selected" and the comparison view's "Export what's on screen"."""
        if not rows:
            return
        target = QFileDialog.getExistingDirectory(self, "Export selected images to…")
        if not target:
            return
        dialog = ExportOptionsDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        destination = Path(target)
        if dialog.format == EXPORT_FORMATS[0]:
            self._export_tiff(rows, destination)
        else:
            self._export_jpeg(
                rows,
                destination,
                bake_scale_bar=dialog.bake_scale_bar,
                shared_contrast=dialog.shared_contrast,
            )
        self.status.emit(f"exported {len(rows)} images to {destination}")

    def _export_tiff(self, rows: list[ImageRow], destination: Path) -> None:
        for row in rows:
            shutil.copy2(row.path, destination / Path(row.path).name)

    def _shared_contrast_limits(self, rows: list[ImageRow]) -> dict[str, tuple[float, float]]:
        """One (lo, hi) per channel, computed from percentiles across every
        selected image in that channel — an auto contrast shared by the batch
        rather than the fixed screen-wide levels."""
        samples: dict[str, list[np.ndarray]] = {}
        for row in rows:
            plane = read_plane(Path(row.path))
            flat = plane.ravel().astype(np.float32)
            if flat.size > 50_000:
                flat = flat[:: flat.size // 50_000]
            samples.setdefault(row.channel or "_", []).append(flat)
        limits: dict[str, tuple[float, float]] = {}
        for channel, arrays in samples.items():
            combined = np.concatenate(arrays)
            lo, hi = np.percentile(combined, [1.0, 99.5])
            if hi <= lo:
                hi = lo + 1.0
            limits[channel] = (float(lo), float(hi))
        return limits

    def _export_jpeg(
        self,
        rows: list[ImageRow],
        destination: Path,
        *,
        bake_scale_bar: bool = False,
        shared_contrast: bool = False,
    ) -> None:
        levels = self._shared_contrast_limits(rows) if shared_contrast else self.levels
        for row in rows:
            path = Path(row.path)
            plane = read_plane(path)
            limits = levels.get(row.channel or "_", (float(plane.min()), float(plane.max())))
            eight_bit = scale_to_uint8(plane, limits)
            image = Image.fromarray(eight_bit, mode="L")
            if bake_scale_bar:
                image = draw_scale_bar(image)
            image.save(destination / f"{path.stem}.jpg", format="JPEG", quality=92)

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
        if len(self.session.plates) > 1:
            # The `plate` column comes from the filename/plate map and can be
            # the same string in two separately loaded plates; the session's
            # name is the disambiguated one, so it says which of them this is.
            lines.insert(1, f"loaded plate: <b>{self.session.plates[row.session_index].name}</b>")
        lines += [
            f"{self.session.label(key)}: <b>{value}</b>"
            for key, value in row.metadata.items()
            if value is not None
        ]
        if row.rating:
            lines.append(f"rating: {'★' * row.rating}")
        lines += ["", f"<i>{row.image_id}</i>"]
        self.metadata.setText("<br>".join(lines))
