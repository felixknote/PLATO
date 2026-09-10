"""A grid of thumbnails for an arbitrary set of embedding rows.

What a lasso is for. Hovering answers "what is this point?"; selecting a
cluster and seeing every image in it answers "what is this *region*?", which
is the question a projection actually poses and the one every comparable tool
(CellProfiler Analyst's Image Gallery, cellxgene's selection, TissUUmaps'
linked feature/physical views) exists to answer.

Loading is the same shape as the browser grid's: a Qt model that never touches
disk in ``data()``, a bounded thread pool behind it, and an LRU of decoded
pixmaps. A selection can easily be thousands of images on a network share, so
the grid shows a capped sample and says so rather than trying to read them all.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    Signal,
)
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QListView,
    QVBoxLayout,
    QWidget,
)

from ..gui.theme import BORDER, IMAGE_BACKGROUND, TEXT_FAINT, TEXT_MUTED
from .preview import PREVIEW_STRIDE, _to_pixmap

ROW_ROLE = Qt.ItemDataRole.UserRole + 1

# Tile edge in pixels.
TILE = 116

# Most images a selection will render. A lasso over a dense cluster can cover
# thousands, and reading thousands of 14 MB TIFFs off a share to fill a grid
# nobody can scan is a way to make the app appear hung. A sample answers "what
# is in here" just as well.
MAX_TILES = 120

# Decoded pixmaps kept in memory. Each is ~116x116 grey, so this is small.
CACHE_LIMIT = 240


class _Signals(QObject):
    loaded = Signal(int, QPixmap)
    failed = Signal(int)


class _LoadTask(QRunnable):
    def __init__(self, generation: int, position: int, path: Path, signals: _Signals) -> None:
        super().__init__()
        self._generation = generation
        self._position = position
        self._path = path
        self._signals = signals

    def run(self) -> None:  # pragma: no cover - worker thread
        try:
            from ..cache import read_plane

            plane = read_plane(self._path, stride=PREVIEW_STRIDE)
            pixmap = _to_pixmap(plane).scaled(
                TILE,
                TILE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        except Exception:  # noqa: BLE001 - one bad file must not stop the grid
            try:
                self._signals.failed.emit(self._position)
            except RuntimeError:
                pass
            return
        try:
            self._signals.loaded.emit(self._position, pixmap)
        except RuntimeError:
            # The model can be torn down while a read is in flight.
            pass


class GalleryModel(QAbstractListModel):
    """Thumbnails for a list of (row index, path) pairs, loaded lazily."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._entries: list[tuple[int, Path]] = []
        self._pixmaps: OrderedDict[int, QPixmap] = OrderedDict()
        self._pending: set[int] = set()
        self._generation = 0
        self._lock = threading.Lock()

        self._placeholder = QPixmap(TILE, TILE)
        self._placeholder.fill(QColor(IMAGE_BACKGROUND))

        self._signals = _Signals()
        self._signals.loaded.connect(self._on_loaded)
        self._signals.failed.connect(self._on_failed)
        self._pool = QThreadPool.globalInstance()

    def set_entries(self, entries: list[tuple[int, Path]]) -> None:
        self.beginResetModel()
        self._entries = entries
        self._pending.clear()
        self._generation += 1
        self.endResetModel()

    def row_at(self, position: int) -> int | None:
        if 0 <= position < len(self._entries):
            return self._entries[position][0]
        return None

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._entries)

    def data(self, index, role: int = Qt.ItemDataRole.DisplayRole):  # noqa: ANN001
        if not index.isValid():
            return None
        position = index.row()
        if role == ROW_ROLE:
            return self._entries[position][0]
        if role == Qt.ItemDataRole.DecorationRole:
            pixmap = self._pixmaps.get(position)
            if pixmap is not None:
                self._pixmaps.move_to_end(position)
                return pixmap
            self._request(position)
            return self._placeholder
        if role == Qt.ItemDataRole.SizeHintRole:
            return QSize(TILE + 8, TILE + 8)
        return None

    def _request(self, position: int) -> None:
        if position in self._pending:
            return
        self._pending.add(position)
        self._pool.start(
            _LoadTask(self._generation, position, self._entries[position][1], self._signals)
        )

    def _on_loaded(self, position: int, pixmap: QPixmap) -> None:
        self._pending.discard(position)
        with self._lock:
            self._pixmaps[position] = pixmap
            self._pixmaps.move_to_end(position)
            while len(self._pixmaps) > CACHE_LIMIT:
                self._pixmaps.popitem(last=False)
        if position < len(self._entries):
            idx = self.index(position, 0)
            self.dataChanged.emit(idx, idx)

    def _on_failed(self, position: int) -> None:
        self._pending.discard(position)


class SelectionGallery(QWidget):
    """Thumbnails of the current selection, with a count above them."""

    row_activated = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.heading = QLabel("Lasso a region to see its images")
        self.heading.setWordWrap(True)
        self.heading.setObjectName("hint")

        self.model = GalleryModel(self)
        self.view = QListView()
        self.view.setModel(self.model)
        self.view.setViewMode(QListView.ViewMode.IconMode)
        self.view.setResizeMode(QListView.ResizeMode.Adjust)
        self.view.setUniformItemSizes(True)
        self.view.setSpacing(3)
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.view.setStyleSheet(
            f"QListView {{ background: {IMAGE_BACKGROUND}; border: 1px solid {BORDER}; }}"
        )
        self.view.doubleClicked.connect(self._on_activated)
        self.view.hide()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.heading)
        layout.addWidget(self.view, 1)
        self.setLayout(layout)

    def show_selection(self, entries: list[tuple[int, Path]], total: int) -> None:
        """Show up to MAX_TILES of ``total`` selected rows."""
        if not entries:
            self.model.set_entries([])
            self.view.hide()
            self.heading.setText(
                f"{total:,} points selected — none of their images were found"
                if total
                else "Lasso a region to see its images"
            )
            return

        shown = entries[:MAX_TILES]
        self.model.set_entries(shown)
        self.view.show()
        if total > len(shown):
            self.heading.setText(
                f"<b>{total:,}</b> points selected · showing {len(shown)}"
            )
        else:
            self.heading.setText(f"<b>{total:,}</b> points selected")

    def _on_activated(self, index) -> None:  # noqa: ANN001
        row = self.model.row_at(index.row())
        if row is not None:
            self.row_activated.emit(row)
