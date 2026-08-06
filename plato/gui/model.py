"""Qt model layer.

The whole reason this app stays responsive at 10^4-10^5 images:

* ``QAbstractListModel`` + ``QListView(IconMode)`` instantiates delegates only
  for visible items. Building a QScrollArea full of QLabels dies around 2000.
* ``data()`` never touches disk. It returns a placeholder and schedules a
  background fetch; the row is refreshed when the pixmap arrives.

Rows can come from any loaded plate (see ``session.py``), each with its own
thumbnail cache file, so loading is keyed on ``(thumb_db_path, image_id)``
rather than a single fixed database.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

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

from ..cache import ThumbnailCache
from ..index.db import ImageRow
from .session import Session

ROW_ROLE = Qt.ItemDataRole.UserRole + 1

_thread_local = threading.local()


def _cache_for_thread(path: Path) -> ThumbnailCache:
    cache = getattr(_thread_local, "cache", None)
    if cache is None or cache.path != path:
        cache = ThumbnailCache(path)
        _thread_local.cache = cache
    return cache


class _LoaderSignals(QObject):
    loaded = Signal(str, bytes)
    missing = Signal(str)


class _LoadTask(QRunnable):
    def __init__(self, thumb_db: Path, image_id: str, key: str, signals: _LoaderSignals) -> None:
        super().__init__()
        self._thumb_db = thumb_db
        self._image_id = image_id
        self._key = key
        self._signals = signals

    def run(self) -> None:  # pragma: no cover - runs on worker thread
        try:
            png = _cache_for_thread(self._thumb_db).get(self._image_id)
        except Exception:  # noqa: BLE001
            png = None
        if png is None:
            self._signals.missing.emit(self._key)
        else:
            self._signals.loaded.emit(self._key, png)


class ThumbnailModel(QAbstractListModel):
    """Flat list of :class:`ImageRow`, with lazily loaded thumbnails."""

    def __init__(
        self,
        session: Session,
        caption_fields: list[str],
        *,
        tile: int = 220,
        memory_items: int = 2000,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._rows: list[ImageRow] = []
        self._index_of: dict[str, int] = {}
        self._pixmaps: OrderedDict[str, QPixmap] = OrderedDict()
        self._pending: set[str] = set()
        self._memory_items = memory_items
        self.caption_fields = caption_fields
        self.tile = tile
        self.blind = False

        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(4)
        self._signals = _LoaderSignals()
        self._signals.loaded.connect(self._on_loaded)
        self._signals.missing.connect(self._on_missing)

        self._placeholder = QPixmap(tile, tile)
        self._placeholder.fill(QColor(60, 60, 66))

    # -- content ----------------------------------------------------------

    @staticmethod
    def _key(row: ImageRow) -> str:
        return f"{row.session_index}:{row.image_id}"

    def set_rows(self, rows: list[ImageRow]) -> None:
        self.beginResetModel()
        self._rows = rows
        self._index_of = {self._key(r): i for i, r in enumerate(rows)}
        self._pending.clear()
        self.endResetModel()

    def row_at(self, index: int) -> ImageRow | None:
        if 0 <= index < len(self._rows):
            return self._rows[index]
        return None

    def rows(self) -> list[ImageRow]:
        return self._rows

    def refresh_row(self, row: ImageRow) -> None:
        position = self._index_of.get(self._key(row))
        if position is None:
            return
        idx = self.index(position, 0)
        self.dataChanged.emit(idx, idx)

    def set_blind(self, blind: bool) -> None:
        self.blind = blind
        if self._rows:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self._rows) - 1, 0))

    def caption(self, row: ImageRow) -> str:
        if self.blind:
            return ""
        parts = [
            str(row.metadata[f])
            for f in self.caption_fields
            if f in row.metadata and row.metadata[f] is not None
        ]
        head = row.well if not parts else f"{row.well}  {' · '.join(parts)}"
        return head

    # -- QAbstractListModel ----------------------------------------------

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def data(self, index, role: int = Qt.ItemDataRole.DisplayRole):  # noqa: ANN001
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        if role == ROW_ROLE:
            return row
        if role == Qt.ItemDataRole.DisplayRole:
            return self.caption(row)
        if role == Qt.ItemDataRole.ToolTipRole:
            if self.blind:
                return "(blinded)"
            meta = "\n".join(
                f"{k}: {v}" for k, v in row.metadata.items() if v is not None
            )
            return f"{row.image_id}\n{row.plate} {row.well}\n{meta}"
        if role == Qt.ItemDataRole.DecorationRole:
            key = self._key(row)
            pixmap = self._pixmaps.get(key)
            if pixmap is not None:
                self._pixmaps.move_to_end(key)
                return pixmap
            self._request(row)
            return self._placeholder
        if role == Qt.ItemDataRole.SizeHintRole:
            return QSize(self.tile + 16, self.tile + 34)
        return None

    # -- loading ----------------------------------------------------------

    def _request(self, row: ImageRow) -> None:
        key = self._key(row)
        if key in self._pending:
            return
        self._pending.add(key)
        thumb_db = self._session.plates[row.session_index].cfg.thumb_db_path
        self._pool.start(_LoadTask(thumb_db, row.image_id, key, self._signals))

    def _on_loaded(self, key: str, png: bytes) -> None:
        self._pending.discard(key)
        pixmap = QPixmap()
        if not pixmap.loadFromData(png, "PNG"):
            return
        self._pixmaps[key] = pixmap
        while len(self._pixmaps) > self._memory_items:
            self._pixmaps.popitem(last=False)
        position = self._index_of.get(key)
        if position is not None:
            idx = self.index(position, 0)
            self.dataChanged.emit(idx, idx)

    def _on_missing(self, key: str) -> None:
        self._pending.discard(key)
        self._pixmaps[key] = self._placeholder
