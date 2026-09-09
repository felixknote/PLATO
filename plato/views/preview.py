"""Hover preview: a small rendering of the original micrograph, plus the
metadata for that point.

Loading happens on a thread-pool worker, never on the GUI thread. These files
are 14 MB 16-bit TIFFs on a network share (~38 ms each when warm, far worse
cold), and doing that in a mouse-move handler would make the plot stutter
exactly while you are trying to point at something.

Two guards make hover-driven loading behave:

* A *generation counter* -- the cursor moves faster than the disk, so results
  for a point you have already left are discarded rather than flickering
  through the preview.
* A small *LRU cache* of rendered pixmaps, so moving back and forth across a
  cluster is instant after the first pass.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..cache import read_plane, scale_to_uint8
from ..gui.theme import BORDER, IMAGE_BACKGROUND, SURFACE, TEXT, TEXT_FAINT, TEXT_MUTED

# Long edge of the rendered preview. Large enough to judge a phenotype,
# small enough that the downscale is cheap.
PREVIEW_PX = 260

# Subsample factor when reading for a preview. The plane is 2720 px and the
# preview is 260, a ~10x reduction, so reading every 4th pixel still supplies
# more detail than the output can show -- and cuts the network read fourfold.
PREVIEW_STRIDE = 4

_CACHE_LIMIT = 96


class _PreviewSignals(QObject):
    ready = Signal(int, int, QPixmap)  # generation, row index, pixmap
    failed = Signal(int, int, str)


class _PreviewTask(QRunnable):
    def __init__(
        self,
        generation: int,
        row_index: int,
        path: Path,
        signals: _PreviewSignals,
    ) -> None:
        super().__init__()
        self._generation = generation
        self._row = row_index
        self._path = path
        self._signals = signals

    def run(self) -> None:  # pragma: no cover - worker thread
        try:
            plane = read_plane(self._path, stride=PREVIEW_STRIDE)
            pixmap = _to_pixmap(plane)
        except Exception as exc:  # noqa: BLE001 - a bad file must not kill hover
            try:
                self._signals.failed.emit(self._generation, self._row, str(exc))
            except RuntimeError:
                pass
            return
        try:
            self._signals.ready.emit(self._generation, self._row, pixmap)
        except RuntimeError:
            # The widget can be destroyed while a load is in flight.
            pass


def _to_pixmap(plane: np.ndarray) -> QPixmap:
    """Contrast-stretch a plane and render it to a preview-sized pixmap.

    Percentile limits per image, unlike the browser grid's fixed screen-wide
    limits. The grid deliberately shares limits so wells stay comparable; here
    a single image is shown on its own, with no neighbour to compare against,
    and the job is simply to make its phenotype visible.
    """
    flat = plane.ravel()
    if flat.size > 20_000:
        flat = flat[:: flat.size // 20_000]
    lo, hi = np.percentile(flat.astype(np.float32), [1.0, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(plane.min()), float(plane.max())
    if hi <= lo:
        hi = lo + 1.0

    eight_bit = np.ascontiguousarray(scale_to_uint8(plane, (float(lo), float(hi))))
    height, width = eight_bit.shape
    image = QImage(eight_bit.data, width, height, width, QImage.Format.Format_Grayscale8)
    # copy() because the QImage above borrows the numpy buffer, which is freed
    # when this function returns.
    return QPixmap.fromImage(image.copy()).scaled(
        PREVIEW_PX,
        PREVIEW_PX,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class PreviewPane(QWidget):
    """Image preview above a metadata block, driven by hovered row index."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.image_label = QLabel("Hover a point")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(PREVIEW_PX, PREVIEW_PX)
        self.image_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.image_label.setStyleSheet(
            f"background-color: {IMAGE_BACKGROUND}; border: 1px solid {BORDER};"
            f"color: {TEXT_FAINT}; border-radius: 3px;"
        )

        self.metadata_label = QLabel("")
        self.metadata_label.setWordWrap(True)
        self.metadata_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.metadata_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.metadata_label.setStyleSheet(f"color: {TEXT_MUTED};")
        self.metadata_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.image_label)
        layout.addWidget(self.metadata_label, 1)
        self.setLayout(layout)

        self._signals = _PreviewSignals()
        self._signals.ready.connect(self._on_ready)
        self._signals.failed.connect(self._on_failed)
        self._pool = QThreadPool.globalInstance()
        self._cache: OrderedDict[int, QPixmap] = OrderedDict()
        self._generation = 0
        self._current_row = -1
        self._lock = threading.Lock()

    # -- api ---------------------------------------------------------------

    def clear(self) -> None:
        self._generation += 1
        self._current_row = -1
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("Hover a point")
        self.metadata_label.setText("")

    def show_row(self, row_index: int, description: str, path: Path | None) -> None:
        """Show metadata immediately; load the image behind it."""
        self._generation += 1
        self._current_row = row_index
        self.metadata_label.setText(description)

        if path is None:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("image not found")
            return

        cached = self._cache.get(row_index)
        if cached is not None:
            self._cache.move_to_end(row_index)
            self.image_label.setPixmap(cached)
            return

        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("loading…")
        self._pool.start(_PreviewTask(self._generation, row_index, path, self._signals))

    # -- worker results ----------------------------------------------------

    def _on_ready(self, generation: int, row_index: int, pixmap: QPixmap) -> None:
        with self._lock:
            self._cache[row_index] = pixmap
            self._cache.move_to_end(row_index)
            while len(self._cache) > _CACHE_LIMIT:
                self._cache.popitem(last=False)
        # Stale result: the cursor has moved on. Cached above anyway, so
        # coming back to this point is instant.
        if generation != self._generation:
            return
        self.image_label.setPixmap(pixmap)

    def _on_failed(self, generation: int, row_index: int, message: str) -> None:
        if generation != self._generation:
            return
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("could not read image")
