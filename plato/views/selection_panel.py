"""The right-hand column: every selected point, as a scrollable list of crops.

Replaces a single hover preview with the *selection*, because one preview can
only answer "what is this point?" and the questions a projection raises are
comparative -- is this cluster one phenotype or three, does this outlier look
like its neighbours.

Three decisions worth stating:

**Crops, not whole fields.** A field of view is mostly background at preview
size; the phenotype is in the cells. Entries show a centre crop by default,
which is the same pixels a whole-image preview would give you at four times
the magnification. The uncropped image is a double-click away -- that is what
the detail viewer is for.

**Entries are widgets, built lazily.** A selection can be thousands of points.
Building a widget per row would stall the GUI thread long before the images
loaded, so entries are created only for rows scrolled into view, and their
images are read on the shared thread pool through the same decode path the
gallery and hover preview use.

**Row indices, never positions.** Selection state lives in frame-row space, so
it survives filtering, recolouring and reprojection.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..gui.theme import (
    ACCENT,
    BORDER,
    IMAGE_BACKGROUND,
    SURFACE,
    SURFACE_RAISED,
    TEXT,
    TEXT_FAINT,
    TEXT_MUTED,
)
from .preview import PREVIEW_STRIDE

# Fraction of the field kept when cropping. 0.5 keeps the centre quarter by
# area, which at typical seeding density is a few dozen cells -- enough to
# read a phenotype, tight enough that they are visibly cells.
CROP_FRACTION = 0.5

# Entry heights are derived from the column width, within these bounds, so
# dragging the splitter really does make the images bigger.
MIN_IMAGE_PX = 90
MAX_IMAGE_PX = 520

# Decoded crops held in memory, keyed by (row, cropped). Small: each is at
# most MAX_IMAGE_PX square, 8-bit grey.
_CACHE_LIMIT = 160

# Entries built at once. A selection of 5,000 points does not need 5,000
# widgets; it needs the ones you can see, plus enough slack to scroll
# smoothly. More are built as the viewport reaches the end of the list.
PAGE_SIZE = 24


class _CropSignals(QObject):
    ready = Signal(int, int, QPixmap)  # generation, row, pixmap
    failed = Signal(int, int)


class _CropTask(QRunnable):
    """Reads one image and renders the crop, off the GUI thread."""

    def __init__(
        self,
        generation: int,
        row: int,
        path: Path,
        target_px: int,
        cropped: bool,
        signals: _CropSignals,
    ) -> None:
        super().__init__()
        self._generation = generation
        self._row = row
        self._path = path
        self._target = target_px
        self._cropped = cropped
        self._signals = signals

    def run(self) -> None:  # pragma: no cover - worker thread
        try:
            from ..cache import read_plane

            from .preview import _to_pixmap

            plane = read_plane(self._path, stride=PREVIEW_STRIDE)
            if self._cropped:
                plane = _centre_crop(plane, CROP_FRACTION)
            pixmap = _to_pixmap(plane).scaled(
                self._target,
                self._target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        except Exception:  # noqa: BLE001 - one bad file must not stop the list
            try:
                self._signals.failed.emit(self._generation, self._row)
            except RuntimeError:
                pass
            return
        try:
            self._signals.ready.emit(self._generation, self._row, pixmap)
        except RuntimeError:
            # The panel can be torn down while a read is in flight.
            pass


def _centre_crop(plane: np.ndarray, fraction: float) -> np.ndarray:
    """The middle ``fraction`` of a plane, by linear dimension."""
    fraction = max(0.05, min(1.0, fraction))
    if fraction >= 1.0:
        return plane
    height, width = plane.shape[:2]
    new_h, new_w = max(1, int(height * fraction)), max(1, int(width * fraction))
    top, left = (height - new_h) // 2, (width - new_w) // 2
    return plane[top : top + new_h, left : left + new_w]


class SelectionEntry(QFrame):
    """One selected point: its crop, its label, and a remove button."""

    clicked = Signal(int)
    activated = Signal(int)
    removed = Signal(int)

    def __init__(self, row: int, caption: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.row = row
        self._current = False

        self.image = QLabel()
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setStyleSheet(f"background: {IMAGE_BACKGROUND}; border-radius: 2px;")
        self.image.setText("…")

        self.caption = QLabel(caption)
        self.caption.setWordWrap(True)
        self.caption.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self.caption.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Removing one point from a selection has to be possible without
        # rebuilding it, so every entry carries its own control.
        self.remove_button = QPushButton("✕")
        self.remove_button.setFixedSize(18, 18)
        self.remove_button.setToolTip("Remove this point from the selection")
        self.remove_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.remove_button.clicked.connect(lambda: self.removed.emit(self.row))

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.caption, 1)
        header.addWidget(self.remove_button, 0, Qt.AlignmentFlag.AlignTop)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(self.image)
        layout.addLayout(header)
        self.setLayout(layout)
        self.set_current(False)

    def set_image_size(self, px: int) -> None:
        self.image.setFixedHeight(px)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self.image.setPixmap(pixmap)

    def set_failed(self) -> None:
        self.image.setPixmap(QPixmap())
        self.image.setText("unreadable")

    def set_current(self, current: bool) -> None:
        """Mark this as the entry whose metadata is shown below the list."""
        self._current = current
        self.setStyleSheet(
            f"SelectionEntry {{ background: {SURFACE_RAISED};"
            f" border: 1px solid {ACCENT}; border-radius: 4px; }}"
            if current
            else f"SelectionEntry {{ background: {SURFACE};"
            f" border: 1px solid {BORDER}; border-radius: 4px; }}"
        )

    def mousePressEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.row)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit(self.row)
        super().mouseDoubleClickEvent(event)


class SelectionPanel(QWidget):
    """A scrollable column of crops, one per selected point."""

    # A row was clicked once: show its metadata, highlight it in the plot.
    row_clicked = Signal(int)
    # A row was double-clicked: open the full, uncropped image.
    row_activated = Signal(int)
    # The selection changed from inside this panel (a point was removed).
    selection_changed = Signal(object)
    compare_requested = Signal()
    # A reveal that could not happen, as a message for the status bar.
    reveal_failed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.rows: list[int] = []
        self._entries: dict[int, SelectionEntry] = {}
        self._built = 0
        self._current_row = -1
        self._generation = 0
        self._cropped = True
        self._image_px = 150
        self._cache: OrderedDict[tuple[int, bool], QPixmap] = OrderedDict()
        self._lock = threading.Lock()
        self._describe = lambda row: ""
        self._path_for = lambda row: None

        self._signals = _CropSignals()
        self._signals.ready.connect(self._on_ready)
        self._signals.failed.connect(self._on_failed)
        self._pool = QThreadPool.globalInstance()

        # -- header
        self.heading = QLabel("Selection")
        self.heading.setObjectName("panelHeading")

        self.count_label = QLabel("Click a point to select it")
        self.count_label.setWordWrap(True)
        self.count_label.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 11px;")

        self.compare_button = QPushButton("Compare selected")
        self.compare_button.setToolTip(
            "Open every selected image side by side, with zoom and pan locked "
            "together."
        )
        self.compare_button.clicked.connect(self.compare_requested.emit)
        self.compare_button.setEnabled(False)

        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Empty the selection")
        self.clear_button.clicked.connect(lambda: self.selection_changed.emit([]))
        self.clear_button.setEnabled(False)

        self.reveal_button = QPushButton("Open in File Explorer")
        self.reveal_button.setToolTip(
            "Open the folder containing the current image, with the file "
            "itself selected."
        )
        self.reveal_button.clicked.connect(self._reveal_current)
        self.reveal_button.setEnabled(False)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(6)
        buttons.addWidget(self.compare_button, 1)
        buttons.addWidget(self.clear_button)

        # -- the list itself
        self.list_layout = QVBoxLayout()
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(6)
        self.list_layout.addStretch(1)
        list_container = QWidget()
        list_container.setLayout(self.list_layout)

        self.scroll = QScrollArea()
        self.scroll.setWidget(list_container)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.verticalScrollBar().valueChanged.connect(self._on_scrolled)

        self.empty_label = QLabel(
            "Click a point to select it.\n\n"
            "Shift-click to add more.\n"
            "Double-click to open the full image.\n"
            "Or use Lasso Analysis to select a region."
        )
        self.empty_label.setWordWrap(True)
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 11px;")

        # -- metadata for the current entry
        self.metadata = QLabel("")
        self.metadata.setWordWrap(True)
        self.metadata.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.metadata.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self.metadata.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        self.crop_button = QPushButton("Showing crops")
        self.crop_button.setCheckable(True)
        self.crop_button.setChecked(True)
        self.crop_button.setToolTip(
            "Crops show the centre of each field at higher magnification.\n"
            "Switch to whole fields to see the full frame in this column; "
            "double-clicking always opens the full image either way."
        )
        self.crop_button.toggled.connect(self._on_crop_toggled)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        layout.addWidget(self.heading)
        layout.addWidget(self.count_label)
        layout.addLayout(buttons)
        layout.addWidget(self.crop_button)
        layout.addWidget(self.reveal_button)
        layout.addWidget(self.empty_label)
        layout.addWidget(self.scroll, 1)
        layout.addWidget(self.metadata)
        self.setLayout(layout)
        self.scroll.hide()

    # -- wiring ------------------------------------------------------------

    def set_providers(self, describe, path_for) -> None:
        """Supply the callbacks that turn a row index into text and a file.

        Injected rather than reaching into the explorer, so the panel has no
        knowledge of datasets, resolvers or frames -- and can be tested with
        two lambdas.
        """
        self._describe = describe
        self._path_for = path_for

    # -- selection ---------------------------------------------------------

    def set_rows(self, rows) -> None:
        """Show ``rows`` as the selection, rebuilding only what changed."""
        rows = [int(r) for r in np.asarray(rows, dtype=np.int64).ravel()]
        if rows == self.rows:
            return
        self.rows = rows
        self._generation += 1
        self._clear_entries()

        count = len(rows)
        self.compare_button.setEnabled(count >= 2)
        self.clear_button.setEnabled(count > 0)
        self.compare_button.setText(
            f"Compare {count} images" if count >= 2 else "Compare selected"
        )

        if not count:
            self.count_label.setText("Click a point to select it")
            self.empty_label.show()
            self.scroll.hide()
            self.metadata.setText("")
            self._current_row = -1
            self.reveal_button.setEnabled(False)
            return

        self.empty_label.hide()
        self.scroll.show()
        self.count_label.setText(
            f"<b>{count:,}</b> point{'s' if count != 1 else ''} selected"
        )
        # The first entry is the one whose metadata is shown, so a single
        # click still behaves like the old hover preview.
        self._current_row = rows[0]
        self.metadata.setText(self._describe(rows[0]))
        self.reveal_button.setEnabled(True)
        self._build_more()

    def _clear_entries(self) -> None:
        for entry in self._entries.values():
            entry.setParent(None)
            entry.deleteLater()
        self._entries.clear()
        self._built = 0
        self.scroll.verticalScrollBar().setValue(0)

    def _build_more(self) -> None:
        """Create the next page of entries."""
        end = min(len(self.rows), self._built + PAGE_SIZE)
        for row in self.rows[self._built : end]:
            entry = SelectionEntry(row, self._short_caption(row))
            entry.set_image_size(self._image_px)
            entry.clicked.connect(self._on_entry_clicked)
            entry.activated.connect(self.row_activated.emit)
            entry.removed.connect(self._on_entry_removed)
            entry.set_current(row == self._current_row)
            # Before the stretch, which must stay last or the entries would
            # bunch against the top with a gap under them.
            self.list_layout.insertWidget(self.list_layout.count() - 1, entry)
            self._entries[row] = entry
            self._load(row)
        self._built = end
        if self._built < len(self.rows):
            self.count_label.setText(
                f"<b>{len(self.rows):,}</b> points selected · "
                f"showing {self._built}"
            )

    def _on_scrolled(self, value: int) -> None:
        """Build the next page as the end of the list comes into view."""
        if self._built >= len(self.rows):
            return
        bar = self.scroll.verticalScrollBar()
        if value >= bar.maximum() - self._image_px:
            self._build_more()

    def _short_caption(self, row: int) -> str:
        """A one-line label for an entry: the condition and where it is."""
        text = self._describe(row)
        # _describe returns rich text for the metadata block; the first line
        # of it is the condition, which is what an entry needs.
        first = text.split("<br>")[0] if text else ""
        return first or f"row {row}"

    # -- images ------------------------------------------------------------

    def _load(self, row: int) -> None:
        key = (row, self._cropped)
        cached = self._cache.get(key)
        entry = self._entries.get(row)
        if entry is None:
            return
        if cached is not None:
            self._cache.move_to_end(key)
            entry.set_pixmap(cached)
            return
        path = self._path_for(row)
        if path is None:
            entry.set_failed()
            entry.image.setText("image not found")
            return
        self._pool.start(
            _CropTask(
                self._generation,
                row,
                path,
                self._image_px,
                self._cropped,
                self._signals,
            )
        )

    def _on_ready(self, generation: int, row: int, pixmap: QPixmap) -> None:
        with self._lock:
            self._cache[(row, self._cropped)] = pixmap
            self._cache.move_to_end((row, self._cropped))
            while len(self._cache) > _CACHE_LIMIT:
                self._cache.popitem(last=False)
        if generation != self._generation:
            return
        entry = self._entries.get(row)
        if entry is not None:
            entry.set_pixmap(pixmap)

    def _on_failed(self, generation: int, row: int) -> None:
        if generation != self._generation:
            return
        entry = self._entries.get(row)
        if entry is not None:
            entry.set_failed()

    def _on_crop_toggled(self, cropped: bool) -> None:
        self._cropped = cropped
        self.crop_button.setText("Showing crops" if cropped else "Showing whole fields")
        self._generation += 1
        for row in list(self._entries):
            self._load(row)

    # -- geometry ----------------------------------------------------------

    def set_image_width(self, width: int) -> None:
        """Scale the entries to the column width.

        Called when the splitter moves. Widening the column is the gesture for
        "show me these bigger", so the images have to follow it -- a fixed
        preview size would make a wide column mostly empty margin.
        """
        px = max(MIN_IMAGE_PX, min(MAX_IMAGE_PX, width - 44))
        # Re-decoding on every pixel of a drag would be pointless work; only
        # a meaningful change is worth a reload.
        if abs(px - self._image_px) < 16:
            return
        self._image_px = px
        for entry in self._entries.values():
            entry.set_image_size(px)
        self._generation += 1
        # The cache is keyed by row, not by size, so it would serve the old
        # size back. Drop it and re-render at the new one.
        with self._lock:
            self._cache.clear()
        for row in list(self._entries):
            self._load(row)

    # -- entry interaction -------------------------------------------------

    def _on_entry_clicked(self, row: int) -> None:
        self.set_current(row)
        self.row_clicked.emit(row)

    def set_current(self, row: int) -> None:
        """Mark one entry as current and show its metadata."""
        self._current_row = int(row)
        for candidate, entry in self._entries.items():
            entry.set_current(candidate == self._current_row)
        self.metadata.setText(self._describe(self._current_row))
        self.reveal_button.setEnabled(self._current_row >= 0)

    def _on_entry_removed(self, row: int) -> None:
        remaining = [r for r in self.rows if r != int(row)]
        self.selection_changed.emit(remaining)

    # -- revealing on disk -------------------------------------------------

    def _reveal_current(self) -> None:
        """Open the file manager on the current entry's image."""
        if self._current_row < 0:
            return
        path = self._path_for(self._current_row)
        if path is None:
            self.reveal_failed.emit("no image on disk for this point")
            return
        message = reveal_in_file_manager(Path(path))
        if message:
            self.reveal_failed.emit(message)


def reveal_in_file_manager(path: Path) -> str:
    """Show ``path`` in the OS file manager, selected. Returns an error or "".

    Each platform has its own way of saying "open the folder AND highlight
    this file", and none of them is QDesktopServices, which can only open a
    folder (or worse, open the image in whatever application claims .tif).

    Falls back to opening the containing folder when the platform is unknown
    or the reveal command is missing, since that is still most of the value.
    """
    path = Path(path)
    if not path.exists():
        return f"file not found: {path}"

    import subprocess
    import sys

    try:
        if sys.platform == "win32":
            # /select, needs the native separator and no quoting of its own;
            # passing the argument string whole is what makes it work with
            # spaces in the path.
            subprocess.run(
                ["explorer", f"/select,{path}"],
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            # explorer.exe returns 1 even when it succeeds, so its exit code
            # says nothing and is deliberately not checked.
            return ""
        if sys.platform == "darwin":
            subprocess.run(["open", "-R", str(path)], check=True)
            return ""
        # Linux: the freedesktop interface, with a plain folder open as the
        # fallback for file managers that do not implement it.
        try:
            subprocess.run(
                [
                    "dbus-send",
                    "--session",
                    "--dest=org.freedesktop.FileManager1",
                    "--type=method_call",
                    "/org/freedesktop/FileManager1",
                    "org.freedesktop.FileManager1.ShowItems",
                    f"array:string:file://{path}",
                    "string:",
                ],
                check=True,
                capture_output=True,
            )
            return ""
        except (OSError, subprocess.CalledProcessError):
            subprocess.run(["xdg-open", str(path.parent)], check=True)
            return ""
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"could not open the file manager: {exc}"
