"""Side-by-side detail comparison: one image per value of a variable.

Each column is one slice (one concentration, one timepoint, ...) showing a
single full image rather than a grid of thumbnails. Stepping moves every
column at once, so position N of each slice is on screen together and the
only thing differing between columns is the compared variable.

The grid view answers "what is in this slice"; this answers "how does the
slice change along the variable", which is the comparison a dose series or a
timepoint course is actually for.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..cache import read_plane, scale_to_uint8
from ..index.db import ImageRow

# Decode target for the comparison columns. Well above the ~900px a column
# actually gets on a wide screen, so the displayed image is still downscaled
# from more data than it shows, but far below the 2720px source -- which is
# what makes the 8-bit conversion cheap enough to step through fluidly.
_DISPLAY_TARGET_PX = 1200

# Cached pixmaps per column before the cache is dropped. The prefetch only
# needs the immediate neighbours; this leaves room to step a few frames each
# way without re-decoding, at roughly 1.8MB per entry.
_CACHE_ENTRIES = 12


class ComparisonColumn(QWidget):
    """One slice: a title, one image, and the caption for what is shown."""

    clicked = Signal(object)

    def __init__(self, title: str, rows: list[ImageRow], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.rows = rows
        self.position = 0
        self._levels: tuple[float, float] | None = None
        self.autoscale = False
        self.selected = False
        # (row index, autoscale) -> decoded pixmap.
        self._cache: dict[tuple[int, bool], QPixmap] = {}

        self.title = QLabel(f"<b>{title}</b>")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.image = QLabel()
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(200, 200)
        self.image.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image.setStyleSheet("background: #111;")
        # The image fills the column, so without this it swallows every click
        # and selecting a column by clicking its picture would not work.
        self.image.installEventFilter(self)

        self.caption = QLabel("")
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption.setWordWrap(True)

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.title)
        layout.addWidget(self.image, 1)
        layout.addWidget(self.caption)
        self.setLayout(layout)

    def set_levels(self, levels: tuple[float, float] | None) -> None:
        """Display limits for this column's channel, or None to autoscale."""
        self._levels = levels

    def set_autoscale(self, enabled: bool) -> None:
        self.autoscale = enabled
        self.show_current()

    def set_selected(self, selected: bool) -> None:
        """Frame this column as the one the arrow keys drive."""
        self.selected = selected
        self.setStyleSheet(
            "ComparisonColumn { border: 2px solid palette(highlight); }"
            if selected
            else "ComparisonColumn { border: 2px solid transparent; }"
        )

    def mousePressEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        self.clicked.emit(self)
        super().mousePressEvent(event)

    def eventFilter(self, watched, event) -> bool:  # noqa: ANN001, N802 - Qt override
        if watched is self.image and event.type() == QEvent.Type.MouseButtonPress:
            self.clicked.emit(self)
        return super().eventFilter(watched, event)

    def current_row(self) -> ImageRow | None:
        if not self.rows:
            return None
        return self.rows[self.position % len(self.rows)]

    def _render(self, index: int) -> QPixmap | None:
        """Decode one image to a display-sized pixmap, memoised.

        Two costs are avoided here. The obvious one is decoding the same file
        again every time you step back and forth. The larger one is that a
        2720x2720 uint16 plane costs ~118ms to scale to 8-bit, and the column
        displays it at roughly 900px -- so the plane is subsampled to ~1200px
        *before* the 8-bit conversion, which is where nearly all the time was
        going. Measured: 127ms -> 41ms per image.

        Subsampling here is display-only. The full-resolution viewer and every
        export path read their own planes and are unaffected.
        """
        if not self.rows:
            return None
        index %= len(self.rows)
        key = (index, self.autoscale)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        row = self.rows[index]
        plane = read_plane(Path(row.path))
        step = max(1, min(plane.shape) // _DISPLAY_TARGET_PX)
        if step > 1:
            plane = plane[::step, ::step]

        levels = None if self.autoscale else self._levels
        if levels is None:
            flat = plane.ravel().astype(np.float32)
            if flat.size > 50_000:
                flat = flat[:: flat.size // 50_000]
            levels = (float(np.percentile(flat, 1.0)), float(np.percentile(flat, 99.5)))
        u8 = scale_to_uint8(plane, levels)
        height, width = u8.shape
        pixmap = QPixmap.fromImage(
            QImage(u8.tobytes(), width, height, width, QImage.Format.Format_Grayscale8)
        )

        # Bounded so a long slice cannot grow without limit; a few entries each
        # side of the current position is all the prefetch needs.
        if len(self._cache) > _CACHE_ENTRIES:
            self._cache.clear()
        self._cache[key] = pixmap
        return pixmap

    def prefetch_neighbours(self) -> None:
        """Decode the images either side of the current one.

        Stepping is the whole interaction here, so the next image should
        already be decoded by the time it is asked for. Called after the
        current image is on screen, so it never delays what you are looking at.
        """
        for offset in (1, -1):
            self._render(self.position + offset)

    def show_current(self) -> None:
        row = self.current_row()
        if row is None:
            self.image.setText("no images in this slice")
            self.caption.setText("")
            return
        pixmap = self._render(self.position)
        if pixmap is not None:
            self.image.setPixmap(
                pixmap.scaled(
                    self.image.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        n = len(self.rows)
        self.caption.setText(
            f"{row.well} · field {row.field or '—'} · {self.position % n + 1}/{n}"
        )

    def resizeEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        super().resizeEvent(event)
        self.show_current()


class ComparisonView(QWidget):
    """A row of ComparisonColumns stepped together."""

    status = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.columns: list[ComparisonColumn] = []
        # Needed for keyPressEvent to see the arrow keys at all.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.strip = QHBoxLayout()
        self.strip.setContentsMargins(0, 0, 0, 0)
        strip_container = QWidget()
        strip_container.setLayout(self.strip)

        self.prev_button = QPushButton("◀ Previous")
        self.next_button = QPushButton("Next ▶")
        self.position_label = QLabel("")
        self.export_button = QPushButton("Export what's on screen…")
        self.autoscale_box = QCheckBox("Autoscale each image")
        self.autoscale_box.setToolTip(
            "Off (default) = every column shares the fixed per-channel contrast, "
            "so a difference between columns is a difference in the sample. "
            "On = each image is stretched to its own range, which recovers faint "
            "detail but makes columns no longer directly comparable."
        )
        self.autoscale_box.toggled.connect(self.set_autoscale)
        # The buttons step every column together; the arrow keys step only the
        # selected one. Both are useful: locked stepping keeps slices aligned,
        # independent stepping lets you hunt for a comparable field in one
        # slice when its Nth image happens to be junk.
        self.prev_button.clicked.connect(lambda: self.step_all(-1))
        self.next_button.clicked.connect(lambda: self.step_all(1))

        # Once the arrow keys have pulled columns out of step there is no way
        # back to a like-for-like view without stepping each one by hand.
        self.realign_button = QPushButton("Realign")
        self.realign_button.setToolTip("Move every column to the selected column's position.")
        self.realign_button.clicked.connect(self.realign)

        controls = QHBoxLayout()
        controls.addWidget(self.prev_button)
        controls.addWidget(self.next_button)
        controls.addWidget(self.realign_button)
        controls.addWidget(self.autoscale_box)
        controls.addWidget(self.position_label, 1)
        controls.addWidget(self.export_button)

        layout = QVBoxLayout()
        layout.addWidget(strip_container, 1)
        layout.addLayout(controls)
        self.setLayout(layout)

    def set_columns(self, columns: list[ComparisonColumn]) -> None:
        for existing in self.columns:
            existing.setParent(None)
            existing.deleteLater()
        self.columns = columns
        for column in columns:
            column.clicked.connect(self.select_column)
            column.set_autoscale(self.autoscale_box.isChecked())
            self.strip.addWidget(column, 1)
        if columns:
            self.select_column(columns[0])
        self.refresh()

    def select_column(self, column: ComparisonColumn) -> None:
        for candidate in self.columns:
            candidate.set_selected(candidate is column)
        # Clicking a child would otherwise leave focus there and the arrow keys
        # would stop reaching keyPressEvent after the first selection.
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self._update_position_label()

    def selected_column(self) -> ComparisonColumn | None:
        for column in self.columns:
            if column.selected:
                return column
        return self.columns[0] if self.columns else None

    def set_autoscale(self, enabled: bool) -> None:
        for column in self.columns:
            column.set_autoscale(enabled)

    def step_all(self, delta: int) -> None:
        for column in self.columns:
            if column.rows:
                column.position = (column.position + delta) % len(column.rows)
        self.refresh()

    def step_selected(self, delta: int) -> None:
        column = self.selected_column()
        if column is not None and column.rows:
            column.position = (column.position + delta) % len(column.rows)
            column.show_current()
            self._update_position_label()
            QTimer.singleShot(0, column.prefetch_neighbours)

    def realign(self) -> None:
        """Put every column back on the selected column's position."""
        column = self.selected_column()
        if column is None:
            return
        for candidate in self.columns:
            if candidate.rows:
                candidate.position = column.position % len(candidate.rows)
        self.refresh()

    def keyPressEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        if event.key() == Qt.Key.Key_Left:
            self.step_selected(-1)
        elif event.key() == Qt.Key.Key_Right:
            self.step_selected(1)
        else:
            super().keyPressEvent(event)

    def refresh(self) -> None:
        for column in self.columns:
            column.show_current()
        self._update_position_label()
        # After every column is on screen, not before -- prefetching first
        # would delay the images the user is waiting for.
        QTimer.singleShot(0, self._prefetch)

    def _prefetch(self) -> None:
        for column in self.columns:
            column.prefetch_neighbours()

    def _update_position_label(self) -> None:
        column = self.selected_column()
        if column is None or not column.rows:
            self.position_label.setText("no images")
            return
        # Columns can be stepped independently, so the label reports the
        # selected one and flags when the others are no longer aligned with it.
        positions = {c.position for c in self.columns if c.rows}
        aligned = len(positions) == 1
        title = column.title.text().replace("<b>", "").replace("</b>", "")
        suffix = "" if aligned else "   (columns not aligned)"
        self.position_label.setText(
            f"image {column.position + 1} of {len(column.rows)} - {title}{suffix}"
        )

    def visible_rows(self) -> list[ImageRow]:
        """Exactly the images currently on screen, one per column."""
        rows = [c.current_row() for c in self.columns]
        return [r for r in rows if r is not None]
