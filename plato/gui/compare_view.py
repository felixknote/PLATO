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
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..cache import read_plane, scale_to_uint8
from ..index.db import ImageRow


class ComparisonColumn(QWidget):
    """One slice: a title, one image, and the caption for what is shown."""

    def __init__(self, title: str, rows: list[ImageRow], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.rows = rows
        self.position = 0
        self._levels: tuple[float, float] | None = None

        self.title = QLabel(f"<b>{title}</b>")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.image = QLabel()
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(200, 200)
        self.image.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image.setStyleSheet("background: #111;")

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

    def current_row(self) -> ImageRow | None:
        if not self.rows:
            return None
        return self.rows[self.position % len(self.rows)]

    def show_current(self) -> None:
        row = self.current_row()
        if row is None:
            self.image.setText("no images in this slice")
            self.caption.setText("")
            return
        plane = read_plane(Path(row.path))
        levels = self._levels
        if levels is None:
            flat = plane.ravel().astype(np.float32)
            if flat.size > 50_000:
                flat = flat[:: flat.size // 50_000]
            levels = (float(np.percentile(flat, 1.0)), float(np.percentile(flat, 99.5)))
        u8 = scale_to_uint8(plane, levels)
        height, width = u8.shape
        image = QImage(u8.tobytes(), width, height, width, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(image).scaled(
            self.image.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image.setPixmap(pixmap)
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

        self.strip = QHBoxLayout()
        self.strip.setContentsMargins(0, 0, 0, 0)
        strip_container = QWidget()
        strip_container.setLayout(self.strip)

        self.prev_button = QPushButton("◀ Previous")
        self.next_button = QPushButton("Next ▶")
        self.position_label = QLabel("")
        self.export_button = QPushButton("Export what's on screen…")
        self.prev_button.clicked.connect(lambda: self.step(-1))
        self.next_button.clicked.connect(lambda: self.step(1))

        controls = QHBoxLayout()
        controls.addWidget(self.prev_button)
        controls.addWidget(self.next_button)
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
            self.strip.addWidget(column, 1)
        self.refresh()

    def step(self, delta: int) -> None:
        for column in self.columns:
            if column.rows:
                column.position = (column.position + delta) % len(column.rows)
        self.refresh()

    def refresh(self) -> None:
        for column in self.columns:
            column.show_current()
        depth = max((len(c.rows) for c in self.columns), default=0)
        current = (self.columns[0].position + 1) if self.columns and self.columns[0].rows else 0
        self.position_label.setText(
            f"image {current} of {depth} per slice" if depth else "no images"
        )

    def visible_rows(self) -> list[ImageRow]:
        """Exactly the images currently on screen, one per column."""
        rows = [c.current_row() for c in self.columns]
        return [r for r in rows if r is not None]
