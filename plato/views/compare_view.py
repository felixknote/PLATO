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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..cache import read_plane, scale_to_uint8
from ..data.index.db import ImageRow
from ..gui.theme import ACCENT, TEXT, TEXT_MUTED

# Decode target for the comparison columns. Well above the ~900px a column
# actually gets on a wide screen, so the displayed image is still downscaled
# from more data than it shows, but far below the 2720px source -- which is
# what makes the 8-bit conversion cheap enough to step through fluidly.
_DISPLAY_TARGET_PX = 1200

# Cached pixmaps per column before the cache is dropped. The prefetch only
# needs the immediate neighbours; this leaves room to step a few frames each
# way without re-decoding, at roughly 1.8MB per entry.
_CACHE_ENTRIES = 12

_MAX_ZOOM = 20.0
_ZOOM_STEP = 1.25

# Narrowest a comparison column may become before the strip scrolls instead of
# shrinking further. Below roughly this, a micrograph stops being judgeable.
MIN_COLUMN_PX = 240


@dataclass(slots=True)
class Viewport:
    """The region every column shows, in normalised image coordinates.

    Shared by all columns rather than held per column: the view exists so that
    the only difference between columns is the compared variable, and columns
    showing different regions of their images would break exactly that. Stored
    normalised (0..1) so it survives images of differing pixel dimensions --
    slices can come from different plates.

    ``zoom`` is 1.0 for the whole image; ``cx``/``cy`` are the centre of the
    visible region.
    """

    zoom: float = 1.0
    cx: float = 0.5
    cy: float = 0.5

    @property
    def is_identity(self) -> bool:
        return self.zoom <= 1.0

    def reset(self) -> None:
        self.zoom, self.cx, self.cy = 1.0, 0.5, 0.5

    def clamp(self) -> None:
        """Keep the visible region inside the image."""
        self.zoom = max(1.0, min(_MAX_ZOOM, self.zoom))
        half = 0.5 / self.zoom
        self.cx = min(1.0 - half, max(half, self.cx))
        self.cy = min(1.0 - half, max(half, self.cy))

    def zoom_at(self, factor: float, fx: float, fy: float) -> None:
        """Scale by ``factor`` keeping the normalised point (fx, fy) put.

        Anchoring on the cursor is what makes wheel-zoom feel like zooming
        into the thing under the pointer rather than into the middle.
        """
        old_zoom = self.zoom
        self.zoom = max(1.0, min(_MAX_ZOOM, self.zoom * factor))
        if self.zoom == old_zoom:
            return
        # The point under the cursor keeps its image coordinate; solve for the
        # centre that holds it in place at the new zoom.
        image_x = self.cx + (fx - 0.5) / old_zoom
        image_y = self.cy + (fy - 0.5) / old_zoom
        self.cx = image_x - (fx - 0.5) / self.zoom
        self.cy = image_y - (fy - 0.5) / self.zoom
        self.clamp()

    def pan_by(self, dx: float, dy: float) -> None:
        """Shift the centre by a normalised delta of the *visible* region."""
        self.cx += dx / self.zoom
        self.cy += dy / self.zoom
        self.clamp()

    def crop_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        """The visible region as pixel bounds (left, top, right, bottom)."""
        half = 0.5 / self.zoom
        left = int(round((self.cx - half) * width))
        right = int(round((self.cx + half) * width))
        top = int(round((self.cy - half) * height))
        bottom = int(round((self.cy + half) * height))
        # At least one pixel each way, and never outside the image.
        left = max(0, min(width - 1, left))
        top = max(0, min(height - 1, top))
        right = max(left + 1, min(width, right))
        bottom = max(top + 1, min(height, bottom))
        return left, top, right, bottom


class ComparisonColumn(QWidget):
    """One slice: a title, one image, and the caption for what is shown."""

    clicked = Signal(object)
    # (factor, anchor_x, anchor_y) and (dx, dy), both normalised. Emitted
    # rather than applied directly: the viewport is shared, so the parent view
    # applies the change and refreshes every column together.
    zoomed = Signal(float, float, float)
    panned = Signal(float, float)

    def __init__(
        self,
        title: str,
        rows: list[ImageRow],
        viewport: Viewport | None = None,
        parent: QWidget | None = None,
        *,
        caption: str = "",
    ) -> None:
        super().__init__(parent)
        self.rows = rows
        self.position = 0
        # A fixed caption, for columns that are one chosen image rather than a
        # slice being stepped through. Empty means "describe the position",
        # which is what a dose series wants.
        self.fixed_caption = caption
        self._levels: tuple[float, float] | None = None
        self.autoscale = False
        self.selected = False
        # Shared with every other column so zoom stays locked across the view.
        self.viewport = viewport if viewport is not None else Viewport()
        # (row index, autoscale) -> 8-bit plane. The *plane* is cached rather
        # than the finished pixmap so that zooming and panning re-crop what is
        # already decoded instead of re-reading the file.
        self._cache: dict[tuple[int, bool], np.ndarray] = {}

        self.title = QLabel(f"<b>{title}</b>")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Where a drag started, in label coordinates; None when not panning.
        self._pan_origin: QPoint | None = None

        self.image = QLabel()
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(200, 200)
        self.image.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image.setObjectName("comparisonImage")
        # The image fills the column, so without this it swallows every click
        # and selecting a column by clicking its picture would not work. It
        # also carries the wheel-zoom and drag-pan for the shared viewport.
        self.image.installEventFilter(self)
        # Move events only arrive without a button held if tracking is on; the
        # pan needs them while the button *is* held, which tracking also covers.
        self.image.setMouseTracking(True)

        self.caption = QLabel("")
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption.setWordWrap(True)
        self.caption.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")

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
            f"ComparisonColumn {{ border: 2px solid {ACCENT}; border-radius: 8px; }}"
            if selected
            else "ComparisonColumn { border: 2px solid transparent; border-radius: 8px; }"
        )
        self.title.setStyleSheet(
            f"color: {TEXT};" if selected else f"color: {TEXT_MUTED};"
        )

    def mousePressEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        self.clicked.emit(self)
        super().mousePressEvent(event)

    def _displayed_rect(self) -> tuple[int, int, int, int] | None:
        """Where the pixmap actually sits inside the label (x, y, w, h).

        The pixmap is centred and aspect-fitted, so the label is generally
        larger than the image drawn in it. Zooming on the cursor needs the
        image's rectangle, not the label's, or the anchor drifts.
        """
        pixmap = self.image.pixmap()
        if pixmap is None or pixmap.isNull():
            return None
        label_w, label_h = self.image.width(), self.image.height()
        pix_w, pix_h = pixmap.width(), pixmap.height()
        if pix_w <= 0 or pix_h <= 0:
            return None
        return ((label_w - pix_w) // 2, (label_h - pix_h) // 2, pix_w, pix_h)

    def _normalised_at(self, pos: QPoint) -> tuple[float, float] | None:
        """Cursor position within the displayed image, as 0..1 fractions."""
        rect = self._displayed_rect()
        if rect is None:
            return None
        x, y, width, height = rect
        fx = (pos.x() - x) / width
        fy = (pos.y() - y) / height
        if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
            return None  # cursor is on the letterboxing, not the image
        return fx, fy

    def eventFilter(self, watched, event) -> bool:  # noqa: ANN001, N802 - Qt override
        if watched is not self.image:
            return super().eventFilter(watched, event)

        kind = event.type()
        if kind == QEvent.Type.MouseButtonPress:
            self.clicked.emit(self)
            if event.button() == Qt.MouseButton.LeftButton:
                self._pan_origin = event.position().toPoint()
        elif kind == QEvent.Type.MouseMove and self._pan_origin is not None:
            rect = self._displayed_rect()
            if rect is not None:
                _, _, width, height = rect
                point = event.position().toPoint()
                delta = point - self._pan_origin
                self._pan_origin = point
                # Dragging moves the image with the cursor, so the viewport
                # centre moves the opposite way.
                self.panned.emit(-delta.x() / width, -delta.y() / height)
        elif kind == QEvent.Type.MouseButtonRelease:
            self._pan_origin = None
        elif kind == QEvent.Type.Wheel:
            anchor = self._normalised_at(event.position().toPoint())
            if anchor is not None:
                steps = event.angleDelta().y() / 120.0
                if steps:
                    self.zoomed.emit(_ZOOM_STEP**steps, anchor[0], anchor[1])
                    return True
        return super().eventFilter(watched, event)

    def current_row(self) -> ImageRow | None:
        if not self.rows:
            return None
        return self.rows[self.position % len(self.rows)]

    def _plane(self, index: int) -> np.ndarray | None:
        """Decode one image to an 8-bit display plane, memoised.

        Two costs are avoided here. The obvious one is decoding the same file
        again every time you step back and forth. The larger one is that a
        2720x2720 uint16 plane costs ~118ms to scale to 8-bit, and the column
        displays it at roughly 900px -- so the plane is subsampled *before* the
        8-bit conversion, which is where nearly all the time was going.
        Measured: 127ms -> 41ms per image.

        How far it is subsampled depends on the zoom: at 1x the column shows
        the whole image and ~1200px is already more data than it can display,
        but zoomed in it shows a fraction of that, so the target scales with
        the zoom to keep real detail available instead of magnifying the
        subsampled pixels into mush. It is capped at the source resolution,
        which is the point past which there is nothing more to read.

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
        target = _DISPLAY_TARGET_PX * max(1.0, self.viewport.zoom)
        step = max(1, int(min(plane.shape) // target))
        if step > 1:
            plane = plane[::step, ::step]

        levels = None if self.autoscale else self._levels
        if levels is None:
            flat = plane.ravel().astype(np.float32)
            if flat.size > 50_000:
                flat = flat[:: flat.size // 50_000]
            levels = (float(np.percentile(flat, 1.0)), float(np.percentile(flat, 99.5)))
        u8 = scale_to_uint8(plane, levels)

        # Bounded so a long slice cannot grow without limit; a few entries each
        # side of the current position is all the prefetch needs.
        if len(self._cache) > _CACHE_ENTRIES:
            self._cache.clear()
        self._cache[key] = u8
        return u8

    def _render(self, index: int) -> QPixmap | None:
        """The visible region of image ``index`` as a pixmap."""
        u8 = self._plane(index)
        if u8 is None:
            return None
        if not self.viewport.is_identity:
            height, width = u8.shape
            left, top, right, bottom = self.viewport.crop_box(width, height)
            u8 = u8[top:bottom, left:right]
        # ascontiguousarray: a crop is a view with a row stride, and QImage
        # reads a flat buffer -- without this the image shears.
        u8 = np.ascontiguousarray(u8)
        height, width = u8.shape
        return QPixmap.fromImage(
            QImage(u8.tobytes(), width, height, width, QImage.Format.Format_Grayscale8)
        )

    def invalidate(self) -> None:
        """Drop decoded planes so they are re-read at the current zoom."""
        self._cache.clear()

    def prefetch_neighbours(self) -> None:
        """Decode the images either side of the current one.

        Stepping is the whole interaction here, so the next image should
        already be decoded by the time it is asked for. Called after the
        current image is on screen, so it never delays what you are looking at.
        Decodes the plane only -- cropping it to the viewport is cheap and
        happens when the image is actually shown.
        """
        for offset in (1, -1):
            self._plane(self.position + offset)

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
        zoom = "" if self.viewport.is_identity else f" · {self.viewport.zoom:.1f}×"
        if self.fixed_caption:
            self.caption.setText(f"{self.fixed_caption}{zoom}")
        else:
            self.caption.setText(
                f"{row.well} · field {row.field or '—'} · {self.position % n + 1}/{n}{zoom}"
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
        # One viewport shared by every column, so zoom and pan stay locked
        # together and the columns always show the same region.
        self.viewport = Viewport()
        # Needed for keyPressEvent to see the arrow keys at all.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.strip = QHBoxLayout()
        self.strip.setContentsMargins(0, 0, 0, 0)
        strip_container = QWidget()
        strip_container.setLayout(self.strip)

        # Columns are laid out in a scroll area rather than directly, because
        # the number of them is no longer bounded by a handful of slices: a
        # comparison of 30 selected points would otherwise divide the width
        # into 30 unreadable slivers. Below MIN_COLUMN_PX each they scroll
        # horizontally instead; at or above it they fill the width as before,
        # so the dose-series case looks exactly as it did.
        self.strip_scroll = QScrollArea()
        self.strip_scroll.setWidget(strip_container)
        self.strip_scroll.setWidgetResizable(True)
        self.strip_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.strip_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._strip_container = strip_container

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

        # Zoom is locked across columns by design (see Viewport), so these act
        # on the shared viewport and every column follows.
        self.zoom_in_button = QPushButton("＋")
        self.zoom_in_button.setToolTip("Zoom in (scroll wheel over an image, or +)")
        self.zoom_in_button.clicked.connect(lambda: self.zoom_by(_ZOOM_STEP))
        self.zoom_out_button = QPushButton("－")
        self.zoom_out_button.setToolTip("Zoom out (scroll wheel over an image, or -)")
        self.zoom_out_button.clicked.connect(lambda: self.zoom_by(1 / _ZOOM_STEP))
        self.reset_zoom_button = QPushButton("Reset zoom")
        self.reset_zoom_button.setToolTip("Back to the whole image (0)")
        self.reset_zoom_button.clicked.connect(self.reset_zoom)

        controls = QHBoxLayout()
        controls.addWidget(self.prev_button)
        controls.addWidget(self.next_button)
        controls.addWidget(self.realign_button)
        controls.addWidget(self.autoscale_box)
        controls.addWidget(self.zoom_out_button)
        controls.addWidget(self.zoom_in_button)
        controls.addWidget(self.reset_zoom_button)
        controls.addWidget(self.position_label, 1)
        controls.addWidget(self.export_button)

        layout = QVBoxLayout()
        layout.addWidget(self.strip_scroll, 1)
        layout.addLayout(controls)
        self.setLayout(layout)

    def set_columns(self, columns: list[ComparisonColumn]) -> None:
        for existing in self.columns:
            existing.setParent(None)
            existing.deleteLater()
        self.columns = columns
        # A new set of columns is a new comparison; carrying the old zoom over
        # would frame a region chosen for images that are no longer shown.
        self.viewport.reset()
        for column in columns:
            column.viewport = self.viewport
            column.clicked.connect(self.select_column)
            column.zoomed.connect(self._on_zoomed)
            column.panned.connect(self._on_panned)
            column.set_autoscale(self.autoscale_box.isChecked())
            self.strip.addWidget(column, 1)
        if columns:
            self.select_column(columns[0])
        self._fit_strip()
        # Stepping, realigning and the position readout are all about moving
        # within a slice. When every column holds exactly one image there is
        # nowhere to step, and the controls would be dead weight.
        steppable = any(len(c.rows) > 1 for c in columns)
        for widget in (
            self.prev_button,
            self.next_button,
            self.realign_button,
            self.position_label,
        ):
            widget.setVisible(steppable)
        self.refresh()

    def _fit_strip(self) -> None:
        """Give every column at least MIN_COLUMN_PX, scrolling if need be."""
        count = len(self.columns)
        if not count:
            self._strip_container.setMinimumWidth(0)
            return
        needed = count * MIN_COLUMN_PX
        # Only force a minimum when the columns would otherwise be too narrow;
        # leaving it at 0 lets them stretch to fill a wide window.
        self._strip_container.setMinimumWidth(
            needed if needed > self.strip_scroll.viewport().width() else 0
        )

    def resizeEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        super().resizeEvent(event)
        # The threshold between "fill the width" and "scroll" depends on the
        # width, so it has to be re-evaluated when the window changes.
        self._fit_strip()

    # -- zoom -------------------------------------------------------------

    def _on_zoomed(self, factor: float, anchor_x: float, anchor_y: float) -> None:
        before = self.viewport.zoom
        self.viewport.zoom_at(factor, anchor_x, anchor_y)
        if self.viewport.zoom != before:
            # Crossing to a new zoom level wants more (or less) source detail
            # than the cached planes were decoded at.
            self._invalidate_columns()
        self._show_columns()

    def _on_panned(self, dx: float, dy: float) -> None:
        if self.viewport.is_identity:
            return  # nothing to pan when the whole image is already shown
        self.viewport.pan_by(dx, dy)
        self._show_columns()

    def zoom_by(self, factor: float) -> None:
        """Zoom about the centre, for the buttons and keyboard."""
        self._on_zoomed(factor, 0.5, 0.5)

    def reset_zoom(self) -> None:
        self.viewport.reset()
        self._invalidate_columns()
        self._show_columns()
        self.status.emit("zoom reset")

    def _invalidate_columns(self) -> None:
        for column in self.columns:
            column.invalidate()

    def _show_columns(self) -> None:
        for column in self.columns:
            column.show_current()
        self._update_position_label()

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
        key = event.key()
        if key == Qt.Key.Key_Left:
            self.step_selected(-1)
        elif key == Qt.Key.Key_Right:
            self.step_selected(1)
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_by(_ZOOM_STEP)
        elif key == Qt.Key.Key_Minus:
            self.zoom_by(1 / _ZOOM_STEP)
        elif key == Qt.Key.Key_0:
            self.reset_zoom()
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
        zoom = "" if self.viewport.is_identity else f"   ·   {self.viewport.zoom:.1f}× zoom"
        self.position_label.setText(
            f"image {column.position + 1} of {len(column.rows)} - {title}{suffix}{zoom}"
        )

    def visible_rows(self) -> list[ImageRow]:
        """Exactly the images currently on screen, one per column."""
        rows = [c.current_row() for c in self.columns]
        return [r for r in rows if r is not None]
