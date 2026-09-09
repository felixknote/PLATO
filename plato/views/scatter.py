"""The embedding scatter plot: draw, hover, click, export.

Built on pyqtgraph rather than matplotlib because this plot is interactive
first and a figure second. ``ScatterPlotItem`` handles tens of thousands of
points at interactive frame rates, and pyqtgraph ships a real vector SVG
exporter, so PNG and SVG both come from the same live scene -- the SVG is
actual vector geometry, not a raster wrapped in an <image> tag.

Points are drawn in one item per colour group rather than one item per point:
pyqtgraph batches a group into a single draw call, so 32k points in 20 groups
is 20 draws, while 32k individually styled points is not usable.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QGraphicsRectItem, QVBoxLayout, QWidget

from ..gui.theme import BORDER, IMAGE_BACKGROUND, TEXT, TEXT_MUTED

pg.setConfigOption("background", IMAGE_BACKGROUND)
pg.setConfigOption("foreground", TEXT_MUTED)
pg.setConfigOption("antialias", True)

# Radius in pixels within which the cursor counts as being over a point.
HOVER_RADIUS_PX = 12

# Above this many points, drop the per-point outline. Outlines are what make a
# sparse plot readable and what makes a dense one a grey smear, because at 30k
# points the 1 px border is most of the mark.
OUTLINE_LIMIT = 4000


class EmbeddingScatter(QWidget):
    """A 2-D scatter with grouped colours, hover reporting and box selection."""

    # Row index into the explorer frame, or -1 when the cursor leaves a point.
    point_hovered = Signal(int)
    point_clicked = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.plot = pg.PlotWidget()
        self.plot.setMenuEnabled(False)
        self.plot.showGrid(x=False, y=False)
        self.plot.setLabel("bottom", "1")
        self.plot.setLabel("left", "2")
        self.plot.getViewBox().setAspectLocked(True)
        # Points carry no meaningful axis units -- they are an arbitrary
        # embedding basis -- so the numbers would invite over-reading.
        for axis in ("bottom", "left"):
            self.plot.getAxis(axis).setStyle(showValues=False)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)
        self.setLayout(layout)

        # Coordinates and row indices of everything currently drawn, in one
        # flat array so hit-testing is a single vectorised distance query
        # rather than a walk over the scatter items.
        self._coords = np.empty((0, 2), dtype=np.float32)
        self._rows = np.empty(0, dtype=np.int64)
        self._items: list[pg.ScatterPlotItem] = []
        self._hovered = -1

        self._highlight = pg.ScatterPlotItem(
            size=16,
            pen=pg.mkPen(TEXT, width=2),
            brush=pg.mkBrush(None),
        )
        self._highlight.setZValue(100)
        self.plot.addItem(self._highlight)
        self._highlight.hide()

        self.legend: pg.LegendItem | None = None

        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.plot.scene().sigMouseClicked.connect(self._on_mouse_clicked)

    # -- drawing ----------------------------------------------------------

    def set_points(
        self,
        coords: np.ndarray,
        row_indices: np.ndarray,
        colours: list[str],
        *,
        groups: dict[str, np.ndarray] | None = None,
        point_size: int = 6,
        opacity: float = 0.85,
        legend_entries: list[tuple[str, str]] | None = None,
        reset_view: bool = True,
    ) -> None:
        """Draw ``coords``, coloured per point, batched by colour group.

        Args:
            coords: (M, 2) plot coordinates.
            row_indices: (M,) index of each point into the explorer frame.
            colours: hex colour per point.
            groups: optional colour -> positional mask, to skip regrouping.
            legend_entries: (label, colour) pairs; None hides the legend.
        """
        self._clear_items()
        self._coords = np.ascontiguousarray(coords, dtype=np.float32)
        self._rows = np.ascontiguousarray(row_indices, dtype=np.int64)
        self._hovered = -1
        self._highlight.hide()

        if groups is None:
            groups = {}
            palette = np.asarray(colours)
            for colour in dict.fromkeys(colours):
                groups[colour] = np.flatnonzero(palette == colour)

        outline = (
            pg.mkPen(IMAGE_BACKGROUND, width=0.5)
            if len(self._coords) <= OUTLINE_LIMIT
            else None
        )
        alpha = int(max(0.0, min(1.0, opacity)) * 255)

        for colour, mask in groups.items():
            if len(mask) == 0:
                continue
            brush_colour = QColor(colour)
            brush_colour.setAlpha(alpha)
            item = pg.ScatterPlotItem(
                x=self._coords[mask, 0],
                y=self._coords[mask, 1],
                size=point_size,
                brush=pg.mkBrush(brush_colour),
                pen=outline,
                # Hit-testing is done here, not by pyqtgraph: its own hover
                # machinery builds a QPainterPath per point, which is what
                # makes a 30k-point plot stutter on every mouse move.
                hoverable=False,
            )
            item.setZValue(0)
            self.plot.addItem(item)
            self._items.append(item)

        self._set_legend(legend_entries)
        if reset_view:
            self.plot.getViewBox().autoRange(padding=0.05)

    def _clear_items(self) -> None:
        for item in self._items:
            self.plot.removeItem(item)
        self._items.clear()

    def _set_legend(self, entries: list[tuple[str, str]] | None) -> None:
        if self.legend is not None:
            self.legend.scene().removeItem(self.legend)
            self.legend = None
        if not entries:
            return
        self.legend = pg.LegendItem(
            offset=(-14, 14),
            labelTextColor=TEXT,
            brush=pg.mkBrush(QColor(30, 33, 39, 220)),
            pen=pg.mkPen(BORDER),
            verSpacing=-4,
        )
        self.legend.setParentItem(self.plot.getPlotItem())
        for label, colour in entries:
            sample = pg.ScatterPlotItem(
                size=9, brush=pg.mkBrush(QColor(colour)), pen=None
            )
            self.legend.addItem(sample, f" {label}")

    # -- interaction ------------------------------------------------------

    def _nearest(self, scene_position: QPointF) -> int:
        """Positional index of the point under the cursor, or -1."""
        if len(self._coords) == 0:
            return -1
        view = self.plot.getViewBox()
        data_point = view.mapSceneToView(scene_position)
        deltas = self._coords - np.array(
            [data_point.x(), data_point.y()], dtype=np.float32
        )
        distances = np.einsum("ij,ij->i", deltas, deltas)
        best = int(np.argmin(distances))

        # The threshold is in pixels, so convert it through the current zoom:
        # a fixed data-space radius would grab half the plot when zoomed out
        # and nothing when zoomed in.
        pixel_size = view.viewPixelSize()[0] or 1e-9
        limit = (HOVER_RADIUS_PX * pixel_size) ** 2
        return best if distances[best] <= limit else -1

    def _on_mouse_moved(self, position) -> None:
        index = self._nearest(position)
        if index == self._hovered:
            return
        self._hovered = index
        if index < 0:
            self._highlight.hide()
            self.point_hovered.emit(-1)
            return
        self._highlight.setData(
            x=[float(self._coords[index, 0])], y=[float(self._coords[index, 1])]
        )
        self._highlight.show()
        self.point_hovered.emit(int(self._rows[index]))

    def _on_mouse_clicked(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        index = self._nearest(event.scenePos())
        if index >= 0:
            event.accept()
            self.point_clicked.emit(int(self._rows[index]))

    def reset_view(self) -> None:
        self.plot.getViewBox().autoRange(padding=0.05)

    # -- export -----------------------------------------------------------

    def export(self, path: str, *, width: int = 1600) -> None:
        """Write the current scene to PNG or SVG, by file extension.

        Both come from the live scene, so whatever is on screen -- method,
        colouring, filtering, zoom and legend -- is what lands in the file.
        """
        import pyqtgraph.exporters as exporters

        plot_item = self.plot.getPlotItem()
        if path.lower().endswith(".svg"):
            exporter = exporters.SVGExporter(plot_item)
        else:
            exporter = exporters.ImageExporter(plot_item)
            # Set width only, and let the exporter derive height from the
            # scene's aspect ratio; setting both distorts the plot.
            exporter.parameters()["width"] = width
        exporter.export(path)
