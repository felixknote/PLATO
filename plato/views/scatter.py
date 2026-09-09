"""The embedding scatter plot: draw, hover, click, export.

Built on pyqtgraph rather than matplotlib because this plot is interactive
first and a figure second. ``ScatterPlotItem`` handles tens of thousands of
points at interactive frame rates, and pyqtgraph ships a real vector SVG
exporter, so PNG and SVG both come from the same live scene -- the SVG is
actual vector geometry, not a raster wrapped in an <image> tag.

Points are drawn in one item per colour group rather than one item per point:
pyqtgraph batches a group into a single draw call, so 32k points in 20 groups
is 20 draws, while 32k individually styled points is not usable.

Two things a plain scatter cannot do, both added here:

*Lasso selection* -- freehand a region and get back the rows inside it. Hover
and click answer "what is this point?"; a lasso answers "what is this
cluster?", which is the question a projection actually poses.

*Density mode* -- at these point counts the scatter overplots badly. For
32k points at the default 6 px mark, the marks alone cover ~38% of the plot
area before clustering concentrates them, so a dense cluster is a solid blob
in which structure and count are both invisible. Density mode replaces the
marks with a kernel density estimate, where shade means how many rather than
merely "at least one".
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainterPath, QPolygonF
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

# Resolution of the density grid. 320 is a little under a typical plot's pixel
# width, so the estimate is finer than the screen without paying for detail
# nothing can display.
DENSITY_GRID = 320

# Kernel width as a fraction of the plot's extent. Wide enough to merge the
# grain of individual points, narrow enough to keep clusters apart.
DENSITY_SIGMA = 0.012

# A lasso shorter than this many points is a stray click-drag, not a selection.
MIN_LASSO_POINTS = 4


def _density_lookup_table() -> np.ndarray:
    """Background -> blue -> cyan -> amber, as a 256-entry RGB table.

    Starts at the plot's own background so empty regions read as empty rather
    than as the low end of a scale, and rises through hues that stay distinct
    in both lightness and chroma.
    """
    global _DENSITY_LUT
    if _DENSITY_LUT is not None:
        return _DENSITY_LUT
    stops = [
        (0.00, QColor(IMAGE_BACKGROUND)),
        (0.15, QColor("#16314f")),
        (0.45, QColor("#2f7fb5")),
        (0.75, QColor("#5bc0d4")),
        (1.00, QColor("#e8a33d")),
    ]
    table = np.zeros((256, 3), dtype=np.ubyte)
    for i in range(256):
        t = i / 255.0
        for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
            if t0 <= t <= t1:
                f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                table[i] = [
                    round(c0.red() + (c1.red() - c0.red()) * f),
                    round(c0.green() + (c1.green() - c0.green()) * f),
                    round(c0.blue() + (c1.blue() - c0.blue()) * f),
                ]
                break
    _DENSITY_LUT = table
    return table


_DENSITY_LUT: np.ndarray | None = None


class EmbeddingScatter(QWidget):
    """A 2-D scatter with grouped colours, hover reporting and box selection."""

    # Row index into the explorer frame, or -1 when the cursor leaves a point.
    point_hovered = Signal(int)
    point_clicked = Signal(int)
    # Row indices inside a freehand lasso; empty when a selection is cleared.
    points_selected = Signal(object)

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

        # -- lasso
        self.lasso_enabled = False
        self._lasso_points: list[tuple[float, float]] = []
        self._lasso_curve = pg.PlotCurveItem(
            pen=pg.mkPen(TEXT, width=1.5, style=Qt.PenStyle.DashLine)
        )
        self._lasso_curve.setZValue(90)
        self.plot.addItem(self._lasso_curve)

        # Points inside the last lasso, drawn over everything so the selection
        # stays visible against whatever it was drawn on top of.
        self._selection = pg.ScatterPlotItem(
            size=7, brush=pg.mkBrush(QColor(TEXT)), pen=None
        )
        self._selection.setZValue(95)
        self.plot.addItem(self._selection)
        self._selection.hide()
        self.selected_rows = np.empty(0, dtype=np.int64)

        # -- density
        self.density_enabled = False
        self._density_image = pg.ImageItem()
        self._density_image.setZValue(-10)
        self.plot.addItem(self._density_image)
        self._density_image.hide()

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

        # A selection indexes rows that may not be on screen any more.
        self.selected_rows = np.empty(0, dtype=np.int64)
        self._selection.hide()
        self._lasso_curve.setData([], [])

        for item in self._items:
            item.setVisible(not self.density_enabled)
        if self.density_enabled:
            self._render_density()

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

    # -- density ----------------------------------------------------------

    def set_density(self, enabled: bool) -> None:
        """Swap between drawing every mark and drawing their density."""
        self.density_enabled = enabled
        for item in self._items:
            item.setVisible(not enabled)
        self._density_image.setVisible(enabled)
        if enabled:
            self._render_density()

    def _render_density(self) -> None:
        """Estimate and draw point density over the plot's extent.

        Bin-then-blur, not a per-point kernel sum. Evaluating a Gaussian from
        every point to every grid cell is O(points x cells) and takes seconds;
        binning to a grid and convolving once is O(cells) after the histogram
        and is what the fast KDE literature (Heer 2021, as used by Embedding
        Atlas) does. Measured here on 32k points at a 320x320 grid: 5-18 ms
        depending on kernel width, against 8.6 s for scipy's exact KDE on a
        2k-point subsample. At display resolution the two are
        indistinguishable, because the screen cannot show detail finer than
        the grid anyway.
        """
        if len(self._coords) == 0:
            self._density_image.hide()
            return

        x, y = self._coords[:, 0], self._coords[:, 1]
        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())
        # A degenerate extent (every point identical) would make a zero-width
        # bin range, which numpy rejects.
        if x_max <= x_min:
            x_min, x_max = x_min - 0.5, x_max + 0.5
        if y_max <= y_min:
            y_min, y_max = y_min - 0.5, y_max + 0.5

        counts, _, _ = np.histogram2d(
            x, y, bins=DENSITY_GRID, range=[[x_min, x_max], [y_min, y_max]]
        )

        from scipy.ndimage import gaussian_filter

        sigma = max(1.0, DENSITY_SIGMA * DENSITY_GRID)
        density = gaussian_filter(counts, sigma=sigma, mode="constant")

        peak = float(density.max())
        if peak <= 0:
            self._density_image.hide()
            return
        # Square root, not linear: a few dense cores otherwise flatten
        # everything else to black, and the sparse structure between clusters
        # is exactly what the mode exists to reveal.
        normalised = np.sqrt(density / peak)

        # histogram2d returns [x][y]; ImageItem in row-major mode reads
        # [row][col] = [y][x], so without the transpose the density is drawn
        # rotated relative to the points it describes.
        self._density_image.setImage(
            normalised.T, autoLevels=False, levels=(0.0, 1.0)
        )
        self._density_image.setRect(x_min, y_min, x_max - x_min, y_max - y_min)
        self._density_image.setLookupTable(_density_lookup_table())
        self._density_image.show()

    # -- lasso ------------------------------------------------------------

    def set_lasso(self, enabled: bool) -> None:
        """Turn freehand selection on. Panning is disabled while it is."""
        self.lasso_enabled = enabled
        self._lasso_points = []
        self._lasso_curve.setData([], [])
        # The view box would otherwise pan under the same drag that draws.
        self.plot.getViewBox().setMouseEnabled(x=not enabled, y=not enabled)
        self.plot.setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        )

    def clear_selection(self) -> None:
        self.selected_rows = np.empty(0, dtype=np.int64)
        self._selection.hide()
        self._lasso_curve.setData([], [])
        self.points_selected.emit(self.selected_rows)

    def _lasso_move(self, position) -> None:
        point = self.plot.getViewBox().mapSceneToView(position)
        self._lasso_points.append((point.x(), point.y()))
        if len(self._lasso_points) >= 2:
            path = np.asarray(self._lasso_points)
            self._lasso_curve.setData(path[:, 0], path[:, 1])

    def _lasso_finish(self) -> None:
        """Close the path and report the rows inside it."""
        path_points = self._lasso_points
        self._lasso_points = []
        if len(path_points) < MIN_LASSO_POINTS or len(self._coords) == 0:
            self._lasso_curve.setData([], [])
            return

        polygon = QPolygonF([QPointF(px, py) for px, py in path_points])
        path = QPainterPath()
        path.addPolygon(polygon)
        path.closeSubpath()

        # Bounding box first: containsPoint is a real geometric test, and
        # running it on every one of 32k points is far slower than discarding
        # the ~90% that cannot possibly be inside.
        bounds = path.boundingRect()
        inside_box = (
            (self._coords[:, 0] >= bounds.left())
            & (self._coords[:, 0] <= bounds.right())
            & (self._coords[:, 1] >= bounds.top())
            & (self._coords[:, 1] <= bounds.bottom())
        )
        candidates = np.flatnonzero(inside_box)
        hits = [
            index
            for index in candidates
            if path.contains(QPointF(float(self._coords[index, 0]), float(self._coords[index, 1])))
        ]

        # Show the closed loop rather than the open trace the cursor drew.
        closed = np.asarray(path_points + [path_points[0]])
        self._lasso_curve.setData(closed[:, 0], closed[:, 1])

        if not hits:
            self.selected_rows = np.empty(0, dtype=np.int64)
            self._selection.hide()
        else:
            picked = np.asarray(hits, dtype=np.int64)
            self._selection.setData(
                x=self._coords[picked, 0], y=self._coords[picked, 1]
            )
            self._selection.show()
            self.selected_rows = self._rows[picked]
        self.points_selected.emit(self.selected_rows)

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
        if self.lasso_enabled and self._lasso_points:
            self._lasso_move(position)
            return
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
        if self.lasso_enabled:
            # A click starts the loop; the next one closes it. Tracking a
            # press-drag-release would mean subclassing the view box, and this
            # is both simpler and easier to correct mid-draw.
            event.accept()
            if self._lasso_points:
                self._lasso_finish()
            else:
                point = self.plot.getViewBox().mapSceneToView(event.scenePos())
                self._lasso_points = [(point.x(), point.y())]
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
