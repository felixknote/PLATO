"""Facet the embedding: one small plot per value of a grouping variable.

Answers a question a single plot cannot: does this cluster exist in every
plate, or only in one? Colouring by plate puts the answer in the overlap,
where it is unreadable at 30k points; splitting by plate puts each group on
its own axes where the shape is obvious.

**Shared axes are the default, and that is the important decision.** Letting
each facet autoscale makes every group fill its panel, so two groups occupying
completely different regions of the embedding look identical, and a tight
cluster looks as spread as a diffuse one. That is a comparison that actively
misleads. Shared axes mean position and spread are comparable across panels,
which is the entire point of faceting. Independent axes remain available for
the case where you only want each group's internal structure.

Panels reuse ``EmbeddingScatter`` rather than reimplementing a plot, so hover,
click, shift-click and lasso behave identically to the single view and all
report into the same selection.

Scale is handled by paging, not by shrinking: a field with 96 wells would give
96 unreadable thumbnails, so the grid shows a page of them with the group
count stated and lets you step through.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..gui import themes
from ..ordering import series_key
from .scatter import EmbeddingScatter

# Legend entries beyond this many wrap the strip past what a header row
# should cost; past it the swatches are still correct, just not all shown.
MAX_STRIP_ENTRIES = 16

# Columns a grid lays out at most, whatever the group count -- a 4-wide grid
# of panels reads better than a long strip. Rows are unbounded: every group
# renders, and the grid's own QScrollArea is how you reach the rest, rather
# than a page count that hides some of the data until you page to it.
MAX_GRID_COLUMNS = 4

# page_size's default: large enough that paging never actually triggers for
# any real grouping -- explorer.py's own MAX_FACET_GROUPS (60) already caps
# how many groups a facet column can offer in the first place, so this only
# needs to be at least that.
NO_PAGE_LIMIT = 1_000

# Smallest a facet may be before it stops showing structure.
MIN_PANEL_PX = 180

# Sort orders offered for the groups.
BY_SIZE = "size"
BY_NAME = "name"
SORT_LABELS = {BY_SIZE: "Largest first", BY_NAME: "By name"}


class _ClickableLabel(QLabel):
    """A QLabel that emits on mouse press -- QLabel has no click signal of
    its own, and a full QPushButton would need its own restyling to read as
    a title rather than a button."""

    clicked = Signal()

    def mousePressEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        self.clicked.emit()
        super().mousePressEvent(event)


class Facet(QWidget):
    """One group's plot, with its name and count above it."""

    # Emitted when the title bar is clicked -- expand this facet to fill the
    # grid, or (if it is already the expanded one) collapse back. A plain
    # click rather than a button: the title is already the one part of a
    # facet that is never a data point, so it costs nothing else to overload.
    title_clicked = Signal()

    def __init__(self, title: str, count: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title_text = title
        self.scatter = EmbeddingScatter()
        self.scatter.setMinimumSize(MIN_PANEL_PX, MIN_PANEL_PX)

        self.title = _ClickableLabel()
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Long category names must not widen the column; elide instead.
        self.title.setTextFormat(Qt.TextFormat.PlainText)
        self.title.setWordWrap(False)
        self.title.setCursor(Qt.CursorShape.PointingHandCursor)
        self.title.setToolTip("Click to expand this plot")
        self.title.clicked.connect(self.title_clicked)
        self._count = count

        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        layout.addWidget(self.title)
        layout.addWidget(self.scatter, 1)
        self.setLayout(layout)
        self.restyle()

    def restyle(self) -> None:
        colours = themes.current()
        self.title.setStyleSheet(
            f"color: {colours.text}; font-size: 11px; font-weight: 600;"
        )
        self._render_title()

    def resizeEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        super().resizeEvent(event)
        self._render_title()

    def _render_title(self) -> None:
        metrics = self.title.fontMetrics()
        width = max(60, self.width() - 12)
        label = f"{self.title_text}  ({self._count:,})"
        self.title.setText(
            metrics.elidedText(label, Qt.TextElideMode.ElideRight, width)
        )
        self.title.setToolTip(label)


class LegendStrip(QWidget):
    """One shared colour legend above the grid, instead of per facet.

    A facet colours the same field as the single plot, so its meaning is
    identical everywhere on the page -- repeating the legend in all twelve
    panels would say nothing twelve times, while every panel omitting it (the
    previous behaviour) left the colours unreadable the moment colour-by and
    display-by were different fields. One strip, read once, applies to every
    panel below it.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QHBoxLayout()
        self._layout.setContentsMargins(6, 2, 6, 2)
        self._layout.setSpacing(14)
        self.setLayout(self._layout)
        # Tracked separately from isVisible(): that also depends on the whole
        # ancestor chain being shown, which export must not care about -- a
        # legend built for a window that happens to be minimised, or a grid
        # under test with no top-level shown, still has entries to draw.
        self.has_entries = False
        self.hide()

    def set_entries(self, entries: list[tuple[str, str]] | None) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.has_entries = bool(entries)
        if not entries:
            self.hide()
            return

        colours = themes.current()
        shown, overflow = entries[:MAX_STRIP_ENTRIES], len(entries) - MAX_STRIP_ENTRIES
        for label, colour in shown:
            entry = QWidget()
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(5)
            swatch = QFrame()
            swatch.setFixedSize(10, 10)
            swatch.setStyleSheet(
                f"background: {colour}; border-radius: 2px; border: none;"
            )
            text = QLabel(label)
            text.setStyleSheet(f"color: {colours.text}; font-size: 11px;")
            row.addWidget(swatch)
            row.addWidget(text)
            entry.setLayout(row)
            self._layout.addWidget(entry)
        if overflow > 0:
            more = QLabel(f"+{overflow} more")
            more.setStyleSheet(f"color: {colours.text_muted}; font-size: 11px;")
            self._layout.addWidget(more)
        self._layout.addStretch(1)
        self.show()

    def restyle(self) -> None:
        colours = themes.current()
        for i in range(self._layout.count()):
            widget = self._layout.itemAt(i).widget()
            if isinstance(widget, QLabel):
                muted = widget.text().startswith("+")
                colour = colours.text_muted if muted else colours.text
                widget.setStyleSheet(f"color: {colour}; font-size: 11px;")


class GridView(QWidget):
    """A page of facets over one grouping variable."""

    # Emitted with (row index) and the usual selection payloads, so the
    # explorer can treat a facet exactly like the single plot.
    point_clicked = Signal(int)
    point_activated = Signal(int)
    points_selected = Signal(object)
    page_changed = Signal(int, int)  # page, total pages
    # The expanded facet's label, or None once collapsed back to the full
    # grid -- explorer.py listens to keep its own "back to grid" affordance
    # (and anything else that cares) in sync with a click on a facet title.
    expanded_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.facets: list[Facet] = []
        self._groups: list[tuple[str, np.ndarray]] = []
        self._page = 0
        # The label of the one facet showing full-size, or None for the
        # ordinary grid. Kept as state here (not a caller-side filter)
        # because collapsing back has to restore every OTHER facet exactly
        # as it was, which only a re-render from the full _groups list does.
        self._expanded_label: str | None = None
        # No real cap: every group renders in one page, and the grid's own
        # QScrollArea is how the rest is reached. Kept as an attribute (with
        # paging still wired through n_pages/set_page) rather than removing
        # the mechanism outright, since a caller can still set it lower.
        self.page_size = NO_PAGE_LIMIT
        self.columns = 0  # 0 = automatic
        self.shared_axes = True

        self.message = QLabel("")
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.setWordWrap(True)

        self.legend = LegendStrip()

        self.grid = QGridLayout()
        self.grid.setContentsMargins(4, 4, 4, 4)
        self.grid.setSpacing(6)
        self._container = QWidget()
        self._container.setLayout(self.grid)

        self.scroll = QScrollArea()
        self.scroll.setWidget(self._container)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.message)
        layout.addWidget(self.legend)
        layout.addWidget(self.scroll, 1)
        self.setLayout(layout)
        self.restyle()

    def restyle(self) -> None:
        colours = themes.current()
        self.message.setStyleSheet(f"color: {colours.text_muted}; padding: 12px;")
        self.legend.restyle()
        for facet in self.facets:
            facet.restyle()

    # -- data --------------------------------------------------------------

    def clear(self) -> None:
        """Drop every facet and its plot.

        Called when the grid is left, not merely hidden: each facet owns a
        pyqtgraph PlotWidget holding its own scene, items and cached arrays,
        and keeping a dozen of those alive behind a hidden widget is a real
        cost that accumulates every time the view is switched.
        """
        self._clear()
        self._groups = []
        self._page = 0
        self._expanded_label = None
        self.legend.set_entries(None)

    def set_groups(
        self,
        groups: list[tuple[str, np.ndarray]],
        *,
        sort: str = BY_SIZE,
    ) -> None:
        """``groups`` is (label, positional indices) per facet.

        Called on every redraw while faceting is active, not only when the
        user picks a new grouping field -- so an expanded facet's label is
        preserved across calls (a filter or colour change must not silently
        collapse it) and only cleared if that label genuinely no longer
        exists in the new groups, which is what switching to a different
        facet column looks like.
        """
        if sort == BY_SIZE:
            groups = sorted(groups, key=lambda g: (-len(g[1]), g[0]))
        else:
            # series_key, not a plain string sort: "By name" on a custom
            # grouping like "Day 1".."Day 16" (or a plain plate facet, "P1"..
            # "P20") must read Day 2 before Day 10, not after it -- see
            # plato.ordering for the same rule the dose/timepoint filters use.
            groups = sorted(groups, key=lambda g: series_key(g[0]))
        self._groups = groups
        self._page = 0
        if self._expanded_label is not None and not any(
            label == self._expanded_label for label, _ in groups
        ):
            self._expanded_label = None

    @property
    def n_groups(self) -> int:
        return len(self._groups)

    @property
    def n_pages(self) -> int:
        if not self._groups:
            return 0
        return max(1, math.ceil(len(self._groups) / max(1, self.page_size)))

    @property
    def page(self) -> int:
        return self._page

    def set_page(self, page: int) -> None:
        self._page = max(0, min(page, max(0, self.n_pages - 1)))

    def current_groups(self) -> list[tuple[str, np.ndarray]]:
        if self._expanded_label is not None:
            return [g for g in self._groups if g[0] == self._expanded_label]
        start = self._page * self.page_size
        return self._groups[start : start + self.page_size]

    @property
    def expanded_label(self) -> str | None:
        return self._expanded_label

    def expand(self, label: str) -> None:
        """Show only ``label``'s facet, filling the grid area."""
        self._expanded_label = label
        self.expanded_changed.emit(label)

    def collapse(self) -> None:
        """Back to the ordinary grid, exactly as it was before expanding."""
        if self._expanded_label is None:
            return
        self._expanded_label = None
        self.expanded_changed.emit(None)

    def column_count(self, n: int) -> int:
        """Columns for ``n`` facets: as square as possible, capped at 4 wide."""
        if self.columns > 0:
            return min(self.columns, MAX_GRID_COLUMNS)
        if n <= 0:
            return 1
        # A square-ish grid reads better than a long strip, and matches the
        # aspect a window usually has -- but never wider than MAX_GRID_COLUMNS,
        # so a page never exceeds a 4x4 layout.
        return max(1, min(n, int(math.ceil(math.sqrt(n))), MAX_GRID_COLUMNS))

    # -- rendering ---------------------------------------------------------

    def render(
        self,
        coords: np.ndarray,
        rows: np.ndarray,
        colours: list[str],
        *,
        point_size: int = 5,
        opacity: float = 0.85,
        symbols: list[str] | None = None,
        selected: np.ndarray | None = None,
        background: str | None = None,
        legend_entries: list[tuple[str, str]] | None = None,
    ) -> None:
        """Draw the current page.

        ``coords``/``rows``/``colours`` cover ALL visible points; each facet
        draws the subset its group indexes. ``legend_entries`` is shown once,
        above every panel, rather than per facet -- see ``LegendStrip``.
        """
        self.legend.set_entries(legend_entries)
        self._clear()
        groups = self.current_groups()
        if not groups:
            self.message.setText("No groups to show.")
            self.message.show()
            return
        self.message.hide()

        # One extent for every panel, computed once over everything on screen
        # rather than per facet -- this is what makes the panels comparable.
        limits = self._shared_limits(coords) if self.shared_axes else None

        # Same idea for the outline: one decision for the whole grid, from
        # the LARGEST facet, so a panel that happens to sit just under
        # OUTLINE_LIMIT never looks different from a denser neighbour in the
        # same grid purely because of its own point count. Bigger wins (grid
        # stays legible over grid stays detailed) because the alternative --
        # every panel outlined, including one at 30k points -- is exactly the
        # "grey smear" OUTLINE_LIMIT exists to avoid.
        outline_count = max((len(indices) for _, indices in groups), default=0)

        columns = self.column_count(len(groups))
        for position, (label, indices) in enumerate(groups):
            facet = Facet(label, len(indices))
            facet.scatter.set_background(background)
            facet.scatter.set_points(
                coords[indices],
                rows[indices],
                [colours[i] for i in indices],
                point_size=point_size,
                opacity=opacity,
                symbols=[symbols[i] for i in indices] if symbols else None,
                legend_entries=None,
                reset_view=limits is None,
                outline_count=outline_count,
            )
            if selected is not None:
                facet.scatter.set_selection(selected, notify=False)
            if limits is not None:
                facet.scatter.set_view_limits(*limits)

            facet.scatter.point_clicked.connect(self.point_clicked.emit)
            facet.scatter.point_activated.connect(self.point_activated.emit)
            facet.scatter.points_selected.connect(self.points_selected.emit)
            # Clicking the expanded facet's own title collapses it back;
            # clicking any facet's title while the grid is showing all of
            # them expands that one. Bound with a default argument (not a
            # bare lambda over the loop variable) so each facet's connection
            # keeps ITS OWN label rather than every one closing over the
            # last value `label` had after the loop finishes.
            facet.title_clicked.connect(
                lambda lbl=label: self.collapse()
                if self._expanded_label == lbl
                else self.expand(lbl)
            )

            self.grid.addWidget(facet, position // columns, position % columns)
            self.facets.append(facet)

        self.page_changed.emit(self._page, self.n_pages)

    @staticmethod
    def _shared_limits(coords: np.ndarray) -> tuple[float, float, float, float]:
        """One extent covering every point, with a small margin."""
        if len(coords) == 0:
            return (-1.0, 1.0, -1.0, 1.0)
        x_min, y_min = (float(v) for v in coords.min(axis=0))
        x_max, y_max = (float(v) for v in coords.max(axis=0))
        pad_x = (x_max - x_min) * 0.05 or 0.5
        pad_y = (y_max - y_min) * 0.05 or 0.5
        return (x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y)

    def set_selection(self, rows) -> None:
        """Ring the same rows in every facet, without re-emitting."""
        for facet in self.facets:
            facet.scatter.set_selection(rows, notify=False)

    def _clear(self) -> None:
        for facet in self.facets:
            self.grid.removeWidget(facet)
            facet.setParent(None)
            facet.deleteLater()
        self.facets.clear()

    # -- export -----------------------------------------------------------

    def export(self, path: str, *, title: str = "", subtitle: str = "") -> None:
        """Write the current page -- headline, legend and every facet -- to one file.

        A facet is a real pyqtgraph scene same as the single plot, but there
        are several of them side by side, and pyqtgraph's exporters draw one
        ``PlotItem`` at a time. So this composes the page a different way:
        paint the legend strip and the facet container widgets, at their
        current laid-out size, exactly as shown -- not the scroll viewport,
        which would crop to whatever fits on screen and silently drop facets
        below the fold.
        """
        from PySide6.QtCore import QPoint, QRectF, QSize
        from PySide6.QtGui import QPainter, QPixmap

        # The container's actual on-screen size, not sizeHint(), IF it is
        # genuinely on screen: each Facet only sets a MINIMUM size and is
        # stretched to fill the scroll viewport's width by QGridLayout, so
        # sizeHint() reflects the layout's PREFERRED size, which can differ
        # from its current stretched size in either direction -- panels on a
        # wide window are stretched wider than sizeHint(), so exporting at
        # sizeHint() would not match what the screen shows, and facet titles
        # (elided to the panel's live width) would elide differently too. So
        # this is an either/or, never a max() of the two: live geometry is
        # authoritative whenever there is a real layout pass to read it from;
        # sizeHint() is purely the fallback for when there is none, which is
        # real -- a GridView under test, or the app's first render before its
        # window has ever been shown. An unshown widget's width()/height()
        # are not zero or otherwise safe to blend with sizeHint() -- Qt hands
        # back a meaningless default (640x480) with no relation to the
        # layout, so this never reads live geometry unless isVisible() is
        # true.
        on_screen = self.isVisible()
        if on_screen:
            width = max(self._container.width(), 1)
            content_height = max(self._container.height(), 1)
            legend_height = self.legend.height() if self.legend.has_entries else 0
        else:
            width = max(self._container.sizeHint().width(), 1)
            content_height = max(self._container.sizeHint().height(), 1)
            legend_height = (
                self.legend.sizeHint().height() if self.legend.has_entries else 0
            )
        # The headline sits above the legend: it names the whole page, the
        # legend decodes the colours within it.
        from . import headline as headline_block

        head_height = headline_block.height(title, subtitle)
        height = head_height + legend_height + content_height

        if path.lower().endswith(".svg"):
            from PySide6.QtSvg import QSvgGenerator

            generator = QSvgGenerator()
            generator.setFileName(path)
            generator.setSize(QSize(width, height))
            generator.setViewBox(QRectF(0, 0, width, height))
            painter = QPainter(generator)
        else:
            pixmap = QPixmap(width, height)
            pixmap.fill(self._page_background())
            painter = QPainter(pixmap)

        try:
            if head_height:
                headline_block.draw(
                    painter,
                    title,
                    subtitle,
                    width=width,
                    background=self._page_background(),
                )
                painter.translate(0, head_height)
            if legend_height:
                self.legend.render(painter, QPoint(0, 0))
            painter.translate(0, legend_height)
            self._container.render(painter, QPoint(0, 0))
        finally:
            painter.end()

        if not path.lower().endswith(".svg"):
            pixmap.save(path)

    def _page_background(self):
        """The fill behind facets that don't cover the whole canvas.

        Only matters when panel counts don't tile the page rectangle exactly
        (a page of 5 in a 3-column grid, say): the gap must match what facets
        already show, not default to whatever QPixmap fills with.
        """
        from PySide6.QtGui import QColor

        if self.facets:
            return self.facets[0].scatter.background_colour()
        return QColor(themes.current().plot_background)


def build_groups(
    values: np.ndarray, *, max_groups: int | None = None
) -> tuple[list[tuple[str, np.ndarray]], int]:
    """Positional indices per distinct value.

    Returns ``(groups, skipped)``. Blank values become their own group rather
    than being dropped: "the points with no value for this field" is a real
    group and often an informative one.
    """
    values = np.asarray(values, dtype=object)
    groups: list[tuple[str, np.ndarray]] = []
    for value in dict.fromkeys(values.tolist()):
        label = str(value) if str(value) else "(blank)"
        groups.append((label, np.flatnonzero(values == value)))
    skipped = 0
    if max_groups is not None and len(groups) > max_groups:
        groups.sort(key=lambda g: (-len(g[1]), g[0]))
        skipped = len(groups) - max_groups
        groups = groups[:max_groups]
    return groups, skipped
