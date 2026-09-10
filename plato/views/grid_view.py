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
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..gui import themes
from .scatter import EmbeddingScatter

# Facets per page. Twelve 300px panels fill a large screen; beyond that they
# stop being readable and paging is the honest answer.
DEFAULT_PAGE_SIZE = 12

# Smallest a facet may be before it stops showing structure.
MIN_PANEL_PX = 180

# Sort orders offered for the groups.
BY_SIZE = "size"
BY_NAME = "name"
SORT_LABELS = {BY_SIZE: "Largest first", BY_NAME: "By name"}


class Facet(QWidget):
    """One group's plot, with its name and count above it."""

    def __init__(self, title: str, count: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title_text = title
        self.scatter = EmbeddingScatter()
        self.scatter.setMinimumSize(MIN_PANEL_PX, MIN_PANEL_PX)

        self.title = QLabel()
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Long category names must not widen the column; elide instead.
        self.title.setTextFormat(Qt.TextFormat.PlainText)
        self.title.setWordWrap(False)
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


class GridView(QWidget):
    """A page of facets over one grouping variable."""

    # Emitted with (row index) and the usual selection payloads, so the
    # explorer can treat a facet exactly like the single plot.
    point_clicked = Signal(int)
    point_activated = Signal(int)
    points_selected = Signal(object)
    page_changed = Signal(int, int)  # page, total pages

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.facets: list[Facet] = []
        self._groups: list[tuple[str, np.ndarray]] = []
        self._page = 0
        self.page_size = DEFAULT_PAGE_SIZE
        self.columns = 0  # 0 = automatic
        self.shared_axes = True

        self.message = QLabel("")
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.setWordWrap(True)

        self.grid = QGridLayout()
        self.grid.setContentsMargins(4, 4, 4, 4)
        self.grid.setSpacing(6)
        container = QWidget()
        container.setLayout(self.grid)

        self.scroll = QScrollArea()
        self.scroll.setWidget(container)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.message)
        layout.addWidget(self.scroll, 1)
        self.setLayout(layout)
        self.restyle()

    def restyle(self) -> None:
        colours = themes.current()
        self.message.setStyleSheet(f"color: {colours.text_muted}; padding: 12px;")
        for facet in self.facets:
            facet.restyle()

    # -- data --------------------------------------------------------------

    def set_groups(
        self,
        groups: list[tuple[str, np.ndarray]],
        *,
        sort: str = BY_SIZE,
    ) -> None:
        """``groups`` is (label, positional indices) per facet."""
        if sort == BY_SIZE:
            groups = sorted(groups, key=lambda g: (-len(g[1]), g[0]))
        else:
            groups = sorted(groups, key=lambda g: g[0])
        self._groups = groups
        self._page = 0

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
        start = self._page * self.page_size
        return self._groups[start : start + self.page_size]

    def column_count(self, n: int) -> int:
        """Columns for ``n`` facets: as square as possible unless pinned."""
        if self.columns > 0:
            return self.columns
        if n <= 0:
            return 1
        # A square-ish grid reads better than a long strip, and matches the
        # aspect a window usually has.
        return max(1, min(n, int(math.ceil(math.sqrt(n)))))

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
    ) -> None:
        """Draw the current page.

        ``coords``/``rows``/``colours`` cover ALL visible points; each facet
        draws the subset its group indexes.
        """
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
            )
            if selected is not None:
                facet.scatter.set_selection(selected, notify=False)
            if limits is not None:
                facet.scatter.set_view_limits(*limits)

            facet.scatter.point_clicked.connect(self.point_clicked.emit)
            facet.scatter.point_activated.connect(self.point_activated.emit)
            facet.scatter.points_selected.connect(self.points_selected.emit)

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
