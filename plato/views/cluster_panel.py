"""Lasso Analysis: what is this region of the embedding made of?

Sits in the left-hand column under the display controls. When a selection
exists, it breaks it down along every categorical column the dataset has and
ranks the fields by how far the selection departs from the rest of the data --
see ``plato.data.cluster_stats`` for why that is the ranking and not raw
prevalence.

The bars are drawn with QPainter rather than pyqtgraph or matplotlib: they are
a row of proportional rectangles in a panel a few hundred pixels wide, and a
plotting library's axes, margins and interaction machinery would cost more
than they add at that size. The scatter still uses pyqtgraph, where the
interaction is the point.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..data.cluster_stats import ClusterComposition, FieldSummary
from ..gui.theme import (
    ACCENT,
    BORDER,
    IMAGE_BACKGROUND,
    SURFACE,
    TEXT,
    TEXT_FAINT,
    TEXT_MUTED,
)

# Bars drawn per field before the rest is summarised as a tail.
TOP_VALUES = 8

BAR_HEIGHT = 18
BAR_GAP = 3
LABEL_WIDTH = 92
COUNT_WIDTH = 82


class RankedBars(QWidget):
    """A ranked bar chart of one field's composition."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._summary: FieldSummary | None = None
        self._shown: list = []
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_summary(self, summary: FieldSummary | None) -> None:
        self._summary = summary
        self._shown = summary.values[:TOP_VALUES] if summary else []
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        rows = len(self._shown)
        return QSize(200, max(1, rows) * (BAR_HEIGHT + BAR_GAP))

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return self.sizeHint()

    def paintEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        if not self._shown:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = painter.font()
        font.setPointSizeF(max(7.5, font.pointSizeF() - 1.5))
        painter.setFont(font)
        metrics = painter.fontMetrics()

        # Scale to the largest bar rather than to 100%, so a cluster split
        # evenly between six values still shows six readable bars instead of
        # six stubs.
        peak = max(v.fraction for v in self._shown) or 1.0
        bar_left = LABEL_WIDTH + 6
        bar_width = max(20, self.width() - bar_left - COUNT_WIDTH)

        for i, share in enumerate(self._shown):
            top = i * (BAR_HEIGHT + BAR_GAP)
            rect_width = max(1.0, bar_width * (share.fraction / peak))

            painter.setPen(QPen(QColor(TEXT_MUTED)))
            label = metrics.elidedText(
                share.value, Qt.TextElideMode.ElideRight, LABEL_WIDTH
            )
            painter.drawText(
                0, top, LABEL_WIDTH, BAR_HEIGHT,
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                label,
            )

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(IMAGE_BACKGROUND))
            painter.drawRect(bar_left, top + 3, bar_width, BAR_HEIGHT - 6)

            # Enrichment drives the colour: a bar that is merely large says
            # the value is common, while a bar that is large AND enriched says
            # it is characteristic of this region.
            painter.setBrush(QColor(_bar_colour(share.enrichment)))
            painter.drawRect(bar_left, top + 3, int(rect_width), BAR_HEIGHT - 6)

            painter.setPen(QPen(QColor(TEXT)))
            painter.drawText(
                bar_left + bar_width + 4, top, COUNT_WIDTH - 4, BAR_HEIGHT,
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                f"{share.count:,}  {share.percent:.1f}%",
            )
        painter.end()


def _bar_colour(enrichment: float) -> str:
    """Blue for background-rate, amber for strongly over-represented."""
    if math.isinf(enrichment) or enrichment >= 4.0:
        return "#e8a33d"
    if enrichment >= 2.0:
        return "#c9a227"
    if enrichment >= 1.25:
        return "#5b9fe3"
    return ACCENT


class FieldBlock(QFrame):
    """One field: its name, its divergence, and its bars."""

    def __init__(self, summary: FieldSummary, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.summary = summary
        self.setStyleSheet(
            f"FieldBlock {{ background: {SURFACE}; border: 1px solid {BORDER};"
            f" border-radius: 4px; }}"
        )

        title = QLabel(f"<b>{summary.label}</b>")
        title.setObjectName("body")

        # Divergence is the ranking, so it should be visible rather than an
        # invisible sort key the user has to take on trust.
        score = QLabel(f"{summary.divergence:.2f}")
        score.setToolTip(
            "How different this field's composition is inside the selection "
            "compared with the rest of the data.\n"
            "0.00 = identical to the background; 1.00 = no overlap at all.\n"
            "Scaled by how much of the selection the field actually annotates."
        )
        score.setObjectName("hintSmall")

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(title, 1)
        header.addWidget(score)

        self.bars = RankedBars()
        self.bars.set_summary(summary)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(4)
        layout.addLayout(header)
        layout.addWidget(self.bars)

        extra = len(summary.values) - TOP_VALUES
        notes = []
        if extra > 0:
            notes.append(f"+{extra} more value{'s' if extra != 1 else ''}")
        if summary.missing:
            notes.append(f"{summary.missing:,} unannotated")
        if notes:
            note = QLabel(" · ".join(notes))
            note.setObjectName("hintSmall")
            layout.addWidget(note)
        self.setLayout(layout)


class ClusterPanel(QWidget):
    """Lasso Analysis: activate the lasso, then read off what was caught."""

    lasso_toggled = Signal(bool)
    additive_toggled = Signal(bool)
    show_images_requested = Signal()
    clear_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._blocks: list[FieldBlock] = []

        self.lasso_button = QPushButton("Draw lasso")
        self.lasso_button.setCheckable(True)
        self.lasso_button.setToolTip(
            "Click on the plot to start an outline, move to trace it, click "
            "again to close.\nEvery point inside becomes selected."
        )
        self.lasso_button.toggled.connect(self._on_lasso_toggled)

        self.additive_box = QCheckBox("Add to selection")
        self.additive_box.setToolTip(
            "Keep what is already selected and add the next lasso to it, so a "
            "cluster can be built from several strokes."
        )
        self.additive_box.toggled.connect(self.additive_toggled.emit)

        self.headline = QLabel("")
        self.headline.setWordWrap(True)
        self.headline.setObjectName("body")

        self.summary_label = QLabel("Draw a lasso to analyse a region.")
        self.summary_label.setWordWrap(True)
        self.summary_label.setObjectName("hint")

        self.images_button = QPushButton("Show images")
        self.images_button.setToolTip(
            "List every selected point in the preview column on the right."
        )
        self.images_button.clicked.connect(self.show_images_requested.emit)
        self.images_button.setEnabled(False)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_requested.emit)
        self.clear_button.setEnabled(False)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(6)
        buttons.addWidget(self.images_button, 1)
        buttons.addWidget(self.clear_button)

        # Fields are ranked, but a specific question ("is this cluster one
        # plate?") wants a specific field, so the ranking can be overridden.
        self.field_box = QComboBox()
        self.field_box.addItem("Most distinctive first", None)
        self.field_box.currentIndexChanged.connect(self._rebuild)
        self.field_box.hide()

        self.blocks_layout = QVBoxLayout()
        self.blocks_layout.setContentsMargins(0, 0, 0, 0)
        self.blocks_layout.setSpacing(6)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)
        layout.addWidget(self.lasso_button)
        layout.addWidget(self.additive_box)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.headline)
        layout.addLayout(buttons)
        layout.addWidget(self.field_box)
        layout.addLayout(self.blocks_layout)
        self.setLayout(layout)

        self._composition: ClusterComposition | None = None

    def _on_lasso_toggled(self, active: bool) -> None:
        self.lasso_button.setText("Drawing — click to finish" if active else "Draw lasso")
        self.lasso_toggled.emit(active)

    def set_lasso_active(self, active: bool) -> None:
        """Reflect a lasso turned on or off from elsewhere."""
        if self.lasso_button.isChecked() != active:
            self.lasso_button.blockSignals(True)
            self.lasso_button.setChecked(active)
            self.lasso_button.blockSignals(False)
        self.lasso_button.setText("Drawing — click to finish" if active else "Draw lasso")

    @property
    def additive(self) -> bool:
        return self.additive_box.isChecked()

    def show_composition(self, composition: ClusterComposition) -> None:
        """Display a breakdown, or the empty state when nothing is selected."""
        self._composition = composition
        has_any = composition.n_selected > 0
        self.images_button.setEnabled(has_any)
        self.clear_button.setEnabled(has_any)

        if not has_any:
            self.summary_label.setText("Draw a lasso to analyse a region.")
            self.headline.setText("")
            self.field_box.hide()
            self._clear_blocks()
            return

        share = (
            composition.n_selected / composition.n_total * 100
            if composition.n_total
            else 0.0
        )
        self.summary_label.setText(
            f"<b>{composition.n_selected:,}</b> of {composition.n_total:,} points "
            f"({share:.1f}%)"
        )
        self.headline.setText(composition.headline)

        # Rebuild the field chooser only when the fields themselves change;
        # otherwise a selection change would reset the user's choice.
        wanted = [(f.label, f.column) for f in composition.fields]
        current = [
            (self.field_box.itemText(i), self.field_box.itemData(i))
            for i in range(1, self.field_box.count())
        ]
        if wanted != current:
            chosen = self.field_box.currentData()
            self.field_box.blockSignals(True)
            self.field_box.clear()
            self.field_box.addItem("Most distinctive first", None)
            for label, column in wanted:
                self.field_box.addItem(label, column)
            index = self.field_box.findData(chosen)
            self.field_box.setCurrentIndex(max(0, index))
            self.field_box.blockSignals(False)
        self.field_box.setVisible(len(wanted) > 1)
        self._rebuild()

    def _rebuild(self) -> None:
        self._clear_blocks()
        if self._composition is None:
            return
        chosen = self.field_box.currentData()
        fields = self._composition.fields
        if chosen is not None:
            fields = [f for f in fields if f.column == chosen]
        if not fields:
            note = QLabel("No categorical metadata to break this selection down by.")
            note.setWordWrap(True)
            note.setObjectName("hint")
            self.blocks_layout.addWidget(note)
            self._blocks = []
            return
        for summary in fields:
            block = FieldBlock(summary)
            self.blocks_layout.addWidget(block)
            self._blocks.append(block)

    def _clear_blocks(self) -> None:
        while self.blocks_layout.count():
            item = self.blocks_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._blocks = []
