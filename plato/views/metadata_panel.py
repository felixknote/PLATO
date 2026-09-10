"""The right-hand column when image tracing is off: metadata, not pixels.

With the image viewer disabled the preview column has nothing to show, but
"what did I just select?" is still the question the column exists to answer.
So it becomes a table of the selection's metadata instead of a list of crops.

Deliberately a QTableWidget over the frame rather than a second selection
model: the rows come from the same workspace selection the plot and the
cluster panel use, so there is nothing here to fall out of step. Only the
presentation differs.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..gui import themes

# Rows shown at once. A lasso can hold thousands; a table that long is not
# read, it is scrolled past, and building it costs more than it tells you.
MAX_ROWS = 500

# Columns that identify a point rather than describe it. Shown last, since
# the descriptive fields are what a selection is usually about.
_TRAILING = ("image_name", "image_path")


class MetadataPanel(QWidget):
    """A table of the current selection's metadata."""

    selection_changed = Signal(object)
    compare_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.rows: list[int] = []
        self._frame = None

        self.heading = QLabel("Selection")
        self.heading.setObjectName("panelHeading")

        self.count_label = QLabel("Image viewer is off")
        self.count_label.setWordWrap(True)

        self.note = QLabel(
            "Points are not traced back to images while the image viewer is "
            "off. Selection, lasso analysis and filtering all still work."
        )
        self.note.setWordWrap(True)

        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.hide()

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(lambda: self.selection_changed.emit([]))
        self.clear_button.setEnabled(False)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.addStretch(1)
        buttons.addWidget(self.clear_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        layout.addWidget(self.heading)
        layout.addWidget(self.count_label)
        layout.addWidget(self.note)
        layout.addWidget(self.table, 1)
        layout.addLayout(buttons)
        self.setLayout(layout)
        self.restyle()

    def restyle(self) -> None:
        """Re-read theme colours. Called on a theme switch."""
        colours = themes.current()
        self.count_label.setStyleSheet(
            f"color: {colours.text_faint}; font-size: 11px;"
        )
        self.note.setStyleSheet(f"color: {colours.text_faint}; font-size: 11px;")

    def set_frame(self, frame) -> None:
        """The metadata to read rows out of."""
        self._frame = frame

    def set_rows(self, rows) -> None:
        rows = [int(r) for r in np.asarray(rows, dtype=np.int64).ravel()]
        self.rows = rows
        self.clear_button.setEnabled(bool(rows))

        if self._frame is None or not rows:
            self.table.hide()
            self.table.setRowCount(0)
            self.count_label.setText(
                "Nothing selected. Click a point, or use Lasso Analysis."
            )
            self.note.show()
            return

        self.note.hide()
        shown = rows[:MAX_ROWS]
        self.count_label.setText(
            f"<b>{len(rows):,}</b> selected"
            + (f" · showing {len(shown)}" if len(rows) > len(shown) else "")
        )

        columns = self._columns()
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(
            [self._label(column) for column in columns]
        )
        self.table.setRowCount(len(shown))
        for position, row in enumerate(shown):
            record = self._frame.iloc[row]
            for column_index, column in enumerate(columns):
                value = str(record.get(column, "") or "")
                self.table.setItem(position, column_index, QTableWidgetItem(value))
        self.table.show()

    def _columns(self) -> list[str]:
        """Which columns to show: those that actually carry values here."""
        if self._frame is None:
            return []
        present = [
            column
            for column in self._frame.columns
            if column not in _TRAILING
            and self._frame[column].astype(str).ne("").any()
        ]
        trailing = [c for c in _TRAILING if c in self._frame.columns]
        return present + trailing

    @staticmethod
    def _label(column: str) -> str:
        from ..data.explorer_model import field_label

        return field_label(column)
