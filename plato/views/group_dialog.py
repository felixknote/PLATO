"""Define a custom grouping over one column's values.

Faceting splits data the way the metadata already splits it, which is not
always the split that matters. Plates P1..P8 were acquired two per day over
four days; "which day" is nowhere in the export, but it is exactly where a
batch effect lives. This dialog is where that mapping gets written.

One row per distinct value, each with an editable class name. Values sharing
a name become one facet. The class field is a free-text combo rather than a
fixed list, so naming a new class is typing it, and reusing one is picking it
from the dropdown -- there is no separate "create class" step to do first.

The grid opens pre-filled from ``custom_groups.suggest_classes`` because
consecutive-values-in-pairs is the common case by a wide margin, and a
dialog that opens on roughly the right answer is faster to correct than an
empty one is to fill.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..data.custom_groups import CustomGrouping, suggest_classes

# Above this many distinct values the grid is a data-entry form, not a
# dialog. Grouping is for a handful of plates or conditions; 200 wells one at
# a time is a job for a CSV, not a popup.
MAX_VALUES = 80


class GroupDialog(QDialog):
    """Assign each value of one column to a named class."""

    def __init__(
        self,
        source: str,
        source_label: str,
        values: list[str],
        existing: CustomGrouping | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.source = source
        self.values = list(values)
        self.setWindowTitle(f"Group {source_label.lower()} values")
        self.setMinimumWidth(420)

        self.name_edit = QLineEdit(existing.name if existing else "")
        self.name_edit.setPlaceholderText("measurement day")
        self.name_edit.setToolTip(
            "What this grouping means. Shown in the Display by dropdown, so "
            "name it after the variable, not the field it groups."
        )

        # The suggestion control. Visible rather than applied silently: it
        # guesses that the column sorts the way the runs happened, which is
        # usually true and worth being able to see and redo.
        self.per_class = QSpinBox()
        self.per_class.setRange(1, max(1, len(values)))
        self.per_class.setValue(2)
        self.per_class.setToolTip("How many consecutive values share a class.")
        fill = QPushButton("Fill")
        fill.setToolTip(
            "Assign consecutive values to classes of this size, in natural "
            "order (P2 before P10). A starting point, not a decision."
        )
        fill.clicked.connect(self._autofill)

        self.prefix_edit = QLineEdit("Day")
        self.prefix_edit.setToolTip("Class names become '<this> 1', '<this> 2', ...")
        self.prefix_edit.setMaximumWidth(90)

        suggest_row = QHBoxLayout()
        suggest_row.setContentsMargins(0, 0, 0, 0)
        suggest_row.addWidget(QLabel("Every"))
        suggest_row.addWidget(self.per_class)
        suggest_row.addWidget(QLabel("as"))
        suggest_row.addWidget(self.prefix_edit)
        suggest_row.addWidget(fill)
        suggest_row.addStretch(1)
        suggest_widget = QWidget()
        suggest_widget.setLayout(suggest_row)

        # One row per value.
        self.editors: dict[str, QComboBox] = {}
        grid = QGridLayout()
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        for index, value in enumerate(self.values):
            label = QLabel(value or "(blank)")
            box = QComboBox()
            box.setEditable(True)
            box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            box.lineEdit().setPlaceholderText("(own class)")
            if existing:
                box.setCurrentText(existing.assignments.get(value, ""))
            grid.addWidget(label, index, 0)
            grid.addWidget(box, index, 1)
            grid.setColumnStretch(1, 1)
            self.editors[value] = box
            # Typing a new class name in any row offers it in every other
            # row's dropdown, so the second plate of a pair is a pick.
            box.lineEdit().editingFinished.connect(self._refresh_class_choices)

        body = QWidget()
        body.setLayout(grid)
        scroll = QScrollArea()
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        # Tall enough for the whole grid when it is short, capped when it is
        # not. A fixed minimum cut the last row in half for the common
        # 8-plate case, which reads as a rendering fault rather than as "there
        # is more below" -- a half-row is not a scroll affordance.
        scroll.setMinimumHeight(min(360, max(140, 34 * len(self.values) + 12)))

        self.summary = QLabel("")
        self.summary.setObjectName("hint")
        self.summary.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        # A third action, because removing a grouping is not "cancel" -- it is
        # a decision, and the only other way to reach it would be clearing
        # every row by hand.
        self.clear_button = buttons.addButton(
            "Remove grouping", QDialogButtonBox.ButtonRole.DestructiveRole
        )
        self.clear_button.clicked.connect(self._clear_all)
        self.clear_button.setEnabled(bool(existing))

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"Group the values of <b>{source_label}</b> into classes."))
        layout.addWidget(QLabel("Name"))
        layout.addWidget(self.name_edit)
        layout.addWidget(suggest_widget)
        layout.addWidget(scroll, 1)
        layout.addWidget(self.summary)
        layout.addWidget(buttons)
        self.setLayout(layout)

        if not existing:
            self._autofill()
        self._refresh_class_choices()

    # -- behaviour ---------------------------------------------------------

    def _autofill(self) -> None:
        suggestion = suggest_classes(
            self.values,
            per_class=self.per_class.value(),
            prefix=self.prefix_edit.text().strip() or "Group",
        )
        for value, box in self.editors.items():
            box.setCurrentText(suggestion.get(value, ""))
        self._refresh_class_choices()

    def _clear_all(self) -> None:
        for box in self.editors.values():
            box.setCurrentText("")
        self._refresh_class_choices()

    def _refresh_class_choices(self) -> None:
        """Offer every class name already in use, in every row's dropdown."""
        names = [n for n in dict.fromkeys(self._assignments().values()) if n]
        for box in self.editors.values():
            current = box.currentText()
            box.blockSignals(True)
            box.clear()
            box.addItems(names)
            box.setCurrentText(current)
            box.blockSignals(False)
        self._refresh_summary(names)

    def _refresh_summary(self, names: list[str]) -> None:
        assigned = sum(1 for v in self._assignments().values() if v)
        unassigned = len(self.values) - assigned
        if not names:
            self.summary.setText("No classes yet — every value keeps its own name.")
            return
        text = f"{len(names)} classes over {assigned} values"
        if unassigned:
            # Said plainly: an unassigned value is its own facet, not a
            # silently dropped one.
            text += f"; {unassigned} unassigned, each keeping its own name"
        self.summary.setText(text)

    def _assignments(self) -> dict[str, str]:
        return {
            value: box.currentText().strip() for value, box in self.editors.items()
        }

    # -- result ------------------------------------------------------------

    def grouping(self) -> CustomGrouping:
        """What was defined. An empty one means "remove this grouping"."""
        return CustomGrouping(
            source=self.source,
            name=self.name_edit.text().strip(),
            assignments={v: c for v, c in self._assignments().items() if c},
        )
