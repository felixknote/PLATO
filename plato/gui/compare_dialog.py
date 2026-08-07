"""Set up a multi-panel comparison.

Pick a base condition (held constant across every panel), then a variable to
compare along and which of its values to show. The result is one panel per
selected value, each pinned to base + that value.

This generalises the two-panel Ctrl+D compare: that one gives you two panels
with independent filters and leaves it to you to keep them comparable. Here
the panels differ in exactly one variable by construction, which is the
comparison you actually want when reading a concentration series or a
timepoint course.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .session import Session

# Above this, a variable is almost certainly an identifier (image_id, path)
# rather than something you compare along, and the value list is unusable.
MAX_COMPARE_VALUES = 40


class CompareSetupDialog(QDialog):
    """Collects: base condition, comparison variable, and which values to show."""

    def __init__(self, session: Session, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Compare conditions")
        self.setMinimumWidth(460)

        self.base_column = QComboBox()
        self.base_value = QComboBox()
        self.variable = QComboBox()

        columns = [c for c in session.filter_columns() if self._usable(c)]
        self.base_column.addItem("(none — compare across everything)", "")
        for column in columns:
            self.base_column.addItem(session.label(column), column)
            self.variable.addItem(session.label(column), column)

        # Default to holding the first real column constant rather than
        # "(none)": comparing a concentration series across *every* antibiotic
        # at once puts four unrelated drugs side by side, which looks like a
        # comparison and is not one. "(none)" stays available deliberately.
        if self.base_column.count() > 1:
            self.base_column.setCurrentIndex(1)
        if self.variable.count() > 1:
            self.variable.setCurrentIndex(1)

        self.base_column.currentIndexChanged.connect(self._reload_base_values)
        self.variable.currentIndexChanged.connect(self._reload_variable_values)

        self.values_box = QGroupBox("Show these values (one panel each)")
        self.values_layout = QVBoxLayout()
        self.values_box.setLayout(self.values_layout)
        self.value_checks: list[QCheckBox] = []

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.values_box)
        scroll.setMinimumHeight(220)

        select_all = QPushButton("All")
        select_all.clicked.connect(lambda: self._set_all(True))
        select_none = QPushButton("None")
        select_none.clicked.connect(lambda: self._set_all(False))

        self.hint = QLabel("")
        self.hint.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Hold constant", self.base_column)
        form.addRow("at value", self.base_value)
        form.addRow("Compare along", self.variable)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

        picks = QVBoxLayout()
        picks.addWidget(select_all)
        picks.addWidget(select_none)
        picks.addStretch(1)
        picker = QWidget()
        picker.setLayout(picks)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(scroll)
        layout.addWidget(self.hint)
        layout.addWidget(buttons)
        self.setLayout(layout)

        self._reload_base_values()
        self._reload_variable_values()

    # -- population ---------------------------------------------------------

    def _usable(self, column: str) -> bool:
        values = self.session.distinct(column)
        return 0 < len(values) <= MAX_COMPARE_VALUES

    def _reload_base_values(self) -> None:
        column = self.base_column.currentData()
        self.base_value.clear()
        self._update_hint()
        if not column:
            self.base_value.setEnabled(False)
            return
        self.base_value.setEnabled(True)
        for value in self.session.distinct(column):
            self.base_value.addItem(str(value), value)

    def _reload_variable_values(self) -> None:
        for check in self.value_checks:
            check.setParent(None)
        self.value_checks.clear()

        column = self.variable.currentData()
        if not column:
            return
        for value in self.session.distinct(column):
            check = QCheckBox(str(value))
            check.setChecked(True)
            check.toggled.connect(self._update_hint)
            self.value_checks.append(check)
            self.values_layout.addWidget(check)
        self._update_hint()

    def _set_all(self, checked: bool) -> None:
        for check in self.value_checks:
            check.setChecked(checked)

    def _update_hint(self) -> None:
        n = len(self.selected_values())
        if not n:
            self.hint.setText("Select at least one value.")
        elif not self.base_column.currentData():
            self.hint.setText(
                f"{n} panels, nothing held constant — each panel will mix every "
                "condition at that value. Pick something to hold constant unless "
                "you mean to compare across the whole screen."
            )
        elif n > 6:
            self.hint.setText(
                f"{n} panels — they will be narrow. Consider fewer values for a "
                "readable side-by-side."
            )
        else:
            self.hint.setText(f"{n} panel(s) will open.")

    # -- result -------------------------------------------------------------

    def selected_values(self) -> list[str]:
        return [c.text() for c in self.value_checks if c.isChecked()]

    def base_filter(self) -> dict[str, list[str]]:
        column = self.base_column.currentData()
        value = self.base_value.currentData()
        return {column: [str(value)]} if column and value is not None else {}

    def variable_column(self) -> str:
        return self.variable.currentData() or ""

    def _accept(self) -> None:
        if not self.selected_values():
            self._update_hint()
            return
        if self.variable_column() and self.variable_column() == self.base_column.currentData():
            self.hint.setText("The comparison variable must differ from the one held constant.")
            return
        self.accept()
