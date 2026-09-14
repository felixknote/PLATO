"""Confirm which values are controls, starting from what the parser found.

The condition parser already recognises the lab's own control labels, so this
opens pre-filled rather than empty. But that recognition is a dictionary of
known names (annotations.CONTROL_LABELS): a screen naming its controls
anything else gets nothing, and would have no way to say so. Hence a dialog
and not silent detection -- the guess is visible, and correcting it is ticking
a box.

One row per distinct value of the chosen column, each with a tick and a class
name. The class is what the control controls FOR ("Vehicle (DMSO)", "WT
(non-targeting gRNA)") -- several values can share one, and different kinds of
control stay distinct rather than collapsing into one "control" bucket, which
would put a solvent control and a non-targeting guide in the same class when
they control for entirely different things.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..data.controls import ControlMarking

# Above this many distinct values the grid stops being a dialog. Controls are
# a handful of conditions; a column with hundreds of values is the wrong
# column to be marking against.
MAX_VALUES = 200


class ControlDialog(QDialog):
    """Tick which values of one column are controls, and name each class."""

    def __init__(
        self,
        columns: list[tuple[str, str]],
        values_for,
        detected: dict[str, str],
        existing: ControlMarking | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """
        Args:
            columns: (column, label) pairs that can be marked against.
            values_for: callable taking a column, returning its distinct values.
            detected: the parser's guess, {value: class}.
            existing: a marking already in force, which wins over the guess.
        """
        super().__init__(parent)
        self.setWindowTitle("Mark controls")
        self.setMinimumWidth(460)
        self._values_for = values_for
        self._detected = dict(detected)
        self._rows: dict[str, tuple[QCheckBox, QComboBox]] = {}

        intro = QLabel(
            "Controls are what every other point is judged against, so they "
            "are shown as their own classes in every colour-by — not "
            "scattered through a legend of genes — and drawn so they stay "
            "visible inside a dense cluster."
        )
        intro.setWordWrap(True)
        intro.setObjectName("muted")

        self.column_box = QComboBox()
        for column, label in columns:
            self.column_box.addItem(label, column)
        start = (existing.source if existing else "") or (
            columns[0][0] if columns else ""
        )
        index = self.column_box.findData(start)
        if index >= 0:
            self.column_box.setCurrentIndex(index)
        self.column_box.currentIndexChanged.connect(self._rebuild)
        self.column_box.setToolTip(
            "Which column says what was in a well. Usually the condition "
            "label; a screen with a dedicated control column can use that."
        )

        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(2, 2, 2, 2)
        self.grid.setHorizontalSpacing(10)
        self.grid.setVerticalSpacing(4)

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.grid_host)
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumHeight(260)

        self.summary = QLabel("")
        self.summary.setObjectName("hint")
        self.summary.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        # Unmarking is a decision, not a cancel -- the only other route would
        # be clearing every tick by hand.
        self.clear_button = buttons.addButton(
            "Unmark all", QDialogButtonBox.ButtonRole.DestructiveRole
        )
        self.clear_button.clicked.connect(self._clear_all)

        reset = QPushButton("Use detected")
        reset.setToolTip("Go back to what the condition parser recognised.")
        reset.clicked.connect(lambda: self._rebuild(force_detected=True))

        layout = QVBoxLayout()
        layout.addWidget(intro)
        layout.addWidget(QLabel("Mark against"))
        layout.addWidget(self.column_box)
        layout.addWidget(self.scroll, 1)
        layout.addWidget(self.summary)
        layout.addWidget(reset)
        layout.addWidget(buttons)
        self.setLayout(layout)

        self._existing = existing
        self._rebuild()

    # -- behaviour ---------------------------------------------------------

    def _rebuild(self, *_args, force_detected: bool = False) -> None:
        """Repopulate the grid for the currently chosen column."""
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._rows.clear()

        column = self.column_box.currentData() or ""
        values = [v for v in self._values_for(column) if str(v).strip()][:MAX_VALUES]

        # An existing marking wins over the guess, unless the guess was asked
        # for explicitly -- and only while it is about the same column.
        prefill = dict(self._detected)
        if not force_detected and self._existing and self._existing.source == column:
            prefill = dict(self._existing.assignments)

        known = list(dict.fromkeys(prefill.values()))
        for row, value in enumerate(values):
            tick = QCheckBox(value)
            name = QComboBox()
            name.setEditable(True)
            name.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            name.lineEdit().setPlaceholderText("what it controls for")
            name.addItems(known)
            assigned = prefill.get(value, "")
            tick.setChecked(bool(assigned))
            name.setCurrentText(assigned)
            name.setEnabled(bool(assigned))
            tick.toggled.connect(name.setEnabled)
            tick.toggled.connect(self._refresh_summary)
            # A ticked row with no name is a control with nothing said about
            # what it controls for; default it to its own value rather than
            # leaving it blank and unlabelled in the legend.
            tick.toggled.connect(
                lambda on, v=value, n=name: self._default_name(on, v, n)
            )
            name.lineEdit().editingFinished.connect(self._refresh_summary)
            self.grid.addWidget(tick, row, 0)
            self.grid.addWidget(name, row, 1)
            self.grid.setColumnStretch(1, 1)
            self._rows[value] = (tick, name)

        self.clear_button.setEnabled(bool(self._assignments()))
        self._refresh_summary()

    @staticmethod
    def _default_name(on: bool, value: str, box: QComboBox) -> None:
        if on and not box.currentText().strip():
            box.setCurrentText(value)

    def _clear_all(self) -> None:
        for tick, _ in self._rows.values():
            tick.setChecked(False)
        self._refresh_summary()

    def _assignments(self) -> dict[str, str]:
        out = {}
        for value, (tick, name) in self._rows.items():
            if tick.isChecked():
                out[value] = name.currentText().strip() or value
        return out

    def _refresh_summary(self, *_args) -> None:
        assignments = self._assignments()
        if not assignments:
            self.summary.setText("Nothing marked — no control classes will be shown.")
            return
        classes = list(dict.fromkeys(assignments.values()))
        self.summary.setText(
            f"{len(assignments)} values marked as {len(classes)} control "
            f"class{'es' if len(classes) != 1 else ''}: "
            + ", ".join(classes[:4])
            + (" …" if len(classes) > 4 else "")
        )

    # -- result ------------------------------------------------------------

    def marking(self) -> ControlMarking:
        """What was confirmed. An empty one means "no controls marked"."""
        return ControlMarking(
            source=self.column_box.currentData() or "",
            assignments=self._assignments(),
        )
