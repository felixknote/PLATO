"""Choose which open embeddings to project together.

The one thing this dialog exists to prevent is combining embeddings that do
not share a vector space -- see plato.data.joint_projection for why that
produces a meaningless fit rather than an error. So compatibility is checked
live as boxes are ticked, not only when the button is pressed.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..data.joint_projection import IncompatibleEmbeddings, check_compatible
from ..data.workspace import EmbeddingEntry
from ..gui import themes


class CombineEmbeddingsDialog(QDialog):
    """Pick two or more open embeddings to project together."""

    def __init__(self, entries: list[EmbeddingEntry], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Combine embeddings")
        self.setMinimumWidth(420)
        self._entries = entries
        self._boxes: dict[str, QCheckBox] = {}

        intro = QLabel(
            "Project several embeddings together as ONE fit, so their "
            "positions are actually comparable -- unlike colouring by "
            "dataset over separately-computed layouts, which have no shared "
            "coordinate system."
        )
        intro.setWordWrap(True)
        intro.setObjectName("muted")

        checks = QVBoxLayout()
        checks.setSpacing(6)
        for entry in entries:
            box = QCheckBox(f"{entry.label()}  —  {entry.n_points:,} × {entry.n_dimensions}d")
            box.toggled.connect(self._revalidate)
            self._boxes[entry.key] = box
            checks.addWidget(box)

        self.status_label = QLabel("Choose at least two embeddings.")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("hint")

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)

        layout = QVBoxLayout()
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        layout.addWidget(intro)
        layout.addLayout(checks)
        layout.addWidget(self.status_label)
        layout.addWidget(self.buttons)
        self.setLayout(layout)
        self.restyle()

    def restyle(self) -> None:
        colours = themes.current()
        self.status_label.setStyleSheet("")  # object name carries the colour

    def _selected(self) -> list[EmbeddingEntry]:
        return [e for e in self._entries if self._boxes[e.key].isChecked()]

    def _revalidate(self) -> None:
        chosen = self._selected()
        ok_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        if len(chosen) < 2:
            self.status_label.setText("Choose at least two embeddings.")
            ok_button.setEnabled(False)
            return
        try:
            check_compatible(chosen)
        except IncompatibleEmbeddings as exc:
            self.status_label.setText(str(exc))
            ok_button.setEnabled(False)
            return
        total = sum(e.n_points for e in chosen)
        self.status_label.setText(
            f"{len(chosen)} embeddings, {total:,} points total. All share "
            f"{chosen[0].n_dimensions} dimensions -- compatible."
        )
        ok_button.setEnabled(True)

    def selected_entries(self) -> list[EmbeddingEntry]:
        return self._selected()
