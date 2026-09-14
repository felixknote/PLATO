"""Choose which open embeddings to project together.

The one thing this dialog exists to prevent is combining embeddings that do
not share a vector space -- see plato.data.joint_projection for why that
produces a meaningless fit rather than an error. So compatibility is checked
live as boxes are ticked, not only when the button is pressed.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ..data.joint_projection import (
    ALIGN_CENTRE,
    ALIGN_LABELS,
    ALIGN_NONE,
    ALIGN_ZSCORE,
    IncompatibleEmbeddings,
    check_compatible,
)
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

        # Alignment. Raw concatenation is the default because the offset IS
        # the measurement at this stage: screens imaged apart sit apart, and
        # seeing how far apart, next to the biological structure, is the
        # point of looking. Centring is a deliberate second question with a
        # real cost -- it cannot tell a batch offset from a genuine
        # difference -- so it is offered with that said plainly rather than
        # applied helpfully.
        align_heading = QLabel("Align datasets before fitting")
        align_heading.setObjectName("panelHeading")
        align_why = QLabel(
            "Datasets imaged at different times sit apart, and seeing that "
            "is usually the point — it is what a batch effect looks like, "
            "at a size you can compare against everything else in the data. "
            "Leave this at None for that.\n\n"
            "Centring is the follow-up question: what is left once the shift "
            "is gone. It cannot tell a batch offset from a real difference "
            "between the screens, so read it as a second picture, not as a "
            "cleaned-up replacement for the first."
        )
        align_why.setWordWrap(True)
        align_why.setObjectName("muted")

        self._align_group = QButtonGroup(self)
        self._align_modes = [ALIGN_NONE, ALIGN_CENTRE, ALIGN_ZSCORE]
        align_box = QVBoxLayout()
        align_box.setSpacing(4)
        tips = {
            ALIGN_NONE: (
                "What the vectors say, untouched. The arms sit apart because "
                "they are apart -- which is the measurement when the screens "
                "were imaged at different times. Stay here unless you have a "
                "specific reason not to."
            ),
            ALIGN_CENTRE: (
                "Subtract each dataset's own mean, so the arms share an "
                "origin. Removes the constant shift and nothing else; "
                "distances within a dataset are unchanged."
            ),
            ALIGN_ZSCORE: (
                "Also divide by each dataset's spread. Use when one screen "
                "is noisier overall, not merely shifted — it asserts the "
                "arms should have equal spread, which is wrong if a real "
                "effect is what widens one."
            ),
        }
        for index, mode in enumerate(self._align_modes):
            button = QRadioButton(ALIGN_LABELS[mode])
            button.setToolTip(tips[mode])
            button.setChecked(mode == ALIGN_NONE)
            self._align_group.addButton(button, index)
            align_box.addWidget(button)

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
        layout.addWidget(align_heading)
        layout.addWidget(align_why)
        layout.addLayout(align_box)
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

    def selected_align(self) -> str:
        """Which alignment was chosen. ``ALIGN_NONE`` unless changed."""
        index = self._align_group.checkedId()
        if 0 <= index < len(self._align_modes):
            return self._align_modes[index]
        return ALIGN_NONE
