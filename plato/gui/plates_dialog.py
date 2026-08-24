"""The list of plates currently loaded, and the way to unload one.

Before this existed, plates only went in. A session that had accumulated six
of them gave no way to see what was loaded — the grid showed a plate name per
tile and nothing said how many plates were behind it — and no way to drop the
one that was added by mistake short of restarting and re-adding the rest.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .session import Session


class PlatesDialog(QDialog):
    """Shows each loaded plate with its image count and source folder."""

    def __init__(self, session: Session, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        self.removed = False
        self.setWindowTitle("Loaded plates")
        self.setMinimumWidth(560)

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.currentRowChanged.connect(self._sync_buttons)

        self.summary = QLabel()

        self.remove_button = QPushButton("Remove selected plate")
        self.remove_button.clicked.connect(self._remove_selected)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)

        actions = QHBoxLayout()
        actions.addWidget(self.remove_button)
        actions.addStretch(1)

        layout = QVBoxLayout()
        layout.addWidget(self.list, 1)
        layout.addWidget(self.summary)
        layout.addLayout(actions)
        layout.addWidget(buttons)
        self.setLayout(layout)

        self._reload()

    def _reload(self) -> None:
        self.list.clear()
        for plate in self.session.plates:
            item = QListWidgetItem(f"{plate.name}\n{plate.db.count()} images  ·  {plate.source}")
            item.setToolTip(str(plate.source))
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)
        plates = len(self.session.plates)
        noun = "plate" if plates == 1 else "plates"
        self.summary.setText(f"{plates} {noun}, {self.session.count()} images in total")
        self._sync_buttons()

    def _sync_buttons(self, *_args: object) -> None:
        self.remove_button.setEnabled(self.list.currentRow() >= 0)

    def _remove_selected(self) -> None:
        position = self.list.currentRow()
        if position < 0:
            return
        plate = self.session.plates[position]
        confirm = QMessageBox.question(
            self,
            "Remove plate",
            f"Remove {plate.name} from this session?\n\n"
            "Its images, flags and ratings stay on disk — only this session "
            "stops showing them. Add it again at any time.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self.session.remove_plate(position)
        self.removed = True
        self._reload()
