"""Every embedding open in this session, all visible at once.

The dropdown this replaces showed exactly one entry at a time, so seeing what
was actually loaded -- three exports, say, one of them a joint combination of
the other two -- meant clicking through the list one by one. This widget
shows every row together: which one is active, what kind it is (export,
computed, joint), and gives each its own close button, without hiding the
others to do it.

Deliberately not a checkbox-per-row "visibility" toggle in the sense of
several embeddings drawn at once -- the explorer is a single scatter, and the
only way several embeddings' positions are actually comparable on one canvas
is a joint projection (see joint_projection.py), which already has its own
entry point (Combine...). What this widget toggles is which ONE is active,
listed rather than hidden behind a combobox, plus which ones exist at all.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..data.workspace import SOURCE_COMPUTED, SOURCE_EXTERNAL, SOURCE_JOINT
from ..gui import themes

KEY_ROLE = Qt.ItemDataRole.UserRole + 1

# Short tag shown after the name, matching EmbeddingEntry.label()'s own
# vocabulary so the two never say different things about the same entry.
_SOURCE_TAG = {
    SOURCE_COMPUTED: "computed",
    SOURCE_EXTERNAL: "external",
    SOURCE_JOINT: "joint",
}


class _Row(QFrame):
    """One embedding: its name, a close button, highlighted while active."""

    clicked = Signal(str)  # entry key
    close_requested = Signal(str)

    def __init__(self, key: str, label: str, tooltip: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        self._active = False

        self.name_label = QLabel(label)
        self.name_label.setWordWrap(True)
        self.name_label.setToolTip(tooltip)

        self.close_button = QPushButton("✕")
        self.close_button.setFixedSize(18, 18)
        self.close_button.setToolTip("Close this embedding")
        self.close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_button.clicked.connect(lambda: self.close_requested.emit(self.key))

        layout = QHBoxLayout()
        layout.setContentsMargins(8, 5, 6, 5)
        layout.setSpacing(6)
        layout.addWidget(self.name_label, 1)
        layout.addWidget(self.close_button)
        self.setLayout(layout)
        self.set_active(False)

    def set_active(self, active: bool) -> None:
        self._active = active
        colours = themes.current()
        self.setStyleSheet(
            f"_Row {{ background: {colours.surface_raised};"
            f" border: 1px solid {colours.accent}; border-radius: 4px; }}"
            if active
            else f"_Row {{ background: {colours.surface};"
            f" border: 1px solid {colours.border}; border-radius: 4px; }}"
        )

    def mousePressEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.key)
        super().mousePressEvent(event)


class OpenEmbeddingsList(QWidget):
    """Every open embedding, one row each, with the active one highlighted."""

    # Emitted when a row is clicked to make it active -- the entry key.
    activated = Signal(str)
    # Emitted when a row's close button is pressed -- the entry key.
    close_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: dict[str, _Row] = {}
        self._active_key: str | None = None

        self.heading = QLabel("Open embeddings")
        self.heading.setObjectName("hint")

        self.list_layout = QVBoxLayout()
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(4)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.heading)
        layout.addLayout(self.list_layout)
        self.setLayout(layout)
        self.hide()  # nothing to show until there are >= 2 entries

    def restyle(self) -> None:
        for row in self._rows.values():
            row.set_active(row.key == self._active_key)

    def set_entries(self, entries, active_key: str | None) -> None:
        """Rebuild the list from ``entries`` (EmbeddingEntry instances).

        Rebuilds wholesale rather than diffing -- the list is short (open
        embeddings in one session, not thousands of images) and this stays
        simple and cannot drift out of sync with the workspace.
        """
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._rows.clear()
        self._active_key = active_key

        for entry in entries:
            tag = _SOURCE_TAG.get(entry.source)
            label = f"{entry.name}" + (f"  ·  {tag}" if tag else "")
            row = _Row(entry.key, label, entry.describe())
            row.clicked.connect(self.activated.emit)
            row.close_requested.connect(self.close_requested.emit)
            row.set_active(entry.key == active_key)
            self._rows[entry.key] = row
            self.list_layout.addWidget(row)

        # Only worth showing once there is a choice to make -- matching the
        # dropdown this replaces, which hid itself under the same condition.
        self.setVisible(len(entries) > 1)
