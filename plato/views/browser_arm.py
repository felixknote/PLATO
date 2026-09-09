"""The plate-browsing arm: one or two thumbnail panels, or an N-way comparison.

This is the original PLATO screen, lifted out of ``MainWindow`` unchanged so
that the window can host it as one tab beside the embedding explorer. The
behaviour is deliberately identical -- the same panels, the same comparison
view, the same export path -- because it is the part of the app that already
works and is used.

What moved is only ownership: the window used to hold ``primary``,
``secondary`` and ``comparison`` directly, and now holds one of these. Menu
actions and shortcuts call through to the same methods by the same names.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QMessageBox, QSplitter, QVBoxLayout, QWidget

from ..data.session import Session
from .browser import BrowserPanel
from .compare_dialog import CompareSetupDialog
from .compare_view import ComparisonColumn, ComparisonView


class BrowserArm(QWidget):
    """Holds the browsing panels and the comparison view."""

    status = Signal(str)
    # Emitted when the comparison opens or closes, so the shell can silence
    # the grid's window-level shortcuts while the comparison owns the keys.
    panel_shortcuts_enabled = Signal(bool)

    def __init__(self, session: Session, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        self.primary: BrowserPanel | None = None
        self.secondary: BrowserPanel | None = None
        self.comparison: ComparisonView | None = None

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.primary = BrowserPanel(self.session, self)
        self.primary.status.connect(self.status.emit)
        self.splitter.addWidget(self.primary)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.splitter)
        self.setLayout(layout)

    # -- plates -----------------------------------------------------------

    def plates_changed(self) -> None:
        """Re-read everything derived from the set of loaded plates."""
        for panel in filter(None, (self.primary, self.secondary)):
            panel.plates_changed()
        # Rows handed to the comparison view carry session_index values that a
        # removal invalidates, and its columns were built from the old plate
        # set either way.
        self.clear_comparison()

    # -- panels -----------------------------------------------------------

    def active_panel(self) -> BrowserPanel | None:
        """The panel keyboard actions apply to, or None.

        Returns None while the comparison view is open: its own keys overlap
        the panel's (0 resets zoom there, clears a rating here).
        """
        if self.comparison is not None:
            return None
        if self.secondary is not None and self.secondary.view.hasFocus():
            return self.secondary
        return self.primary

    def open_current(self) -> None:
        panel = self.active_panel()
        if panel is not None:
            panel.open_viewer(panel.current_index())

    def set_compare(self, enabled: bool) -> None:
        if self.primary is None:
            return
        if enabled and self.secondary is None:
            self.secondary = BrowserPanel(self.session, self)
            self.secondary.status.connect(self.status.emit)
            self.secondary.set_blind(self.primary.blind)
            self.splitter.addWidget(self.secondary)
            self.splitter.setSizes([1, 1])
        elif not enabled and self.secondary is not None:
            self.secondary.setParent(None)
            self.secondary.deleteLater()
            self.secondary = None

    def compare_by_variable(self, *, on_two_panel_compare_off=None) -> bool:
        """One panel per value of a chosen variable, all sharing a base filter.

        Returns True if a comparison was opened.
        """
        if self.session.is_empty:
            return False
        dialog = CompareSetupDialog(self.session, self)
        if dialog.exec() != CompareSetupDialog.DialogCode.Accepted:
            return False

        column = dialog.variable_column()
        values = dialog.selected_values()
        base = dialog.base_filter()
        if not column or not values:
            return False

        self.clear_comparison()
        # The two-panel compare and the N-panel compare are the same screen
        # real estate; leaving both on would stack unrelated panels.
        if on_two_panel_compare_off is not None:
            on_two_panel_compare_off()

        limits = self.session.display_limits()
        columns = []
        for value in values:
            rows = self.session.query({**base, column: [value]})
            col = ComparisonColumn(f"{self.session.label(column)}: {value}", rows)
            # Same fixed per-channel limits the grid uses, so a difference
            # between columns is a difference in the sample, not in scaling.
            channel = rows[0].channel if rows else None
            col.set_levels(limits.get(channel or "_"))
            columns.append(col)

        self.comparison = ComparisonView(self)
        self.comparison.status.connect(self.status.emit)
        self.comparison.export_button.clicked.connect(self.export_comparison)
        self.comparison.set_columns(columns)
        self.splitter.addWidget(self.comparison)
        self.panel_shortcuts_enabled.emit(False)
        self.comparison.setFocus(Qt.FocusReason.OtherFocusReason)

        if self.primary is not None:
            self.primary.setVisible(False)
        base_text = ", ".join(f"{k}={v[0]}" for k, v in base.items()) or "all data"
        self.status.emit(
            f"comparing {column} across {len(values)} values — {base_text}"
        )
        return True

    def export_comparison(self) -> None:
        """Export exactly the images currently on screen, one per slice."""
        if self.comparison is None:
            return
        rows = self.comparison.visible_rows()
        if not rows:
            QMessageBox.information(self, "Export", "Nothing on screen to export.")
            return
        # Reuses the panel's export path so format choice, scale-bar baking and
        # shared contrast behave identically to exporting from the grid.
        exporter = self.primary or BrowserPanel(self.session, self)
        exporter.export_rows(rows)

    def clear_comparison(self) -> None:
        if self.comparison is not None:
            self.comparison.setParent(None)
            self.comparison.deleteLater()
            self.comparison = None
            self.panel_shortcuts_enabled.emit(True)
        if self.primary is not None:
            self.primary.setVisible(True)

    def exit_comparison(self) -> bool:
        if self.comparison is None:
            return False
        self.clear_comparison()
        self.status.emit("comparison closed")
        return True

    def set_blind(self, enabled: bool) -> None:
        for panel in filter(None, (self.primary, self.secondary)):
            panel.set_blind(enabled)
