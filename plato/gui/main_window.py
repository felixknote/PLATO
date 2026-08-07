"""Main window: empty state until data is loaded, then one or two browser
panels, shortcuts, persisted session state."""

from __future__ import annotations

import csv
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .compare_dialog import CompareSetupDialog
from .load_dialog import LoadPlateDialog
from .panel import BrowserPanel
from .session import Session
from .settings import SettingsDialog

ORGANISATION = "plato"


class MainWindow(QMainWindow):
    def __init__(self, session: Session) -> None:
        super().__init__()
        self.session = session
        self.settings = QSettings(ORGANISATION, "plato")

        self.splitter: QSplitter | None = None
        self.primary: BrowserPanel | None = None
        self.secondary: BrowserPanel | None = None
        # Panels of an N-way comparison; empty when not comparing.
        self.comparison_panels: list[BrowserPanel] = []

        self.setStatusBar(QStatusBar())
        self._build_menu()
        self._build_shortcuts()

        if session.is_empty:
            self._show_empty_state()
        else:
            self._build_browser_ui()

        self.showMaximized()
        self._restore_session()

    # -- empty state --------------------------------------------------------

    def _show_empty_state(self) -> None:
        self.setWindowTitle("PLATO")
        label = QLabel("No data loaded")
        font = label.font()
        font.setPointSize(18)
        label.setFont(font)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        load_button = QPushButton("Load Data")
        load_button.setFixedWidth(200)
        load_button.clicked.connect(self._load_data)

        layout = QVBoxLayout()
        layout.addStretch(1)
        layout.addWidget(label, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(load_button, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def _load_data(self) -> None:
        dialog = LoadPlateDialog(self, title="Load data")
        if dialog.exec() != LoadPlateDialog.DialogCode.Accepted or dialog.result_config is None:
            return
        self.session.add_plate(dialog.result_config)
        self._build_browser_ui()

    def _add_plate(self) -> None:
        dialog = LoadPlateDialog(self, title="Add plate")
        if dialog.exec() != LoadPlateDialog.DialogCode.Accepted or dialog.result_config is None:
            return
        self.session.add_plate(dialog.result_config)
        for panel in filter(None, (self.primary, self.secondary)):
            panel.refresh()
        self._show_status(f"added plate — {self.session.count()} images total")

    # -- browser chrome -------------------------------------------------------

    def _build_browser_ui(self) -> None:
        if self.splitter is None:
            self.splitter = QSplitter(Qt.Orientation.Horizontal)
            self.primary = BrowserPanel(self.session, self)
            self.primary.status.connect(self._show_status)
            self.splitter.addWidget(self.primary)
            self.setCentralWidget(self.splitter)
        else:
            self.primary.refresh()
        self.setWindowTitle(f"PLATO — {self.session.count()} images")

    def _build_menu(self) -> None:
        data_menu = self.menuBar().addMenu("&Data")

        load_action = QAction("Load Data…", self)
        load_action.triggered.connect(self._load_data)
        data_menu.addAction(load_action)

        self.add_plate_action = QAction("Add Plate…", self)
        self.add_plate_action.triggered.connect(self._add_plate)
        data_menu.addAction(self.add_plate_action)

        export = QAction("Export flags and ratings as CSV…", self)
        export.triggered.connect(self.export_annotations)
        data_menu.addAction(export)

        view_menu = self.menuBar().addMenu("&View")

        self.compare_action = QAction("Compare two conditions", self, checkable=True)
        self.compare_action.setShortcut("Ctrl+D")
        self.compare_action.toggled.connect(self.set_compare)
        view_menu.addAction(self.compare_action)

        compare_by = QAction("Compare along a variable…", self)
        compare_by.setShortcut("Ctrl+Shift+D")
        compare_by.triggered.connect(self.compare_by_variable)
        view_menu.addAction(compare_by)

        close_compare = QAction("Close comparison", self)
        close_compare.setShortcut("Ctrl+Shift+W")
        close_compare.triggered.connect(self.exit_comparison)
        view_menu.addAction(close_compare)

        self.blind_action = QAction("Blinded review", self, checkable=True)
        self.blind_action.setShortcut("Ctrl+B")
        self.blind_action.setStatusTip(
            "Hide metadata and randomise order while scoring phenotypes."
        )
        self.blind_action.toggled.connect(self.set_blind)
        view_menu.addAction(self.blind_action)

        settings_action = QAction("Settings…", self)
        settings_action.triggered.connect(self.open_settings)
        view_menu.addAction(settings_action)

        help_menu = self.menuBar().addMenu("&Help")
        keys = QAction("Keyboard shortcuts", self)
        keys.triggered.connect(self.show_shortcuts)
        help_menu.addAction(keys)

    def _build_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Return), self, self._open_current)
        QShortcut(QKeySequence(Qt.Key.Key_Enter), self, self._open_current)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, lambda: self._active_or_none() and self._active().toggle_flag())
        QShortcut(QKeySequence(Qt.Key.Key_R), self, lambda: self._active_or_none() and self._active().jump_random())
        for digit in range(1, 6):
            QShortcut(
                QKeySequence(str(digit)),
                self,
                lambda d=digit: self._active_or_none() and self._active().set_rating(d),
            )
        QShortcut(
            QKeySequence(Qt.Key.Key_0),
            self,
            lambda: self._active_or_none() and self._active().set_rating(None),
        )

    def _show_status(self, message: str) -> None:
        self.statusBar().showMessage(message, 4000)

    # -- panels -----------------------------------------------------------

    def _active_or_none(self) -> bool:
        return self.primary is not None

    def _active(self) -> BrowserPanel:
        if self.secondary is not None and self.secondary.view.hasFocus():
            return self.secondary
        return self.primary

    def _open_current(self) -> None:
        if self.primary is None:
            return
        panel = self._active()
        panel.open_viewer(panel.current_index())

    def set_compare(self, enabled: bool) -> None:
        if self.primary is None or self.splitter is None:
            return
        if enabled and self.secondary is None:
            self.secondary = BrowserPanel(self.session, self)
            self.secondary.status.connect(self._show_status)
            self.secondary.set_blind(self.blind_action.isChecked())
            self.splitter.addWidget(self.secondary)
            self.splitter.setSizes([1, 1])
        elif not enabled and self.secondary is not None:
            self.secondary.setParent(None)
            self.secondary.deleteLater()
            self.secondary = None

    def compare_by_variable(self) -> None:
        """One panel per value of a chosen variable, all sharing a base condition."""
        if self.session.is_empty or self.splitter is None:
            return
        dialog = CompareSetupDialog(self.session, self)
        if dialog.exec() != CompareSetupDialog.DialogCode.Accepted:
            return

        column = dialog.variable_column()
        values = dialog.selected_values()
        base = dialog.base_filter()
        if not column or not values:
            return

        self._clear_comparison()
        # The two-panel compare and the N-panel compare are the same screen
        # real estate; leaving both on would stack unrelated panels.
        if self.compare_action.isChecked():
            self.compare_action.setChecked(False)

        for value in values:
            panel = BrowserPanel(self.session, self)
            panel.status.connect(self._show_status)
            panel.set_blind(self.blind_action.isChecked())
            panel.set_locked_filters({**base, column: [value]})
            panel.set_title(f"{self.session.label(column)}: {value}")
            panel.set_compact(True)
            self.splitter.addWidget(panel)
            self.comparison_panels.append(panel)

        if self.primary is not None:
            self.primary.setVisible(False)
        self.splitter.setSizes([1] * self.splitter.count())
        base_text = ", ".join(f"{k}={v[0]}" for k, v in base.items()) or "all data"
        self._show_status(f"comparing {column} across {len(values)} values — {base_text}")

    def _clear_comparison(self) -> None:
        for panel in self.comparison_panels:
            panel.setParent(None)
            panel.deleteLater()
        self.comparison_panels.clear()
        if self.primary is not None:
            self.primary.setVisible(True)

    def exit_comparison(self) -> None:
        if not self.comparison_panels:
            return
        self._clear_comparison()
        self._show_status("comparison closed")

    def set_blind(self, enabled: bool) -> None:
        for panel in filter(None, (self.primary, self.secondary, *self.comparison_panels)):
            panel.set_blind(enabled)
        self._show_status(
            "Blinded review on — metadata hidden, order randomised"
            if enabled
            else "Blinded review off"
        )

    # -- data -------------------------------------------------------------

    def export_annotations(self) -> None:
        rows = self.session.export_annotations()
        if not rows:
            QMessageBox.information(self, "Export", "Nothing has been flagged or rated yet.")
            return
        target, _ = QFileDialog.getSaveFileName(
            self, "Save annotations", "annotations.csv", "CSV (*.csv)"
        )
        if not target:
            return
        # Field names are the union across every row, not row 0's keys: plates
        # in one session can carry different plate-map columns (one screen's
        # "antibiotic" is another's "compound"), and DictWriter raises on any
        # key it was not told about. Taking the union keeps a mixed session
        # exportable, with blanks where a plate has no such column.
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with Path(target).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(rows)
        self._show_status(f"exported {len(rows)} annotations to {target}")

    def open_settings(self) -> None:
        SettingsDialog(self).exec()

    def show_shortcuts(self) -> None:
        QMessageBox.information(
            self,
            "Keyboard shortcuts",
            "Browser\n"
            "  Arrows      move between thumbnails\n"
            "  Enter       open full resolution\n"
            "  Space       flag / unflag selection\n"
            "  1-5 / 0     rate selection / clear rating\n"
            "  R           jump to a random image\n"
            "  Ctrl+D      compare two conditions\n"
            "  Ctrl+Shift+D  compare along a variable (one panel per value)\n"
            "  Ctrl+Shift+W  close comparison\n"
            "  Ctrl+B      blinded review\n\n"
            "Image window\n"
            "  Left/Right  step through the current filter\n"
            "  A           toggle per-image autoscale\n"
            "  S           toggle scale bar\n"
            "  D           measure distance (click two points)\n"
            "  Esc         close",
        )

    # -- session ----------------------------------------------------------

    def _restore_session(self) -> None:
        search = self.settings.value("search", "")
        if search and self.primary is not None:
            self.primary.search.setText(str(search))

    def closeEvent(self, event) -> None:  # noqa: ANN001, N802
        if self.primary is not None:
            self.settings.setValue("search", self.primary.search.text())
        super().closeEvent(event)
