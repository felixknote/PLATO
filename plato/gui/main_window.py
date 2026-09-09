"""Application shell: an empty state until data is loaded, then the analysis
arms as tabs.

The window owns no analysis logic of its own. It holds the menu, the status
bar, the window-level shortcuts and the session, and delegates everything else
to whichever arm is on screen:

* **Plate Browser** -- the thumbnail grid, two-panel compare and N-way
  comparison (``views.browser_arm``).
* **Embedding Explorer** -- UMAP/t-SNE over precomputed features
  (``views.explorer``).

Both arms read the same ``Session``, so a plate loaded once is available to
both, and the image viewer opened from either is the same window.
"""

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
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..data.session import DuplicatePlateError, Session
from ..views.browser_arm import BrowserArm
from ..views.explorer import EmbeddingExplorer
from .branding import DISPLAY_STACK, WORDMARK_TRACKING, logo_pixmap
from .load_dialog import LoadPlateDialog
from .plates_dialog import PlatesDialog
from .settings import SettingsDialog
from .theme import TEXT, TEXT_FAINT, TEXT_MUTED

ORGANISATION = "plato"

BROWSER_TAB = 0
EXPLORER_TAB = 1

# Where the explorer looks for embedding exports on first open. A default, not
# a requirement -- the Browse button accepts any folder.
DEFAULT_EMBEDDING_ROOT = Path(r"Z:\Analysis\DINO")


class MainWindow(QMainWindow):
    def __init__(self, session: Session) -> None:
        super().__init__()
        self.session = session
        self.settings = QSettings(ORGANISATION, "plato")

        self.tabs: QTabWidget | None = None
        self.browser: BrowserArm | None = None
        self.explorer: EmbeddingExplorer | None = None

        self.setStatusBar(QStatusBar())
        self._build_menu()
        self._build_shortcuts()

        if session.is_empty:
            self._show_empty_state()
        else:
            self._build_tabs()

        self.showMaximized()
        self._restore_session()

    # -- empty state --------------------------------------------------------

    def _show_empty_state(self) -> None:
        self.setWindowTitle("PLATO")

        mark = QLabel()
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = logo_pixmap(132)
        if not pixmap.isNull():
            mark.setPixmap(pixmap)

        title = QLabel("PLATO")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Tracking is what makes five capitals read as a mark rather than a
        # word; the display face carries the rest of the character.
        title.setStyleSheet(
            f"color: {TEXT}; font-family: {DISPLAY_STACK}; font-size: 42px;"
            f"font-weight: 600; letter-spacing: {WORDMARK_TRACKING}em;"
            # The tracking adds space after the final letter too, which throws
            # the mark visibly off-centre under an image; pad the other side to
            # put it back.
            f"padding-left: {WORDMARK_TRACKING}em;"
        )

        subtitle = QLabel("Plate-map-aware browsing for high-content screens")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 14px; font-family: {DISPLAY_STACK};"
            "letter-spacing: 0.02em;"
        )

        hint = QLabel("Choose an image folder and a plate map to get started.")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 12px;")

        load_button = QPushButton("Load Data")
        load_button.setFixedWidth(200)
        load_button.setDefault(True)
        load_button.clicked.connect(self._load_data)

        # The explorer needs no plate map -- it reads a precomputed embedding
        # export -- so it is reachable without loading images first. Without
        # this the empty state is a dead end for anyone who only wants to look
        # at embedding space.
        explorer_button = QPushButton("Open Embedding Explorer")
        explorer_button.setFixedWidth(200)
        explorer_button.clicked.connect(self._open_explorer_from_empty)

        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.addStretch(1)
        layout.addWidget(mark, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(14)
        layout.addWidget(title, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(18)
        layout.addWidget(load_button, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(explorer_button, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(6)
        layout.addWidget(hint, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def _load_data(self) -> None:
        dialog = LoadPlateDialog(self, title="Load data")
        if dialog.exec() != LoadPlateDialog.DialogCode.Accepted or dialog.result_config is None:
            return
        if self._add_to_session(dialog.result_config) is None:
            return
        self._build_tabs()

    def _open_explorer_from_empty(self) -> None:
        self._build_tabs(browser_enabled=False)
        if self.tabs is not None:
            self.tabs.setCurrentIndex(EXPLORER_TAB)

    def _add_plate(self) -> None:
        dialog = LoadPlateDialog(self, title="Add plate")
        if dialog.exec() != LoadPlateDialog.DialogCode.Accepted or dialog.result_config is None:
            return
        plate = self._add_to_session(dialog.result_config)
        if plate is None:
            return
        self._plates_changed()
        self._show_status(
            f"added {plate.name} — {len(self.session.plates)} plates, "
            f"{self.session.count()} images total"
        )

    def _add_to_session(self, cfg) -> object | None:  # noqa: ANN001
        """Add one plate, reporting a re-add rather than silently doubling it.

        Adding an already-loaded plate is a mistake, not a request: the grid
        would show every image twice and flagging one copy would leave the
        other unflagged, which is worse than nothing happening.
        """
        try:
            return self.session.add_plate(cfg)
        except DuplicatePlateError as exc:
            QMessageBox.information(
                self,
                "Add plate",
                f"{exc} is already loaded — nothing was added.\n\n"
                "Loading it twice would show every image twice and split its "
                "flags across two copies.",
            )
            return None

    def _plates_changed(self) -> None:
        """Push a change in the set of loaded plates through the whole window.

        Filters, pooled display limits and the title all derive from which
        plates are loaded, so a plain refresh() is not enough.
        """
        if self.browser is not None:
            self.browser.plates_changed()
        self._update_title()

    def _update_title(self) -> None:
        plates = len(self.session.plates)
        if not plates:
            self.setWindowTitle("PLATO")
            return
        suffix = "" if plates == 1 else f" · {plates} plates"
        self.setWindowTitle(f"PLATO — {self.session.count()} images{suffix}")

    def _manage_plates(self) -> None:
        if self.session.is_empty:
            QMessageBox.information(self, "Plates", "No plates are loaded.")
            return
        dialog = PlatesDialog(self.session, self)
        dialog.exec()
        if dialog.removed:
            if self.session.is_empty:
                self._teardown_ui()
                self._show_status("all plates removed")
                return
            self._plates_changed()
            self._show_status(
                f"{len(self.session.plates)} plates, {self.session.count()} images total"
            )

    def _teardown_ui(self) -> None:
        if self.browser is not None:
            self.browser.clear_comparison()
        if self.compare_action.isChecked():
            self.compare_action.setChecked(False)
        self.tabs = None
        self.browser = None
        self.explorer = None
        self._show_empty_state()

    # -- tabs ---------------------------------------------------------------

    def _build_tabs(self, *, browser_enabled: bool = True) -> None:
        """Create the tab shell, or refresh it if it already exists."""
        if self.tabs is not None:
            if self.browser is not None:
                self.browser.plates_changed()
            self._update_title()
            return

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        if browser_enabled and not self.session.is_empty:
            self.browser = BrowserArm(self.session, self)
            self.browser.status.connect(self._show_status)
            self.browser.panel_shortcuts_enabled.connect(
                self._set_panel_shortcuts_enabled
            )
            self.tabs.addTab(self.browser, "Plate Browser")
        else:
            # A placeholder keeps the explorer at a stable tab index whether or
            # not images are loaded, so the menu and shortcuts do not have to
            # care which case they are in. It carries the Load button itself:
            # a tab that only says "load a plate" is a dead end, since the
            # menu is the one place a user looking at this screen is not
            # looking.
            self.tabs.addTab(self._browser_placeholder(), "Plate Browser")

        work_dir = (
            self.session.plates[0].cfg.project.work_dir
            if self.session.plates
            else Path(".plato")
        )
        self.explorer = EmbeddingExplorer(self.session, work_dir, self)
        self.explorer.status.connect(self._show_status)
        self.tabs.addTab(self.explorer, "Embedding Explorer")

        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)
        self._update_title()

        if DEFAULT_EMBEDDING_ROOT.is_dir():
            self.explorer.set_dataset_root(DEFAULT_EMBEDDING_ROOT)

    def _browser_placeholder(self) -> QWidget:
        """The empty Plate Browser tab: says what is missing, and fixes it."""
        message = QLabel("No plates loaded")
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 15px;")

        hint = QLabel("Choose an image folder and a plate map to browse images.")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 12px;")

        button = QPushButton("Load Data")
        button.setFixedWidth(200)
        button.clicked.connect(self._load_data)

        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.addStretch(1)
        layout.addWidget(message, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(10)
        layout.addWidget(button, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)

        container = QWidget()
        container.setLayout(layout)
        return container

    def _on_tab_changed(self, index: int) -> None:
        """Only the browser wants the grid's single-key shortcuts.

        They are window-level, so they would otherwise fire while the explorer
        is on screen -- where R, 0-5 and Space mean nothing and Space would
        flag an image the user cannot see.
        """
        self._set_panel_shortcuts_enabled(
            index == BROWSER_TAB and self.browser is not None
        )
        for action in self._browser_actions:
            action.setEnabled(index == BROWSER_TAB and self.browser is not None)

    def _build_menu(self) -> None:
        data_menu = self.menuBar().addMenu("&Data")

        load_action = QAction("Load Data…", self)
        load_action.triggered.connect(self._load_data)
        data_menu.addAction(load_action)

        self.add_plate_action = QAction("Add Plate…", self)
        self.add_plate_action.triggered.connect(self._add_plate)
        data_menu.addAction(self.add_plate_action)

        self.manage_plates_action = QAction("Loaded Plates…", self)
        self.manage_plates_action.triggered.connect(self._manage_plates)
        data_menu.addAction(self.manage_plates_action)

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

        view_menu.addSeparator()
        browser_tab = QAction("Plate Browser", self)
        browser_tab.setShortcut("Ctrl+1")
        browser_tab.triggered.connect(lambda: self._select_tab(BROWSER_TAB))
        view_menu.addAction(browser_tab)

        explorer_tab = QAction("Embedding Explorer", self)
        explorer_tab.setShortcut("Ctrl+2")
        explorer_tab.triggered.connect(lambda: self._select_tab(EXPLORER_TAB))
        view_menu.addAction(explorer_tab)

        view_menu.addSeparator()
        settings_action = QAction("Settings…", self)
        settings_action.triggered.connect(self.open_settings)
        view_menu.addAction(settings_action)

        help_menu = self.menuBar().addMenu("&Help")
        keys = QAction("Keyboard shortcuts", self)
        keys.triggered.connect(self.show_shortcuts)
        help_menu.addAction(keys)

        # Actions that only mean something on the browser tab, greyed out
        # elsewhere rather than silently doing nothing.
        self._browser_actions = [
            self.compare_action,
            compare_by,
            close_compare,
            self.blind_action,
        ]

    def _select_tab(self, index: int) -> None:
        if self.tabs is not None and index < self.tabs.count():
            self.tabs.setCurrentIndex(index)

    def _build_shortcuts(self) -> None:
        def on_active(action: str, *args) -> None:
            """Run a BrowserPanel method, if there is a panel to run it on."""
            panel = self._active()
            if panel is not None:
                getattr(panel, action)(*args)

        self._panel_shortcuts = [
            QShortcut(QKeySequence(Qt.Key.Key_Return), self, self._open_current),
            QShortcut(QKeySequence(Qt.Key.Key_Enter), self, self._open_current),
            QShortcut(QKeySequence(Qt.Key.Key_Space), self, lambda: on_active("toggle_flag")),
            QShortcut(QKeySequence(Qt.Key.Key_R), self, lambda: on_active("jump_random")),
            *(
                QShortcut(
                    QKeySequence(str(digit)),
                    self,
                    lambda d=digit: on_active("set_rating", d),
                )
                for digit in range(1, 6)
            ),
            QShortcut(QKeySequence(Qt.Key.Key_0), self, lambda: on_active("set_rating", None)),
        ]

    def _set_panel_shortcuts_enabled(self, enabled: bool) -> None:
        """Silence the grid's keys while something else owns the window.

        These are window-level shortcuts, so they fire ahead of the focused
        widget's keyPressEvent. Two things need them off: the comparison view,
        whose own keys overlap (0 resets zoom there, clears a rating here), and
        the explorer tab, where none of them apply.
        """
        for shortcut in self._panel_shortcuts:
            shortcut.setEnabled(enabled)

    def _show_status(self, message: str) -> None:
        self.statusBar().showMessage(message, 4000)

    # -- browser delegation -------------------------------------------------

    def _active(self):
        return self.browser.active_panel() if self.browser is not None else None

    def _open_current(self) -> None:
        if self.browser is not None:
            self.browser.open_current()

    def set_compare(self, enabled: bool) -> None:
        if self.browser is not None:
            self.browser.set_compare(enabled)

    def compare_by_variable(self) -> None:
        if self.browser is None:
            return

        def turn_off_two_panel() -> None:
            if self.compare_action.isChecked():
                self.compare_action.setChecked(False)

        self.browser.compare_by_variable(on_two_panel_compare_off=turn_off_two_panel)

    def exit_comparison(self) -> None:
        if self.browser is not None:
            self.browser.exit_comparison()

    def set_blind(self, enabled: bool) -> None:
        if self.browser is not None:
            self.browser.set_blind(enabled)
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
        # key it was not told about.
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
            "Tabs\n"
            "  Ctrl+1      plate browser\n"
            "  Ctrl+2      embedding explorer\n\n"
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
            "Embedding explorer\n"
            "  hover       preview the original image\n"
            "  click       open it at full resolution\n"
            "  scroll      zoom · drag  pan\n\n"
            "Comparison view\n"
            "  ←/→         step the selected column (click one to select)\n"
            "  scroll      zoom all columns, centred on the cursor\n"
            "  drag        pan all columns\n"
            "  + / −       zoom in / out\n"
            "  0           reset zoom\n\n"
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
        if search and self.browser is not None and self.browser.primary is not None:
            self.browser.primary.search.setText(str(search))

    def closeEvent(self, event) -> None:  # noqa: ANN001, N802
        if self.browser is not None and self.browser.primary is not None:
            self.settings.setValue("search", self.browser.primary.search.text())
        super().closeEvent(event)
