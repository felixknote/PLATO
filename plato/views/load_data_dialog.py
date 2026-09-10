"""Load several folders of embeddings and plates in one pass.

Replaces the old "pick one folder, get one embedding" flow. A user adds one
or more root folders (a screen's parent directory, a lab drive's top level,
whatever they have), each is scanned recursively for embedding exports and
plate image folders, and everything found is offered as a checked list to
load in bulk.

Scanning runs on a worker thread via ``plato.data.discovery.scan`` -- never
on the GUI thread, since the roots here are sometimes large network shares or
mechanical disks. See that module for why the walk is bounded rather than
exhaustive.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..data.discovery import EMBEDDING, PLATE, ScanResult, scan
from ..gui import themes

PATH_ROLE = Qt.ItemDataRole.UserRole + 1
KIND_ROLE = Qt.ItemDataRole.UserRole + 2


class _ScanSignals(QObject):
    finished = Signal(object)  # ScanResult
    failed = Signal(str)


class _ScanTask(QRunnable):
    def __init__(self, roots: list[Path], signals: _ScanSignals) -> None:
        super().__init__()
        self._roots = roots
        self._signals = signals

    def run(self) -> None:  # pragma: no cover - worker thread
        try:
            result = scan(self._roots)
        except Exception as exc:  # noqa: BLE001 - surfaced in the dialog
            try:
                self._signals.failed.emit(str(exc))
            except RuntimeError:
                pass
            return
        try:
            self._signals.finished.emit(result)
        except RuntimeError:
            pass


class LoadDataDialog(QDialog):
    """Add folders, scan them, choose what to load."""

    def __init__(self, parent: QWidget | None = None, *, start: Path | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Load data")
        self.resize(640, 560)

        self.roots: list[Path] = [start] if start is not None else []
        self._pool = QThreadPool.globalInstance()
        self._signals = _ScanSignals()
        self._signals.finished.connect(self._on_scanned)
        self._signals.failed.connect(self._on_scan_failed)

        # -- roots
        self.root_list = QListWidget()
        self.root_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.root_list.setMaximumHeight(90)

        add_button = QPushButton("Add folder…")
        add_button.clicked.connect(self._add_root)
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self._remove_selected_roots)
        scan_button = QPushButton("Scan")
        scan_button.clicked.connect(self._start_scan)
        self.scan_button = scan_button

        root_buttons = QHBoxLayout()
        root_buttons.addWidget(add_button)
        root_buttons.addWidget(remove_button)
        root_buttons.addStretch(1)
        root_buttons.addWidget(scan_button)

        # -- results
        self.status_label = QLabel(
            "Add one or more folders, then Scan. Each is searched "
            "recursively for embedding exports and plate image folders."
        )
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("hint")

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["", "Found", "Detail"])
        self.tree.setColumnWidth(0, 28)
        self.tree.setColumnWidth(1, 380)
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)

        self.select_all_box = QCheckBox("Select all")
        self.select_all_box.setChecked(True)
        self.select_all_box.toggled.connect(self._set_all_checked)

        # -- buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
        )
        self.load_button = QPushButton("Load selected")
        self.load_button.setEnabled(False)
        self.load_button.setDefault(True)
        self.buttons.addButton(
            self.load_button, QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Folders to search"))
        layout.addWidget(self.root_list)
        layout.addLayout(root_buttons)
        layout.addWidget(self.status_label)
        layout.addWidget(self.tree, 1)
        layout.addWidget(self.select_all_box)
        layout.addWidget(self.buttons)
        self.setLayout(layout)

        self._refresh_root_list()
        self.restyle()

    def restyle(self) -> None:
        colours = themes.current()
        self.status_label.setStyleSheet(f"color: {colours.text_faint}; font-size: 11px;")

    # -- roots ---------------------------------------------------------

    def _refresh_root_list(self) -> None:
        self.root_list.clear()
        for root in self.roots:
            self.root_list.addItem(str(root))

    def _add_root(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Add a folder to search")
        if not chosen:
            return
        path = Path(chosen)
        if path not in self.roots:
            self.roots.append(path)
            self._refresh_root_list()

    def _remove_selected_roots(self) -> None:
        selected = {item.text() for item in self.root_list.selectedItems()}
        self.roots = [r for r in self.roots if str(r) not in selected]
        self._refresh_root_list()

    # -- scanning --------------------------------------------------------

    def _start_scan(self) -> None:
        if not self.roots:
            self.status_label.setText("Add at least one folder first.")
            return
        self.scan_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.tree.clear()
        self.status_label.setText("Scanning…")
        self._pool.start(_ScanTask(list(self.roots), self._signals))

    def _on_scan_failed(self, message: str) -> None:
        self.scan_button.setEnabled(True)
        self.status_label.setText(f"Scan failed: {message}")

    def _on_scanned(self, result: ScanResult) -> None:
        self.scan_button.setEnabled(True)
        self.tree.clear()

        for found in result.found:
            item = QTreeWidgetItem(
                ["", "Embedding" if found.kind == EMBEDDING else "Plate images", found.detail]
            )
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Checked)
            item.setData(0, PATH_ROLE, str(found.path))
            item.setData(0, KIND_ROLE, found.kind)
            item.setToolTip(1, str(found.path))
            self.tree.addTopLevelItem(item)
            # The path column carries the actual location; put it under the
            # kind label so it stays visible without a fourth column fighting
            # the dialog for width.
            item.setText(1, f"{item.text(1)} — {found.path.name}")

        count = self.tree.topLevelItemCount()
        self.load_button.setEnabled(count > 0)
        parts = [f"{count} found"]
        if result.truncated:
            parts.append(
                f"stopped early after {result.directories_visited:,} folders "
                f"({result.seconds:.0f}s) — narrow the search folder for a "
                "complete scan"
            )
        else:
            parts.append(f"{result.directories_visited:,} folders checked, {result.seconds:.1f}s")
        self.status_label.setText(" · ".join(parts) if count else "Nothing found under these folders.")

    def _set_all_checked(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.tree.topLevelItemCount()):
            self.tree.topLevelItem(i).setCheckState(0, state)

    # -- result ------------------------------------------------------------

    def selected(self) -> tuple[list[Path], list[Path]]:
        """(embedding directories, plate directories) that were checked."""
        embeddings: list[Path] = []
        plates: list[Path] = []
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item.checkState(0) != Qt.CheckState.Checked:
                continue
            path = Path(item.data(0, PATH_ROLE))
            if item.data(0, KIND_ROLE) == EMBEDDING:
                embeddings.append(path)
            else:
                plates.append(path)
        return embeddings, plates
