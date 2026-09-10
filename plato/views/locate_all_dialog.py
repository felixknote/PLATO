"""Locate source images for every open embedding, in one popup.

The old flow located images for whichever embedding happened to be current,
one dialog per dataset -- so with several embeddings open (the workspace now
supports that; see plato.data.workspace) locating them all meant reopening
the same dialog once per entry with no memory of what had already been
resolved.

This mirrors the "Browse embeddings" flow instead: one list, one row per open
embedding, each independently scanned and accepted. An entry already resolved
shows its current folder and match quality; choosing a new folder for one row
never disturbs the others.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtWidgets import (
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtWidgets import QDialog as _QDialog

from ..data.explorer_model import ImageResolver
from ..data.workspace import EmbeddingEntry
from ..gui import themes

MAX_SUGGESTIONS = 8


class _ScanSignals(QObject):
    done = Signal(str, object)  # entry key, ImageResolver
    failed = Signal(str, str)  # entry key, message


class _ScanTask(QRunnable):
    """Indexes one candidate folder against one entry's frame, off the GUI thread."""

    def __init__(self, key: str, root: Path, frame, signals: _ScanSignals) -> None:
        super().__init__()
        self._key = key
        self._root = root
        self._frame = frame
        self._signals = signals

    def run(self) -> None:  # pragma: no cover - worker thread
        try:
            resolver = ImageResolver.for_root(self._root, self._frame)
        except Exception as exc:  # noqa: BLE001 - reported in the dialog
            try:
                self._signals.failed.emit(self._key, str(exc))
            except RuntimeError:
                pass
            return
        try:
            self._signals.done.emit(self._key, resolver)
        except RuntimeError:
            pass


class _EntryRow(QWidget):
    """One embedding: its name, current status, and a Choose folder button."""

    choose_requested = Signal(str)  # entry key
    suggestion_clicked = Signal(str, str)  # entry key, path

    def __init__(self, entry: EmbeddingEntry, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = entry.key
        self.entry = entry
        # Result actually accepted for this row -- distinct from the entry's
        # resolver, which is only updated when the whole dialog is accepted,
        # so cancelling the dialog leaves every entry exactly as it was.
        self.pending_resolver: ImageResolver | None = entry.resolver

        self.name_label = QLabel(f"<b>{entry.label()}</b>")
        self.name_label.setWordWrap(True)

        self.status_label = QLabel(self._initial_status())
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("hint")

        self.choose_button = QPushButton("Choose folder…")
        self.choose_button.clicked.connect(lambda: self.choose_requested.emit(self.key))

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        self.progress.hide()

        self.suggestions = QListWidget()
        self.suggestions.setMaximumHeight(96)
        self.suggestions.itemDoubleClicked.connect(
            lambda item: self.suggestion_clicked.emit(
                self.key, item.data(Qt.ItemDataRole.UserRole)
            )
        )
        self.suggestions.hide()

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.name_label, 1)
        top.addWidget(self.choose_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        layout.addLayout(top)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.suggestions)
        self.setLayout(layout)
        self.restyle()

    def restyle(self) -> None:
        colours = themes.current()
        border = colours.border
        self.setStyleSheet(
            f"_EntryRow {{ background: {colours.surface}; "
            f"border: 1px solid {border}; border-radius: 4px; }}"
        )

    def _initial_status(self) -> str:
        if self.entry.resolver is not None and self.entry.resolver.report is not None:
            report = self.entry.resolver.report
            percent = report.fraction * 100
            mark = "✓" if report.ok else "⚠"
            return (
                f"{mark} {self.entry.resolver.root}\n"
                f"{report.n_files:,} images, {percent:.0f}% of sampled rows resolved"
            )
        return "Not located yet."

    # -- driving the row from outside --------------------------------------

    def set_scanning(self, root: Path) -> None:
        self.status_label.setText(f"Looking in {root}…")
        self.progress.show()
        self.suggestions.hide()
        self.choose_button.setEnabled(False)

    def set_result(self, resolver: ImageResolver) -> None:
        self.progress.hide()
        self.choose_button.setEnabled(True)
        report = resolver.report
        if report is None:
            self.status_label.setText("Nothing to check against.")
            return

        if report.ok:
            self.pending_resolver = resolver
            percent = report.fraction * 100
            if report.fraction >= 0.95:
                self.status_label.setText(
                    f"✓ {resolver.root}\n{report.n_files:,} images found, "
                    f"every sampled row resolved."
                )
            else:
                self.status_label.setText(
                    f"⚠ {resolver.root}\n{report.n_files:,} images found, but "
                    f"only {percent:.0f}% of sampled rows resolved. This folder "
                    f"probably holds part of the dataset."
                )
            self._offer_neighbours(resolver.root)
            return

        self.status_label.setText(f"✗ {resolver.root}\n{report.describe()}.")
        self._offer_neighbours(resolver.root)

    def set_failed(self, message: str) -> None:
        self.progress.hide()
        self.choose_button.setEnabled(True)
        self.status_label.setText(f"Could not read that folder:\n{message}")

    def _offer_neighbours(self, root: Path) -> None:
        candidates: list[Path] = []
        parent = root.parent
        if parent.is_dir() and parent != root:
            candidates.append(parent)
        for source in (root, parent):
            try:
                candidates.extend(
                    d for d in sorted(source.iterdir()) if d.is_dir() and d != root
                )
            except OSError:
                continue
            if len(candidates) >= MAX_SUGGESTIONS:
                break

        self.suggestions.clear()
        for candidate in candidates[:MAX_SUGGESTIONS]:
            label = "⬆ " if candidate == root.parent else "   "
            item = QListWidgetItem(f"{label}{candidate.name or candidate}")
            item.setData(Qt.ItemDataRole.UserRole, str(candidate))
            item.setToolTip(str(candidate))
            self.suggestions.addItem(item)
        self.suggestions.setVisible(self.suggestions.count() > 0)


class LocateAllDialog(_QDialog):
    """Locate images for every embedding open in the workspace, at once."""

    def __init__(self, entries: list[EmbeddingEntry], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Locate source data")
        self.setMinimumSize(620, 520)

        self._pool = QThreadPool.globalInstance()
        self._signals = _ScanSignals()
        self._signals.done.connect(self._on_scanned)
        self._signals.failed.connect(self._on_failed)
        self._rows: dict[str, _EntryRow] = {}

        intro = QLabel(
            "Choose the folder holding the raw microscopy images for each "
            "embedding below. Any arrangement works — one folder per plate, "
            "arms in subfolders, or everything together."
        )
        intro.setWordWrap(True)
        intro.setObjectName("muted")

        rows_layout = QVBoxLayout()
        rows_layout.setSpacing(8)
        for entry in entries:
            row = _EntryRow(entry)
            row.choose_requested.connect(self._choose_for)
            row.suggestion_clicked.connect(self._scan)
            self._rows[entry.key] = row
            rows_layout.addWidget(row)
        rows_layout.addStretch(1)

        rows_container = QWidget()
        rows_container.setLayout(rows_layout)
        scroll = QScrollArea()
        scroll.setWidget(rows_container)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        layout.addWidget(intro)
        layout.addWidget(scroll, 1)
        layout.addWidget(self.buttons)
        self.setLayout(layout)

    def restyle(self) -> None:
        for row in self._rows.values():
            row.restyle()

    # -- scanning ------------------------------------------------------

    def _choose_for(self, key: str) -> None:
        row = self._rows.get(key)
        if row is None:
            return
        start = str(row.pending_resolver.root) if row.pending_resolver else ""
        chosen = QFileDialog.getExistingDirectory(self, "Choose the image folder", start)
        if chosen:
            self._scan(key, chosen)

    def _scan(self, key: str, root: str) -> None:
        row = self._rows.get(key)
        if row is None:
            return
        path = Path(root)
        row.set_scanning(path)
        self._pool.start(_ScanTask(key, path, row.entry.frame, self._signals))

    def _on_scanned(self, key: str, resolver: ImageResolver) -> None:
        row = self._rows.get(key)
        if row is not None:
            row.set_result(resolver)

    def _on_failed(self, key: str, message: str) -> None:
        row = self._rows.get(key)
        if row is not None:
            row.set_failed(message)

    # -- result ----------------------------------------------------------

    def results(self) -> dict[str, ImageResolver]:
        """entry key -> resolver, for every row that ended up with one.

        A row whose resolver is unchanged from what the entry already had is
        still included; the caller applying results back onto the workspace
        is a cheap no-op for those and it keeps this simple rather than
        tracking "did this one actually change."
        """
        return {
            key: row.pending_resolver
            for key, row in self._rows.items()
            if row.pending_resolver is not None
        }
