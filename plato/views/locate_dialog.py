"""Choosing where a dataset's images live, with evidence rather than a verdict.

The old flow was a folder picker followed by "None of this dataset's images
were found under <path>", which is unhelpful in the two ways that matter: it
does not say what it *did* find, and it offers nothing to do next.

This dialog indexes whatever folder you pick, reports what it saw -- how many
image files, how many of the dataset's rows they cover -- and suggests
neighbouring folders when the pick is close but wrong (a parent, a sibling
holding the other arm). A partial match is offered as a partial match rather
than rejected, because "this folder has the CRISPRi half" is a legitimate
answer.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..data.explorer_model import ImageResolver
from ..gui.theme import TEXT, TEXT_FAINT, TEXT_MUTED

# How many nearby folders to offer when the chosen one is a poor match.
MAX_SUGGESTIONS = 8


class _ScanSignals(QObject):
    done = Signal(object)  # ImageResolver
    failed = Signal(str)


class _ScanTask(QRunnable):
    """Indexes a folder off the GUI thread."""

    def __init__(self, root: Path, frame: pd.DataFrame, signals: _ScanSignals) -> None:
        super().__init__()
        self._root = root
        self._frame = frame
        self._signals = signals

    def run(self) -> None:  # pragma: no cover - worker thread
        try:
            resolver = ImageResolver.for_root(self._root, self._frame)
        except Exception as exc:  # noqa: BLE001 - reported in the dialog
            try:
                self._signals.failed.emit(str(exc))
            except RuntimeError:
                pass
            return
        try:
            self._signals.done.emit(resolver)
        except RuntimeError:
            pass


class LocateDataDialog(QDialog):
    """Pick a folder, see what is in it, and accept it if it fits."""

    def __init__(
        self,
        frame: pd.DataFrame,
        start: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Locate source data")
        self.setMinimumWidth(560)
        self.frame = frame
        self.resolver: ImageResolver | None = None
        self._pool = QThreadPool.globalInstance()
        self._signals = _ScanSignals()
        self._signals.done.connect(self._on_scanned)
        self._signals.failed.connect(self._on_failed)

        intro = QLabel(
            "Choose the folder holding the raw microscopy images these "
            "embeddings were computed from. Any arrangement works — one "
            "folder per plate, arms in subfolders, or everything together."
        )
        intro.setWordWrap(True)
        intro.setObjectName("muted")

        self.path_label = QLabel("No folder chosen")
        self.path_label.setWordWrap(True)
        self.path_label.setObjectName("body")

        choose = QPushButton("Choose folder…")
        choose.clicked.connect(self._choose)

        path_row = QHBoxLayout()
        path_row.setContentsMargins(0, 0, 0, 0)
        path_row.addWidget(self.path_label, 1)
        path_row.addWidget(choose)
        path_widget = QWidget()
        path_widget.setLayout(path_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.hide()

        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setObjectName("muted")

        self.suggestions = QListWidget()
        self.suggestions.setMaximumHeight(130)
        self.suggestions.itemDoubleClicked.connect(self._use_suggestion)
        self.suggestions.hide()

        self.suggestion_label = QLabel("Nearby folders — double-click to try one:")
        self.suggestion_label.setObjectName("hint")
        self.suggestion_label.hide()

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
        layout.addWidget(path_widget)
        layout.addWidget(self.progress)
        layout.addWidget(self.result_label)
        layout.addWidget(self.suggestion_label)
        layout.addWidget(self.suggestions)
        layout.addWidget(self.buttons)
        self.setLayout(layout)

        if start is not None:
            self._scan(start)

    # -- choosing ---------------------------------------------------------

    def _choose(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose the image folder")
        if chosen:
            self._scan(Path(chosen))

    def _use_suggestion(self, item: QListWidgetItem) -> None:
        self._scan(Path(item.data(Qt.ItemDataRole.UserRole)))

    def _scan(self, root: Path) -> None:
        self.path_label.setText(str(root))
        self.result_label.setText("Looking…")
        self.progress.show()
        self.suggestions.hide()
        self.suggestion_label.hide()
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self._pool.start(_ScanTask(root, self.frame, self._signals))

    # -- results ----------------------------------------------------------

    def _on_failed(self, message: str) -> None:
        self.progress.hide()
        self.result_label.setText(f"Could not read that folder:\n{message}")

    def _on_scanned(self, resolver: ImageResolver) -> None:
        self.progress.hide()
        report = resolver.report
        if report is None:
            self.result_label.setText("Nothing to check against.")
            return

        if report.ok:
            self.resolver = resolver
            self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
            percent = report.fraction * 100
            if report.fraction >= 0.95:
                self.result_label.setText(
                    f"✓ {report.n_files:,} images found, and every sampled row "
                    f"resolved. This is the right folder."
                )
            else:
                # A partial match is a real answer -- one arm's folder for a
                # two-arm dataset -- so it is offered, with the shortfall
                # stated rather than hidden.
                self.result_label.setText(
                    f"⚠ {report.n_files:,} images found, but only {percent:.0f}% "
                    f"of sampled rows resolved.\nThis folder probably holds "
                    f"part of the dataset. Use it anyway, or try a parent "
                    f"folder to cover all of it."
                )
                self._offer_neighbours(resolver.root)
            return

        self.resolver = None
        self.result_label.setText(report.describe() + ".")
        self._offer_neighbours(resolver.root)

    def _offer_neighbours(self, root: Path) -> None:
        """List the parent and siblings, which are the usual near-misses."""
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

        has_any = self.suggestions.count() > 0
        self.suggestions.setVisible(has_any)
        self.suggestion_label.setVisible(has_any)
