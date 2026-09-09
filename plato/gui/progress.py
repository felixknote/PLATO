"""A modal progress dialog that runs indexing/thumbnailing off the GUI thread.

Building a plate is minutes of work: parsing thousands of filenames, then
decoding and downscaling every image. Doing that inline froze the whole window
with no feedback, which is indistinguishable from a crash -- and on a network
share it is long enough that people kill the app.

The work runs on a QThread and reports back through signals. The dialog is
modal so a second load cannot start on top of the first, but the event loop
keeps running, so the window repaints and the Cancel button responds.

Cancellation is cooperative and *between* images: a half-written thumbnail
cache is still a valid incremental cache -- the next run picks up where this
one stopped, because entries are keyed by mtime and size.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .theme import TEXT_MUTED


class WorkerCancelled(RuntimeError):
    """Raised inside the worker thread when the user cancels."""


class _Worker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, job: Callable) -> None:
        super().__init__()
        self._job = job
        self.cancelled = False

    def run(self) -> None:
        def report(done: int, total: int, message: str) -> None:
            if self.cancelled:
                raise WorkerCancelled
            self.progress.emit(done, total, message)

        try:
            result = self._job(report)
        except WorkerCancelled:
            self.finished.emit(None)
            return
        except Exception as exc:  # noqa: BLE001 - reported in the dialog
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)


class ProgressDialog(QDialog):
    """Runs ``job(report)`` on a worker thread, showing its progress.

    ``job`` receives a ``report(done, total, message)`` callable. A ``total``
    of 0 shows a busy indicator, for phases whose size is not known upfront.
    """

    def __init__(self, job: Callable, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(420)
        # No close button: the way out is Cancel, which stops the worker
        # cleanly rather than leaving it running against a deleted dialog.
        self.setWindowFlags(
            (self.windowFlags() | Qt.WindowType.CustomizeWindowHint)
            & ~Qt.WindowType.WindowCloseButtonHint
        )

        self.result = None
        self.error: str | None = None
        self.cancelled = False

        self.label = QLabel("Starting…")
        self.label.setWordWrap(True)

        self.bar = QProgressBar()
        self.bar.setRange(0, 0)  # busy until the first real total arrives
        self.bar.setTextVisible(True)

        self.detail = QLabel("")
        self.detail.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel)

        layout = QVBoxLayout()
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        layout.addWidget(self.label)
        layout.addWidget(self.bar)
        layout.addWidget(self.detail)
        layout.addWidget(self.cancel_button, 0, Qt.AlignmentFlag.AlignRight)
        self.setLayout(layout)

        self._thread = QThread(self)
        self._worker = _Worker(job)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    # -- worker callbacks --------------------------------------------------

    def _on_progress(self, done: int, total: int, message: str) -> None:
        self.label.setText(message)
        if total > 0:
            self.bar.setRange(0, total)
            self.bar.setValue(done)
            self.detail.setText(f"{done:,} of {total:,}")
        else:
            self.bar.setRange(0, 0)
            self.detail.setText("")

    def _on_finished(self, result) -> None:
        self.result = result
        self.cancelled = result is None and self._worker.cancelled
        self._shutdown()
        self.accept() if not self.cancelled else self.reject()

    def _on_failed(self, message: str) -> None:
        self.error = message
        self._shutdown()
        self.reject()

    def _cancel(self) -> None:
        self._worker.cancelled = True
        self.cancel_button.setEnabled(False)
        self.label.setText("Cancelling…")

    def _shutdown(self) -> None:
        self._thread.quit()
        self._thread.wait(5000)


def run_with_progress(
    job: Callable, title: str, parent: QWidget | None = None
) -> tuple[object, str | None, bool]:
    """Run ``job`` behind a progress dialog.

    Returns ``(result, error, cancelled)``.
    """
    dialog = ProgressDialog(job, title, parent)
    dialog.exec()
    return dialog.result, dialog.error, dialog.cancelled
