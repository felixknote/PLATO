"""Image Statistics: an opt-in pass that measures pixels, not metadata.

Kept deliberately apart from Display, and off until asked for. Everything else
in the explorer works from data already on disk -- the embedding, the metadata
frame -- and costs nothing to change. This reads every image in the dataset,
which on a network share is real time and real bandwidth, so it is a button
you press rather than a mode you can wander into.

The panel is a single button until a pass has run. Only then does the
colour-by control appear, because until then it would offer six fields whose
every value is unknown.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..data.image_stats import STAT_HELP, STAT_LABELS, STAT_NAMES
from ..gui.theme import TEXT_FAINT, TEXT_MUTED


class StatsPanel(QWidget):
    """Compute-on-demand image statistics, and colour by one of them."""

    compute_requested = Signal()
    cancel_requested = Signal()
    # The statistic to colour by, or None to leave colouring to metadata.
    stat_selected = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._has_stats = False

        self.compute_button = QPushButton("Compute image statistics")
        self.compute_button.setToolTip(
            "Read every image in this dataset and measure brightness, "
            "contrast, focus and saturation.\n\n"
            "This is the only part of the explorer that reads the images in "
            "bulk, so it is off until you ask for it. Roughly 6 ms per image "
            "across all cores; the result is cached, so it runs once per "
            "dataset."
        )
        self.compute_button.clicked.connect(self.compute_requested.emit)

        self.explain = QLabel(
            "Colour points by what the pixels do, rather than by metadata — "
            "useful for telling a real cluster from an acquisition artefact."
        )
        self.explain.setWordWrap(True)
        self.explain.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 11px;")

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.hide()

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self.progress_label.hide()

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        self.cancel_button.hide()

        # Hidden until a pass has actually produced numbers.
        self.stat_row = QWidget()
        self.stat_box = QComboBox()
        self.stat_box.addItem("Off (colour by metadata)", None)
        for name in STAT_NAMES:
            self.stat_box.addItem(STAT_LABELS[name], name)
        self.stat_box.currentIndexChanged.connect(self._on_stat_changed)
        stat_layout = QHBoxLayout()
        stat_layout.setContentsMargins(0, 0, 0, 0)
        stat_layout.addWidget(QLabel("Colour by"))
        stat_layout.addWidget(self.stat_box, 1)
        self.stat_row.setLayout(stat_layout)
        self.stat_row.hide()

        self.stat_help = QLabel("")
        self.stat_help.setWordWrap(True)
        self.stat_help.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 10px;")
        self.stat_help.hide()

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self.summary.hide()

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)
        layout.addWidget(self.explain)
        layout.addWidget(self.compute_button)
        layout.addWidget(self.progress)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.cancel_button)
        layout.addWidget(self.summary)
        layout.addWidget(self.stat_row)
        layout.addWidget(self.stat_help)
        self.setLayout(layout)

    # -- state -------------------------------------------------------------

    def set_running(self, running: bool) -> None:
        for widget in (self.progress, self.progress_label, self.cancel_button):
            widget.setVisible(running)
        self.compute_button.setEnabled(not running)
        if running:
            self.progress.setValue(0)
            self.progress_label.setText("starting…")
            self.explain.hide()

    def set_progress(self, done: int, total: int) -> None:
        fraction = (done / total) if total else 0.0
        self.progress.setValue(int(fraction * 1000))
        self.progress_label.setText(f"{done:,} of {total:,} images — {fraction * 100:.0f}%")

    def set_available(self, measured: int, total: int) -> None:
        """A pass has finished; reveal the colouring control."""
        self._has_stats = measured > 0
        self.stat_row.setVisible(self._has_stats)
        self.summary.setVisible(self._has_stats)
        self.explain.setVisible(not self._has_stats)
        if not self._has_stats:
            self.summary.setText("")
            return
        if measured < total:
            self.summary.setText(
                f"Measured <b>{measured:,}</b> of {total:,} images "
                f"({total - measured:,} could not be read)."
            )
        else:
            self.summary.setText(f"Measured all <b>{measured:,}</b> images.")
        self.compute_button.setText("Recompute image statistics")
        self._update_help()

    def reset(self) -> None:
        """Forget the current dataset's statistics (a new dataset loaded)."""
        self._has_stats = False
        self.stat_box.blockSignals(True)
        self.stat_box.setCurrentIndex(0)
        self.stat_box.blockSignals(False)
        self.stat_row.hide()
        self.stat_help.hide()
        self.summary.hide()
        self.explain.show()
        self.compute_button.setText("Compute image statistics")
        self.compute_button.setEnabled(True)

    @property
    def current_stat(self) -> str | None:
        return self.stat_box.currentData()

    def _on_stat_changed(self) -> None:
        self._update_help()
        self.stat_selected.emit(self.current_stat)

    def _update_help(self) -> None:
        name = self.current_stat
        if name and name in STAT_HELP:
            self.stat_help.setText(STAT_HELP[name])
            self.stat_help.show()
        else:
            self.stat_help.hide()
