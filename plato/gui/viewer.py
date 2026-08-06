"""Full-resolution image window.

Zoom/pan is delegated to pyqtgraph rather than reimplemented. The window keeps
a reference to the *current filtered list* so left/right steps through exactly
what you were browsing, not through the directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
import tifffile
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..config import Config
from ..index.db import ImageRow

pg.setConfigOption("imageAxisOrder", "row-major")


def _load_stack(path: Path) -> np.ndarray:
    array = np.asarray(tifffile.imread(path))
    if array.ndim == 2:
        return array[np.newaxis, ...]
    if array.ndim > 3:
        array = array.reshape(-1, *array.shape[-2:])
    return array


def mask_path_for(cfg: Config, row: ImageRow) -> Path | None:
    if not cfg.masks.enabled:
        return None
    stem = Path(row.path).stem
    try:
        name = cfg.masks.pattern.format(
            stem=stem,
            plate=row.plate,
            well=row.well,
            row=row.well[0],
            col=row.well[1:],
            field=row.field or "",
            channel=row.channel or "",
        )
    except KeyError:
        return None
    candidate = Path(cfg.masks.dir) / name
    return candidate if candidate.exists() else None


class ImageWindow(QMainWindow):
    """Shows one image, navigable within a list of rows."""

    def __init__(
        self,
        cfg: Config,
        rows: list[ImageRow],
        start: int,
        levels: dict[str, tuple[float, float]],
        *,
        blind: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self.rows = rows
        self.position = start
        self.levels = levels
        self.blind = blind
        self.autoscale = False

        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()

        self.mask_item = pg.ImageItem()
        self.mask_item.setZValue(10)
        self.mask_item.setOpacity(0.45)
        self.image_view.getView().addItem(self.mask_item)
        self.mask_item.hide()

        self.metadata_label = QLabel()
        self.metadata_label.setWordWrap(True)
        self.metadata_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.metadata_label.setMinimumWidth(240)
        self.metadata_label.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.autoscale_box = QCheckBox("Autoscale this image")
        self.autoscale_box.setToolTip(
            "Off = fixed per-channel limits shared across the screen, so wells "
            "stay comparable. On = per-image contrast, useful for faint detail."
        )
        self.autoscale_box.toggled.connect(self._set_autoscale)

        self.mask_box = QCheckBox("Show mask overlay (M)")
        self.mask_box.toggled.connect(self._set_mask_visible)
        self.mask_box.setEnabled(cfg.masks.enabled)

        side = QVBoxLayout()
        side.addWidget(self.metadata_label, 1)
        side.addWidget(self.autoscale_box)
        side.addWidget(self.mask_box)

        layout = QHBoxLayout()
        layout.addWidget(self.image_view, 4)
        layout.addLayout(side, 1)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.setStatusBar(QStatusBar())
        self.resize(1100, 800)

        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self.step(1))
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self.step(-1))
        QShortcut(QKeySequence(Qt.Key.Key_M), self, self.mask_box.toggle)
        QShortcut(QKeySequence(Qt.Key.Key_A), self, self.autoscale_box.toggle)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self.close)

        self.show_current()

    # -- navigation -------------------------------------------------------

    def step(self, delta: int) -> None:
        if not self.rows:
            return
        self.position = (self.position + delta) % len(self.rows)
        self.show_current()

    def current_row(self) -> ImageRow:
        return self.rows[self.position]

    def show_current(self) -> None:
        row = self.current_row()
        try:
            stack = _load_stack(Path(row.path))
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"cannot read {row.path}: {exc}")
            return

        view_box = self.image_view.getView()
        state = view_box.getState()
        self.image_view.setImage(
            stack,
            autoLevels=False,
            autoRange=False,
            autoHistogramRange=False,
        )
        view_box.setState(state)
        self._apply_levels(stack, row)
        self._load_mask(row)
        self._update_labels(row)

    def _apply_levels(self, stack: np.ndarray, row: ImageRow) -> None:
        if self.autoscale:
            lo, hi = float(np.percentile(stack, 1)), float(np.percentile(stack, 99.5))
        else:
            lo, hi = self.levels.get(row.channel or "_", (float(stack.min()), float(stack.max())))
        if hi <= lo:
            hi = lo + 1.0
        self.image_view.setLevels(lo, hi)

    def _set_autoscale(self, enabled: bool) -> None:
        self.autoscale = enabled
        self.show_current()

    def _set_mask_visible(self, visible: bool) -> None:
        self.mask_item.setVisible(visible and self.mask_item.image is not None)

    def _load_mask(self, row: ImageRow) -> None:
        path = mask_path_for(self.cfg, row)
        if path is None:
            self.mask_item.clear()
            self.mask_item.hide()
            return
        mask = np.asarray(tifffile.imread(path))
        while mask.ndim > 2:
            mask = mask[0]
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        labelled = mask > 0
        rgba[labelled] = (255, 80, 80, 255)
        self.mask_item.setImage(rgba, autoLevels=False)
        self.mask_item.setVisible(self.mask_box.isChecked())

    def _update_labels(self, row: ImageRow) -> None:
        if self.blind:
            self.setWindowTitle(f"[blinded] image {self.position + 1}/{len(self.rows)}")
            self.metadata_label.setText(
                "<b>Blinded review</b><br>Metadata hidden. Turn blind mode off in "
                "the browser window to reveal it."
            )
        else:
            self.setWindowTitle(f"{row.image_id}  ({self.position + 1}/{len(self.rows)})")
            lines = [
                f"<b>{row.plate} {row.well}</b>",
                f"field {row.field or '-'} · channel {row.channel or '-'}",
                "",
            ]
            lines += [
                f"{key}: <b>{value}</b>"
                for key, value in row.metadata.items()
                if value is not None
            ]
            lines += ["", f"<i>{row.path}</i>"]
            self.metadata_label.setText("<br>".join(lines))
        scaling = "per-image" if self.autoscale else "fixed screen-wide"
        self.statusBar().showMessage(
            f"contrast: {scaling}   ←/→ step   A autoscale   M mask   Esc close"
        )
