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

from ..index.db import ImageRow
from .scalebar import bar_length_um
from .session import Session
from .settings import get_nm_per_pixel, get_scale_bar_fraction

pg.setConfigOption("imageAxisOrder", "row-major")


def _load_stack(path: Path) -> np.ndarray:
    array = np.asarray(tifffile.imread(path))
    if array.ndim == 2:
        return array[np.newaxis, ...]
    if array.ndim > 3:
        array = array.reshape(-1, *array.shape[-2:])
    return array


class ImageWindow(QMainWindow):
    """Shows one image, navigable within a list of rows."""

    def __init__(
        self,
        session: Session,
        rows: list[ImageRow],
        start: int,
        levels: dict[str, tuple[float, float]],
        *,
        blind: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # The session rather than one plate's Config: left/right steps through
        # the filtered list, which in a multi-plate session crosses from one
        # plate into another, so a single Config would be wrong for most of
        # the rows in the window. (It was also never read.)
        self.session = session
        self.rows = rows
        self.position = start
        self.levels = levels
        self.blind = blind
        self.autoscale = False

        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()

        self.scale_bar = pg.ScaleBar(size=1, suffix="m", offset=(-20, -20))
        self.scale_bar.setParentItem(self.image_view.getView())
        self.scale_bar.hide()

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

        self.scale_bar_box = QCheckBox("Show scale bar (S)")
        self.scale_bar_box.toggled.connect(self._set_scale_bar_visible)

        self.measure_box = QCheckBox("Measure distance (D)")
        self.measure_box.setToolTip(
            "Click two points on the image to measure the distance between them "
            "(in pixels, and in µm using the pixel size from Settings)."
        )
        self.measure_box.toggled.connect(self._set_measure_active)

        self.measure_line = pg.PlotDataItem(pen=pg.mkPen("#e8a33d", width=2))
        self.measure_line.setZValue(20)
        self.image_view.getView().addItem(self.measure_line)
        self.measure_line.hide()

        self.measure_label = pg.TextItem(color="#e8a33d", anchor=(0.5, 1.2))
        self.measure_label.setZValue(20)
        self.image_view.getView().addItem(self.measure_label)
        self.measure_label.hide()

        self._measure_first_point: tuple[float, float] | None = None

        side = QVBoxLayout()
        side.addWidget(self.metadata_label, 1)
        side.addWidget(self.autoscale_box)
        side.addWidget(self.scale_bar_box)
        side.addWidget(self.measure_box)

        layout = QHBoxLayout()
        layout.addWidget(self.image_view, 4)
        layout.addLayout(side, 1)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.setStatusBar(QStatusBar())
        self.resize(1600, 1100)

        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self.step(1))
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self.step(-1))
        QShortcut(QKeySequence(Qt.Key.Key_A), self, self.autoscale_box.toggle)
        QShortcut(QKeySequence(Qt.Key.Key_S), self, self.scale_bar_box.toggle)
        QShortcut(QKeySequence(Qt.Key.Key_D), self, self.measure_box.toggle)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self.close)

        self.image_view.getView().scene().sigMouseClicked.connect(self._on_scene_clicked)

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
        self._update_labels(row)
        if self.scale_bar_box.isChecked():
            self._update_scale_bar()

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

    def _set_scale_bar_visible(self, visible: bool) -> None:
        if visible:
            self._update_scale_bar()
            self.scale_bar.show()
        else:
            self.scale_bar.hide()

    def _set_measure_active(self, active: bool) -> None:
        self._measure_first_point = None
        if not active:
            self.measure_line.hide()
            self.measure_label.hide()
        self.statusBar().showMessage(
            "click two points to measure" if active else "measurement off", 4000
        )

    def _on_scene_clicked(self, event) -> None:  # noqa: ANN001
        if not self.measure_box.isChecked():
            return
        view_box = self.image_view.getView()
        if not view_box.sceneBoundingRect().contains(event.scenePos()):
            return
        point = view_box.mapSceneToView(event.scenePos())
        x, y = point.x(), point.y()

        if self._measure_first_point is None:
            self._measure_first_point = (x, y)
            self.measure_line.setData([x], [y])
            self.measure_line.show()
            self.measure_label.hide()
            return

        x0, y0 = self._measure_first_point
        self.measure_line.setData([x0, x], [y0, y])
        distance_px = float(np.hypot(x - x0, y - y0))
        nm_per_pixel = get_nm_per_pixel()
        if nm_per_pixel > 0:
            distance_um = distance_px * nm_per_pixel / 1000.0
            label = f"{distance_px:.1f} px = {distance_um:.3g} µm"
        else:
            label = f"{distance_px:.1f} px"
        self.measure_label.setText(label)
        self.measure_label.setPos((x0 + x) / 2, (y0 + y) / 2)
        self.measure_label.show()
        self.statusBar().showMessage(label, 8000)
        self._measure_first_point = None

    def _update_scale_bar(self) -> None:
        nm_per_pixel = get_nm_per_pixel()
        if nm_per_pixel <= 0:
            return
        width_px = self.image_view.getImageItem().image.shape[-1]
        um_length = bar_length_um(width_px, nm_per_pixel, get_scale_bar_fraction())
        px_length = um_length * 1000.0 / nm_per_pixel
        self.scale_bar.size = px_length
        self.scale_bar.text.setText(f"{um_length:g} µm")
        self.scale_bar.updateBar()

    def _update_labels(self, row: ImageRow) -> None:
        if self.blind:
            self.setWindowTitle(f"[blinded] image {self.position + 1}/{len(self.rows)}")
            self.metadata_label.setText(
                "<b>Blinded review</b><br>Metadata hidden. Turn blind mode off in "
                "the browser window to reveal it."
            )
        else:
            plates = self.session.plates
            source = plates[row.session_index] if row.session_index < len(plates) else None
            # image_id is a path relative to one plate's image root, so it is
            # not unique across a multi-plate list. Naming the loaded plate in
            # the title is what keeps two identically named files apart as you
            # step from one plate into the next.
            prefix = f"{source.name} · " if source is not None and len(plates) > 1 else ""
            self.setWindowTitle(
                f"{prefix}{row.image_id}  ({self.position + 1}/{len(self.rows)})"
            )
            lines = [
                f"<b>{row.plate} {row.well}</b>",
                f"field {row.field or '-'} · channel {row.channel or '-'}",
                "",
            ]
            if source is not None and len(plates) > 1:
                lines.insert(1, f"loaded plate: <b>{source.name}</b>")
            lines += [
                f"{key}: <b>{value}</b>"
                for key, value in row.metadata.items()
                if value is not None
            ]
            lines += ["", f"<i>{row.path}</i>"]
            self.metadata_label.setText("<br>".join(lines))
        scaling = "per-image" if self.autoscale else "fixed screen-wide"
        self.statusBar().showMessage(
            f"contrast: {scaling}   ←/→ step   A autoscale   S scale bar   D measure   Esc close"
        )
