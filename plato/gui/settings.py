"""App-wide preferences, persisted via QSettings (not per-plate Config).

Pixel size and scale-bar length, both used to draw the scale bar in the
viewer and to bake it into exported images.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QWidget,
)

from ..data.locations import APPLICATION, ORGANISATION

DEFAULT_NM_PER_PIXEL = 108.0
# Bar length as a fraction of the image's width, so it scales with field of
# view / magnification instead of being pinned to one physical length.
#
# 1%, not 2%: the rule rounds UP to the nearest 1/2/5 x 10^n um, so at the
# 108 nm/px default a 2% target (5.9 um) became a 10 um bar spanning a visibly
# large part of the field. 1% targets 2.9 um and lands on 5 um, which reads as
# a reference mark rather than a feature of the image. Still adaptive: a
# coarser pixel size moves it back up to 10 um on its own.
DEFAULT_SCALE_BAR_FRACTION = 0.01


def get_nm_per_pixel() -> float:
    settings = QSettings(ORGANISATION, APPLICATION)
    return float(settings.value("nm_per_pixel", DEFAULT_NM_PER_PIXEL))


def set_nm_per_pixel(value: float) -> None:
    settings = QSettings(ORGANISATION, APPLICATION)
    settings.setValue("nm_per_pixel", float(value))


def get_scale_bar_fraction() -> float:
    settings = QSettings(ORGANISATION, APPLICATION)
    return float(settings.value("scale_bar_fraction", DEFAULT_SCALE_BAR_FRACTION))


def set_scale_bar_fraction(value: float) -> None:
    settings = QSettings(ORGANISATION, APPLICATION)
    settings.setValue("scale_bar_fraction", float(value))


def get_show_timepoint() -> bool:
    settings = QSettings(ORGANISATION, APPLICATION)
    return settings.value("show_timepoint", True, type=bool)


def set_show_timepoint(value: bool) -> None:
    settings = QSettings(ORGANISATION, APPLICATION)
    settings.setValue("show_timepoint", bool(value))


class SettingsDialog(QDialog):
    """Edits app-wide preferences: pixel size and scale-bar length."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")

        self.nm_per_pixel = QDoubleSpinBox()
        self.nm_per_pixel.setRange(0.1, 100_000.0)
        self.nm_per_pixel.setDecimals(2)
        self.nm_per_pixel.setSuffix(" nm/px")
        self.nm_per_pixel.setValue(get_nm_per_pixel())

        self.scale_bar_percent = QDoubleSpinBox()
        self.scale_bar_percent.setRange(0.1, 50.0)
        self.scale_bar_percent.setDecimals(1)
        self.scale_bar_percent.setSuffix(" % of image width")
        self.scale_bar_percent.setValue(get_scale_bar_fraction() * 100)

        self.show_timepoint = QCheckBox("Offer timepoint as a filter")
        self.show_timepoint.setToolTip(
            "Plate folders named like P13_T1 carry a timepoint suffix. When on, "
            "the suffix is offered as its own filter so you can slice across "
            "plates (all T1 wells, regardless of plate). Off hides it — useful "
            "if your folder names end in something that is not a timepoint."
        )
        self.show_timepoint.setChecked(get_show_timepoint())

        form = QFormLayout()
        form.addRow("Pixel size", self.nm_per_pixel)
        form.addRow("Scale bar length", self.scale_bar_percent)
        form.addRow("Timepoint", self.show_timepoint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.setLayout(form)

    def _accept(self) -> None:
        set_nm_per_pixel(self.nm_per_pixel.value())
        set_scale_bar_fraction(self.scale_bar_percent.value() / 100)
        set_show_timepoint(self.show_timepoint.isChecked())
        self.accept()
