"""Thumbnail tile painting."""

from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QStyle, QStyledItemDelegate, QStyleOptionViewItem

from .model import ROW_ROLE
from .theme import ACCENT, FLAG, TEXT_MUTED

FLAG_COLOR = QColor(FLAG)
SELECT_COLOR = QColor(ACCENT)
CAPTION_COLOR = QColor(TEXT_MUTED)
TILE_RADIUS = 6


class ThumbnailDelegate(QStyledItemDelegate):
    def __init__(self, tile: int = 180, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self.tile = tile

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:  # noqa: ANN001
        return QSize(self.tile + 16, self.tile + 34)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:  # noqa: ANN001
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = option.rect
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)

        # Rounded tile behind the thumbnail, so selection and hover read as
        # the tile lighting up rather than a hard rectangle of colour.
        if selected or hovered:
            tint = QColor(SELECT_COLOR)
            tint.setAlpha(70 if selected else 28)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(tint)
            painter.drawRoundedRect(rect.adjusted(2, 2, -2, -2), TILE_RADIUS, TILE_RADIUS)

        pixmap: QPixmap | None = index.data(Qt.ItemDataRole.DecorationRole)
        image_rect = QRect(rect.left() + 8, rect.top() + 6, self.tile, self.tile)
        if pixmap is not None and not pixmap.isNull():
            scaled = pixmap.scaled(
                image_rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            target = QRect(0, 0, scaled.width(), scaled.height())
            target.moveCenter(image_rect.center())
            painter.drawPixmap(target, scaled)

        row = index.data(ROW_ROLE)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if selected:
            painter.setPen(QPen(SELECT_COLOR, 2))
            painter.drawRoundedRect(image_rect.adjusted(-2, -2, 2, 2), 4, 4)
        if row is not None and row.flagged:
            painter.setPen(QPen(FLAG_COLOR, 3))
            painter.drawRoundedRect(image_rect.adjusted(1, 1, -1, -1), 4, 4)
        if row is not None and row.rating:
            painter.setPen(FLAG_COLOR)
            painter.drawText(
                image_rect.adjusted(4, 2, -4, -2),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
                "★" * int(row.rating),
            )

        caption = index.data(Qt.ItemDataRole.DisplayRole) or ""
        if caption:
            painter.setPen(option.palette.text().color() if selected else CAPTION_COLOR)
            text_rect = QRect(rect.left() + 4, image_rect.bottom() + 6, rect.width() - 8, 22)
            metrics = QFontMetrics(option.font)
            elided = metrics.elidedText(
                caption, Qt.TextElideMode.ElideRight, text_rect.width()
            )
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignHCenter, elided)
        painter.restore()
