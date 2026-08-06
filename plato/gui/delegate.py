"""Thumbnail tile painting."""

from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QStyle, QStyledItemDelegate, QStyleOptionViewItem

from .model import ROW_ROLE

FLAG_COLOR = QColor("#e8a33d")
SELECT_COLOR = QColor("#4a90d9")


class ThumbnailDelegate(QStyledItemDelegate):
    def __init__(self, tile: int = 180, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self.tile = tile

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:  # noqa: ANN001
        return QSize(self.tile + 16, self.tile + 34)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:  # noqa: ANN001
        painter.save()
        rect = option.rect
        selected = bool(option.state & QStyle.StateFlag.State_Selected)

        if selected:
            painter.fillRect(rect, QColor(SELECT_COLOR.red(), SELECT_COLOR.green(), SELECT_COLOR.blue(), 60))

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
        if row is not None and row.flagged:
            painter.setPen(QPen(FLAG_COLOR, 3))
            painter.drawRect(image_rect.adjusted(1, 1, -1, -1))
        if row is not None and row.rating:
            painter.setPen(FLAG_COLOR)
            painter.drawText(
                image_rect.adjusted(4, 2, -4, -2),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
                "★" * int(row.rating),
            )

        caption = index.data(Qt.ItemDataRole.DisplayRole) or ""
        if caption:
            painter.setPen(option.palette.text().color())
            text_rect = QRect(rect.left() + 4, image_rect.bottom() + 4, rect.width() - 8, 22)
            metrics = QFontMetrics(option.font)
            elided = metrics.elidedText(
                caption, Qt.TextElideMode.ElideRight, text_rect.width()
            )
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignHCenter, elided)
        painter.restore()
