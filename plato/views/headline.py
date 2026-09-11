"""Drawing a title block above an exported plot.

An exported file leaves the app carrying nothing. On screen the dataset,
method and encoding are all readable from the Explorer's sidebar; in a thesis
figure or a message to a collaborator, a bare scatter of coloured dots is
unidentifiable, and the filename that did encode those facts is gone the
moment the file is renamed or dropped into a document.

Both export paths compose the page differently -- the single plot hands a
``PlotItem`` to a pyqtgraph exporter, the facet grid paints widgets into a
painter -- so this module owns only the two things they share: how tall the
block is, and how to paint it. Composing the final page stays with each
exporter.

**The ink has to follow the plot's own background, not the app theme.** A
plot exported with a transparent or explicitly light/dark ground is
independent of whether the app is in dark mode (see
``EmbeddingScatter.background_colour``), so a headline drawn in the theme's
text colour would be invisible on half of them -- black on a dark export,
white on a light one. ``ink_for`` picks from the actual fill instead.
"""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter

# Space around and between the two lines, in device-independent pixels.
MARGIN_X = 14
MARGIN_TOP = 12
MARGIN_BOTTOM = 10
GAP = 3

TITLE_PT = 13
SUBTITLE_PT = 10


def _fonts() -> tuple[QFont, QFont]:
    title = QFont()
    title.setPointSize(TITLE_PT)
    title.setBold(True)
    subtitle = QFont()
    subtitle.setPointSize(SUBTITLE_PT)
    return title, subtitle


def ink_for(background: QColor) -> tuple[QColor, QColor]:
    """Title and subtitle colours that stay legible on ``background``.

    A fully transparent export has no ground of its own and will be composited
    over something unknown; mid-grey ink is the only choice that survives
    both a light and a dark host, so that is what it gets rather than a guess
    that is unreadable half the time.
    """
    if background.alpha() < 8:
        return QColor("#808080"), QColor("#808080")
    # Rec. 601 luma: good enough to choose between black and white ink, and
    # it does not need the sRGB linearisation a contrast ratio would.
    luma = (
        0.299 * background.red()
        + 0.587 * background.green()
        + 0.114 * background.blue()
    )
    if luma > 140:
        return QColor("#101827"), QColor("#53607a")
    return QColor("#e6ebf2"), QColor("#94a3b8")


def height(title: str, subtitle: str) -> int:
    """Pixels the block needs, or 0 when there is nothing to draw.

    Callers add this to the page height and translate the plot down by it, so
    a headline never overlaps the data it describes.
    """
    if not title and not subtitle:
        return 0
    title_font, subtitle_font = _fonts()
    total = MARGIN_TOP + MARGIN_BOTTOM
    if title:
        total += QFontMetrics(title_font).height()
    if subtitle:
        total += QFontMetrics(subtitle_font).height()
    if title and subtitle:
        total += GAP
    return total


def draw(
    painter: QPainter,
    title: str,
    subtitle: str,
    *,
    width: int,
    background: QColor,
) -> None:
    """Paint the block into the top ``height()`` pixels of ``painter``.

    Elides rather than wraps: a title block that grows a second line changes
    the page height a caller already committed to when it asked for
    ``height()``.
    """
    if not title and not subtitle:
        return
    title_font, subtitle_font = _fonts()
    title_ink, subtitle_ink = ink_for(background)
    available = max(width - 2 * MARGIN_X, 1)
    y = MARGIN_TOP

    painter.save()
    try:
        if title:
            metrics = QFontMetrics(title_font)
            painter.setFont(title_font)
            painter.setPen(title_ink)
            text = metrics.elidedText(title, Qt.TextElideMode.ElideRight, available)
            painter.drawText(
                QRect(MARGIN_X, y, available, metrics.height()),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                text,
            )
            y += metrics.height() + (GAP if subtitle else 0)
        if subtitle:
            metrics = QFontMetrics(subtitle_font)
            painter.setFont(subtitle_font)
            painter.setPen(subtitle_ink)
            text = metrics.elidedText(subtitle, Qt.TextElideMode.ElideRight, available)
            painter.drawText(
                QRect(MARGIN_X, y, available, metrics.height()),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                text,
            )
    finally:
        painter.restore()
