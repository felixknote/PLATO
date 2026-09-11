"""Collapsible sidebar sections, and the accordion that coordinates them.

**The problem this solves.** The Embedding Explorer's left column had seven
``QGroupBox``es stacked in one ``QVBoxLayout`` inside a ``QScrollArea``, every
one permanently expanded, with the t-SNE panel contributing four *nested*
group boxes of its own. Nothing was ever hidden, so nothing was ever findable:
the sidebar was a long scroll of unrelated controls all at the same volume,
and reaching Filters meant scrolling past Encoding and Grouping whether or not
you cared about them.

**The rule.** One section is open at a time. Opening one closes the rest, so
the column's height is bounded by its single tallest section rather than by
the sum of all of them, and the sidebar stops scrolling.

**Why that is safe here.** The usual objection to an accordion is that it
hides state -- you cannot see that a filter is active if the filter section is
shut. So a closed section is not silent: each shows a ``summary`` on its
header row stating its current value ("UMAP - 5,000 pts", "Colour: Gene",
"3 active"). The whole configuration stays readable with everything closed,
and ``set_badge`` puts an accent dot on sections whose state is actively
changing what is on screen. Hiding a control never hides its effect.

The header is a real button: focusable, space/enter activates it, and it
reports its expanded/collapsed state through Qt's accessible name.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractButton,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStyle,
    QStyleOption,
    QStylePainter,
    QVBoxLayout,
    QWidget,
)


class _Chevron(QLabel):
    """The open/closed indicator.

    A text glyph rather than an icon: it inherits the header's colour from the
    global stylesheet, needs no asset, and survives a theme switch without a
    ``restyle`` of its own.
    """

    def __init__(self) -> None:
        super().__init__("›")  # single right-pointing angle quote
        self.setObjectName("sectionChevron")
        self.setFixedWidth(14)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_open(self, is_open: bool) -> None:
        # Rotating a QLabel means a QTransform and a custom paint path;
        # swapping the glyph carries the same information for none of that.
        self.setText("˅" if is_open else "›")


class SectionHeader(QAbstractButton):
    """The clickable row: title on the left, summary and badge on the right."""

    def __init__(self, title: str) -> None:
        super().__init__()
        self.setObjectName("sectionHeader")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        self._chevron = _Chevron()
        self._title = QLabel(title)
        self._title.setObjectName("sectionTitle")
        self._summary = QLabel("")
        self._summary.setObjectName("sectionSummary")
        self._summary.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        # The summary yields first when the column is narrow: the title is
        # what you navigate by, the summary is only a reminder. Preferred
        # rather than Ignored -- Ignored let the row collapse the label to
        # width 0, so every summary was "visible" and measured a correct
        # sizeHint while painting nothing at all. Elide long values instead,
        # and keep the full text in the tooltip.
        self._summary.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self._summary.setMinimumWidth(0)

        row = QHBoxLayout()
        row.setContentsMargins(8, 7, 8, 7)
        row.setSpacing(6)
        row.addWidget(self._chevron)
        row.addWidget(self._title)
        row.addStretch(1)
        row.addWidget(self._summary)
        self.setLayout(row)

        self.toggled.connect(self._on_toggled)
        self._on_toggled(False)

    def _on_toggled(self, is_open: bool) -> None:
        self._chevron.set_open(is_open)
        # Summary and body say the same thing; showing both is noise.
        self._summary.setVisible(not is_open)
        state = "expanded" if is_open else "collapsed"
        self.setAccessibleName(f"{self._title.text()}, {state}")

    def title_text(self) -> str:
        return self._title.text()

    def summary_text(self) -> str:
        """The full summary, before any eliding."""
        return getattr(self, "_full_summary", "")

    def set_summary(self, text: str) -> None:
        self._full_summary = text
        self._summary.setToolTip(text)
        self._apply_elide()

    def _apply_elide(self) -> None:
        """Fit the summary to whatever width the row can spare.

        A dataset name or a facet column can be far longer than the column,
        and a label that simply overflows would push the title out of the
        row. Qt does not elide a QLabel on its own.
        """
        text = getattr(self, "_full_summary", "")
        if not text:
            self._summary.setText("")
            return
        # Whatever is left after the chevron, the title and the spacing.
        available = max(
            0, self.width() - self._title.sizeHint().width() - 14 - 6 * 3 - 16
        )
        metrics = self._summary.fontMetrics()
        if available <= 0 or metrics.horizontalAdvance(text) <= available:
            self._summary.setText(text)
        else:
            self._summary.setText(
                metrics.elidedText(text, Qt.TextElideMode.ElideRight, available)
            )

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._apply_elide()

    def set_badge(self, on: bool) -> None:
        """Mark the section as actively changing what is displayed."""
        self._summary.setProperty("badge", "on" if on else "off")
        # A dynamic property used as a stylesheet selector only takes effect
        # after a re-polish; Qt does not re-evaluate the sheet on its own.
        style = self._summary.style()
        style.unpolish(self._summary)
        style.polish(self._summary)

    def has_badge(self) -> bool:
        return self._summary.property("badge") == "on"

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # QAbstractButton paints nothing by default, and the global sheet
        # styles #sectionHeader, so hand the drawing to the style system.
        option = QStyleOption()
        option.initFrom(self)
        painter = QStylePainter(self)
        painter.drawPrimitive(QStyle.PrimitiveElement.PE_Widget, option)


class Section(QFrame):
    """One collapsible section: a header row and a body widget."""

    toggled = Signal(bool)

    def __init__(self, title: str, body: QWidget) -> None:
        super().__init__()
        self.setObjectName("section")
        self.title = title
        self.header = SectionHeader(title)
        self.body = body
        self.body.setVisible(False)
        # A section is exactly as tall as its header plus (when open) its
        # body. Without this the default Preferred/Preferred policy lets the
        # column's spare height be shared out between all seven, so a closed
        # section grows a large empty gap under its one-line header and the
        # accordion looks broken rather than compact.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        # Never let a body's preferred width push the section wider than the
        # column it was given -- the forms inside ask for their natural width
        # and would otherwise overflow the sidebar and clip on the right.
        self.body.setMinimumWidth(0)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.header)
        layout.addWidget(self.body)
        self.setLayout(layout)

        self.header.toggled.connect(self._on_header)

    def _on_header(self, is_open: bool) -> None:
        self.body.setVisible(is_open)
        self.toggled.emit(is_open)

    def is_open(self) -> bool:
        return self.header.isChecked()

    def set_open(self, is_open: bool) -> None:
        self.header.setChecked(is_open)

    def set_summary(self, text: str) -> None:
        self.header.set_summary(text)

    def set_badge(self, on: bool) -> None:
        self.header.set_badge(on)


class Accordion(QWidget):
    """A column of sections where at most one is open.

    Sections are added in workflow order. Every open/close goes through
    ``_on_toggled``, so the one-at-a-time invariant holds no matter whether
    the change came from a click, the keyboard, or code.
    """

    section_opened = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._sections: dict[str, Section] = {}
        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self.setLayout(self._layout)
        self._guard = False
        # The column is what fixes the width; sections wrap into it.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)

    def add(self, key: str, title: str, body: QWidget) -> Section:
        section = Section(title, body)
        section.toggled.connect(lambda is_open, k=key: self._on_toggled(k, is_open))
        self._sections[key] = section
        self._layout.addWidget(section)
        return section

    def add_widget(self, widget: QWidget) -> None:
        """Put a plain widget in the column, outside any section."""
        self._layout.addWidget(widget)

    def add_stretch(self) -> None:
        self._layout.addStretch(1)

    def _on_toggled(self, key: str, is_open: bool) -> None:
        # Re-entrancy guard: closing the others below re-enters this slot.
        if self._guard or not is_open:
            return
        self._guard = True
        try:
            for other, section in self._sections.items():
                if other != key and section.is_open():
                    section.set_open(False)
        finally:
            self._guard = False
        self.section_opened.emit(key)

    def open_section(self, key: str) -> None:
        section = self._sections.get(key)
        if section is not None and not section.is_open():
            section.set_open(True)

    def section(self, key: str) -> Section | None:
        return self._sections.get(key)

    def set_summary(self, key: str, text: str) -> None:
        section = self._sections.get(key)
        if section is not None:
            section.set_summary(text)

    def set_badge(self, key: str, on: bool) -> None:
        section = self._sections.get(key)
        if section is not None:
            section.set_badge(on)

    def keys(self) -> list[str]:
        return list(self._sections)

    def open_key(self) -> str | None:
        for key, section in self._sections.items():
            if section.is_open():
                return key
        return None
