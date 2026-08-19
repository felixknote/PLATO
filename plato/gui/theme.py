"""Application-wide dark theme.

Scientific image browsing wants a dark, low-chroma interface: the images are
the only thing that should be bright, and a light grey chrome around a 16-bit
micrograph both fights it for attention and skews how you judge intensity by
eye. Everything here is neutral grey except one accent colour, used only where
something is selected or active.

Applied once to the QApplication, so every window, dialog and popup inherits
it -- there is no per-widget styling anywhere else in the GUI beyond object
names used as selectors below.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

# Neutral greys, dark to light. Kept a touch blue to avoid the muddy look flat
# greys get next to a greyscale image.
BACKGROUND = "#16181d"
SURFACE = "#1e2127"
SURFACE_RAISED = "#252932"
BORDER = "#333945"
BORDER_STRONG = "#3f4653"

TEXT = "#e4e7ec"
TEXT_MUTED = "#9aa3b2"
TEXT_FAINT = "#6b7482"

ACCENT = "#4a90d9"
ACCENT_HOVER = "#5b9fe3"
ACCENT_PRESSED = "#3d7cbd"
FLAG = "#e8a33d"

# The canvas an image sits on: darker than the panel so the image edge reads
# as an edge, without being pure black, which makes dark pixels unjudgeable.
IMAGE_BACKGROUND = "#0d0f12"

STYLESHEET = f"""
QWidget {{
    background-color: {BACKGROUND};
    color: {TEXT};
    font-size: 13px;
}}

QMainWindow, QDialog {{
    background-color: {BACKGROUND};
}}

/* -- menus ----------------------------------------------------------- */

QMenuBar {{
    background-color: {SURFACE};
    border-bottom: 1px solid {BORDER};
    padding: 2px 4px;
}}
QMenuBar::item {{
    padding: 6px 12px;
    background: transparent;
    border-radius: 5px;
}}
QMenuBar::item:selected {{ background-color: {SURFACE_RAISED}; }}
QMenuBar::item:pressed {{ background-color: {ACCENT}; color: #ffffff; }}

QMenu {{
    background-color: {SURFACE_RAISED};
    border: 1px solid {BORDER_STRONG};
    border-radius: 8px;
    padding: 6px;
}}
QMenu::item {{
    padding: 7px 28px 7px 22px;
    border-radius: 5px;
}}
QMenu::item:selected {{ background-color: {ACCENT}; color: #ffffff; }}
QMenu::item:disabled {{ color: {TEXT_FAINT}; }}
QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 5px 8px;
}}

/* -- buttons --------------------------------------------------------- */

QPushButton {{
    background-color: {SURFACE_RAISED};
    border: 1px solid {BORDER_STRONG};
    border-radius: 6px;
    padding: 7px 14px;
    color: {TEXT};
}}
QPushButton:hover {{
    background-color: {BORDER};
    border-color: {ACCENT};
}}
QPushButton:pressed {{ background-color: {ACCENT_PRESSED}; color: #ffffff; }}
QPushButton:disabled {{
    background-color: {SURFACE};
    color: {TEXT_FAINT};
    border-color: {BORDER};
}}
QPushButton:default {{
    background-color: {ACCENT};
    border-color: {ACCENT};
    color: #ffffff;
}}
QPushButton:default:hover {{ background-color: {ACCENT_HOVER}; }}

/* -- text entry and combos ------------------------------------------- */

QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {{
    background-color: {SURFACE};
    border: 1px solid {BORDER_STRONG};
    border-radius: 6px;
    padding: 7px 10px;
    selection-background-color: {ACCENT};
    selection-color: #ffffff;
}}
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
    border-color: {ACCENT};
}}
QLineEdit::placeholder {{ color: {TEXT_FAINT}; }}

QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox::down-arrow {{
    /* Qt has no built-in chevron here; a small triangle drawn with borders. */
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_MUTED};
    width: 0;
    height: 0;
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {SURFACE_RAISED};
    border: 1px solid {BORDER_STRONG};
    border-radius: 6px;
    selection-background-color: {ACCENT};
    selection-color: #ffffff;
    outline: none;
    padding: 4px;
}}

/* -- lists ------------------------------------------------------------ */

QListWidget, QListView {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    outline: none;
    padding: 3px;
}}
QListWidget::item {{
    padding: 5px 8px;
    border-radius: 5px;
}}
QListWidget::item:hover {{ background-color: {SURFACE_RAISED}; }}
QListWidget::item:selected {{ background-color: {ACCENT}; color: #ffffff; }}

/* The thumbnail grid paints its own tiles; keep the view itself flat so the
   delegate's selection tint is the only selection cue. */
QListView#thumbnailGrid {{
    background-color: {BACKGROUND};
    border: none;
}}

/* -- group boxes (the filter sidebar) --------------------------------- */

QGroupBox {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 6px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    color: {TEXT_MUTED};
    font-size: 12px;
    text-transform: uppercase;
}}

/* -- checkboxes ------------------------------------------------------- */

QCheckBox {{ spacing: 8px; padding: 3px; background: transparent; }}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {BORDER_STRONG};
    border-radius: 4px;
    background-color: {SURFACE};
}}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}

/* -- splitters and scrollbars ----------------------------------------- */

QSplitter::handle {{ background-color: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QSplitter::handle:hover {{ background-color: {ACCENT}; }}

QScrollArea {{ border: none; background: transparent; }}

QScrollBar:vertical {{
    background: transparent;
    width: 11px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_STRONG};
    border-radius: 5px;
    min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: {TEXT_FAINT}; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 11px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER_STRONG};
    border-radius: 5px;
    min-width: 28px;
}}
QScrollBar::handle:horizontal:hover {{ background: {TEXT_FAINT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

/* -- status bar, tooltips, labels -------------------------------------- */

QStatusBar {{
    background-color: {SURFACE};
    border-top: 1px solid {BORDER};
    color: {TEXT_MUTED};
}}
QStatusBar::item {{ border: none; }}

QToolTip {{
    background-color: {SURFACE_RAISED};
    color: {TEXT};
    border: 1px solid {BORDER_STRONG};
    border-radius: 6px;
    padding: 6px 9px;
}}

QLabel {{ background: transparent; }}

/* The canvas behind a comparison image. */
QLabel#comparisonImage {{
    background-color: {IMAGE_BACKGROUND};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}

/* Section headings in side panels. */
QLabel#panelHeading {{
    color: {TEXT_MUTED};
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
}}

QDialogButtonBox QPushButton {{ min-width: 84px; }}
"""


def apply_theme(app: QApplication) -> None:
    """Apply the dark theme to the whole application.

    Both a palette and a stylesheet are set: the stylesheet covers the widgets
    PLATO builds, while the palette catches what it does not style directly --
    native file dialogs, pyqtgraph's own widgets in the viewer -- which would
    otherwise render light against everything else.
    """
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BACKGROUND))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(SURFACE_RAISED))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(SURFACE_RAISED))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(SURFACE_RAISED))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(TEXT_FAINT))
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(TEXT_FAINT)
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(TEXT_FAINT)
    )
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)
