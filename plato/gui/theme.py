"""Application-wide dark theme.

Scientific image browsing wants a dark, low-chroma interface: the images are
the only thing that should be bright, and a light grey chrome around a 16-bit
micrograph both fights it for attention and skews how you judge intensity by
eye. Everything here is low-chroma navy -- the logo's own ground -- except two
reserved hues: cyan for selected/active, magenta for controls and flags.

Those two are held out of the plot palettes (views/palettes.py) so that a
coloured mark in a plot can never be mistaken for an interface state. Colour
on screen is data; the interface is not.

Applied once to the QApplication, so every window, dialog and popup inherits
it -- there is no per-widget styling anywhere else in the GUI beyond object
names used as selectors below.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette

from .branding import DISPLAY_STACK
from PySide6.QtWidgets import QApplication

# Neutral greys, dark to light. Kept a touch blue to avoid the muddy look flat
# greys get next to a greyscale image.
BACKGROUND = "#0a0f1c"
SURFACE = "#111827"
SURFACE_RAISED = "#18202f"
BORDER = "#222d42"
BORDER_STRONG = "#2f3d56"

TEXT = "#e6ebf2"
TEXT_MUTED = "#94a3b8"
TEXT_FAINT = "#64748b"

# Cyan = selected/active, magenta = control/flagged. Both are held out of the
# categorical scales in views/palettes.py; see themes.DARK_COLOURS.
ACCENT = "#34c7d4"
ACCENT_HOVER = "#4fdbe8"
ACCENT_PRESSED = "#1f9fad"
FLAG = "#ea45cc"

# The canvas an image sits on: darker than the panel so the image edge reads
# as an edge, without being pure black, which makes dark pixels unjudgeable.
IMAGE_BACKGROUND = "#05080f"

# The stylesheet as a TEMPLATE rather than a baked string.
#
# Every rule below is substituted at build time from a ThemeColours instance,
# so light and dark share one definition and a rule added for one applies to
# both. `STYLESHEET` remains as the dark sheet for anything importing it.
_TEMPLATE = """
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
    color: {TEXT};
    font-family: {DISPLAY_STACK};
    font-size: 13px;
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

/* -- sliders ----------------------------------------------------------- */

/* Point size and opacity. Unstyled, these fell through to Fusion's own
   drawing, which paints the groove from the QPalette Highlight and so kept
   rendering the pre-navy blue after the theme changed -- the one control in
   the sidebar still wearing the old accent. */

QSlider::groove:horizontal {{
    height: 4px;
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {TEXT};
    border: none;
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 6px;
}}
QSlider::handle:horizontal:hover {{ background: #ffffff; }}
QSlider::handle:horizontal:disabled {{ background: {TEXT_FAINT}; }}
QSlider::sub-page:horizontal:disabled {{ background: {BORDER_STRONG}; }}

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

/* Section headings in side panels, and the tab bar: the display face here
   ties the chrome to the wordmark without touching body text.

   Sentence case, not tracked-out caps. A heading that labels a control the
   user operates should read as a title, not as a filing-cabinet tag; caps
   plus letter-spacing was applied here AND on QGroupBox::title, so the
   sidebar carried the same shouted treatment twice. Tracking is kept for
   the wordmark alone, where five wide capitals genuinely need it. */
QLabel#panelHeading {{
    color: {TEXT};
    font-family: {DISPLAY_STACK};
    font-size: 13px;
    font-weight: 600;
}}

/* Semantic label roles.

   These exist so that panels do NOT bake colours into their own stylesheets.
   A widget that does setStyleSheet(f"color: {{TEXT}}") captures whatever the
   constant held at import time and can never be restyled, which is what made
   the first light-mode attempt leave half the interface dark. Setting an
   object name instead means the colour comes from this sheet, and re-applying
   the sheet restyles the widget for free. */
QLabel#hint {{
    color: {TEXT_FAINT};
    font-size: 11px;
}}
QLabel#hintSmall {{
    color: {TEXT_FAINT};
    font-size: 10px;
}}
QLabel#muted {{
    color: {TEXT_MUTED};
}}
QLabel#mutedSmall {{
    color: {TEXT_MUTED};
    font-size: 11px;
}}
QLabel#body {{
    color: {TEXT};
    font-size: 11px;
}}
QLabel#message {{
    color: {TEXT_MUTED};
    padding: 18px;
}}
/* The canvas an image or a plot sits on. Deliberately dark in both themes --
   a light surround destroys the contrast a micrograph is judged on. */
QLabel#imageCanvas {{
    background: {IMAGE_BACKGROUND};
    border-radius: 2px;
}}

/* -- collapsible sidebar sections -------------------------------------- */

/* A section is a header row plus a body. Closed, it is one line; open, the
   body sits directly under it on the surface colour. Only the OPEN one gets
   a surface of its own -- a column of identical raised cards is exactly the
   undifferentiated stack the accordion replaced. */

QFrame#section {{
    background: transparent;
    border: none;
}}

QAbstractButton#sectionHeader {{
    background: transparent;
    border: none;
    border-radius: 6px;
    text-align: left;
}}
QAbstractButton#sectionHeader:hover {{ background-color: {SURFACE}; }}
QAbstractButton#sectionHeader:checked {{
    background-color: {SURFACE_RAISED};
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    border-bottom-left-radius: 0;
    border-bottom-right-radius: 0;
}}
/* Keyboard focus must be visible, and the hover tint is not enough on its
   own for someone tabbing through. */
QAbstractButton#sectionHeader:focus {{
    background-color: {SURFACE};
    border: 1px solid {ACCENT};
}}

QLabel#sectionTitle {{
    font-family: {DISPLAY_STACK};
    font-size: 13px;
    font-weight: 600;
    color: {TEXT};
}}
QAbstractButton#sectionHeader:!checked QLabel#sectionTitle {{
    color: {TEXT_MUTED};
}}

/* The closed-state value. Muted, so the column reads as a list of titles
   first and values second. */
QLabel#sectionSummary {{
    font-size: 11px;
    color: {TEXT_FAINT};
}}
/* "This section is changing what you are looking at right now." The accent
   is the interface's one reserved hue for active state, so an active filter
   announces itself in the same colour as everything else that is live. */
QLabel#sectionSummary[badge="on"] {{
    color: {ACCENT};
    font-weight: 600;
}}

QLabel#sectionChevron {{
    color: {TEXT_FAINT};
    font-size: 13px;
}}

/* The open body. Squared off at the top so it reads as continuous with its
   header rather than as a separate card. */
QWidget#sectionBody {{
    background-color: {SURFACE_RAISED};
    border-bottom-left-radius: 6px;
    border-bottom-right-radius: 6px;
}}

/* Inside a section body the group-box chrome is redundant -- the section IS
   the grouping. This flattens the nested boxes (the t-SNE panel's four) to
   plain labelled blocks. */
QWidget#sectionBody QGroupBox {{
    background: transparent;
    border: none;
    border-radius: 0;
    margin-top: 10px;
    padding-top: 2px;
}}
QWidget#sectionBody QGroupBox::title {{
    color: {TEXT_MUTED};
    font-size: 12px;
    font-weight: 600;
}}

QTabBar::tab {{
    background: {SURFACE};
    color: {TEXT_MUTED};
    font-family: {DISPLAY_STACK};
    font-size: 13px;
    letter-spacing: 0.05em;
    padding: 8px 20px;
    border: 1px solid {BORDER};
    border-bottom: none;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background: {SURFACE_RAISED};
    color: {TEXT};
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover:!selected {{ color: {TEXT}; }}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    top: -1px;
}}

QDialogButtonBox QPushButton {{ min-width: 84px; }}
"""


def build_stylesheet(colours) -> str:
    """Render the template for one palette.

    Takes a ``plato.gui.themes.ThemeColours``. Kept here rather than in
    themes.py so the rules and their substitution stay in one file.
    """
    return _TEMPLATE.format(
        BACKGROUND=colours.background,
        SURFACE=colours.surface,
        SURFACE_RAISED=colours.surface_raised,
        BORDER=colours.border,
        BORDER_STRONG=colours.border_strong,
        TEXT=colours.text,
        TEXT_MUTED=colours.text_muted,
        TEXT_FAINT=colours.text_faint,
        ACCENT=colours.accent,
        ACCENT_HOVER=colours.accent_hover,
        ACCENT_PRESSED=colours.accent_pressed,
        IMAGE_BACKGROUND=colours.image_background,
        DISPLAY_STACK=DISPLAY_STACK,
    )


def sync_module_constants(colours) -> None:
    """Point this module's names at ``colours``.

    Code that did ``from .theme import TEXT`` at import time keeps whatever it
    captured -- nothing can change that -- but anything reading
    ``theme.TEXT`` from now on sees the live theme. Widgets that need to
    repaint implement ``restyle()``; see plato.gui.themes.apply.
    """
    globals().update(
        BACKGROUND=colours.background,
        SURFACE=colours.surface,
        SURFACE_RAISED=colours.surface_raised,
        BORDER=colours.border,
        BORDER_STRONG=colours.border_strong,
        TEXT=colours.text,
        TEXT_MUTED=colours.text_muted,
        TEXT_FAINT=colours.text_faint,
        ACCENT=colours.accent,
        ACCENT_HOVER=colours.accent_hover,
        ACCENT_PRESSED=colours.accent_pressed,
        FLAG=colours.flag,
        IMAGE_BACKGROUND=colours.image_background,
        STYLESHEET=build_stylesheet(colours),
    )


STYLESHEET = _TEMPLATE.format(
    BACKGROUND=BACKGROUND,
    SURFACE=SURFACE,
    SURFACE_RAISED=SURFACE_RAISED,
    BORDER=BORDER,
    BORDER_STRONG=BORDER_STRONG,
    TEXT=TEXT,
    TEXT_MUTED=TEXT_MUTED,
    TEXT_FAINT=TEXT_FAINT,
    ACCENT=ACCENT,
    ACCENT_HOVER=ACCENT_HOVER,
    ACCENT_PRESSED=ACCENT_PRESSED,
    IMAGE_BACKGROUND=IMAGE_BACKGROUND,
    DISPLAY_STACK=DISPLAY_STACK,
)



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
