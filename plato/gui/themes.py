"""Light and dark palettes, and the machinery to switch between them at runtime.

``theme.py`` holds the original dark constants and the big stylesheet. This
adds a second palette and a way to change which one is live.

**The constraint that shapes this design.** Widgets across the app do
``from ..gui.theme import TEXT`` at import time and bake the value into an
f-string stylesheet. Rebinding the module attribute afterwards does nothing to
a widget that has already been built -- its stylesheet string still holds the
old hex. So switching themes needs two things:

1. re-apply the application stylesheet and palette (fixes everything styled
   by the global sheet, which is most of the chrome), and
2. tell widgets that painted themselves with baked-in colours to re-read
   them, via ``restyle()``.

Rather than hunt every such widget, ``apply()`` walks the widget tree and
calls ``restyle()`` on anything that defines it. A widget that does not
define it is one whose appearance comes entirely from the global sheet, and
needs nothing.

The plot background is deliberately NOT tied to the app theme: a dark
interface with a white plot is exactly what you want when the plot is destined
for a figure. See ``plato.views.scatter`` for that setting.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QWidget

DARK = "dark"
LIGHT = "light"


@dataclass(frozen=True, slots=True)
class ThemeColours:
    """Every colour the chrome uses. One instance is live at a time."""

    key: str
    name: str
    background: str
    surface: str
    surface_raised: str
    border: str
    border_strong: str
    text: str
    text_muted: str
    text_faint: str
    accent: str
    accent_hover: str
    accent_pressed: str
    flag: str
    image_background: str


DARK_COLOURS = ThemeColours(
    key=DARK,
    name="Dark",
    # Kept a touch blue: flat greys look muddy next to a greyscale image.
    background="#16181d",
    surface="#1e2127",
    surface_raised="#252932",
    border="#333945",
    border_strong="#3f4653",
    text="#e4e7ec",
    text_muted="#9aa3b2",
    text_faint="#6b7482",
    accent="#4a90d9",
    accent_hover="#5b9fe3",
    accent_pressed="#3d7cbd",
    flag="#e8a33d",
    # Darker than the panel so an image edge reads as an edge, but not black,
    # which makes dark pixels unjudgeable.
    image_background="#0d0f12",
)

LIGHT_COLOURS = ThemeColours(
    key=LIGHT,
    name="Light",
    # Warm-neutral rather than pure white: a full-white chrome around a
    # greyscale micrograph glares and skews how you judge intensity by eye,
    # which is the same reason the dark theme is not pure black.
    background="#f4f5f7",
    surface="#ffffff",
    surface_raised="#eceef1",
    border="#d3d7de",
    border_strong="#b8bec8",
    text="#1c1f25",
    text_muted="#5a6270",
    text_faint="#868e9b",
    # Darkened from the dark theme's accent to keep contrast on a light ground.
    accent="#2b6cb0",
    accent_hover="#3a7cc0",
    accent_pressed="#225a94",
    flag="#b9761a",
    # Still dark: an image canvas is a viewing surface, not chrome, and a
    # white surround around a fluorescence image destroys the contrast the
    # image is being judged on.
    image_background="#101216",
)

THEMES = {DARK: DARK_COLOURS, LIGHT: LIGHT_COLOURS}

_current = DARK_COLOURS


def current() -> ThemeColours:
    """The live palette."""
    return _current


def stylesheet(colours: ThemeColours) -> str:
    """The application stylesheet, built for one palette.

    Generated from the same template as the original dark sheet rather than
    duplicated, so a rule added for one theme applies to both.
    """
    from . import theme as _theme

    return _theme.build_stylesheet(colours)


def qt_palette(colours: ThemeColours) -> QPalette:
    """A QPalette matching ``colours``, for widgets Qt draws itself."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(colours.background))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(colours.text))
    palette.setColor(QPalette.ColorRole.Base, QColor(colours.surface))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(colours.surface_raised))
    palette.setColor(QPalette.ColorRole.Text, QColor(colours.text))
    palette.setColor(QPalette.ColorRole.Button, QColor(colours.surface_raised))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(colours.text))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(colours.accent))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(colours.surface_raised))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(colours.text))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(colours.text_faint))
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(colours.text_faint)
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.ButtonText,
        QColor(colours.text_faint),
    )
    return palette


def apply(key: str, app: QApplication | None = None) -> ThemeColours:
    """Make ``key`` the live theme and restyle everything already built."""
    global _current
    colours = THEMES.get(key, DARK_COLOURS)
    _current = colours

    # Keep the legacy module-level names in step, so code that imported them
    # at module scope at least agrees with the live theme from now on.
    from . import theme as _theme

    _theme.sync_module_constants(colours)

    app = app or QApplication.instance()
    if app is None:
        return colours
    app.setPalette(qt_palette(colours))
    app.setStyleSheet(stylesheet(colours))

    # Widgets that baked colours into their own stylesheets cannot be fixed by
    # the global sheet; give each a chance to rebuild.
    for widget in app.allWidgets():
        restyle = getattr(widget, "restyle", None)
        if callable(restyle):
            try:
                restyle()
            except Exception:  # noqa: BLE001 - one bad widget must not abort the switch
                pass
    return colours


def toggle(app: QApplication | None = None) -> ThemeColours:
    """Switch between light and dark."""
    return apply(LIGHT if _current.key == DARK else DARK, app)


def is_dark() -> bool:
    return _current.key == DARK


def restyle_tree(root: QWidget) -> None:
    """Call ``restyle()`` on ``root`` and every descendant that has one.

    For widgets created *after* a theme change, or built outside the
    application's widget list at the moment ``apply`` ran.
    """
    for widget in [root, *root.findChildren(QWidget)]:
        restyle = getattr(widget, "restyle", None)
        if callable(restyle):
            try:
                restyle()
            except Exception:  # noqa: BLE001
                pass
