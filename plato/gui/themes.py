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
    # The ground a PLOT sits on. Separate from image_background because a
    # scatter is chrome-like (its marks are theme colours) while a micrograph
    # is not (its pixels are the data).
    plot_background: str


# The chrome descends from the logo (scripts/make_logo.py): a deep navy plate
# with a handful of wells glowing cyan and magenta. That artwork already was
# the identity; the interface used to run a generic Windows blue alongside it.
#
# Two hues are now RESERVED, and mean one thing each:
#   cyan    -- selected / active. The accent.
#   magenta -- control wells, and anything flagged. The reference points every
#              other point is judged against.
# Both are held out of the categorical scales (see views/palettes.py), so a
# coloured mark in a plot is never the same hue as an interface state.
# Everything else is low-chroma navy, so colour anywhere on screen is data.
DARK_COLOURS = ThemeColours(
    key=DARK,
    name="Dark",
    # The logo's TILE_BG family. Navy rather than neutral grey: it reads as
    # the dark plate the app is a window onto, and flat greys look muddy
    # next to a greyscale image anyway.
    background="#0a0f1c",
    surface="#111827",
    surface_raised="#18202f",
    border="#222d42",
    border_strong="#2f3d56",
    text="#e6ebf2",
    text_muted="#94a3b8",
    text_faint="#64748b",
    # The logo's cyan, held back from its full #1af8fe: at full saturation a
    # 1 px border of it vibrates against the navy. Raw #1af8fe is reserved
    # for the live lasso stroke alone -- see views/scatter.py.
    accent="#34c7d4",
    accent_hover="#4fdbe8",
    accent_pressed="#1f9fad",
    flag="#ea45cc",
    # Darker than the panel so an image edge reads as an edge, but not black,
    # which makes dark pixels unjudgeable.
    image_background="#05080f",
    plot_background="#05080f",
)

LIGHT_COLOURS = ThemeColours(
    key=LIGHT,
    name="Light",
    # Cool-neutral rather than pure white: a full-white chrome around a
    # greyscale micrograph glares and skews how you judge intensity by eye,
    # which is the same reason the dark theme is not pure black. Tinted
    # toward the same navy as the dark theme so the two read as one product.
    background="#f2f4f8",
    surface="#ffffff",
    surface_raised="#e8ecf3",
    border="#ced6e2",
    border_strong="#aeb9c9",
    text="#101827",
    text_muted="#53607a",
    # 3.31:1 on the page ground. The obvious lighter #8492a8 measured 2.86:1,
    # under the 3:1 floor, and this colour carries hint text.
    text_faint="#7a8799",
    # The same two reserved hues as the dark theme, darkened to hold contrast
    # on a light ground: #34c7d4 on white is 2.05:1 and unusable for a state
    # that has to be noticed. These keep the hue and buy the contrast back.
    accent="#0e7c8c",
    accent_hover="#0f8fa1",
    accent_pressed="#0a616e",
    flag="#a81f8c",
    # Still dark: an image canvas is a viewing surface, not chrome, and a
    # white surround around a fluorescence image destroys the contrast the
    # image is being judged on.
    image_background="#101216",
    # A scatter plot is NOT an image canvas. Its marks are drawn in the
    # theme's own palette on a ground that should match the chrome, so
    # "Follow app theme" gives a light plot in light mode. Keeping it dark
    # here made the light theme look half-applied.
    plot_background="#ffffff",
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
