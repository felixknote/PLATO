"""The categorical palette should suit the plot's actual background.

PLATO's default categorical palette is tuned for a dark ground (see
palette.py's own docstring): 17 of its 20 colours fall below WCAG's 3:1
minimum contrast for graphical objects against white. Switching Background to
Light/white (or to a theme-following/transparent ground while the app theme
is light) must suggest a palette that is actually legible there, unless the
user has explicitly picked a palette themselves.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from plato.views import palettes  # noqa: E402
from plato.views.explorer import EmbeddingExplorer  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _relative_luminance(hexcolor: str) -> float:
    colour = QColor(hexcolor)

    def lin(v: float) -> float:
        v = v / 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = lin(colour.red()), lin(colour.green()), lin(colour.blue())
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(hex1: str, hex2: str) -> float:
    l1, l2 = _relative_luminance(hex1), _relative_luminance(hex2)
    l1, l2 = max(l1, l2), min(l1, l2)
    return (l1 + 0.05) / (l2 + 0.05)


# -- the palette data itself --------------------------------------------------


def test_deep_palette_is_legible_on_white():
    deep = palettes.get("deep")
    for colour in deep.colours:
        assert _contrast(colour, "#ffffff") >= 3.0, (
            f"{colour} fails WCAG graphics-object contrast (3:1) against white"
        )


def test_default_plato_palette_is_not_claimed_legible_on_white():
    """Documents the actual problem this fix addresses: the app's own
    default categorical scale is not usable on a light ground, which is
    exactly why switching to it must be automatic rather than left for a
    user to discover the hard way."""
    plato = palettes.get("plato")
    passing = [c for c in plato.colours if _contrast(c, "#ffffff") >= 3.0]
    assert len(passing) < len(plato.colours) / 2


# -- categorical_for_background: the pure selection logic --------------------


def test_light_background_selects_deep():
    assert palettes.categorical_for_background("light").key == "deep"


def test_dark_background_selects_plato():
    assert palettes.categorical_for_background("dark").key == "plato"


def test_theme_background_follows_app_theme():
    assert palettes.categorical_for_background("theme", theme_is_dark=True).key == "plato"
    assert palettes.categorical_for_background("theme", theme_is_dark=False).key == "deep"


def test_transparent_background_follows_app_theme_too():
    """Transparent has no ground of its own -- it is composited over
    whatever the app looks like most of the time it is actually viewed."""
    assert palettes.categorical_for_background("transparent", theme_is_dark=True).key == "plato"
    assert palettes.categorical_for_background("transparent", theme_is_dark=False).key == "deep"


# -- explorer wiring: suggestion vs. an explicit user choice ------------------


def _make_explorer(app) -> EmbeddingExplorer:
    """A widget with just enough state for _on_background_changed and
    _on_palette_changed to run their palette-selection logic without a real
    projection: self.result stays None, so _redraw's first line returns
    immediately, and set_background on an EmbeddingScatter/GridView with no
    points is a harmless no-op.
    """
    explorer = EmbeddingExplorer.__new__(EmbeddingExplorer)
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QComboBox

    from plato.views.grid_view import GridView
    from plato.views.scatter import BACKGROUND_CHOICES, EmbeddingScatter

    explorer.result = None
    explorer.frame = None
    explorer._redraw_timer = QTimer()
    explorer._stat_column = None
    explorer._palette_chosen_by_user = False
    explorer.scatter = EmbeddingScatter()
    explorer.grid = GridView()
    explorer.palette_box = QComboBox()
    explorer.background_box = QComboBox()
    for key, label in BACKGROUND_CHOICES:
        explorer.background_box.addItem(label, key)
    return explorer


def test_background_change_suggests_deep_when_no_explicit_choice(app):
    explorer = _make_explorer(app)
    explorer._rebuild_palette_options()
    assert explorer.palette_box.currentData() == "plato"

    index = explorer.background_box.findData("light")
    explorer.background_box.setCurrentIndex(index)
    explorer._on_background_changed()

    assert explorer.palette_box.currentData() == "deep"
    assert not explorer._palette_chosen_by_user, (
        "an automatic suggestion must not be mistaken for the user's own "
        "explicit choice, or the next background change will refuse to "
        "re-suggest anything"
    )


def test_explicit_palette_choice_survives_a_background_change(app):
    explorer = _make_explorer(app)
    explorer._rebuild_palette_options()

    # The user picks Okabe-Ito by hand.
    index = explorer.palette_box.findData("okabe_ito")
    explorer.palette_box.setCurrentIndex(index)
    explorer._on_palette_changed()
    assert explorer._palette_chosen_by_user

    bg_index = explorer.background_box.findData("light")
    explorer.background_box.setCurrentIndex(bg_index)
    explorer._on_background_changed()

    assert explorer.palette_box.currentData() == "okabe_ito", (
        "switching background must never override a palette the user chose"
    )


def test_switching_back_to_dark_suggests_plato_again(app):
    explorer = _make_explorer(app)
    explorer._rebuild_palette_options()

    light_index = explorer.background_box.findData("light")
    explorer.background_box.setCurrentIndex(light_index)
    explorer._on_background_changed()
    assert explorer.palette_box.currentData() == "deep"

    dark_index = explorer.background_box.findData("dark")
    explorer.background_box.setCurrentIndex(dark_index)
    explorer._on_background_changed()
    assert explorer.palette_box.currentData() == "plato"
